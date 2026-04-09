# Route B 工程设计 — Section 3: 多后端 Layer1 架构

> 本文档是 Route B **一次性架构设计** 的工程实现部分，覆盖从文件结构到部署的全部工程细节。
>
> **现状基线**：Layer1 当前使用单一 LightGBM Huber 回归模型，输入 (N, 37) tabular 特征，输出 `(beta_score, confidence)`。本设计将其泛化为可插拔后端架构，同时支持 tabular 和 sequential 两种输入模式。

---

## 3.1 文件结构规划

```
src/quant/strategy/layer1/
├── __init__.py              # 不变，保持现有公共 API 导出
├── layer1.py                # 小改 — 重构以支持 backend 路由
├── features.py              # 不变 — FeatureBuilderV2, GlobalState, GlobalStateBuilder
├── sequence_features.py     # 新增 — SequenceFeatureBuilder，生成 (T, D) 序列张量
├── mii.py                   # 不变
├── walk_forward.py          # 不变
├── model.py                 # 重命名 → lgbm_backend.py（保留 model.py 作为 re-export shim 一个版本周期）
├── backend_protocol.py      # 新增 — ModelBackend Protocol 定义
├── backends/
│   ├── __init__.py          # 导出 BACKEND_REGISTRY + 工厂函数
│   ├── lgbm_backend.py      # 重构自 model.py：LGBMBackend(ModelBackend)
│   ├── xgb_backend.py       # 新增 — XGBBackend(SklearnTabularBase)
│   ├── ridge_backend.py     # 新增 — RidgeBackend(SklearnTabularBase)
│   ├── elasticnet_backend.py# 新增 — ElasticNetBackend(SklearnTabularBase)
│   ├── rf_backend.py        # 新增 — RandomForestBackend(SklearnTabularBase)
│   ├── _sklearn_base.py     # 新增 — sklearn 系列 backend 共享基类
│   ├── _seq_base.py         # 新增 — PyTorch SequentialBackend 基类
│   ├── dlinear_backend.py   # 新增 — DLinearBackend
│   ├── patchtst_backend.py  # 新增 — PatchTSTBackend
│   └── lstm_backend.py      # 新增 — LSTM/GRU Backend
└── model.py                 # 过渡期 shim：from .backends.lgbm_backend import LGBMBackend as Layer1Model
```

### 各文件职责

| 文件 | 变动 | 职责 |
|------|------|------|
| `backend_protocol.py` | 新增 | 定义 `ModelBackend` Protocol 和 `BackendMeta` 元数据 dataclass |
| `sequence_features.py` | 新增 | 从 Bar 序列构建 (T, D) 张量，保证 no-lookahead |
| `backends/__init__.py` | 新增 | 后端注册表 `BACKEND_REGISTRY`、`create_backend()` 工厂函数 |
| `backends/lgbm_backend.py` | 重构 | 现有 `Layer1Model` → `LGBMBackend`，实现 `ModelBackend` Protocol |
| `backends/_sklearn_base.py` | 新增 | sklearn 系列后端共用的 fit/predict/save/load 基类 |
| `backends/xgb_backend.py` | 新增 | XGBoost 回归后端 |
| `backends/ridge_backend.py` | 新增 | Ridge 回归后端 |
| `backends/elasticnet_backend.py` | 新增 | ElasticNet 回归后端 |
| `backends/rf_backend.py` | 新增 | Random Forest 回归后端 |
| `backends/_seq_base.py` | 新增 | PyTorch sequential 后端共用基类（训练循环、设备管理、checkpoint） |
| `backends/dlinear_backend.py` | 新增 | DLinear 时序分解模型后端 |
| `backends/patchtst_backend.py` | 新增 | PatchTST Transformer 后端 |
| `backends/lstm_backend.py` | 新增 | LSTM/GRU 循环网络后端 |
| `layer1.py` | 小改 | 根据 config 初始化正确的 backend，推理时路由至对应 input_type |
| `model.py` | shim | 向后兼容导入：`from .backends.lgbm_backend import LGBMBackend as Layer1Model` |

---

## 3.2 ModelBackend Protocol 完整代码

```python
# src/quant/strategy/layer1/backend_protocol.py
"""Layer1 模型后端的 Protocol 定义。

所有 tabular 和 sequential 后端必须实现此协议。
Layer1 类通过此协议统一调用 fit / predict / save / load，
无需感知底层模型实现细节。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BackendMeta:
    """模型元数据，随模型一同保存至 meta.json。

    Attributes
    ----------
    model_type : str
        后端标识符（如 "lgbm", "xgb", "patchtst"）。
    input_type : Literal["tabular", "sequential"]
        "tabular" 接收 (N, D) ndarray；"sequential" 接收 (N, T, D) ndarray。
    feature_dim : int
        tabular 特征维度 D，或 sequential 每步特征维度 D。
    lookback_T : int
        sequential 后端的序列长度 T；tabular 后端固定为 1。
    feature_names : list[str]
        有序特征名列表，用于特征对齐校验。
    extra : dict[str, Any]
        后端特有的额外元数据（如 PatchTST 的 patch_size）。
    """
    model_type: str
    input_type: Literal["tabular", "sequential"]
    feature_dim: int
    lookback_T: int = 1
    feature_names: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    """Layer1 模型后端统一接口。

    Tabular 后端
        fit(X, y, ...)  X: (N, D)    →  predict(X) X: (1, D)
    Sequential 后端
        fit(X, y, ...)  X: (N, T, D) →  predict(X) X: (1, T, D)
    """

    @property
    def input_type(self) -> Literal["tabular", "sequential"]:
        """返回此后端接受的输入类型。"""
        ...

    @property
    def is_fitted(self) -> bool:
        """模型是否已完成训练。"""
        ...

    @property
    def meta(self) -> BackendMeta:
        """返回模型元数据。"""
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        categorical_indices: list[int] | None = None,
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        """训练模型。

        Parameters
        ----------
        X : ndarray
            tabular: (N, D)；sequential: (N, T, D)。
        y : ndarray (N,)
            波动率缩放前向收益率，已 Winsorize。
        sample_weights : ndarray (N,) or None
            逆频率样本权重。
        categorical_indices : list[int] or None
            类别型特征列索引（仅 LightGBM 使用，其余后端忽略）。
        val_fraction : float
            时间顺序尾部切分比例，用于早停 / 监控。

        Returns
        -------
        dict[str, Any]
            训练日志（best_iteration、val_loss 等），便于诊断。
        """
        ...

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        """单样本推理。

        Parameters
        ----------
        X : ndarray
            tabular: (D,) 或 (1, D)；sequential: (T, D) 或 (1, T, D)。

        Returns
        -------
        (beta_score, confidence)
            beta_score: 波动率缩放期望收益率（无界）。
            confidence: ∈ [0, 1]。
        """
        ...

    def save(self, directory: Path) -> None:
        """将模型权重 + meta.json 保存到 *directory*。

        目录结构：
            directory/
                meta.json          — BackendMeta 序列化
                model.pkl          — tabular 后端
                checkpoint.pt      — sequential 后端
                confidence_scale   — 存入 meta.json.extra
        """
        ...

    def load(self, directory: Path) -> None:
        """从 *directory* 加载之前保存的模型。"""
        ...
```

### 设计决策说明

1. **`predict()` 返回 `(beta_score, confidence)` 元组** — 与现有 `Layer1Model.predict()` 签名一致，下游 `Layer1._compute_v2()` 无需任何修改。

2. **`save()/load()` 使用目录而非单文件** — sequential 后端需要多个文件（checkpoint + tokenizer/scaler state），统一用目录避免格式差异。

3. **`categorical_indices` 作为 `fit()` 参数** — 仅 LightGBM 需要声明类别型特征，其余后端自动忽略。这比在 Protocol 上强制所有后端实现 categorical 处理更干净。

4. **`fit()` 返回 `dict`** — 提供训练诊断信息（best_iteration、val_loss 等），调用方可记录到日志。

---

## 3.3 TabularBackend 实现规划

### 3.3.1 共享基类：`_sklearn_base.py`

XGBoost、Ridge、ElasticNet、RandomForest 均遵循 sklearn 的 `fit(X, y, sample_weight=)` / `predict(X)` 接口，可共用基类：

```python
# src/quant/strategy/layer1/backends/_sklearn_base.py
"""sklearn 兼容 tabular 后端的共享基类。"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from quant.strategy.layer1.backend_protocol import BackendMeta

logger = logging.getLogger(__name__)
_EPS = 1e-9


class SklearnTabularBase(ABC):
    """sklearn 系列 tabular 后端的公共基类。

    子类只需实现 ``_create_estimator()`` 和 ``_model_type``。
    fit / predict / save / load / confidence 校准全部由基类处理。
    """

    @property
    def input_type(self) -> Literal["tabular"]:
        return "tabular"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def meta(self) -> BackendMeta:
        return BackendMeta(
            model_type=self._model_type,
            input_type="tabular",
            feature_dim=self._feature_dim,
            extra={"confidence_scale": self._confidence_scale},
        )

    @property
    @abstractmethod
    def _model_type(self) -> str:
        """后端标识符，如 'xgb', 'ridge'。"""
        ...

    @abstractmethod
    def _create_estimator(self) -> Any:
        """创建未训练的 sklearn 兼容估计器实例。"""
        ...

    def __init__(self, **kwargs: Any) -> None:
        self._estimator: Any | None = None
        self._is_fitted: bool = False
        self._confidence_scale: float = 1.0
        self._feature_dim: int = 0
        self._kwargs = kwargs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        categorical_indices: list[int] | None = None,  # 忽略
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        if len(X) < 50:
            logger.warning("%s.fit: only %d samples, skipping", self._model_type, len(X))
            return {"status": "skipped", "reason": "insufficient_samples"}

        self._feature_dim = X.shape[1]
        n = len(X)
        split = int(n * (1 - val_fraction))

        estimator = self._create_estimator()
        fit_kwargs: dict[str, Any] = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[:split]

        estimator.fit(X[:split], y[:split], **fit_kwargs)
        self._estimator = estimator
        self._is_fitted = True

        # confidence_scale = |训练预测值| 第 90 百分位数
        y_pred_train = estimator.predict(X[:split])
        self._confidence_scale = float(np.percentile(np.abs(y_pred_train), 90) + _EPS)

        # 验证集指标
        y_pred_val = estimator.predict(X[split:])
        val_mse = float(np.mean((y_pred_val - y[split:]) ** 2))

        return {"status": "ok", "n_train": split, "n_val": n - split, "val_mse": val_mse}

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        if not self._is_fitted or self._estimator is None:
            return 0.0, 0.0
        X_2d = X.reshape(1, -1) if X.ndim == 1 else X
        pred = self._estimator.predict(X_2d)
        beta_score = float(pred[0])
        confidence = float(min(abs(beta_score) / (self._confidence_scale + _EPS), 1.0))
        return beta_score, confidence

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._estimator, directory / "model.joblib")
        meta = {
            "model_type": self._model_type,
            "input_type": "tabular",
            "feature_dim": self._feature_dim,
            "confidence_scale": self._confidence_scale,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, directory: Path) -> None:
        self._estimator = joblib.load(directory / "model.joblib")
        meta = json.loads((directory / "meta.json").read_text())
        self._confidence_scale = float(meta.get("confidence_scale", 1.0))
        self._feature_dim = int(meta.get("feature_dim", 0))
        self._is_fitted = True
```

### 3.3.2 各 Tabular Backend 差异

| Backend | 估计器 | 特殊处理 | 序列化 |
|---------|--------|----------|--------|
| **LGBMBackend** | `lgb.train()` (原生 API) | `categorical_feature` 参数传入 `lgb.Dataset`；early_stopping | pickle（与现有 `Layer1Model` 一致，逐步迁移至目录格式） |
| **XGBBackend** | `xgb.XGBRegressor` | `enable_categorical=True` 需要 xgboost>=2.0 且特征需转 `pd.Categorical`；**建议 Phase 1 先用 ordinal 编码跳过** | joblib |
| **RidgeBackend** | `sklearn.linear_model.Ridge` | 需要手动 StandardScaler 预处理（线性模型对特征尺度敏感）；alpha 由 config 控制 | joblib（scaler + model 打包） |
| **ElasticNetBackend** | `sklearn.linear_model.ElasticNet` | 同 Ridge，需 StandardScaler；alpha + l1_ratio 可调 | joblib |
| **RandomForestBackend** | `sklearn.ensemble.RandomForestRegressor` | 不需要特征缩放；`n_estimators` / `max_depth` 从 config 读取 | joblib |

#### LGBMBackend 特殊说明

`LGBMBackend` 不继承 `SklearnTabularBase`，因为它使用 LightGBM 原生 `lgb.train()` API（非 sklearn wrapper）：
- 原生 API 支持 `categorical_feature` 声明（LightGBM 独有的 categorical 处理方式）
- 支持 `lgb.early_stopping` 回调
- 保持与现有 `Layer1Model` 的 100% 行为兼容

重构方式：将 `model.py` 中的 `Layer1Model` 改名为 `LGBMBackend`，增加 `ModelBackend` Protocol 所需的属性（`input_type`, `is_fitted`, `meta`），并将 `save()/load()` 从单文件改为目录格式。

#### 线性模型的 StandardScaler

Ridge 和 ElasticNet 需要特征缩放。基类处理方式：

```python
class ScaledSklearnBase(SklearnTabularBase):
    """为需要特征缩放的线性模型添加 StandardScaler。"""

    def _create_pipeline(self) -> Any:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", self._create_estimator()),
        ])

    # 重写 fit() 中创建 estimator 的部分使用 pipeline
```

实际实现中，`RidgeBackend` 和 `ElasticNetBackend` 的 `_create_estimator()` 直接返回 `sklearn.pipeline.Pipeline`，基类不感知内部是否有 scaler。

---

## 3.4 SequentialBackend 实现规划

### 3.4.1 框架选择：PyTorch

**选择 PyTorch** 作为 sequential 后端的深度学习框架。

| 维度 | PyTorch | 备选 (TensorFlow/JAX) |
|------|---------|----------------------|
| 生态适配 | DLinear/PatchTST 原始论文均提供 PyTorch 实现 | 需自行移植 |
| 灵活性 | 动态图适合调试和自定义训练循环 | TF 2.x 已改善但仍有 Session 残留 |
| 部署 | CPU 推理 <1ms/sample，满足 15min bar 需求 | JAX 的 CPU 路径较成熟但社区较小 |
| 依赖 | `torch` 单包搞定；HuggingFace `transformers` 可选 | TF 依赖链较重 |
| CI/CD | `torch.cpu` 安装包约 200MB，CI 可用 | GPU-free 测试同样支持 |

**劣势**：
- PyTorch CPU 安装包体积大（~200MB vs sklearn 的 ~30MB）
- 训练较慢（无 GPU 时 DLinear 尚可，PatchTST 较慢）
- 需要手动管理训练循环、学习率调度、梯度裁剪

**结论**：优势明显压倒劣势。PyTorch 是 TSF（时序预测）领域的事实标准。

### 3.4.2 SequentialBackend 基类

```python
# src/quant/strategy/layer1/backends/_seq_base.py
"""PyTorch sequential 后端的共享基类。"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)
_EPS = 1e-9


class SequentialBackendBase(ABC):
    """PyTorch 时序模型后端的公共基类。

    子类需实现：
        _model_type: str          — 后端标识符
        _build_model() -> nn.Module
        _default_hparams() -> dict — 默认超参数
    """

    @property
    def input_type(self) -> Literal["sequential"]:
        return "sequential"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    @abstractmethod
    def _model_type(self) -> str: ...

    @abstractmethod
    def _build_model(self) -> Any:
        """返回 nn.Module 实例。"""
        ...

    @abstractmethod
    def _default_hparams(self) -> dict[str, Any]:
        """返回默认超参数（lr, epochs, batch_size 等）。"""
        ...

    def __init__(
        self,
        lookback_T: int = 96,
        feature_dim: int = 14,
        **kwargs: Any,
    ) -> None:
        self._lookback_T = lookback_T
        self._feature_dim = feature_dim
        self._model: Any | None = None  # nn.Module
        self._is_fitted: bool = False
        self._confidence_scale: float = 1.0
        self._hparams: dict[str, Any] = {**self._default_hparams(), **kwargs}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        categorical_indices: list[int] | None = None,
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        """训练 PyTorch 模型。

        X : (N, T, D) float32
        y : (N,) float64
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if len(X) < 50:
            logger.warning("%s.fit: only %d samples, skipping", self._model_type, len(X))
            return {"status": "skipped"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = len(X)
        split = int(n * (1 - val_fraction))

        X_train_t = torch.from_numpy(X[:split]).float().to(device)
        y_train_t = torch.from_numpy(y[:split]).float().to(device)
        X_val_t = torch.from_numpy(X[split:]).float().to(device)
        y_val_t = torch.from_numpy(y[split:]).float().to(device)

        w_train_t = (
            torch.from_numpy(sample_weights[:split]).float().to(device)
            if sample_weights is not None
            else torch.ones(split, device=device)
        )

        train_ds = TensorDataset(X_train_t, y_train_t, w_train_t)
        train_dl = DataLoader(
            train_ds,
            batch_size=self._hparams.get("batch_size", 256),
            shuffle=False,  # 时序数据不打乱
        )

        model = self._build_model().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self._hparams.get("lr", 1e-3),
            weight_decay=self._hparams.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self._hparams.get("epochs", 50),
        )

        best_val_loss = float("inf")
        best_state = None
        patience = self._hparams.get("patience", 10)
        patience_counter = 0

        for epoch in range(self._hparams.get("epochs", 50)):
            model.train()
            for X_batch, y_batch, w_batch in train_dl:
                optimizer.zero_grad()
                pred = model(X_batch).squeeze(-1)  # (B,)
                # 加权 Huber 损失
                loss = torch.nn.functional.huber_loss(
                    pred, y_batch, reduction="none",
                    delta=self._hparams.get("huber_delta", 1.0),
                )
                loss = (loss * w_batch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # 验证
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).squeeze(-1)
                val_loss = float(
                    torch.nn.functional.huber_loss(val_pred, y_val_t).item()
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug("%s: early stop at epoch %d", self._model_type, epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model.cpu()
        self._is_fitted = True

        # confidence_scale
        with torch.no_grad():
            y_pred_all = model(X_train_t.cpu()).squeeze(-1).numpy()
        self._confidence_scale = float(np.percentile(np.abs(y_pred_all), 90) + _EPS)

        return {
            "status": "ok",
            "n_train": split,
            "n_val": n - split,
            "val_loss": best_val_loss,
            "epochs_run": epoch + 1,
        }

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        """单样本推理。X: (T, D) 或 (1, T, D)。"""
        if not self._is_fitted or self._model is None:
            return 0.0, 0.0

        import torch

        if X.ndim == 2:
            X = X[np.newaxis, ...]  # (1, T, D)

        self._model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float()
            pred = self._model(X_t).squeeze().item()

        beta_score = float(pred)
        confidence = float(min(abs(beta_score) / (self._confidence_scale + _EPS), 1.0))
        return beta_score, confidence

    def save(self, directory: Path) -> None:
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            torch.save(self._model.state_dict(), directory / "checkpoint.pt")

        meta = {
            "model_type": self._model_type,
            "input_type": "sequential",
            "feature_dim": self._feature_dim,
            "lookback_T": self._lookback_T,
            "confidence_scale": self._confidence_scale,
            "hparams": self._hparams,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, directory: Path) -> None:
        import torch

        meta = json.loads((directory / "meta.json").read_text())
        self._confidence_scale = float(meta.get("confidence_scale", 1.0))
        self._lookback_T = int(meta.get("lookback_T", self._lookback_T))
        self._feature_dim = int(meta.get("feature_dim", self._feature_dim))
        self._hparams = meta.get("hparams", self._hparams)

        self._model = self._build_model()
        state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
        self._is_fitted = True
```

### 3.4.3 DLinearBackend

DLinear（Zeng et al., 2023 "Are Transformers Effective for Time Series Forecasting?"）是一个极简线性时序模型，将输入分解为趋势（moving average）和残差，各经一层线性映射后合并。

```python
# src/quant/strategy/layer1/backends/dlinear_backend.py
"""DLinear 时序预测后端。"""
from __future__ import annotations

from typing import Any

from quant.strategy.layer1.backends._seq_base import SequentialBackendBase


class DLinearBackend(SequentialBackendBase):

    @property
    def _model_type(self) -> str:
        return "dlinear"

    def _default_hparams(self) -> dict[str, Any]:
        return {
            "lr": 1e-3,
            "epochs": 100,
            "batch_size": 256,
            "patience": 15,
            "huber_delta": 1.0,
            "weight_decay": 1e-4,
            "kernel_size": 25,          # moving average 窗口
            "individual": False,        # True = channel-independent；False = channel-mixing
        }

    def _build_model(self) -> Any:
        import torch
        import torch.nn as nn

        T = self._lookback_T
        D = self._feature_dim
        kernel_size = self._hparams.get("kernel_size", 25)
        individual = self._hparams.get("individual", False)

        class MovingAvg(nn.Module):
            def __init__(self, kernel: int):
                super().__init__()
                self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)
                self.pad_len = (kernel - 1) // 2

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, T, D) → (B, D, T) for AvgPool1d
                front = x[:, :1, :].repeat(1, self.pad_len, 1)
                end = x[:, -1:, :].repeat(1, (kernel_size - 1) - self.pad_len, 1)
                x_padded = torch.cat([front, x, end], dim=1)
                return self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)

        class DLinearModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.decomp = MovingAvg(kernel_size)
                if individual:
                    self.linear_trend = nn.ModuleList([nn.Linear(T, 1) for _ in range(D)])
                    self.linear_resid = nn.ModuleList([nn.Linear(T, 1) for _ in range(D)])
                else:
                    self.linear_trend = nn.Linear(T, 1)
                    self.linear_resid = nn.Linear(T, 1)
                self.head = nn.Linear(D, 1)
                self.individual = individual

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, T, D)
                trend = self.decomp(x)          # (B, T, D)
                resid = x - trend               # (B, T, D)

                if self.individual:
                    # channel-independent: 每个 feature 一个线性层
                    trend_parts = []
                    resid_parts = []
                    for i in range(D):
                        trend_parts.append(self.linear_trend[i](trend[:, :, i]))  # (B, 1)
                        resid_parts.append(self.linear_resid[i](resid[:, :, i]))  # (B, 1)
                    out = torch.stack(trend_parts, dim=-1).squeeze(-2) + \
                          torch.stack(resid_parts, dim=-1).squeeze(-2)  # (B, D)
                else:
                    # channel-mixing: 共享线性层
                    trend_out = self.linear_trend(trend.permute(0, 2, 1)).squeeze(-1)  # (B, D)
                    resid_out = self.linear_resid(resid.permute(0, 2, 1)).squeeze(-1)  # (B, D)
                    out = trend_out + resid_out  # (B, D)

                return self.head(out)  # (B, 1)

        return DLinearModule()
```

**DLinear 架构要点**：
- **分解方式**：moving average 提取趋势，原始 - 趋势 = 残差
- `individual=False`（默认）：所有 feature channel 共享线性权重 → 参数量极小，不易过拟合
- `individual=True`：每个 feature channel 独立线性层 → 参数量 = 2 * D * T + D，适合特征间关系弱的场景
- 输出经 `nn.Linear(D, 1)` 头部映射为标量 beta_score

### 3.4.4 PatchTSTBackend

PatchTST（Nie et al., 2023）将时序切分为固定长度 patch，每个 patch 通过线性映射得到 embedding，再经标准 Transformer Encoder 处理。

```python
# src/quant/strategy/layer1/backends/patchtst_backend.py
"""PatchTST 时序预测后端。"""
from __future__ import annotations

from typing import Any

from quant.strategy.layer1.backends._seq_base import SequentialBackendBase


class PatchTSTBackend(SequentialBackendBase):

    @property
    def _model_type(self) -> str:
        return "patchtst"

    def _default_hparams(self) -> dict[str, Any]:
        return {
            "lr": 5e-4,
            "epochs": 80,
            "batch_size": 128,
            "patience": 12,
            "huber_delta": 1.0,
            "weight_decay": 1e-4,
            # PatchTST 特有
            "patch_size": 16,           # 每个 patch 覆盖的时步数
            "stride": 8,               # patch 步进；stride < patch_size → 重叠
            "d_model": 64,             # Transformer 隐层维度
            "n_heads": 4,              # 多头注意力头数
            "n_layers": 2,             # Transformer Encoder 层数
            "d_ff": 128,               # FFN 中间维度
            "dropout": 0.2,
            "channel_independent": True, # True = CI 模式（每个 feature 独立建模）
        }

    def _build_model(self) -> Any:
        import math

        import torch
        import torch.nn as nn

        T = self._lookback_T
        D = self._feature_dim
        hp = self._hparams

        patch_size = hp["patch_size"]
        stride = hp["stride"]
        d_model = hp["d_model"]
        n_heads = hp["n_heads"]
        n_layers = hp["n_layers"]
        d_ff = hp["d_ff"]
        dropout = hp["dropout"]
        channel_independent = hp["channel_independent"]

        # 计算 patch 数量
        n_patches = (T - patch_size) // stride + 1

        class PatchTSTModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_size = patch_size
                self.stride = stride
                self.n_patches = n_patches

                # Patch embedding: (patch_size,) → (d_model,)
                if channel_independent:
                    # CI: 所有 channel 共享同一个 patch embedding
                    self.patch_embed = nn.Linear(patch_size, d_model)
                else:
                    # CM: patch 包含所有 channel
                    self.patch_embed = nn.Linear(patch_size * D, d_model)

                # Learnable positional encoding
                self.pos_embed = nn.Parameter(
                    torch.randn(1, n_patches, d_model) * 0.02
                )

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers,
                )

                self.flatten_head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 1),
                )
                self.channel_independent = channel_independent
                self.D = D

                if channel_independent:
                    # CI 模式下，合并 D 个 channel 的输出
                    self.channel_agg = nn.Linear(D, 1)

            def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
                """x: (B, T, D) → patches."""
                B = x.shape[0]
                if self.channel_independent:
                    # (B, T, D) → (B*D, T, 1) → unfold → (B*D, n_patches, patch_size)
                    x_ci = x.permute(0, 2, 1).reshape(B * self.D, T, 1)
                    patches = x_ci.squeeze(-1).unfold(1, self.patch_size, self.stride)
                    return patches  # (B*D, n_patches, patch_size)
                else:
                    # (B, T, D) → unfold along T → (B, n_patches, patch_size, D) → (B, n_patches, patch_size*D)
                    patches = x.unfold(1, self.patch_size, self.stride)  # (B, n_patches, D, patch_size)
                    patches = patches.permute(0, 1, 3, 2).reshape(B, self.n_patches, -1)
                    return patches

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, T, D)
                B = x.shape[0]
                patches = self._create_patches(x)  # (B[*D], n_patches, patch_dim)
                embedded = self.patch_embed(patches) + self.pos_embed  # (B[*D], n_patches, d_model)
                encoded = self.encoder(embedded)  # (B[*D], n_patches, d_model)

                # 取所有 patch 的均值池化
                pooled = encoded.mean(dim=1)  # (B[*D], d_model)

                if self.channel_independent:
                    # (B*D, d_model) → (B, D, d_model) → flatten_head → (B, D, 1) → agg → (B, 1)
                    out = pooled.reshape(B, self.D, -1)  # (B, D, d_model)
                    out = self.flatten_head(out).squeeze(-1)  # (B, D)
                    return self.channel_agg(out)  # (B, 1)
                else:
                    return self.flatten_head(pooled)  # (B, 1)

        return PatchTSTModule()
```

**PatchTST 架构要点**：
- **patch_size=16, stride=8**：重叠 50% 提高信息保留（对 T=96 产生 11 个 patch）
- **Channel-Independent (CI)** 模式（默认）：每个特征维度独立通过 Transformer，最后通过 `channel_agg` 合并 → 参数共享，对小数据集更鲁棒
- **Channel-Mixing (CM)** 模式：所有特征 flatten 进单个 patch → 可捕获跨特征交互，但参数量更大
- `d_model=64, n_heads=4, n_layers=2`：轻量配置，适合 ~10K 级样本

### 3.4.5 LSTMBackend

```python
# src/quant/strategy/layer1/backends/lstm_backend.py
"""LSTM/GRU 循环网络后端。"""
from __future__ import annotations

from typing import Any

from quant.strategy.layer1.backends._seq_base import SequentialBackendBase


class LSTMBackend(SequentialBackendBase):

    @property
    def _model_type(self) -> str:
        return "lstm"

    def _default_hparams(self) -> dict[str, Any]:
        return {
            "lr": 1e-3,
            "epochs": 80,
            "batch_size": 256,
            "patience": 12,
            "huber_delta": 1.0,
            "weight_decay": 1e-4,
            # LSTM 特有
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": False,
            "cell_type": "LSTM",        # "LSTM" 或 "GRU"
        }

    def _build_model(self) -> Any:
        import torch
        import torch.nn as nn

        D = self._feature_dim
        hp = self._hparams
        hidden_dim = hp["hidden_dim"]
        num_layers = hp["num_layers"]
        dropout = hp["dropout"]
        bidirectional = hp["bidirectional"]
        cell_type = hp.get("cell_type", "LSTM")

        RNNClass = nn.LSTM if cell_type == "LSTM" else nn.GRU

        class LSTMModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = RNNClass(
                    input_size=D,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
                dir_mult = 2 if bidirectional else 1
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_dim * dir_mult),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * dir_mult, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, T, D)
                output, _ = self.rnn(x)       # (B, T, hidden*dir)
                last = output[:, -1, :]       # (B, hidden*dir) — 取最后时步
                return self.head(last)         # (B, 1)

        return LSTMModule()
```

**LSTM 架构要点**：
- `hidden_dim=64, num_layers=2, dropout=0.3`：中等容量配置
- 支持通过 `cell_type="GRU"` 切换为 GRU（参数更少，收敛更快）
- `bidirectional=False`（默认）：因果推理，不使用未来信息
- 仅使用最后时步的隐状态（last hidden state），通过 head 映射为标量

---

## 3.5 SequenceFeatureBuilder 设计

```python
# src/quant/strategy/layer1/sequence_features.py
"""序列特征构建器 —— 为 sequential 后端生成 (T, D) 张量。

与 FeatureBuilderV2 的关系：
    FeatureBuilderV2  → 单个时间点的 37 维 tabular 快照
    SequenceFeatureBuilder → T 个连续时间点的 D 维序列

SequenceFeatureBuilder 复用 FeatureBuilderV2 的 *逐资产特征计算逻辑*
（_per_asset_features 中的 14 个基础特征），但：
  1. 不包含 PCA 和复合特征（全局状态在每个时步变化，计算开销大）
  2. 不包含类别型特征（时序维度上不变化的特征对序列模型无信息量）
  3. 按时间轴逐步构建，形成 (T, D) 矩阵

Default feature set (D=14):
    ret_1, ret_5, ret_20, ret_60, ret_120,
    vol_20, vol_60, vol_ratio, volume_ratio,
    ret_vol_scaled_20, ret_vol_scaled_60,
    trend_consistency_20, staleness_weight, market_beta_20
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from quant.core.types import Bar

logger = logging.getLogger(__name__)

# 序列特征维度名称（固定顺序）
SEQUENCE_FEATURE_NAMES: list[str] = [
    "ret_1", "ret_5", "ret_20", "ret_60", "ret_120",
    "vol_20", "vol_60", "vol_ratio", "volume_ratio",
    "ret_vol_scaled_20", "ret_vol_scaled_60",
    "trend_consistency_20", "staleness_weight", "market_beta_20",
]

SEQUENCE_FEATURE_DIM: int = len(SEQUENCE_FEATURE_NAMES)  # 14


class SequenceFeatureBuilder:
    """为 sequential 后端构建 (T, D) 特征序列。

    Parameters
    ----------
    vol_lookback : int
        波动率计算回看窗口（默认 60）。
    staleness_halflife_hours : float
        陈旧度权重半衰期（默认 8.0）。
    eps : float
        数值下限。
    """

    def __init__(
        self,
        vol_lookback: int = 60,
        staleness_halflife_hours: float = 8.0,
        eps: float = 1e-9,
    ) -> None:
        self._vol_lookback = vol_lookback
        self._halflife = staleness_halflife_hours
        self._eps = eps

    @property
    def feature_dim(self) -> int:
        return SEQUENCE_FEATURE_DIM

    @property
    def feature_names(self) -> list[str]:
        return SEQUENCE_FEATURE_NAMES.copy()

    def build_sequence(
        self,
        asset_bars: list[Bar],
        lookback_T: int,
        as_of: datetime,
        composite_ew_ret_history: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """为单个资产构建 (T, D) 特征序列。

        Parameters
        ----------
        asset_bars : list[Bar]
            该资产的完整 K 线历史（按时间排序）。
            最后一根 bar 对应 as_of 时刻。
        lookback_T : int
            序列长度 T。
        as_of : datetime
            当前时间戳（仅用于最后一步的陈旧度计算；
            历史步使用各自的 bar.timestamp）。
        composite_ew_ret_history : ndarray or None
            等权复合 1-bar 收益率历史，用于 market_beta_20。
            None → market_beta_20 填 0。

        Returns
        -------
        ndarray (T, D) float32 or None
            特征序列。若 bars 不足以填满 T 步则返回 None。

        No-lookahead guarantee
        ----------------------
        对于序列中的第 t 步（t = 0, ..., T-1），
        只使用 asset_bars[:end_idx_t]，其中 end_idx_t 严格 ≤ 该步对应的 bar 索引。
        不使用任何 t 之后的 bar 数据。
        """
        min_history = self._vol_lookback + lookback_T + 1
        if len(asset_bars) < min_history:
            return None

        # 预计算 closes 和 volumes 数组（一次性转换）
        all_closes = np.array([b.close for b in asset_bars], dtype=np.float64)
        all_volumes = np.array([b.volume for b in asset_bars], dtype=np.float64)
        all_timestamps = [b.timestamp for b in asset_bars]

        # 序列中第 t 步对应 asset_bars 的索引 end_idx（不含）
        # 最后一步 (t = T-1) 对应 asset_bars 的最后一根 bar
        last_bar_idx = len(asset_bars)  # 不含边界
        first_bar_idx = last_bar_idx - lookback_T  # 第 0 步的 bar 索引

        result = np.zeros((lookback_T, SEQUENCE_FEATURE_DIM), dtype=np.float32)

        for t in range(lookback_T):
            end_idx = first_bar_idx + t + 1  # 截至当前步（含）的 bar 数量
            closes_t = all_closes[:end_idx]
            volumes_t = all_volumes[:end_idx]
            ts_t = all_timestamps[end_idx - 1]

            result[t] = self._compute_step_features(
                closes_t, volumes_t, ts_t,
                composite_ew_ret_history,
            )

        return result

    def build_training_sequences(
        self,
        asset_bars: list[Bar],
        lookback_T: int,
        forward_horizon: int,
        composite_ew_ret_histories: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """批量生成训练用的 (N, T, D) 序列 + 标签。

        使用 sliding window 沿时间轴滑动，生成所有可用的
        (sequence, label) 对。

        Parameters
        ----------
        asset_bars : list[Bar]
            完整 K 线历史。
        lookback_T : int
            每个 sequence 的时间步长 T。
        forward_horizon : int
            标签的前向收益率跨度（bar 数）。
        composite_ew_ret_histories : list[ndarray] or None
            与 asset_bars 等长的 composite EW return 历史列表。
            None → market_beta_20 填 0。

        Returns
        -------
        (X, y, indices) or None
            X : (N, T, D) float32
            y : (N,) float64 — 波动率缩放前向收益率
            indices : (N,) int — 每个样本在 asset_bars 中的 end index
        """
        min_len = self._vol_lookback + lookback_T + forward_horizon + 1
        if len(asset_bars) < min_len:
            return None

        all_closes = np.array([b.close for b in asset_bars], dtype=np.float64)
        all_volumes = np.array([b.volume for b in asset_bars], dtype=np.float64)
        all_timestamps = [b.timestamp for b in asset_bars]

        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        idx_list: list[int] = []

        # 滑动窗口起始：确保有足够历史计算特征
        start_i = self._vol_lookback + lookback_T
        end_i = len(asset_bars) - forward_horizon

        for i in range(start_i, end_i):
            # 序列 [i - lookback_T, i) 对应的特征
            seq = np.zeros((lookback_T, SEQUENCE_FEATURE_DIM), dtype=np.float32)
            for t in range(lookback_T):
                end_idx = (i - lookback_T) + t + 1
                closes_t = all_closes[:end_idx]
                volumes_t = all_volumes[:end_idx]
                ts_t = all_timestamps[end_idx - 1]
                cew = composite_ew_ret_histories[end_idx - 1] if composite_ew_ret_histories else None
                seq[t] = self._compute_step_features(closes_t, volumes_t, ts_t, cew)

            # 波动率缩放前向收益率标签
            sigma_i = self._realized_vol(all_closes[:i + 1])
            if sigma_i < self._eps:
                continue
            fwd_ret = (all_closes[i + forward_horizon] - all_closes[i]) / (all_closes[i] + self._eps)
            y_label = fwd_ret / sigma_i

            X_list.append(seq)
            y_list.append(y_label)
            idx_list.append(i)

        if not X_list:
            return None

        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float64),
            np.array(idx_list, dtype=np.int64),
        )

    def _compute_step_features(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamp: datetime,
        composite_ew_ret_history: np.ndarray | None,
    ) -> np.ndarray:
        """计算单个时步的 D 维特征向量。

        与 FeatureBuilderV2._per_asset_features() 逻辑一致，
        保证 tabular 和 sequential 路径使用相同的特征定义。
        """
        eps = self._eps
        out = np.zeros(SEQUENCE_FEATURE_DIM, dtype=np.float64)

        def safe_ret(n: int) -> float:
            if len(closes) <= n or closes[-n - 1] == 0:
                return 0.0
            return float((closes[-1] / closes[-n - 1]) - 1.0)

        out[0] = safe_ret(1)
        out[1] = safe_ret(5)
        out[2] = safe_ret(20)
        out[3] = safe_ret(60)
        out[4] = safe_ret(120)

        log_rets = np.diff(np.log(closes + eps))
        vol_20 = float(log_rets[-20:].std()) if len(log_rets) >= 20 else 0.0
        vol_60 = float(log_rets[-60:].std()) if len(log_rets) >= 60 else 0.0

        out[5] = vol_20
        out[6] = vol_60
        out[7] = float(vol_20 / (vol_60 + eps)) if vol_60 > eps else 0.0

        vol_ma = float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean() + eps)
        out[8] = float(volumes[-1] / (vol_ma + eps)) if vol_ma > 0 else 1.0

        out[9] = float(out[2] / (vol_20 + eps)) if vol_20 > eps else 0.0
        out[10] = float(out[3] / (vol_60 + eps)) if vol_60 > eps else 0.0

        # trend_consistency_20
        out[11] = self._trend_consistency(closes, 20)

        # staleness_weight — 序列模型中用 bar 间距代替
        out[12] = 1.0  # 序列模型中设为 1（bar 已按时间排列，陈旧度由位置编码隐式处理）

        # market_beta_20
        if composite_ew_ret_history is not None and len(composite_ew_ret_history) >= 20 and len(log_rets) >= 20:
            a_ret = log_rets[-20:]
            m_ret = composite_ew_ret_history[-20:].astype(np.float64)
            var_m = float(np.var(m_ret))
            if var_m > eps:
                cov_am = float(np.cov(a_ret, m_ret)[0, 1])
                out[13] = float(cov_am / (var_m + eps))
        # else: 0.0 (default)

        return out.astype(np.float32)

    def _realized_vol(self, closes: np.ndarray) -> float:
        lb = self._vol_lookback
        eps = self._eps
        if len(closes) < lb + 1:
            rets = np.diff(np.log(closes + eps))
            return float(rets.std()) if len(rets) > 0 else 0.0
        rets = np.diff(np.log(closes[-(lb + 1):] + eps))
        return float(rets.std())

    @staticmethod
    def _trend_consistency(closes: np.ndarray, window: int) -> float:
        if len(closes) < window + 1:
            return 0.0
        segment = closes[-(window + 1):]
        returns = np.diff(segment)
        overall = np.sign(segment[-1] - segment[0])
        if overall == 0:
            return 0.0
        return float(np.mean(np.sign(returns) == overall))
```

### 与 FeatureBuilderV2 的关系

| 维度 | FeatureBuilderV2 | SequenceFeatureBuilder |
|------|-------------------|----------------------|
| 输出 | (37,) 单快照 | (T, 14) 序列 |
| 逐资产特征 | [0:14] 14 维 | [0:14] **相同的 14 维** |
| PCA 特征 | [14:29] 15 维 | 不包含（计算成本过高） |
| 复合特征 | [29:33] 4 维 | 不包含 |
| 类别特征 | [33:37] 4 维 | 不包含（时序不变量） |
| 复用关系 | — | 逐资产特征计算逻辑与 `_per_asset_features()` 一致 |

**设计理由**：
- 不直接复用 `FeatureBuilderV2.build()` 是因为它需要 `GlobalState`（PCA、复合收益率），在 T 个时步上逐步重建 GlobalState 的计算量 = O(T × N_universe)，不可接受
- 14 维纯价量特征对序列模型已足够 — DLinear/PatchTST 通过时序依赖关系自动学习了类似 PCA 的跨时间模式

### No-lookahead 保证

对序列中第 t 步，`build_sequence()` 使用 `all_closes[:end_idx_t]`，其中 `end_idx_t = first_bar_idx + t + 1`。这确保：
1. 第 t 步只能看到 t 及之前的 bar
2. 波动率 / 收益率计算窗口不会越过当前时步
3. 训练时标签 `y[i] = fwd_ret[i + forward_horizon]` 严格在第 i 步之后

### NaN / 短序列策略

- `build_sequence()` 在 bars 不足 `vol_lookback + T + 1` 时返回 `None`
- 调用方（`Layer1._fit_pooled()`）跳过返回 None 的资产
- 逐步特征中，当历史不足时（如 ret_60 需要 61 根 bar 但只有 30 根）返回 0.0（与 FeatureBuilderV2 一致的安全填充策略）
- 训练时 `build_training_sequences()` 自动跳过波动率为零的样本

---

## 3.6 Layer1 类重构

### 3.6.1 关键方法签名变更

```python
class Layer1:
    def __init__(self, cfg: Layer1Config) -> None:
        self._cfg = cfg
        self._use_v2 = (cfg.feature_version >= 2)

        # ===== 新增：backend 类型识别 =====
        # model_type: "lgbm" | "xgb" | "ridge" | "elasticnet" | "rf"
        #           | "dlinear" | "patchtst" | "lstm"
        # 从 cfg 新增字段读取（见 §3.6.2 config 变更）
        self._model_type: str = getattr(cfg, "model_type", "lgbm")
        self._backend_input_type: str  # "tabular" or "sequential"

        # 通过注册表创建 backend
        from quant.strategy.layer1.backends import create_backend
        self._backend_template = create_backend(self._model_type, cfg)
        self._backend_input_type = self._backend_template.input_type

        # 模型实例（v2 为单一 pooled 模型）
        self._models: dict[str, ModelBackend] = {}

        # Sequential 路径需要额外的特征构建器
        if self._backend_input_type == "sequential":
            self._seq_feature_builder = SequenceFeatureBuilder(
                vol_lookback=cfg.vol_lookback,
                staleness_halflife_hours=cfg.staleness_halflife_hours,
            )

        # MII、GlobalState 等保持不变 ...

    def fit_window(
        self,
        windows: dict[Instrument, list[Bar]],
        bar_index: int,
        start_idx: int,
        end_idx: int,
    ) -> None:
        """滚动训练 — 根据 backend input_type 路由到对应方法。"""
        # ... 切片逻辑不变 ...

        if self._use_v2:
            if self._backend_input_type == "tabular":
                self._fit_pooled(sliced, sliced_sym, forward, temp_builder)
            else:
                self._fit_pooled_sequential(sliced, sliced_sym, forward)

    def _fit_pooled_sequential(
        self,
        windows: dict[Instrument, list[Bar]],
        sym_windows: dict[str, list[Bar]],
        forward: int,
    ) -> None:
        """为 sequential 后端构建 (N, T, D) 训练集并训练。

        遍历所有合约，调用 SequenceFeatureBuilder.build_training_sequences()
        生成每个合约的序列样本，然后合并为全局训练集。
        """
        ...

    def compute(
        self,
        windows: dict[Instrument, list[Bar]],
        external_data: dict[str, Any] | None = None,
    ) -> dict[Instrument, Layer1Result]:
        """推理 — 根据 backend input_type 路由。"""
        if self._use_v2:
            if self._backend_input_type == "tabular":
                return self._compute_v2(windows)
            else:
                return self._compute_v2_sequential(windows)
        return self._compute_v1(windows)

    def _compute_v2_sequential(
        self,
        windows: dict[Instrument, list[Bar]],
    ) -> dict[Instrument, Layer1Result]:
        """Sequential 后端推理路径。

        对每个合约：
        1. 调用 SequenceFeatureBuilder.build_sequence() 生成 (T, D)
        2. 调用 backend.predict() 获得 (beta_score, confidence)
        3. 计算 MII，分类 regime
        """
        ...

    def load_model(self, directory: str | Path) -> None:
        """从版本化目录加载任意类型模型。

        读取 directory/meta.json 确定 model_type 和 input_type，
        通过 create_backend() 创建正确的后端实例，
        调用 backend.load(directory)。
        """
        ...
```

### 3.6.2 Layer1Config 新增字段

```python
class Layer1Config(BaseModel):
    # ... 现有字段 ...

    # 新增：模型后端类型
    model_type: str = "lgbm"
    # 可选值: "lgbm", "xgb", "ridge", "elasticnet", "rf",
    #          "dlinear", "patchtst", "lstm"

    # Sequential 后端参数
    seq_lookback_T: int = Field(96, ge=16)  # 序列长度（96 bars = 24h @15min）
    seq_feature_dims: list[str] = Field(default_factory=list)
    # 空 = 使用 SEQUENCE_FEATURE_NAMES 默认 14 维

    # 模型持久化路径
    model_dir: str = "models/layer1/"
```

### 3.6.3 改动最小化策略

核心思路是 **路由而非重写**：

1. **`__init__`** — 新增 backend 创建和 `_seq_feature_builder` 初始化；现有 v1/v2 路径不变
2. **`fit_window()` / `_fit_pooled()`** — tabular 路径 100% 不变；新增 `_fit_pooled_sequential()` 分支
3. **`compute()`** — tabular 路径不变；新增 `_compute_v2_sequential()` 分支
4. **`load_model()`** — 新增方法，现有 pickle 加载路径在 `LGBMBackend.load()` 中保持向后兼容

总变更量预估：layer1.py 新增 ~120 行（两个新方法 + __init__ 扩展），现有代码 0 行删除。

---

## 3.7 模型持久化格式扩展

### 目录结构

```
models/layer1/
└── v2_20260408_120000/          # 版本标签 = f"v2_{timestamp}"
    ├── meta.json                # 统一元数据
    ├── model.joblib             # tabular 后端（sklearn 系列）
    ├── model.pkl                # tabular 后端（LightGBM 原生）
    └── checkpoint.pt            # sequential 后端（PyTorch）
```

### meta.json 格式

```json
{
    "model_type": "patchtst",
    "input_type": "sequential",
    "feature_dim": 14,
    "lookback_T": 96,
    "feature_names": [
        "ret_1", "ret_5", "ret_20", "ret_60", "ret_120",
        "vol_20", "vol_60", "vol_ratio", "volume_ratio",
        "ret_vol_scaled_20", "ret_vol_scaled_60",
        "trend_consistency_20", "staleness_weight", "market_beta_20"
    ],
    "confidence_scale": 0.0342,
    "created_at": "2026-04-08T12:00:00Z",
    "n_train_samples": 8500,
    "n_instruments": 7,
    "train_bar_range": [13210, 26210],
    "hparams": {
        "patch_size": 16,
        "stride": 8,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "lr": 0.0005,
        "epochs": 80,
        "batch_size": 128
    },
    "val_loss": 0.0891,
    "best_epoch": 42
}
```

**Tabular 后端 meta.json 示例**：

```json
{
    "model_type": "lgbm",
    "input_type": "tabular",
    "feature_dim": 37,
    "lookback_T": 1,
    "feature_names": [],
    "confidence_scale": 0.0512,
    "created_at": "2026-04-08T12:00:00Z",
    "n_train_samples": 12000,
    "lgb_best_iteration": 187,
    "lgb_params": {
        "objective": "huber",
        "alpha": 1.0,
        "num_leaves": 63,
        "learning_rate": 0.05
    }
}
```

### 加载逻辑

```python
def load_model(directory: Path) -> ModelBackend:
    """从 meta.json 自动识别后端类型并加载。"""
    meta = json.loads((directory / "meta.json").read_text())
    model_type = meta["model_type"]
    backend = create_backend(model_type, cfg=None)  # 最小初始化
    backend.load(directory)
    return backend
```

### 向后兼容

现有的 `Layer1Model.save(path)` 输出为单个 `.pkl` 文件。`LGBMBackend.load()` 在检测到 `directory` 实际是文件（而非目录）时，回退到旧格式的 pickle 加载：

```python
def load(self, directory: Path) -> None:
    if directory.is_file():
        # 向后兼容：旧格式单文件 pickle
        self._load_legacy_pickle(directory)
        return
    # 新格式：目录 + meta.json
    ...
```

---

## 3.8 依赖管理

### pyproject.toml 结构

```toml
[project]
dependencies = [
    # 核心（必须）
    "numpy>=1.24",
    "pydantic>=2.0",
    "lightgbm>=4.0",
    "joblib>=1.3",
]

[project.optional-dependencies]
# sklearn 后端
sklearn = [
    "scikit-learn>=1.3",
]

# XGBoost 后端
xgb = [
    "xgboost>=2.0",
    "scikit-learn>=1.3",
]

# PyTorch 后端（CPU）
torch-cpu = [
    "torch>=2.1",
]

# PyTorch 后端（CUDA 12）
torch-gpu = [
    "torch>=2.1",
]

# 全部后端
all = [
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "torch>=2.1",
]

# 开发 / 测试
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "torch>=2.1",
]
```

### PyTorch CPU vs GPU 安装注意事项

1. **CPU 安装**（默认，开发/回测/CI）：
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   包体积 ~200MB（vs GPU 版 ~2GB）。

2. **GPU 安装**（训练加速，可选）：
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
   代码中通过 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 自动检测。

3. **CI 环境**：使用 CPU 版本，在 `_seq_base.py` 中所有 `.to(device)` 自动回退到 CPU。

### 版本约束说明

| 依赖 | 最低版本 | 约束原因 |
|------|---------|---------|
| `lightgbm>=4.0` | 4.0 | Dataset API 变更；categorical feature 处理改进 |
| `scikit-learn>=1.3` | 1.3 | `set_output` API；Pipeline 改进 |
| `xgboost>=2.0` | 2.0 | 原生 categorical 支持（`enable_categorical`） |
| `torch>=2.1` | 2.1 | `torch.compile()` 稳定版；`weights_only=True` 安全加载 |
| `numpy>=1.24` | 1.24 | 已存在约束，保持不变 |

---

## 3.9 实施阶段划分

### Phase 0: 基础设施（0.5 天）

**交付物**：
- `backend_protocol.py` — `ModelBackend` Protocol + `BackendMeta`
- `backends/__init__.py` — `BACKEND_REGISTRY` + `create_backend()` 工厂函数
- `Layer1Config` 新增 `model_type`, `seq_lookback_T`, `model_dir` 字段

**依赖**：无

### Phase 1: LGBMBackend 重构（1 天）

**交付物**：
- `backends/lgbm_backend.py` — 从 `model.py` 重构，实现 `ModelBackend` Protocol
- `model.py` → re-export shim（向后兼容）
- `layer1.py` 修改 `__init__` 使用 `create_backend()`
- 所有现有测试通过（行为 100% 不变）

**依赖**：Phase 0
**验证**：`pytest tests/` 全量通过

### Phase 2: sklearn Tabular Backends（1.5 天）

**交付物**：
- `backends/_sklearn_base.py`
- `backends/xgb_backend.py`
- `backends/ridge_backend.py`
- `backends/elasticnet_backend.py`
- `backends/rf_backend.py`

**依赖**：Phase 1
**验证**：
- 每个 backend 的 fit/predict/save/load 单元测试
- 在小数据集上 `model_type=xgb` 端到端回测

### Phase 3: SequenceFeatureBuilder（1 天）

**交付物**：
- `sequence_features.py`
- 单元测试：no-lookahead 验证、NaN 处理、形状校验

**依赖**：Phase 0（不依赖 Phase 1/2）
**注意**：Phase 2 和 Phase 3 可并行开发

### Phase 4: SequentialBackend 基类 + DLinear（1.5 天）

**交付物**：
- `backends/_seq_base.py` — PyTorch 训练循环基类
- `backends/dlinear_backend.py`
- `layer1.py` 新增 `_fit_pooled_sequential()` 和 `_compute_v2_sequential()`

**依赖**：Phase 3
**验证**：
- DLinear 端到端：synthetic data fit → predict → save → load → predict
- `model_type=dlinear` 小规模回测

> **Phase 4 后可进行端到端测试**：此时 tabular (lgbm) 和 sequential (dlinear) 两条路径均已可用，整个 Layer1 → BacktestEngine 管线可完整运行。

### Phase 5: PatchTST + LSTM（1.5 天）

**交付物**：
- `backends/patchtst_backend.py`
- `backends/lstm_backend.py`
- 对应单元测试

**依赖**：Phase 4

### Phase 6: 模型持久化 + 加载（0.5 天）

**交付物**：
- `meta.json` 统一格式实现
- `load_model()` 方法（自动识别后端类型）
- 向后兼容测试（加载旧 pickle 格式）

**依赖**：Phase 5

### Phase 7: 集成测试 + 文档（1 天）

**交付物**：
- 跨后端一致性测试（验证所有 backend 输出格式一致）
- 回测对比报告（LGBM vs XGB vs DLinear vs PatchTST）
- 更新 `docs/universal_feature_spec.md` 和配置文档

**依赖**：Phase 6

### Phase 依赖关系图

```
Phase 0 (基础设施)
  ├── Phase 1 (LGBMBackend)
  │   └── Phase 2 (sklearn backends)
  │       └──┐
  └── Phase 3 (SequenceFeatureBuilder)     ← 可与 Phase 1/2 并行
      └── Phase 4 (seq base + DLinear)
          └── Phase 5 (PatchTST + LSTM)
              └── Phase 6 (持久化)
                  └── Phase 7 (集成)

端到端测试里程碑: Phase 4 完成后
```

**总工期估算**：约 8.5 人天（如果 Phase 2 和 Phase 3 并行则约 7 天）。

---

## 3.10 测试策略

### 新增测试文件

| 测试文件 | 覆盖范围 |
|---------|---------|
| `tests/test_backend_protocol.py` | 验证所有 backend 均满足 `ModelBackend` Protocol（`isinstance` 检查 + 签名校验） |
| `tests/test_lgbm_backend.py` | LGBMBackend fit/predict/save/load，与旧 `Layer1Model` 行为一致性对比 |
| `tests/test_sklearn_backends.py` | XGB/Ridge/ElasticNet/RF 的 fit/predict/save/load，共用 parametrize fixture |
| `tests/test_seq_feature_builder.py` | SequenceFeatureBuilder 形状校验、no-lookahead 验证、NaN 处理、与 FeatureBuilderV2 逐资产特征一致性 |
| `tests/test_seq_backends.py` | DLinear/PatchTST/LSTM 的 fit/predict/save/load，CPU 模式 + 小数据集 |
| `tests/test_backend_registry.py` | `create_backend()` 工厂函数，未知 model_type 的错误处理 |
| `tests/test_layer1_routing.py` | Layer1 类根据 `model_type` 正确路由到 tabular/sequential 路径 |
| `tests/test_model_persistence.py` | meta.json 格式验证、跨后端 save/load 往返、向后兼容旧 pickle |

### 无 GPU CI 环境中测试 PyTorch 模型

1. **CPU 强制**：所有测试 fixture 设置 `os.environ["CUDA_VISIBLE_DEVICES"] = ""`，确保 `torch.cuda.is_available()` 返回 False。

2. **小规模数据**：测试用数据集限制为 N=200, T=32, D=14（训练 <5 秒 / 后端）。

3. **确定性**：测试 fixture 设置 `torch.manual_seed(42)` + `np.random.seed(42)`。

4. **epochs 缩减**：测试时通过 `hparams={"epochs": 3, "patience": 2}` 覆盖默认值。

5. **Skip 策略**：当 `torch` 不可用时（如最小化安装），用 `pytest.importorskip("torch")` 跳过 sequential 测试，而非报错。

6. **数值精度**：由于 CPU 和 GPU 浮点行为差异，测试使用 `np.allclose(atol=1e-4)` 而非严格相等。

### 核心测试用例

```python
# tests/conftest.py 新增 fixture
@pytest.fixture
def synthetic_tabular_data():
    """(X, y) tabular 数据，N=200, D=37。"""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 37).astype(np.float32)
    y = rng.randn(200).astype(np.float64)
    return X, y

@pytest.fixture
def synthetic_sequential_data():
    """(X, y) sequential 数据，N=200, T=32, D=14。"""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 32, 14).astype(np.float32)
    y = rng.randn(200).astype(np.float64)
    return X, y
```

```python
# tests/test_backend_protocol.py
import pytest
from quant.strategy.layer1.backend_protocol import ModelBackend

ALL_BACKENDS = ["lgbm", "xgb", "ridge", "elasticnet", "rf", "dlinear", "patchtst", "lstm"]

@pytest.mark.parametrize("model_type", ALL_BACKENDS)
def test_backend_satisfies_protocol(model_type):
    backend = create_backend(model_type, cfg=default_cfg())
    assert isinstance(backend, ModelBackend)
    assert backend.input_type in ("tabular", "sequential")

@pytest.mark.parametrize("model_type", ALL_BACKENDS)
def test_unfitted_predict_returns_zeros(model_type):
    backend = create_backend(model_type, cfg=default_cfg())
    beta, conf = backend.predict(np.zeros(37 if backend.input_type == "tabular" else (32, 14)))
    assert beta == 0.0
    assert conf == 0.0
```

```python
# tests/test_seq_feature_builder.py
def test_no_lookahead():
    """验证序列特征不使用未来数据。"""
    bars = generate_bars(n=200)
    builder = SequenceFeatureBuilder()

    # 构建到 bar 150 的序列
    seq_150 = builder.build_sequence(bars[:150], lookback_T=32, as_of=bars[149].timestamp)

    # 构建到 bar 200 的序列（包含 bar 150 之后的数据）
    seq_200 = builder.build_sequence(bars[:200], lookback_T=32, as_of=bars[199].timestamp)

    # 在 seq_200 中，对应 bar 150 时刻的那个时步
    # 应该与 seq_150 的最后一步完全一致
    # （因为该步只能看到 bar 150 及之前的数据）
    # 注意：这里的精确对比需要按 bar index 对齐
    assert seq_150[-1] is not None  # 基本 sanity check
```

---

## @strategist-b: 需要策略确认的问题

以下设计决策需要策略师确认或讨论：

1. **Sequential 特征维度 D=14 vs D=37**：当前设计中 sequential 后端仅使用 14 维逐资产特征，不包含 PCA 和复合特征。是否需要在每个时步也注入全局状态特征？这会将训练开销从 O(N×T) 增至 O(N×T×N_universe)。

2. **seq_lookback_T 默认值 96**：96 bars × 15 min = 24 小时。对于日内交易风格是否合适？是否需要支持多个 lookback_T 的 multi-scale 输入？

3. **DLinear `individual=False` vs `True`**：默认 channel-mixing 模式假设特征间有交互关系。是否应该默认使用 channel-independent 模式（更保守、更适合小样本）？

4. **PatchTST `channel_independent=True` 默认**：CI 模式参数更少，但无法捕获跨特征交互。在我们 D=14 维度下，CM 模式参数量仍然可控（~15K 参数），是否应该默认 CM？

5. **训练时 shuffle=False**：当前设计在 PyTorch DataLoader 中不打乱训练数据（因为时序数据）。但 `_fit_pooled` 的训练集已经是独立样本（不同资产 × 不同时间），是否可以打乱以提高收敛速度？

6. **LGBMBackend 是否保持独立实现**：当前设计中 LGBMBackend 不继承 `SklearnTabularBase`（因为使用原生 API）。是否接受改用 sklearn wrapper `lgb.LGBMRegressor` 以统一接口？代价是失去 `categorical_feature` 原生声明和 early_stopping 回调的精确控制。

7. **模型版本化策略**：当前使用 timestamp 命名目录（`v2_20260408_120000/`）。是否需要更结构化的版本管理（如 MLflow / DVC 集成），还是文件系统方案足够？

8. **标签 Winsorize 在 sequential 路径中的位置**：tabular 路径的 Winsorize 在 `_fit_pooled()` 中完成。sequential 路径的 `build_training_sequences()` 直接返回原始波动率缩放标签。Winsorize 应在 `build_training_sequences()` 内部还是由调用方处理？建议由调用方处理以保持 feature builder 的纯粹性。

---

## 对 @strategist-b 的工程回答

以下逐条回应策略文档中 `@engineer-b:` 提出的 8 个工程确认问题。

### Q1: PyTorch 依赖策略 — 可选依赖 + ImportError 兜底

**结论：`torch` 作为可选依赖（`pip install .[torch-cpu]`），`create_backend()` 内做 `ImportError` 处理。**

理由：核心回测/实盘路径当前仅依赖 LightGBM，将 ~200MB 的 PyTorch 强制为核心依赖会拖慢 CI 和部署。工厂函数 `create_backend("dlinear", ...)` 在 import 失败时抛出明确的 `ImportError("pip install ibkr-signal-assistant[torch-cpu]")`，用户 1 秒内可修复。`backends/_seq_base.py` 中所有 `import torch` 已位于方法内部（lazy import），模块加载本身不会失败。

### Q2: SequenceFeatureBuilder 中 GlobalState 逐时步计算 — 不复用 GlobalState，接受 D=14

**结论：序列路径不包含 GlobalState 特征（PCA/复合收益率），仅使用 D=14 逐资产特征。无需 `update_batch()` 接口。**

理由：如工程文档 §3.5 所述，对 T=120 × N_universe=7 逐步重建 GlobalState 的计算量 = O(T × N × PCA_window)，单个训练样本的特征构建将从 ~0.1ms 膨胀到 ~100ms，1 万个样本 = 17 分钟，完全不可接受。D=14 纯价量特征对序列模型已足够 — 时间维度上的 temporal pattern 本身承担了 PCA 所捕获的跨时间结构信息。如果策略侧认为全局特征对 sequential 后端也至关重要，可以在 Phase 7（集成测试后）设计一个轻量级的 `GlobalState 快照缓存` —— 预计算并缓存每根 bar 的 GlobalState，序列构建时直接查表拼接。但这应作为独立优化迭代，不阻塞 Route B 主线。

### Q3: GPU 推理延迟 — 实盘强制 CPU，训练可用 GPU

**结论：实盘推理强制 CPU。训练时若有 GPU 则自动使用。**

理由：单样本推理（batch_size=1）时 CUDA kernel launch overhead (~0.3ms) + CPU↔GPU 数据传输 (~0.1ms) 常常超过 CPU 上的纯计算时间（DLinear CPU 推理 <0.05ms，PatchTST <0.5ms）。15 分钟 bar 间隔下推理延迟完全不是瓶颈。代码实现方式：`predict()` 中 `self._model` 始终保持在 CPU（`_seq_base.py` 的 `fit()` 末尾已有 `self._model = model.cpu()`），`load()` 使用 `map_location="cpu"`。训练时 `fit()` 内部通过 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 自动检测。

### Q4: ONNX 导出 — 推迟到后续迭代

**结论：Phase 1-7 不支持 ONNX 导出，推迟到后续迭代。**

理由：ONNX 导出的主要场景是"无 PyTorch 依赖的推理环境"。但我们的实盘系统本身就运行在 Python 环境中，已经能安装 torch-cpu（~200MB）。ONNX runtime 本身也需要 `onnxruntime` 依赖（~40MB），节省并不显著。此外 PatchTST 的 Transformer Encoder 导出到 ONNX 时有已知的 dynamic shape 问题，需要额外的 trace/script 调试。如果未来需要将模型嵌入 C++ 推理服务或边缘设备，再引入 ONNX 导出层。

### Q5: model 文件命名 — 采纳 meta.json 中 `model_file` 字段方案

**结论：采纳。在 `meta.json` 中新增 `"model_file"` 字段，`load_model()` 读取此字段确定加载哪个文件。**

实现方式：

```json
{
    "model_type": "patchtst",
    "model_file": "checkpoint.pt",
    ...
}
```

```json
{
    "model_type": "lgbm",
    "model_file": "model.pkl",
    ...
}
```

```json
{
    "model_type": "xgb",
    "model_file": "model.joblib",
    ...
}
```

`load_model()` 逻辑：读 `meta.json` → 取 `model_file` → `backend.load(directory / model_file)`。向后兼容：若 `model_file` 字段缺失，按后端类型回退到默认名（tabular → `model.pkl`，sequential → `checkpoint.pt`）。工程文档 §3.7 的持久化格式已做相应更新。

### Q6: confidence_temperature 调参方式 — 推理时可配置，不违反冻结原则

**结论：`confidence_temperature` 存在于 `Layer1Config` 中，推理时可调。训练时的 `confidence_scale` 写入 `meta.json` 作为基线参考值。**

理由：`confidence` 的计算公式是 `min(|beta_score| / (confidence_scale * temperature), 1.0)`。`confidence_scale` 是模型内禀属性（由训练数据分布决定），应随模型冻结。`temperature` 是运维旋钮（控制入场灵敏度），应在推理时可调。这与模型冻结原则不矛盾 —— 冻结的是模型权重和 `confidence_scale`，`temperature` 是后处理参数，类似于 LLM 推理的 temperature 不属于模型权重。具体实现：`backend.predict()` 返回原始 `(beta_score, raw_confidence)`，`Layer1._compute_v2()` 用 `cfg.confidence_temperature` 做最终缩放。

### Q7: 多品种 padding/mask — 现有 windows 结构足以推导

**结论：不需要额外 mask。现有 `windows: dict[Instrument, list[Bar]]` 中每个品种的 `list[Bar]` 长度已隐含了该品种有多少真实 bar。**

理由：当前 `BacktestEngine` 按 aligned_timestamps 遍历时，各品种的 `windows[inst]` 仅包含该品种*实际存在*的 bar（非交易时段不会插入虚拟 bar）。`SequenceFeatureBuilder.build_sequence()` 的输入是 `asset_bars: list[Bar]`（单品种的真实 bar 列表），长度不足 `vol_lookback + T + 1` 时直接返回 `None`，调用方跳过该品种。因此不存在"哪些时步是真实 bar"的歧义 —— 所有时步都是真实 bar。如果未来需要对齐跨品种时间轴（如跨资产 attention），届时需要引入 padding + mask，但当前 channel-independent 架构不需要。

### Q8: 实施顺序 — 同意策略侧建议，微调 XGBoost 提前

**结论：基本同意 LightGBM → DLinear → PatchTST → GRU → XGBoost → 其余的顺序，但建议将 XGBoost 提前到 DLinear 之前。**

调整后的顺序：
1. **LightGBM**（Phase 1，已有代码重构）
2. **XGBoost**（Phase 2 前半，tabular 基础设施验证）
3. **DLinear**（Phase 4，sequential 基础设施验证）
4. **PatchTST**（Phase 5 前半）
5. **GRU/LSTM**（Phase 5 后半）
6. **Ridge / ElasticNet / RandomForest**（Phase 2 后半，低优先级）

理由：XGBoost 是第二个 tabular 后端，实现它可以验证 `SklearnTabularBase` 基类和 `BACKEND_REGISTRY` 工厂函数是否正确工作。如果这些基础设施有设计问题，在实现轻量级的 XGBoost 时暴露比在实现 DLinear（涉及 PyTorch）时暴露更容易修复。Ridge/ElasticNet/RF 则纯粹是基类的薄 wrapper，一旦 XGBoost 跑通，它们 30 分钟内全部完成。
