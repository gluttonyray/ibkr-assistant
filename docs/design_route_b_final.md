# Route B 设计文档 -- 多后端 Layer1 架构（最终版）

> **作者**：@strategist-b + @engineer-b 协作
> **日期**：2026-04-08
> **状态**：Approved -- 待实施

---

## 1. 背景与目标

### 1.1 现状问题

当前 Layer1 存在两个架构瓶颈：

**训练/推理紧耦合**：`BacktestEngine._run_loop()` 中每 130 bar 触发 `layer1.fit_window()` 在线重训练，与推理共享同一事件循环。这导致实盘推理阻塞、内存峰值翻倍、回测不可复现、模型不可审计。

**单一模型后端**：Layer1 仅支持 LightGBM Huber 回归，输入固定为 `FeatureBuilderV2` 产出的 (N, 37) tabular 快照。37 维手工特征工程存在信息瓶颈，无法捕捉原始序列中的时序模式。

### 1.2 Route B 目标

1. **可插拔后端架构**：通过统一的 `ModelBackend` Protocol，同时支持 tabular（LightGBM, XGBoost, Ridge, ElasticNet, RandomForest）和 sequential（DLinear, PatchTST, LSTM/GRU）两类后端
2. **离线训练分离**：训练从 BacktestEngine 中移除，成为独立脚本 `scripts/train_layer1.py`；推理加载冻结模型只做 forward pass
3. **零破坏迁移**：LightGBM 路径 100% 行为兼容，`model_type="lgbm"` 时与现有系统输出一致

---

## 2. 最终架构决策

以下为两轮协作中确定的所有设计决策：

| # | 决策点 | 最终结论 |
|---|--------|---------|
| 1 | PyTorch 依赖方式 | 可选依赖 `pip install .[torch-cpu]`，`create_backend()` 在 import 失败时抛出 `ImportError` 并附安装指引 |
| 2 | 序列特征中的 GlobalState | 序列路径不包含 PCA/复合收益率特征（D=14 纯价量特征），逐时步重建 GlobalState 计算开销不可接受 |
| 3 | GPU 推理策略 | 实盘推理强制 CPU（kernel launch overhead 在 batch=1 时更慢）；训练时自动检测 GPU |
| 4 | ONNX 导出 | 推迟到后续迭代，当前 Python 环境直接安装 torch-cpu 即可 |
| 5 | 模型文件命名 | `meta.json` 中新增 `"model_file"` 字段指定实际文件名，消除扩展名硬编码 |
| 6 | confidence_temperature | 推理时可调，存在于 `Layer1Config` 中；训练时的 `confidence_scale` 写入 meta.json 作为基线 |
| 7 | 多品种 padding/mask | 不需要额外 mask，`windows[inst]` 的 `list[Bar]` 长度已隐含真实 bar 信息 |
| 8 | 深度学习框架 | PyTorch（DLinear/PatchTST 原论文均为 PyTorch；TSF 领域事实标准） |
| 9 | SequenceFeatureBuilder 方案 | 新建独立类（方案 B），不扩展 FeatureBuilderV2（输入形状根本不同） |
| 10 | Sequential 置信度映射 | 与 tabular 一致：`confidence = min(\|beta_score\| / confidence_scale, 1.0)`，`confidence_scale = P90(\|y_pred_train\|)` |
| 11 | LGBMBackend 继承关系 | 独立实现，不继承 SklearnTabularBase（保留原生 `lgb.train()` API + categorical_feature + early_stopping） |
| 12 | 模型版本化 | 文件系统目录 `v2_{timestamp}/`，不引入 MLflow/DVC |
| 13 | 标签 Winsorize 位置 | 由调用方（`_fit_pooled_sequential()`）在拿到原始标签后执行 `np.clip`，SequenceFeatureBuilder 保持纯粹 |
| 14 | Lookback T 默认值 | 120 bars（30 小时，约 2 个完整交易日），范围 [60, 480] |
| 15 | DLinear channel 模式 | `individual=True`（channel-independent），D 较小时参数量可控且更不易过拟合 |
| 16 | PatchTST channel 模式 | `channel_independent=True`（CI 模式），参数共享，对小数据集更鲁棒 |
| 17 | 训练 DataLoader shuffle | `shuffle=False`，时序数据保持时间顺序 |
| 18 | 实施优先级 | LGBM -> XGBoost -> DLinear -> PatchTST -> GRU -> Ridge/ElasticNet/RF |

---

## 3. 文件结构

```
src/quant/strategy/layer1/
+-- __init__.py              # 不变，保持现有公共 API 导出
+-- layer1.py                # 小改 -- 重构以支持 backend 路由
+-- features.py              # 不变 -- FeatureBuilderV2, GlobalState, GlobalStateBuilder
+-- sequence_features.py     # 新增 -- SequenceFeatureBuilder，生成 (T, D) 序列张量
+-- mii.py                   # 不变
+-- walk_forward.py          # 不变
+-- model.py                 # 过渡期 shim：from .backends.lgbm_backend import LGBMBackend as Layer1Model
+-- backend_protocol.py      # 新增 -- ModelBackend Protocol 定义
+-- backends/
    +-- __init__.py          # 导出 BACKEND_REGISTRY + create_backend() 工厂函数
    +-- lgbm_backend.py      # 重构自 model.py：LGBMBackend(ModelBackend)
    +-- xgb_backend.py       # 新增 -- XGBBackend(SklearnTabularBase)
    +-- ridge_backend.py     # 新增 -- RidgeBackend(SklearnTabularBase)
    +-- elasticnet_backend.py# 新增 -- ElasticNetBackend(SklearnTabularBase)
    +-- rf_backend.py        # 新增 -- RandomForestBackend(SklearnTabularBase)
    +-- _sklearn_base.py     # 新增 -- sklearn 系列 backend 共享基类
    +-- _seq_base.py         # 新增 -- PyTorch SequentialBackend 基类
    +-- dlinear_backend.py   # 新增 -- DLinearBackend
    +-- patchtst_backend.py  # 新增 -- PatchTSTBackend
    +-- lstm_backend.py      # 新增 -- LSTM/GRU Backend
```

### 各文件职责

| 文件 | 变动 | 职责 |
|------|------|------|
| `backend_protocol.py` | 新增 | `ModelBackend` Protocol + `BackendMeta` 元数据 dataclass |
| `sequence_features.py` | 新增 | 从 Bar 序列构建 (T, D) 张量，保证 no-lookahead |
| `backends/__init__.py` | 新增 | 后端注册表 `BACKEND_REGISTRY` + `create_backend()` 工厂函数 |
| `backends/lgbm_backend.py` | 重构 | 现有 `Layer1Model` -> `LGBMBackend`，实现 `ModelBackend` Protocol |
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

## 4. 核心接口

### 4.1 ModelBackend Protocol

```python
# src/quant/strategy/layer1/backend_protocol.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BackendMeta:
    """模型元数据，随模型一同保存至 meta.json。"""
    model_type: str
    input_type: Literal["tabular", "sequential"]
    feature_dim: int
    lookback_T: int = 1
    feature_names: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    """Layer1 模型后端统一接口。

    Tabular 后端:  fit(X, y, ...) X: (N, D)   -> predict(X) X: (1, D)
    Sequential 后端: fit(X, y, ...) X: (N, T, D) -> predict(X) X: (1, T, D)
    """

    @property
    def input_type(self) -> Literal["tabular", "sequential"]: ...

    @property
    def is_fitted(self) -> bool: ...

    @property
    def meta(self) -> BackendMeta: ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        categorical_indices: list[int] | None = None,
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        """训练模型。返回训练日志。"""
        ...

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        """单样本推理，返回 (beta_score, confidence)。"""
        ...

    def save(self, directory: Path) -> None:
        """将模型权重 + meta.json 保存到目录。"""
        ...

    def load(self, directory: Path) -> None:
        """从目录加载模型。"""
        ...
```

**设计要点**：

1. `predict()` 返回 `(beta_score, confidence)` 元组 -- 与现有 `Layer1Model.predict()` 签名一致，下游 `Layer1._compute_v2()` 无需修改
2. `save()/load()` 使用目录而非单文件 -- sequential 后端需要多个文件（checkpoint + meta），统一用目录
3. `categorical_indices` 作为 `fit()` 参数 -- 仅 LightGBM 使用，其余后端忽略
4. `fit()` 返回 `dict` -- 训练诊断信息（best_iteration、val_loss 等）

### 4.2 后端注册与工厂

```python
# src/quant/strategy/layer1/backends/__init__.py
from typing import Callable
from quant.config.schema import Layer1Config

_REGISTRY: dict[str, Callable[[Layer1Config], ModelBackend]] = {}

def register_backend(name: str):
    """装饰器：注册模型后端工厂函数。"""
    def decorator(factory):
        _REGISTRY[name] = factory
        return factory
    return decorator

def create_backend(name: str, cfg: Layer1Config) -> ModelBackend:
    """根据名称创建模型后端实例。未安装依赖时抛出 ImportError。"""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](cfg)
```

### 4.3 SklearnTabularBase 基类

XGBoost、Ridge、ElasticNet、RandomForest 共用此基类。子类只需实现 `_create_estimator()` 和 `_model_type`。

```python
# src/quant/strategy/layer1/backends/_sklearn_base.py
class SklearnTabularBase(ABC):
    """sklearn 系列 tabular 后端的公共基类。

    fit / predict / save / load / confidence 校准全部由基类处理。
    """

    @property
    def input_type(self) -> Literal["tabular"]:
        return "tabular"

    @property
    @abstractmethod
    def _model_type(self) -> str: ...

    @abstractmethod
    def _create_estimator(self) -> Any: ...

    def fit(self, X, y, *, sample_weights=None, categorical_indices=None, val_fraction=0.2):
        # 80/20 时间顺序切分
        # estimator.fit(X_train, y_train, sample_weight=...)
        # confidence_scale = P90(|y_pred_train|) + eps
        # 返回 val_mse 等诊断信息
        ...

    def predict(self, X) -> tuple[float, float]:
        # confidence = min(|beta_score| / confidence_scale, 1.0)
        ...

    def save(self, directory: Path):
        # joblib.dump(estimator, directory / "model.joblib")
        # meta.json 写入 model_type, feature_dim, confidence_scale, model_file
        ...

    def load(self, directory: Path):
        # joblib.load + meta.json 读取
        ...
```

**各 Tabular Backend 差异**：

| Backend | 估计器 | 特殊处理 | 序列化 |
|---------|--------|----------|--------|
| **LGBMBackend** | `lgb.train()` 原生 API | `categorical_feature` 声明 + `lgb.early_stopping` | pickle（向后兼容） |
| **XGBBackend** | `xgb.XGBRegressor` | Phase 1 用 ordinal 编码（跳过 `enable_categorical`） | joblib |
| **RidgeBackend** | `sklearn.pipeline.Pipeline(StandardScaler + Ridge)` | 线性模型需 StandardScaler 预处理 | joblib |
| **ElasticNetBackend** | `sklearn.pipeline.Pipeline(StandardScaler + ElasticNet)` | 同 Ridge | joblib |
| **RandomForestBackend** | `sklearn.ensemble.RandomForestRegressor` | 不需要特征缩放 | joblib |

### 4.4 SequentialBackendBase 基类

PyTorch 时序模型后端共用此基类。子类需实现 `_model_type`、`_build_model()`、`_default_hparams()`。

训练循环核心特性：
- AdamW 优化器 + CosineAnnealingLR 调度
- 加权 Huber 损失（delta 可配置）
- 梯度裁剪（max_norm=1.0）
- 早停（patience 可配置）
- 训练在 GPU（若可用），推理强制 CPU
- confidence_scale = P90(|y_pred_train|)（与 tabular 一致）

### 4.5 Sequential 模型架构

#### DLinear

输入分解为趋势（moving average, kernel_size=25）和残差，各经一层线性映射后合并。`individual=True`：每个特征维度独立线性层。参数量极小，不易过拟合。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| kernel_size | 25 | 移动平均窗口（25 bars ~ 1 RTH 交易日） |
| individual | True | 每个特征维度独立线性层 |
| lr | 1e-3 | |
| epochs | 100 | |
| patience | 15 | |

#### PatchTST

将时序切分为固定长度 patch，每个 patch 通过线性映射得到 embedding，经 Transformer Encoder 处理。Channel-Independent (CI) 模式：每个特征维度独立通过 Transformer，最后聚合。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| patch_size | 16 | 每个 patch 覆盖 16 个时步 |
| stride | 8 | 50% 重叠 |
| d_model | 64 | Transformer 隐层维度 |
| n_heads | 4 | 多头注意力头数 |
| n_layers | 2 | Encoder 层数 |
| d_ff | 128 | FFN 中间维度 |
| dropout | 0.2 | |
| channel_independent | True | CI 模式 |

Token 数（T=120, P=16, S=8）：`(120-16)/8 + 1 = 14` tokens。

#### LSTM/GRU

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_dim | 64 | 隐层维度 |
| num_layers | 2 | RNN 层数 |
| dropout | 0.3 | 层间 dropout |
| bidirectional | False | 因果推理禁止双向 |
| cell_type | GRU | GRU 优先（参数更少，收敛更快） |

输出：取最后时步隐状态 `h[-1]`，经线性头映射为标量 beta_score。

### 4.6 SequenceFeatureBuilder

新增独立类，不扩展 FeatureBuilderV2。

```python
# src/quant/strategy/layer1/sequence_features.py

SEQUENCE_FEATURE_NAMES = [
    "ret_1", "ret_5", "ret_20", "ret_60", "ret_120",
    "vol_20", "vol_60", "vol_ratio", "volume_ratio",
    "ret_vol_scaled_20", "ret_vol_scaled_60",
    "trend_consistency_20", "staleness_weight", "market_beta_20",
]
SEQUENCE_FEATURE_DIM = 14

class SequenceFeatureBuilder:
    def build_sequence(
        self,
        asset_bars: list[Bar],
        lookback_T: int,
        as_of: datetime,
        composite_ew_ret_history: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """返回 (T, 14) float32 or None（bars 不足时）。"""
        ...

    def build_training_sequences(
        self,
        asset_bars: list[Bar],
        lookback_T: int,
        forward_horizon: int,
        composite_ew_ret_histories: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """批量 sliding window，返回 (X: (N,T,D), y: (N,), indices: (N,))。"""
        ...
```

**与 FeatureBuilderV2 的关系**：

| 维度 | FeatureBuilderV2 | SequenceFeatureBuilder |
|------|-------------------|----------------------|
| 输出 | (37,) 单快照 | (T, 14) 序列 |
| 逐资产特征 | [0:14] 14 维 | [0:14] **相同的 14 维** |
| PCA 特征 | [14:29] 15 维 | 不包含（计算成本不可接受） |
| 复合特征 | [29:33] 4 维 | 不包含 |
| 类别特征 | [33:37] 4 维 | 不包含（时序不变量） |

**No-lookahead 保证**：第 t 步只使用 `asset_bars[:end_idx_t]`，训练标签 `y[i]` 严格在第 i 步之后。

**NaN / 短序列策略**：bars 不足 `vol_lookback + T + 1` 时返回 None，调用方跳过。历史不足的特征返回 0.0（安全填充）。

---

## 5. Layer1Config 新增字段

```python
class Layer1Config(BaseModel):
    # --- 现有字段保持不变 ---

    # --- Route B 新增 ---

    # 模型后端类型
    model_type: str = Field(
        "lgbm",
        description="后端名称。"
        "Tabular: 'lgbm', 'xgb', 'ridge', 'elasticnet', 'rf'. "
        "Sequential: 'dlinear', 'patchtst', 'lstm', 'gru'."
    )

    # Sequential 后端专用参数
    seq_lookback_T: int = Field(
        120, ge=60, le=480,
        description="序列 lookback 窗口长度（bars）。"
        "120 bars x 15min = 30 小时 ~ 2 交易日。"
    )
    seq_feature_set: list[str] = Field(
        default_factory=lambda: [
            "ret_1", "ret_5", "ret_20", "ret_60", "ret_120",
            "vol_20", "vol_60", "vol_ratio", "volume_ratio",
            "ret_vol_scaled_20", "ret_vol_scaled_60",
            "trend_consistency_20", "staleness_weight", "market_beta_20",
        ],
        description="序列模型的特征维度名称列表。"
    )

    # DLinear 参数
    dlinear_kernel_size: int = Field(25, ge=3)

    # PatchTST 参数
    patchtst_patch_len: int = Field(16, ge=4)
    patchtst_stride: int = Field(8, ge=1)
    patchtst_d_model: int = Field(64, ge=16)
    patchtst_n_heads: int = Field(4, ge=1)
    patchtst_n_layers: int = Field(2, ge=1)
    patchtst_dropout: float = Field(0.2, ge=0, le=0.5)

    # LSTM/GRU 参数
    rnn_hidden_dim: int = Field(64, ge=16)
    rnn_num_layers: int = Field(2, ge=1)
    rnn_dropout: float = Field(0.2, ge=0, le=0.5)
    rnn_cell_type: str = Field("gru", description="'lstm' or 'gru'")

    # 训练参数（Sequential 专用）
    seq_learning_rate: float = Field(1e-3, gt=0)
    seq_epochs: int = Field(50, ge=1)
    seq_batch_size: int = Field(256, ge=16)
    seq_early_stopping_patience: int = Field(10, ge=1)

    # confidence 温度（推理时可调，训练时的 confidence_scale 写入 meta.json）
    confidence_temperature: float = Field(
        1.0, gt=0,
        description="confidence_scale 的乘数。>1 降低 confidence，<1 提高。"
    )

    # 模型持久化路径
    model_dir: str = "models/layer1/"

    # Pydantic validator
    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        ALL_BACKENDS = {
            "lgbm", "xgb", "ridge", "elasticnet", "rf",
            "dlinear", "patchtst", "lstm", "gru",
        }
        if v not in ALL_BACKENDS:
            raise ValueError(f"Unknown model_type '{v}'. Available: {sorted(ALL_BACKENDS)}")
        return v
```

---

## 6. 训练/推理分离

Route B 继承并扩展 `design_train_infer_separation.md` 的 4-Phase 方案。以下为 Route B 调整后的关键组件。

### 6.1 GlobalStateBuilder frozen 模式（不变）

推理时 PCA 状态冻结，不重新拟合。接口新增 `frozen: bool = False` 参数 + `save_pca_state()` / `load_pca_state()` 方法。`update()` 中复合收益率和 PCA 投影照常计算，但跳过 `_recompute_pca()`。

### 6.2 scripts/train_layer1.py 多后端路由

```
1. load_config(args.config)
2. 校验 pca_universe_symbols 非空
3. 加载数据
4. 根据 cfg.model_type 通过 create_backend() 创建 ModelBackend
5. 根据 backend.input_type 选择特征构建器：
   - "tabular" -> 现有 FeatureBuilderV2 + GlobalStateBuilder 逻辑
   - "sequential" -> SequenceFeatureBuilder（D=14，无 GlobalState 依赖）
6. 构建 X, y（sequential: sliding window 生成 (N, T, D)）
7. Winsorize 标签（调用方统一执行 np.clip）
8. 计算逆频率样本权重
9. backend.fit(X, y, sample_weights=...)
10. OOS 验证（predict 接口统一，无需分支）
11. backend.save(model_dir / version_tag/)
12. global_builder.save_pca_state(...)
13. 保存 meta.json（含 model_type, model_file, confidence_scale 等）
```

### 6.3 BacktestEngine 推理解耦（不变）

BacktestEngine 删除所有训练触发逻辑，成为纯推理引擎。仅调用 `layer1.compute(windows)`。

### 6.4 Layer1.load_model() 多后端支持

```python
class Layer1:
    def load_model(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        model_type = meta.get("model_type", "lgbm")

        # 通过工厂创建正确的后端
        backend = create_backend(model_type, self._cfg)
        backend.load(model_dir)
        self._models[_POOLED_KEY] = backend

        # PCA 冻结状态（与后端无关）
        if self._use_v2:
            self._global_builder.load_pca_state(model_dir / "global_state.pkl")
```

### 6.5 Layer1 类路由机制

```python
class Layer1:
    def __init__(self, cfg: Layer1Config) -> None:
        # ... 现有初始化 ...
        self._model_type = cfg.model_type
        self._backend_template = create_backend(self._model_type, cfg)
        self._backend_input_type = self._backend_template.input_type

        if self._backend_input_type == "sequential":
            self._seq_feature_builder = SequenceFeatureBuilder(...)

    def compute(self, windows, external_data=None):
        if self._use_v2:
            if self._backend_input_type == "tabular":
                return self._compute_v2(windows)      # 现有路径，不变
            else:
                return self._compute_v2_sequential(windows)  # 新增
        return self._compute_v1(windows)

    def fit_window(self, windows, bar_index, start_idx, end_idx):
        if self._use_v2:
            if self._backend_input_type == "tabular":
                self._fit_pooled(...)           # 现有路径，不变
            else:
                self._fit_pooled_sequential(...)  # 新增
```

改动策略：**路由而非重写**。layer1.py 新增 ~120 行，现有代码 0 行删除。

### 6.6 模型持久化格式

```
models/layer1/
+-- v2_20260408_120000/
    +-- meta.json              # 统一元数据（含 model_file 字段）
    +-- global_state.pkl       # PCA 冻结状态
    +-- model.pkl              # LightGBM 后端
    +-- model.joblib           # sklearn 后端
    +-- checkpoint.pt          # PyTorch 后端
```

**meta.json 示例（sequential）**：

```json
{
    "model_type": "patchtst",
    "input_type": "sequential",
    "model_file": "checkpoint.pt",
    "feature_dim": 14,
    "lookback_T": 120,
    "feature_names": ["ret_1", "ret_5", "..."],
    "confidence_scale": 0.0342,
    "created_at": "2026-04-08T12:00:00Z",
    "n_train_samples": 8500,
    "hparams": {"patch_size": 16, "stride": 8, "d_model": 64, "...": "..."},
    "val_loss": 0.0891
}
```

**meta.json 示例（tabular）**：

```json
{
    "model_type": "lgbm",
    "input_type": "tabular",
    "model_file": "model.pkl",
    "feature_dim": 37,
    "lookback_T": 1,
    "confidence_scale": 0.0512,
    "created_at": "2026-04-08T12:00:00Z",
    "n_train_samples": 12000,
    "lgb_best_iteration": 187
}
```

**向后兼容**：`LGBMBackend.load()` 检测到参数为文件（而非目录）时，回退到旧格式 pickle 加载。

---

## 7. 实施阶段

### Phase 0: 基础设施（0.5 天）

**交付物**：
- `backend_protocol.py` -- `ModelBackend` Protocol + `BackendMeta`
- `backends/__init__.py` -- `BACKEND_REGISTRY` + `create_backend()` 工厂函数
- `Layer1Config` 新增字段 + validator

**依赖**：无

### Phase 1: LGBMBackend 重构（1 天）

**交付物**：
- `backends/lgbm_backend.py` -- 从 `model.py` 重构，实现 `ModelBackend` Protocol
- `model.py` -> re-export shim（向后兼容）
- `layer1.py` 修改 `__init__` 使用 `create_backend()`
- 所有现有测试通过（行为 100% 不变）

**依赖**：Phase 0
**验证**：`pytest tests/` 全量通过

### Phase 2: XGBoost + sklearn Backends（1.5 天）

**交付物**：
- `backends/_sklearn_base.py`
- `backends/xgb_backend.py`（第一个非 LightGBM tabular 后端，验证基础设施）
- `backends/ridge_backend.py`、`backends/elasticnet_backend.py`、`backends/rf_backend.py`

**依赖**：Phase 1
**验证**：每个 backend 的 fit/predict/save/load 单元测试

### Phase 3: SequenceFeatureBuilder（1 天）

**交付物**：
- `sequence_features.py`
- 单元测试：no-lookahead 验证、NaN 处理、形状校验

**依赖**：Phase 0（不依赖 Phase 1/2，可并行开发）

### Phase 4: SequentialBackend 基类 + DLinear（1.5 天）

**交付物**：
- `backends/_seq_base.py` -- PyTorch 训练循环基类
- `backends/dlinear_backend.py`
- `layer1.py` 新增 `_fit_pooled_sequential()` 和 `_compute_v2_sequential()`

**依赖**：Phase 3
**验证**：DLinear 端到端测试 + `model_type=dlinear` 小规模回测

> **端到端测试里程碑**：Phase 4 完成后，tabular (lgbm) 和 sequential (dlinear) 两条路径均已可用，整个 Layer1 -> BacktestEngine 管线可完整运行。

### Phase 5: PatchTST + LSTM/GRU（1.5 天）

**交付物**：
- `backends/patchtst_backend.py`
- `backends/lstm_backend.py`
- 对应单元测试

**依赖**：Phase 4

### Phase 6: 模型持久化 + 加载（0.5 天）

**交付物**：
- `meta.json` 统一格式实现（含 `model_file` 字段）
- `load_model()` 方法（自动识别后端类型）
- 向后兼容测试（加载旧 pickle 格式）

**依赖**：Phase 5

### Phase 7: 集成测试 + 文档（1 天）

**交付物**：
- 跨后端一致性测试（验证所有 backend 输出格式一致）
- 回测对比报告（LGBM vs XGB vs DLinear vs PatchTST）
- 更新配置文档

**依赖**：Phase 6

### 依赖关系图

```
Phase 0 (基础设施)
  +-- Phase 1 (LGBMBackend)
  |   +-- Phase 2 (XGB + sklearn backends)
  |       +--+
  +-- Phase 3 (SequenceFeatureBuilder)     <- 可与 Phase 1/2 并行
      +-- Phase 4 (seq base + DLinear)
          +-- Phase 5 (PatchTST + LSTM)
              +-- Phase 6 (持久化)
                  +-- Phase 7 (集成)

端到端测试里程碑: Phase 4 完成后
```

**总工期**：约 7-8.5 人天（Phase 2 和 Phase 3 并行时约 7 天）。

---

## 8. 测试策略

### 新增测试文件

| 测试文件 | 覆盖范围 |
|---------|---------|
| `tests/test_backend_protocol.py` | 所有 backend 满足 `ModelBackend` Protocol（isinstance 检查 + 签名校验 + 未训练时 predict 返回 zeros） |
| `tests/test_lgbm_backend.py` | LGBMBackend fit/predict/save/load，与旧 `Layer1Model` 行为一致性对比 |
| `tests/test_sklearn_backends.py` | XGB/Ridge/ElasticNet/RF 的 fit/predict/save/load，共用 parametrize fixture |
| `tests/test_seq_feature_builder.py` | SequenceFeatureBuilder 形状校验、no-lookahead 验证、NaN 处理、与 FeatureBuilderV2 逐资产特征一致性 |
| `tests/test_seq_backends.py` | DLinear/PatchTST/LSTM 的 fit/predict/save/load，CPU 模式 + 小数据集 |
| `tests/test_backend_registry.py` | `create_backend()` 工厂函数，未知 model_type 错误处理 |
| `tests/test_layer1_routing.py` | Layer1 类根据 `model_type` 正确路由到 tabular/sequential 路径 |
| `tests/test_model_persistence.py` | meta.json 格式验证、跨后端 save/load 往返、向后兼容旧 pickle |

### 无 GPU CI 环境中测试 PyTorch 模型

1. **CPU 强制**：所有测试 fixture 设置 `os.environ["CUDA_VISIBLE_DEVICES"] = ""`
2. **小规模数据**：N=200, T=32, D=14（训练 <5 秒 / 后端）
3. **确定性**：`torch.manual_seed(42)` + `np.random.seed(42)`
4. **Epochs 缩减**：测试时 `hparams={"epochs": 3, "patience": 2}`
5. **Skip 策略**：`pytest.importorskip("torch")` 跳过 sequential 测试（torch 未安装时）
6. **数值精度**：`np.allclose(atol=1e-4)` 而非严格相等

---

## 9. 风险分析

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| Sequential 模型过拟合 | 高 | 回测漂亮但实盘失效 | Huber 损失 + early stopping + dropout；OOS 验证必须通过才部署 |
| PyTorch 安装/版本冲突 | 中 | CI 构建失败 | 可选依赖 + `importorskip` + CPU-only 安装路径 |
| 特征一致性漂移 | 中 | tabular/sequential 路径输出特征语义不一致 | 单元测试验证逐资产 14 维特征在两条路径上完全一致 |
| 训练/推理 PCA 不一致 | 低 | 推理时 PCA 投影方向翻转 | 冻结模式 + 显式 `pca_universe_symbols` 校验 |
| Sequential 训练耗时 | 中 | 训练流水线阻塞 | DLinear CPU <1 分钟；PatchTST/LSTM 推荐 GPU；训练脚本支持断点续训 |
| 置信度跨后端不可比 | 低 | 不同 backend 的 confidence 量纲不一致 | 统一使用 P90 归一化 + `confidence_temperature` 运维旋钮 |
| 向后兼容断裂 | 低 | 旧模型无法加载 | `LGBMBackend.load()` 检测文件 vs 目录自动回退 |

---

## 10. 依赖管理

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "pydantic>=2.0",
    "lightgbm>=4.0",
    "joblib>=1.3",
]

[project.optional-dependencies]
sklearn = ["scikit-learn>=1.3"]
xgb = ["xgboost>=2.0", "scikit-learn>=1.3"]
torch-cpu = ["torch>=2.1"]
torch-gpu = ["torch>=2.1"]
all = ["scikit-learn>=1.3", "xgboost>=2.0", "torch>=2.1"]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "scikit-learn>=1.3", "xgboost>=2.0", "torch>=2.1"]
```

**PyTorch 安装**：
- CPU（默认）：`pip install torch --index-url https://download.pytorch.org/whl/cpu`（~200MB）
- GPU：`pip install torch --index-url https://download.pytorch.org/whl/cu121`（~2GB）
- 代码通过 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 自动检测

**版本约束**：

| 依赖 | 最低版本 | 约束原因 |
|------|---------|---------|
| lightgbm>=4.0 | Dataset API + categorical 改进 |
| scikit-learn>=1.3 | Pipeline 改进 |
| xgboost>=2.0 | 原生 categorical 支持 |
| torch>=2.1 | `weights_only=True` 安全加载 |
