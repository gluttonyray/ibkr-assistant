# Route B 架构设计 — Section 2: 策略层

> 作者：@strategist
> 日期：2026-04-08
> 状态：Draft — 待审批

---

## 2.1 模型后端分类与特征输入

Route B 的核心设计决策是将模型后端分为两类，共享统一的推理接口但接受不同形状的输入。

### Tabular 后端

**输入形状：(N, 37)**

沿用现有 `FeatureBuilderV2` 产出的 37 维快照特征向量。每个样本 `(asset, time)` 对应一行独立的特征快照，时间维度信息已被手工聚合到动量/波动率等衍生特征中。

**适用模型：** LightGBM, XGBoost, Ridge, ElasticNet, Random Forest

**优势：**
- 与现有架构完全兼容，零迁移成本
- 树模型天然支持类别型特征（indices 33-36），无需额外编码
- 训练速度快，CPU 即可（LightGBM 300 轮 ~10 秒）
- 可解释性好（特征重要度、SHAP）

**局限：**
- 手工特征工程的信息瓶颈——37 维快照无法捕捉原始序列中的时序模式
- 收益率回看窗口固定在 {1, 5, 20, 60, 120} bars，无法自适应发现最优窗口

### Sequential 后端

**输入形状：(N, T, D)**

原始/轻度处理的时间序列张量。`T` 为 lookback window 长度，`D` 为每个时间步的特征维度。模型自主从序列中学习时序模式。

**适用模型：** DLinear, PatchTST, LSTM/GRU

**优势：**
- 端到端学习时序依赖关系，无需手工设定回看窗口
- PatchTST 的 attention 机制可发现非局部时序模式
- DLinear 的分解结构天然捕捉趋势 + 季节性

**局限：**
- 需要 GPU 训练（推理可 CPU）
- 过拟合风险高于树模型，需要更谨慎的正则化
- 类别型特征无法直接输入序列（需单独处理或丢弃）
- 多品种对齐问题更复杂

### 关键区别总结

| 维度 | Tabular | Sequential |
|------|---------|------------|
| 输入形状 | (N, 37) | (N, T, D) |
| 时间信息 | 手工聚合至快照 | 原始序列保留 |
| 特征工程 | 重（FeatureBuilderV2） | 轻（标准化即可） |
| 类别特征 | 原生支持 | 作为 static covariates 单独注入 |
| 训练设备 | CPU | GPU（推荐）/ CPU（DLinear） |
| 样本效率 | 高 | 低（需更多数据） |

---

## 2.2 ModelBackend Protocol 设计

定义统一的 `ModelBackend` Protocol，使 `Layer1` 类对具体模型实现无感知。

```python
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pathlib import Path


@runtime_checkable
class ModelBackend(Protocol):
    """Layer1 模型后端的统一协议。

    所有后端必须实现此协议，确保 Layer1 类可以无差别地
    调用 fit / predict / save / load。

    类型约定：
    - Tabular 后端: X 形状 (N, 37)，2D ndarray
    - Sequential 后端: X 形状 (N, T, D)，3D ndarray
    """

    @property
    def input_type(self) -> Literal["tabular", "sequential"]:
        """声明此后端期望的输入类型。

        Layer1 根据此属性决定调用 FeatureBuilderV2（tabular）
        还是 SequenceFeatureBuilder（sequential）。
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """模型是否已完成训练/加载。"""
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """训练模型。

        Parameters
        ----------
        X : ndarray
            Tabular: (N, 37)  Sequential: (N, T, D)
        y : ndarray (N,)
            波动率缩放前向收益率（已 Winsorize）。
        sample_weights : ndarray (N,) or None
            逆频率权重。
        """
        ...

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        """单样本推理，返回 (beta_score, confidence)。

        Parameters
        ----------
        X : ndarray
            Tabular: (37,)  Sequential: (T, D) 或 (1, T, D)

        Returns
        -------
        beta_score : float
            方向性预测（正=看多，负=看空）。
        confidence : float ∈ [0, 1]
            校准后的预测置信度。
        """
        ...

    def save(self, path: str | Path) -> None:
        """序列化模型到指定路径。"""
        ...

    def load(self, path: str | Path) -> None:
        """从指定路径加载模型。"""
        ...
```

### 现有 Layer1Model 的适配

现有 `Layer1Model` 已天然满足此 Protocol 的核心接口（`fit`, `predict`, `save`, `load`）。仅需新增一个属性：

```python
class Layer1Model:  # 现有类
    @property
    def input_type(self) -> Literal["tabular", "sequential"]:
        return "tabular"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
```

### 后端注册与工厂

```python
# src/quant/strategy/layer1/backends/__init__.py

from typing import Callable
from quant.config.schema import Layer1Config

_REGISTRY: dict[str, Callable[[Layer1Config], ModelBackend]] = {}


def register_backend(name: str):
    """装饰器：注册模型后端工厂函数。"""
    def decorator(factory: Callable[[Layer1Config], ModelBackend]):
        _REGISTRY[name] = factory
        return factory
    return decorator


def create_backend(name: str, cfg: Layer1Config) -> ModelBackend:
    """根据名称创建模型后端实例。"""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](cfg)
```

注册示例：

```python
@register_backend("lightgbm")
def _create_lgb(cfg: Layer1Config) -> ModelBackend:
    return Layer1Model(cfg)

@register_backend("xgboost")
def _create_xgb(cfg: Layer1Config) -> ModelBackend:
    return XGBoostBackend(cfg)

@register_backend("dlinear")
def _create_dlinear(cfg: Layer1Config) -> ModelBackend:
    return DLinearBackend(cfg)
```

---

## 2.3 Sequential 模型设计原则

### 2.3.1 Lookback Window T

| 参数 | 推荐范围 | 推荐默认值 | 理由 |
|------|----------|-----------|------|
| T（bars） | 60–240 | 120 | 120 bars × 15min = 30 小时 ≈ 2 个完整交易日。覆盖日内模式（美股 RTH 26 bars，港股 RTH 18 bars）+ 隔夜变化。|

**下界 60 bars 的理由：**
- 现有 `vol_lookback=60` 用于波动率缩放，低于此值无法可靠估计波动率
- DLinear 的趋势/季节性分解至少需要 1-2 个完整交易日才有意义

**上界 240 bars 的理由：**
- 240 bars = 60 小时 ≈ 4-5 交易日，超过此范围的金融时间序列自相关性急剧衰减
- 内存开销线性增长：`N_samples × T × D × 4 bytes`；T=240, D=12 时，单样本 ~11 KB
- PatchTST 的 attention 复杂度为 O(T/P)^2，过大的 T 导致训练缓慢

**多品种 T 一致性：** 所有品种使用相同的 T 值。不同交易时段的品种（美股 26 bars/天 vs 港股 18 bars/天）通过 padding 对齐（见 2.3.5）。

### 2.3.2 DLinear 设计

DLinear 将输入序列分解为趋势（moving average）和残差（remainder），分别通过两个独立的线性层映射到预测输出。

**特征选择（D 维）：**
DLinear 适合低维、有明确时序结构的输入。推荐 D=6 的核心子集：

| 索引 | 特征 | 理由 |
|------|------|------|
| 0 | vol_scaled_close_ret_1 | 波动率缩放 1-bar 收益率，核心价格信号 |
| 1 | vol_scaled_close_ret_5 | 5-bar 中频动量 |
| 2 | vol_20 | 短期波动率水平 |
| 3 | volume_ratio | 成交量异常度 |
| 4 | composite_ew_ret_1 | 全品种等权 1-bar 收益率（市场因子） |
| 5 | composite_iv_ret_1 | 逆波动率加权 1-bar 收益率（质量因子） |

**DLinear 特有参数：**

```python
@dataclass
class DLinearParams:
    kernel_size: int = 25     # 移动平均窗口，25 bars ≈ 1 交易日
    individual: bool = True   # True = 每个特征维度独立线性层
                              # False = 所有维度共享（当 D 小时推荐 True）
```

**kernel_size 建议：** 25（约 1 个美股 RTH 交易日）。这使趋势分量捕获日级方向，残差分量捕获日内波动。

### 2.3.3 PatchTST 设计

PatchTST 将长序列切分为固定大小的 patch，每个 patch 作为一个 token 输入 Transformer encoder。

**Patch 参数建议：**

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| patch_len (P) | 12 | 12 bars × 15min = 3 小时，约半个 RTH 交易时段 |
| stride (S) | 6 | 50% overlap，平衡分辨率与计算成本 |
| d_model | 64 | 输入维度 D=6-12 → 64 维 embedding 足够 |
| n_heads | 4 | d_model/n_heads = 16，标准配置 |
| n_layers | 2 | 序列长度有限（T/S ≈ 20 tokens），2 层足以 |
| dropout | 0.2 | 金融数据噪声大，需较强正则化 |

**Token 数计算：** T=120, P=12, S=6 → `(120-12)/6 + 1 = 19` tokens。Attention 矩阵仅 19×19，计算量可控。

**Channel Independence：** 推荐启用（PatchTST 原论文的核心贡献），即每个特征维度独立处理再聚合。这避免了跨维度的虚假相关，对金融数据尤为重要。

### 2.3.4 LSTM/GRU 设计

**参数建议：**

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| hidden_dim | 64 | 与 PatchTST d_model 一致，防止过拟合 |
| num_layers | 2 | 2 层 GRU，配合 residual connection |
| dropout | 0.2 | 层间 dropout |
| bidirectional | False | 因果序列，禁止双向 |
| cell_type | GRU | 优先 GRU（参数更少，收敛更快，金融场景经验上优于 LSTM） |

**推理输出：** 取最后一个时间步的隐状态 `h[-1]`，经一个线性层映射到标量 `beta_score`。

### 2.3.5 Pooled 训练下的序列对齐

Pooled 训练（多品种共享一个模型）在 sequential 模式下面临一个 tabular 模式没有的对齐问题：不同品种的交易时段不同，导致同一时间窗口内有效 bar 数不同。

**方案：左 padding + staleness mask**

```
美股 ES（26 bars/天）:  [bar_1, bar_2, ..., bar_120]      # 完整
港股 HSI（18 bars/天）: [pad, pad, ..., bar_1, ..., bar_84] # 左侧填零
```

具体规则：
1. 所有品种使用统一的 T=120 窗口
2. 非交易时段的 bar 填零（zero padding）
3. 序列特征中加入一个二值维度 `is_active`（1=真实 bar，0=padding）
4. 模型训练时传入 padding mask，使 attention / RNN 忽略 padding 位置

**为什么不用插值/forward fill：** 非交易时段的价格没有发生变化，forward fill 收益率会人为引入零收益率 bar，污染波动率估计。显式 padding + mask 更干净。

**Pooled 训练的额外约束：**
- 类别型特征（asset_class, exchange, currency, region）作为 static covariates 注入，而非时序维度。具体方式是通过一个 embedding 层将 4 个整数映射为一个固定向量，与序列 encoder 的输出拼接后送入最终的线性 head。
- `sample_weights`（逆频率权重）沿用现有 `_inverse_frequency_weights()` 逻辑，无变化。

---

## 2.4 特征构建架构重设计

### 方案对比

| 方案 | 优势 | 劣势 |
|------|------|------|
| **A: 扩展 FeatureBuilderV2** | 单一入口，改动少 | V2 的 `build()` 返回 1D 快照，改为返回 2D 序列是破坏性变更；`FEATURE_DIM=37` 硬编码在类级别 |
| **B: 新建 SequenceFeatureBuilder** | 职责清晰；不影响现有 tabular 路径；D 可独立于 37 维变化 | 新增一个类，需维护两套特征构建逻辑 |

**推荐方案：B（新建 SequenceFeatureBuilder）**

理由：
1. **输入形状根本不同**：V2 输出 `(37,)` 1D 向量，序列构建器输出 `(T, D)` 2D 矩阵。强行统一接口会导致返回类型歧义。
2. **D ≠ 37**：序列模型的每个时间步只需 6-12 维原始/轻处理特征，不需要类别编码和 PCA 动量（这些是快照级聚合，对序列模型无意义）。
3. **前视风险隔离**：序列构建器的滑窗逻辑与快照构建器不同，分离可降低前视偏差引入的风险。
4. **渐进式迁移**：新建类不破坏 tabular 路径，可并行开发测试。

### SequenceFeatureBuilder 接口

```python
class SequenceFeatureBuilder:
    """为 Sequential 后端构建 (T, D) 特征张量。

    每个时间步 t ∈ [0, T) 包含 D 维特征，均为数值型。
    类别型特征作为 static covariates 单独返回。

    Parameters
    ----------
    lookback_T : int
        序列长度 T。
    feature_set : list[str]
        启用的特征名称列表。
    vol_lookback : int
        波动率缩放回看窗口。
    """

    def build(
        self,
        instrument: Instrument,
        asset_bars: list[Bar],
        global_bars: dict[str, list[Bar]],
        as_of_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (sequence, static_covariates)。

        Returns
        -------
        sequence : ndarray (T, D)
            时序特征张量。若实际 bar 数 < T，左侧 zero-pad。
        static_covariates : ndarray (4,)
            [asset_class, exchange, currency, region] 整数编码。
        """
        ...
```

### D 维特征推荐核心子集

按信噪比和时序结构评估，推荐以下 D=8 维特征（可配置扩展至 12 维）：

**核心 8 维（推荐默认）：**

| 索引 | 特征名 | 来源 | 理由 |
|------|--------|------|------|
| 0 | vol_scaled_ret_1 | 逐 bar 计算 | 核心价格信号，波动率归一化后跨品种可比 |
| 1 | vol_20 | 滚动 20-bar std | 波动率状态，regime 信息的直接代理 |
| 2 | vol_ratio | vol_20 / vol_60 | 波动率变化趋势（vol regime shift 指标） |
| 3 | volume_ratio | vol[-1] / mean(vol[-20:]) | 成交量异常度，流动性信号 |
| 4 | trend_consistency_20 | 滚动计算 | 趋势持续性 |
| 5 | composite_ew_ret_1 | GlobalState | 市场因子（1-bar 等权收益率） |
| 6 | pc1_score | GlobalState PCA | 市场主成分暴露 |
| 7 | is_active | 二值 | Padding mask（0=非交易时段/padding） |

**可选扩展 4 维（D=12）：**

| 索引 | 特征名 | 理由 |
|------|--------|------|
| 8 | pc2_score | 第二主成分（通常为美/港分化因子） |
| 9 | composite_iv_ret_1 | 质量因子 |
| 10 | spread_pct | bid-ask spread 占比（若可获取） |
| 11 | hour_of_day_sin | 时间编码（正弦），捕获日内季节性 |

### Sliding Window 构建（避免前视偏差）

```
训练时（bar_index = i, T = 120, forward_horizon = 26）:

  序列窗口:  bars[i-T : i]     # 严格只看过去 T 根 bar
  标签:      bars[i + forward_horizon].close  # 前向收益率

  逐步滑动: i = T, T+1, T+2, ..., len(bars) - forward_horizon

  GlobalState 在每个 i 处使用 bars[:i]（无前视）
  vol_scaled_ret_1 在每个时间步 t 使用 bars[t-vol_lookback:t+1]（无前视）
```

**关键约束：** `SequenceFeatureBuilder.build()` 的 `as_of_index` 参数标定当前时间。所有特征计算严格限制在 `[:as_of_index+1]` 范围内。GlobalState 的 PCA/复合收益率也必须使用截至该时间的数据。

---

## 2.5 置信度统一化

两类后端均必须输出 `(beta_score, confidence)` 二元组，其中 `confidence ∈ [0, 1]`。

### Tabular 后端

沿用现有机制，无需修改：

```
confidence = min(|beta_score| / confidence_scale, 1.0)
confidence_scale = P90(|y_pred_train|) + eps
```

此方法适用于所有 tabular 后端（LightGBM、XGBoost、Ridge 等），因为它们的 `predict()` 输出均为无界标量。

Ridge / ElasticNet 的额外说明：线性模型的预测值分布比树模型更集中（方差更小），P90 归一化仍然有效，但 `confidence_scale` 数值可能偏小，导致 confidence 整体偏高。若实测发现此问题，可改用 P75 或引入温度参数 `confidence_temperature`。

### Sequential 后端

Sequential 模型的输出层是一个线性 head（`nn.Linear(hidden_dim, 1)`），输出无界标量 `beta_score`。置信度映射方案：

**方案：训练集 P90 归一化（与 tabular 一致）**

```python
# 训练完成后
y_pred_train = model.predict_batch(X_train)  # 批量推理
confidence_scale = np.percentile(np.abs(y_pred_train), 90) + eps

# 推理时
beta_score = model.forward(x).item()
confidence = min(abs(beta_score) / confidence_scale, 1.0)
```

此方案的关键优势是 tabular 和 sequential 后端使用**完全相同的 confidence 计算逻辑**，不需要在 Layer1/Layer2 层面做分支判断。

**已考虑但排除的替代方案：**

| 方案 | 排除理由 |
|------|----------|
| MC Dropout（多次采样取 std） | 推理延迟 ~5x，日内交易不可接受 |
| Deep Ensemble | 需训练 3-5 个模型，存储和计算成本过高 |
| 额外 confidence head | 需要额外的校准数据和损失函数，增加复杂度 |

### confidence_scale 的序列化

`confidence_scale` 作为模型产出物的一部分保存在 `model.pkl`（或对应的序列化文件）中。这已由现有 `Layer1Model.save()` 实现，所有新的 `ModelBackend` 实现必须遵循相同约定。

---

## 2.6 Layer1Config 扩展

### 新增配置项

```python
class Layer1Config(BaseModel):
    # --- 现有字段保持不变 ---
    # ...

    # --- Route B 新增 ---

    # 模型后端类型
    model_type: str = Field(
        "lightgbm",
        description="模型后端名称。"
        "Tabular: 'lightgbm', 'xgboost', 'ridge', 'elasticnet', 'random_forest'. "
        "Sequential: 'dlinear', 'patchtst', 'lstm', 'gru'."
    )

    # Sequential 后端专用参数
    seq_lookback_T: int = Field(
        120, ge=60, le=480,
        description="序列 lookback 窗口长度（bars）。"
        "120 bars × 15min = 30 小时 ≈ 2 交易日。"
    )
    seq_feature_set: list[str] = Field(
        default_factory=lambda: [
            "vol_scaled_ret_1", "vol_20", "vol_ratio", "volume_ratio",
            "trend_consistency_20", "composite_ew_ret_1", "pc1_score", "is_active",
        ],
        description="序列模型的特征维度名称列表。"
    )

    # DLinear 参数
    dlinear_kernel_size: int = Field(
        25, ge=3,
        description="DLinear 移动平均窗口。25 bars ≈ 1 个 RTH 交易日。"
    )

    # PatchTST 参数
    patchtst_patch_len: int = Field(12, ge=4)
    patchtst_stride: int = Field(6, ge=1)
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

    # confidence 温度（可选微调）
    confidence_temperature: float = Field(
        1.0, gt=0,
        description="confidence_scale 的乘数。>1 降低 confidence，<1 提高。"
    )
```

### 模型类型验证

```python
from pydantic import field_validator

TABULAR_BACKENDS = {"lightgbm", "xgboost", "ridge", "elasticnet", "random_forest"}
SEQUENTIAL_BACKENDS = {"dlinear", "patchtst", "lstm", "gru"}
ALL_BACKENDS = TABULAR_BACKENDS | SEQUENTIAL_BACKENDS

class Layer1Config(BaseModel):
    # ...

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        if v not in ALL_BACKENDS:
            raise ValueError(
                f"Unknown model_type '{v}'. "
                f"Available: {sorted(ALL_BACKENDS)}"
            )
        return v
```

### 框架选择建议

**推荐：PyTorch**

| 维度 | PyTorch | JAX |
|------|---------|-----|
| 生态成熟度 | PatchTST 官方实现为 PyTorch；TSLib、Hugging Face 均基于 PyTorch | 金融时序库较少 |
| 调试体验 | 动态图，pdb 可逐行调试 | 需要 jit 后才有性能，调试需 jax.debug.print |
| 部署 | `torch.jit.trace` / ONNX 导出 | jax2tf 可行但生态不成熟 |
| GPU 支持 | CUDA/MPS 均稳定 | Apple Silicon 支持不完整 |
| 团队熟悉度 | 主流选择 | 学习曲线陡峭 |

决策：使用 **PyTorch** 作为 sequential 后端的唯一框架。DLinear 和 PatchTST 代码可直接从 TSLib / Time-Series-Library 适配。

**依赖管理：** `torch` 为可选依赖。Tabular 后端不需要 PyTorch。

```
# requirements-sequential.txt（仅 sequential 后端需要）
torch >= 2.0
```

运行时若未安装 torch 且 `model_type` 为 sequential 类型，`create_backend()` 应抛出 `ImportError` 并附带清晰的安装指引。

---

## 2.7 训练/推理分离在 Route B 中的影响

现有设计文档（`design_train_infer_separation.md`）定义了 4-Phase 实施计划。以下分析 Route B 对每个 Phase 的影响。

### 不变的部分

| 内容 | 理由 |
|------|------|
| Phase 1 核心（GlobalStateBuilder frozen 模式） | PCA 冻结与模型后端无关，tabular/sequential 均需要 |
| Phase 2 核心（离线训练脚本框架） | `scripts/train_layer1.py` 的 CLI + 数据加载 + OOS 验证 + meta.json 逻辑不变 |
| Phase 3 核心（BacktestEngine 推理解耦） | 删除训练触发逻辑对所有后端一致 |
| Phase 4 核心（测试 + 对比基线） | 验证流程不变 |
| Layer2 处理策略 | ICWeightCalculator 保持在线滚动更新，与 Layer1 后端选择无关 |
| PCA 品种域约束 | 显式配置 `pca_universe_symbols` 的要求不变 |

### 需要调整的部分

#### Phase 1 扩展：ModelBackend Protocol + SequenceFeatureBuilder

| 新增内容 | 说明 |
|----------|------|
| `ModelBackend` Protocol 定义 | 新文件 `src/quant/strategy/layer1/backends/protocol.py` |
| 后端注册机制 | 新文件 `src/quant/strategy/layer1/backends/__init__.py` |
| `SequenceFeatureBuilder` | 新文件 `src/quant/strategy/layer1/seq_features.py` |
| `Layer1Config` 新增字段 | 修改 `src/quant/config/schema.py` |
| `Layer1Model` 适配 Protocol | 修改 `src/quant/strategy/layer1/model.py`（新增 `input_type` 属性） |

#### Phase 2 扩展：离线训练脚本支持多后端

```
# 调整后的 train_layer1.py 流程

1. load_config(args.config)
2. 校验 pca_universe_symbols 非空
3. 加载数据
4. 根据 cfg.model_type 创建 ModelBackend（通过工厂）
5. 根据 backend.input_type 选择特征构建器：
   - "tabular" → 现有 FeatureBuilderV2 逻辑
   - "sequential" → SequenceFeatureBuilder（新增）
6. 构建 X, y
7. backend.fit(X, y, sample_weights)
8. OOS 验证（预测接口统一，无需分支）
9. backend.save(model_dir / "model.pkl")
10. 保存 meta.json（新增 model_type 字段）
```

#### Phase 2 新增：模型产出物变化

```
models/
  layer1/
    lgb_v2_20260401_13000/         # Tabular：与现有一致
      model.pkl                     # lgb.Booster + confidence_scale
      global_state.pkl
      meta.json

    dlinear_v2_20260401_13000/     # Sequential：新增格式
      model.pt                     # PyTorch state_dict
      model_config.json            # 模型超参数（hidden_dim 等）
      confidence_scale.json        # {"confidence_scale": 0.847}
      global_state.pkl             # PCA 冻结状态（同上）
      meta.json                    # 新增 "model_type": "dlinear"
```

**Sequential 模型序列化决策：**
- 使用 `torch.save(model.state_dict(), ...)` 而非 `pickle`
- 模型结构参数单独存为 `model_config.json`，加载时先重建结构再加载权重
- 这比 `torch.save(model, ...)` 更安全（不依赖代码路径）

#### Phase 3 调整：Layer1.load_model() 支持多后端

```python
class Layer1:
    def load_model(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)

        # 从 meta.json 读取 model_type
        meta_path = model_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        model_type = meta.get("model_type", "lightgbm")

        # 通过工厂创建正确的后端
        backend = create_backend(model_type, self._cfg)
        backend.load(model_dir / "model.pkl")  # 或 model.pt
        self._models[_POOLED_KEY] = backend

        # PCA 冻结状态（与后端无关）
        if self._use_v2:
            self._global_builder.load_pca_state(model_dir / "global_state.pkl")

        # feature_version 校验
        if meta.get("feature_version") != self._cfg.feature_version:
            raise ValueError(...)
```

#### Phase 3 调整：Layer1.compute() 路由

```python
def _compute_v2(self, windows):
    backend = self._models.get(_POOLED_KEY)

    # 根据 backend.input_type 选择特征构建方式
    if backend is not None and backend.input_type == "sequential":
        # 使用 SequenceFeatureBuilder
        for inst, bars in windows.items():
            seq, static_cov = self._seq_feature_builder.build(...)
            beta_score, confidence = backend.predict(seq)
            # ...
    else:
        # 现有 FeatureBuilderV2 路径（不变）
        for inst, bars in windows.items():
            feat = self._feature_builder_v2.build(...)
            beta_score, confidence = backend.predict(feat)
            # ...
```

### 修订后的 Phase 计划

| Phase | 原计划 | Route B 调整 |
|-------|--------|-------------|
| **1a** | GlobalStateBuilder frozen + Layer1.load_model() | 不变 |
| **1b** | *(新增)* | ModelBackend Protocol + 注册机制 + Layer1Model 适配 |
| **1c** | *(新增)* | Layer1Config 新增字段 + validator |
| **2a** | 离线训练脚本 | 调整为多后端路由 |
| **2b** | *(新增)* | SequenceFeatureBuilder 实现 |
| **2c** | *(新增)* | 首个 Sequential 后端（DLinear，最简单）+ 端到端测试 |
| **3** | 推理路径解耦 | Layer1.compute() 新增 sequential 分支 |
| **4** | 验证 + 清理 | 新增 sequential vs tabular 对比基线 |

---

## @engineer-b: 需要工程确认的问题

1. **PyTorch 依赖策略：** `torch` 作为可选依赖（`pip install .[sequential]`），还是作为核心依赖？前者需要在 `create_backend()` 中做 `ImportError` 处理。

2. **SequenceFeatureBuilder 中的 GlobalState 复用：** 序列构建器需要逐时间步计算 `composite_ew_ret_1` 和 `pc1_score`，这意味着需要对每个 `t ∈ [0, T)` 调用一次 `GlobalStateBuilder.update()`。这在训练时可能很慢（T=120 意味着特征构建慢 120 倍）。工程确认：是否可以接受，还是需要设计批量化的 `GlobalStateBuilder.update_batch()` 接口？

3. **GPU 推理延迟：** 单样本推理时，PyTorch GPU kernel launch 的开销可能比 CPU 推理更高（小 batch 场景）。实盘推理是否应强制 CPU？建议先 benchmark 再决策。

4. **Sequential 模型的 ONNX 导出：** 是否需要在 Phase 2 就支持 ONNX 导出（用于无 PyTorch 依赖的推理环境），还是推迟到后续迭代？

5. **`model.pt` vs `model.pkl` 命名：** Sequential 后端使用 `torch.save()` 保存 state_dict，建议统一扩展名为 `.pt`。但这会破坏 `Layer1.load_model()` 中硬编码的 `model.pkl` 路径。建议在 `meta.json` 中新增 `"model_file"` 字段指定实际文件名，消除扩展名假设。工程确认是否可行？

6. **`confidence_temperature` 的调参方式：** 该参数是在训练时固定（写入 meta.json），还是在推理时可配置（仅存在于 Layer1Config 中）？建议后者（推理时可调），但需确认这是否违反"模型冻结"原则。

7. **多品种 padding 对齐在回测引擎中的实现：** 当前 `BacktestEngine` 按 aligned_timestamps 遍历 bar，不同品种的 bar 已经时间对齐。Sequential 特征构建器需要知道某个时间步是否为"真实 bar"——这个信息目前是否可从 `windows: dict[Instrument, list[Bar]]` 中推导？还是需要引擎传递额外的 mask？

8. **XGBoost / Ridge / ElasticNet / Random Forest 后端的优先级：** Route B 设计覆盖了全部 5 个 tabular 和 3 个 sequential 后端。建议实施顺序为：LightGBM（已有）→ DLinear → PatchTST → GRU → XGBoost → 其余。工程确认此顺序是否合理？

---

## 对 @engineer-b 的策略回答

以下逐条回答工程设计文档（`design_route_b_engineering.md`）末尾 `@strategist-b` 部分的 8 个策略问题。

### Q1: Sequential 特征维度 D=14 vs D=37 — 是否需要在每个时步注入全局状态特征？

**结论：不需要。D=14（仅逐资产价量特征）是正确的设计。**

理由：工程师准确识别了核心矛盾——逐时步重建 GlobalState 的开销为 O(T x N_universe)，在 T=96、N_universe=7 的场景下，训练特征构建速度下降约 100 倍，这是不可接受的。更关键的是，PCA 得分和复合收益率本质上是*跨品种的截面统计量*，其信息含量在时间维度上变化缓慢（PCA 每 20 bars 更新一次），对序列模型来说几乎是常数。序列模型的价值在于从原始价量序列中自主发现时序依赖——如果我们把 PCA 塞进去，反而是在用手工聚合的信号替代模型本应自主学习的模式。14 维逐资产特征 + 类别型 static covariates 的组合已经足够。

### Q2: seq_lookback_T 默认值 96 vs 策略文档建议的 120 — 是否需要 multi-scale 输入？

**结论：接受 T=96 作为默认值，不需要 multi-scale 输入。**

理由：策略文档建议的 T=120 基于"覆盖 2 个完整交易日"的直觉，但 T=96（24 小时 = 1.5 个 RTH 交易日）已经覆盖了完整的日内周期模式（开盘-午盘-收盘）。工程师选择 96 使得 PatchTST 的 patch 对齐更整齐（96/16=6 个 patch，96/8=12 个 patch，均为整数），这对 Transformer 的位置编码和 attention 计算有实际好处。T=120 并不会带来质的信息增量，因为美股和港股的 RTH 分别只有 26 和 18 个 15min bars，120 bars 的额外 24 bars 几乎全是盘后/盘前低流动性时段。Multi-scale 输入（如同时输入 T=32 和 T=96）增加工程复杂度但回报不确定，推迟到有 backtest 证据表明单尺度不足时再考虑。

### Q3: DLinear `individual=False` vs `True` — 默认 channel-mixing 还是 channel-independent？

**结论：默认 `individual=False`（channel-mixing），但将 `individual` 作为可配置超参数暴露。**

理由：D=14 的特征之间存在已知的强交互关系——例如 `ret_20`（索引 2）和 `ret_vol_scaled_20`（索引 9）高度相关，`vol_20`（索引 5）和 `vol_ratio`（索引 7）是构成/被构成关系。Channel-mixing 模式让模型利用这些交互，参数量极小（2 x D x T + D = 2 x 14 x 96 + 14 = 2702），远不构成过拟合风险。相比之下，channel-independent 模式在特征间无关联假设下才占优，但我们的 14 维特征明确不满足这一假设。工程师默认 `individual=False` 是正确的。

### Q4: PatchTST `channel_independent=True` 默认 — CI 还是 CM？

**结论：保持默认 `channel_independent=True`（CI 模式）。**

理由：这看起来与 Q3 的结论矛盾，但 PatchTST 和 DLinear 的架构性质不同。DLinear 是线性模型，channel-mixing 仅增加极少参数（~2700）；PatchTST 的 CM 模式将 patch 维度从 `patch_size` 膨胀为 `patch_size x D = 16 x 14 = 224`，经过 2 层 Transformer 后参数量显著增加，且 attention 矩阵在高维 patch embedding 上的过拟合风险远高于 DLinear 的线性权重。PatchTST 原论文（Nie et al., 2023）在多个 benchmark 上验证了 CI 模式在小样本场景优于 CM，我们 ~10K 级训练样本正属于此范围。参数共享带来的正则化效果比跨特征交互的信息增益更有价值。保持 CI=True 默认，CM 作为高级用户的可选配置。

### Q5: 训练时 shuffle=False — 是否可以打乱 Pooled 训练集？

**结论：Sequential 后端应设 `shuffle=True`。**

理由：工程师的疑虑是合理的——`_fit_pooled` 的训练集虽然源自时序数据，但每个样本是一个独立的 `(asset, time)` 对，已经通过 sliding window 切好了固定长度的序列。样本间的时序依赖性已经被封装在每个 (T, D) 张量内部，样本之间不存在需要保持的顺序关系。不打乱会导致同一资产的连续样本组成 mini-batch，使梯度估计产生偏差（同一资产的 vol_20 等特征高度自相关），损害 SGD 收敛。注意：80/20 时间切分（train/val）仍然必须严格保持——只在训练集内部打乱，验证集不打乱。这与 tabular 路径中 LightGBM 的 `train_data = lgb.Dataset(X_train, ...)` 逻辑一致（LightGBM 内部自行采样）。

### Q6: LGBMBackend 是否改用 sklearn wrapper `lgb.LGBMRegressor`？

**结论：不改。LGBMBackend 保持原生 `lgb.train()` API，不使用 sklearn wrapper。**

理由：现有 `Layer1Model` 依赖两个 sklearn wrapper 不提供的关键功能：(1) `categorical_feature` 原生声明——FeatureBuilderV2 的 4 个类别型特征（索引 33-36）通过 `lgb.Dataset(categorical_feature=...)` 直接传入，避免了 one-hot 编码的维度膨胀和信息损失；(2) `lgb.early_stopping()` 回调的精确控制——当前代码使用 `stopping_rounds=30` 和 `lgb.log_evaluation(period=-1)` 的组合，在 sklearn wrapper 中需要额外的 `fit_params` 和 `eval_set` 传递，接口更脆弱。接口统一的收益（代码形式上更一致）不足以抵消功能损失的风险。LGBMBackend 的 `fit/predict/save/load` 已经满足 `ModelBackend` Protocol，接口在 Protocol 层面已经统一，不需要在实现层面进一步抹平差异。

### Q7: 模型版本化策略 — 文件系统 vs MLflow/DVC？

**结论：文件系统方案足够。不引入 MLflow 或 DVC。**

理由：当前系统的模型管理需求非常简单——每周训练一次，产出一个模型目录，模型数量以单位数计。MLflow/DVC 解决的是"团队协作、大规模实验追踪、模型 lineage 审计"等问题，在单人/小团队的日内交易系统中是明显的过度工程。文件系统方案 + `meta.json`（包含 OOS 指标、训练参数、时间戳）已经提供了必要的可审计性。如果未来需要对比数十个模型版本的 OOS 表现，一个简单的 Python 脚本扫描 `models/layer1/*/meta.json` 即可生成对比表，无需引入额外的依赖和学习成本。风险低，收益低，不引入。

### Q8: 标签 Winsorize 在 sequential 路径中的位置

**结论：由调用方（`_fit_pooled_sequential`）处理 Winsorize，不在 `build_training_sequences()` 内部执行。**

理由：工程师的建议完全正确。`SequenceFeatureBuilder` 是一个*特征构建器*，其职责是将原始 bar 数据转换为 (T, D) 张量和原始标签。Winsorize 是一个*训练预处理*操作，与特征构建逻辑正交。将 Winsorize 放在调用方有三个好处：(1) 特征构建器保持纯粹——其输出是客观的波动率缩放收益率，不包含任何训练偏好；(2) Winsorize 的 sigma 阈值（`label_winsorize_sigma=5.0`）来自 `Layer1Config`，在 `_fit_pooled_sequential` 中可以直接访问 `self._cfg`，而 SequenceFeatureBuilder 不应该依赖完整的 Layer1Config；(3) 与 tabular 路径一致——现有 `_fit_pooled()` 在特征构建后、`model.fit()` 前执行 `np.clip(y, -winsorize_sigma, winsorize_sigma)`，sequential 路径在相同位置做相同操作，保持了代码结构的对称性。
