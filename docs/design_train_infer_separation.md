# Layer1 训练/推理分离设计方案

> 作者：@strategist-l1 (策略) + @engineer (工程)
> 日期：2026-04-07
> 状态：Draft — 待用户审批后进入实施

---

## 1. 问题陈述

### 1.1 现状

当前 Layer1 的训练和推理**紧耦合在同一个事件循环中**：

```
BacktestEngine._run_loop()
  for bar_index, timestamp in aligned_timestamps:
      ...
      # 步骤 4b：每 130 bar 触发重训练（在线学习）
      if scheduler.should_retrain(bar_index):
          layer1.fit_window(...)      # <-- 训练
          scheduler.mark_retrained(bar_index)

      # 步骤 5：Layer1 推理
      layer1_results = layer1.compute(windows)  # <-- 推理
      ...
```

涉及文件：
- `src/quant/engine/backtest.py` 第 225-267 行（训练触发逻辑）
- `src/quant/strategy/layer1/layer1.py`（`fit_window()` + `compute()` 共存）
- `src/quant/strategy/layer1/walk_forward.py`（调度器嵌入回测引擎）

### 1.2 问题

| 问题 | 影响 |
|------|------|
| 训练耗时不可控 | 实盘环境中 `fit_window()` 可能阻塞推理循环数分钟 |
| 内存峰值 | 训练期间同时持有训练数据 + 推理窗口，内存翻倍 |
| 训练失败风险 | 若训练异常，模型状态不确定，后续推理结果不可信 |
| 回测结果不可复现 | 在线重训练使回测结果依赖于训练成功与否的随机性 |
| 模型不可审计 | 无法事后检查"回测使用的是哪个版本的模型" |

### 1.3 目标状态

```
离线训练（独立流程，跑一次产出模型文件）
  scripts/train_layer1.py --data-dir ... --train-end 2026-04-01
      ↓ 产出
  models/layer1/v2_20260401_13000/
      model.pkl + global_state.pkl + meta.json

推理（回测 / 实盘，加载冻结模型，只做 forward pass）
  main.py --backtest --data ... --model models/layer1/v2_20260401_13000/
      ↓ 加载模型
  BacktestEngine._run_loop() 中只调用 layer1.compute()，不再调用 fit_window()
```

---

## 2. 策略设计决策

### 2.1 WalkForward 重新定位

**决策：WalkForward 从推理循环中移除，转为离线训练流水线的 OOS 验证工具。**

- Walk-Forward 的本质是 Out-of-Sample 验证方法，用于评估模型泛化能力。
- 回测和实盘推理应该只做 forward pass：加载冻结模型 → `model.predict()` → 产生信号。
- `WalkForwardScheduler` 类代码保留不变（纯计数器，代码质量好），迁移到离线训练流水线中作为组件使用。

### 2.2 训练窗口与重训练频率

| 参数 | 值 | 理由 |
|------|----|------|
| 训练窗口 | `retrain_window_bars`（默认 13,000 bars ~ 100 交易日） | 覆盖约 5 个月的 15min bar，平衡市场非平稳性与样本量 |
| 重训练频率 | 每周一次 | 日内交易策略需要适应市场微观结构变化；月度太慢，日度计算成本过高 |
| 标签前瞻 | `forward_horizon_bars`（默认 26 bars ~ 6.5 小时） | 保持不变 |
| 窗口类型 | 滚动窗口（非全量历史） | 金融市场非平稳，远历史数据引入噪声 |

### 2.3 Layer2 处理

**决策：Layer2 ICWeightCalculator 保持在线滚动更新，不进行离线化。**

理由：
- `ICWeightCalculator` 是轻量级统计量（deque + Spearman rank correlation），O(N) 复杂度，无迭代优化。
- IC 权重时效性高于 Layer1 模型，冻结反而降低适应性。
- 当前 `Layer2Strategy` Protocol 中不包含训练接口，架构已天然分离。

注意：当前 `BacktestEngine._run_loop()` 未调用 `layer2.update_ic()`，IC 权重在回测中始终使用等权。这是已有功能缺口，不在本次重构范围内。

---

## 3. 工程设计

### 3.1 模型产出物与版本化存储

#### 目录结构

```
models/
  layer1/
    v2_20260401_13000/          # {feature_version}_{train_end}_{window_bars}
      model.pkl                  # lgb.Booster + confidence_scale (pickle)
      global_state.pkl           # GlobalStateBuilder PCA 快照 (pickle)
      meta.json                  # 人类可读的元数据 (JSON)
    v2_20260408_13000/
      ...
```

#### model.pkl 内容

沿用现有 `Layer1Model.save()` 格式：
```python
{
    "model": lgb.Booster,          # LightGBM 模型对象
    "confidence_scale": float,      # |y_pred_train| 的 P90
}
```

#### global_state.pkl 内容（新增）

```python
{
    "pca_components": np.ndarray,      # shape (n_components, n_symbols)
    "pca_syms_fitted": list[str],      # PCA 拟合时的品种列表和顺序
}
```

仅保存需要冻结的 PCA 状态。`_score_history` 和 `_ew_bar_ret_history`
属于推理时的滑窗统计，从零开始积累即可（warmup 期间自然填充）。

#### meta.json 内容（新增）

```json
{
    "feature_version": 2,
    "train_start": "2026-01-06T00:00:00Z",
    "train_end": "2026-04-01T00:00:00Z",
    "train_window_bars": 13000,
    "forward_horizon_bars": 26,
    "pca_n_components": 5,
    "pca_universe_symbols": ["ES", "NQ", "YM", "RTY", "HSI", "MHI", "HHI"],
    "training_symbols": ["ES", "NQ", "YM", "RTY", "HSI", "MHI", "HHI"],
    "n_training_samples": 285000,
    "oos_ic_mean": 0.032,
    "oos_ic_ir": 0.45,
    "oos_sharpe": 1.15,
    "confidence_scale": 0.847,
    "lgb_best_iteration": 187,
    "created_at": "2026-04-07T18:30:00Z"
}
```

#### 序列化格式选择

| 方案 | 优势 | 劣势 | 决策 |
|------|------|------|------|
| pickle | 与现有 `Layer1Model.save()` 一致；单文件存储复合对象 | 反序列化安全风险 | **采用**（本地生产，不从外部加载） |
| LightGBM `save_model` | 跨语言兼容 | 无法保存 confidence_scale 等额外状态 | 不采用 |
| joblib | numpy memmap 加速 | 此场景无额外优势 | 不采用 |
| JSON (meta) | 人类可读，跨版本兼容 | 无法存储二进制对象 | **仅用于元数据** |

### 3.2 GlobalStateBuilder 冻结模式

**核心挑战：训练和推理时 PCA 状态一致性。**

#### 状态分类

| 状态 | 训练时 | 推理时 | 是否序列化 |
|------|--------|--------|------------|
| `_pca_components` | 从训练数据拟合 | **冻结**（从文件加载） | 是 |
| `_pca_syms_fitted` | 从训练数据确定 | **冻结**（从文件加载） | 是 |
| `_score_history` | 逐步积累 | 从零开始滚动更新 | 否 |
| `_ew_bar_ret_history` | 逐步积累 | 从零开始滚动更新 | 否 |
| `_last_pca_bar` | 控制 PCA 更新频率 | 冻结模式下无意义 | 否 |

#### 接口变更

```python
class GlobalStateBuilder:
    def __init__(self, ..., frozen: bool = False) -> None:
        self._frozen = frozen
        # ... 其余不变

    def update(self, bar_index, windows, pca_universe=None) -> GlobalState:
        # ... 波动率缩放收益率计算（不变）
        # ... EW 历史更新（不变）

        # PCA 更新：冻结模式下跳过
        if not self._frozen:
            needs_pca_update = (
                self._pca_components is None
                or (bar_index - self._last_pca_bar) >= self._pca_update_freq
            )
            if needs_pca_update:
                self._recompute_pca(windows, pca_syms)
                self._last_pca_bar = bar_index
        # else: 使用已加载的 _pca_components，跳过重拟合

        # ... PCA 投影 + momentum 计算（不变）
        # ... 复合收益率计算（不变）

    def save_pca_state(self, path: str | Path) -> None:
        """序列化 PCA 冻结状态。"""
        payload = {
            "pca_components": self._pca_components,
            "pca_syms_fitted": getattr(self, "_pca_syms_fitted", []),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_pca_state(self, path: str | Path) -> None:
        """加载 PCA 冻结状态，并启用冻结模式。"""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._pca_components = payload["pca_components"]
        self._pca_syms_fitted = payload["pca_syms_fitted"]
        self._frozen = True
```

**改动量评估**：约 20 行新增代码，`update()` 核心逻辑仅增加一个 `if not self._frozen` 条件分支。

#### PCA 冻结状态的时间点

**规则：序列化保存的 PCA 状态必须取训练窗口最后一次 `_recompute_pca()` 的结果。**

当前 `_fit_pooled()` 内部创建临时 `GlobalStateBuilder`（layer1.py:133-138），遍历训练窗口时每 `pca_update_freq` 根 bar 调用一次 `_recompute_pca()`。由于 `_pca_components` 被最后一次调用覆盖，当前代码天然满足此规则。但此约束必须在设计文档中明确声明：离线训练脚本在调用 `save_pca_state()` 时，必须确保训练已完整遍历整个训练窗口，使得 PCA 状态反映训练窗口末尾的市场结构。

#### PCA 品种域约束

**规则：离线训练模式下，`Layer1Config.pca_universe_symbols` 必须显式配置，禁止使用默认空列表。**

理由：空列表意味着"使用所有可用品种"，但训练和推理时可用品种可能不同（品种退市/新增），导致 PCA 载荷矩阵维度不匹配。显式配置固定品种域可消除此风险。

**实现**：离线训练脚本启动时，若 `cfg.strategy.layer1.pca_universe_symbols` 为空列表，应抛出 `ValueError("pca_universe_symbols must be explicitly configured for offline training")`，而非静默使用所有品种。

### 3.3 BacktestEngine 变更

#### 删除内容

| 位置 | 内容 | 行号 |
|------|------|------|
| `__init__` 属性 | `self._training_universe` | 110 |
| `__init__` 属性 | `self._scheduler` | 114 |
| `_setup()` | `WalkForwardScheduler` 初始化 | 147-150 |
| `_run_loop()` | 步骤 4b 训练触发块 | 225-267 |
| `_run_loop()` | 训练 Universe 数据合并到推理窗口 | 286-294 |
| `run()` 签名 | `training_universe` 参数 | 119 |
| import | `from quant.strategy.layer1.walk_forward import WalkForwardScheduler` | 44 |

#### 保留内容

- `layer1` 构造参数（传入已加载模型的 Layer1 实例）
- 步骤 5（`layer1.compute(windows)`）及后续所有逻辑不变
- `BacktestResult` 数据结构不变

#### 新增内容

无。BacktestEngine 只做减法——它变成一个纯推理引擎。

### 3.4 Layer1 类变更

#### 推理路径（保持不变）

`compute()` → `_compute_v2()` / `_compute_v1()` 不需要任何修改。

#### 训练路径（保留但不再由 BacktestEngine 调用）

`fit()` 和 `fit_window()` 方法保留在 `Layer1` 类中，但只在离线训练脚本中调用。这保持了 Layer1 类的训练能力，同时不影响推理路径。

#### 新增：模型加载方法

```python
class Layer1:
    def load_model(self, model_dir: str | Path) -> None:
        """从版本化目录加载预训练模型 + PCA 状态。

        Parameters
        ----------
        model_dir : path
            包含 model.pkl、global_state.pkl、meta.json 的目录。
        """
        model_dir = Path(model_dir)

        # 加载 LightGBM 模型
        model = Layer1Model(self._cfg)
        model.load(model_dir / "model.pkl")
        self._models[_POOLED_KEY] = model

        # 加载 PCA 冻结状态
        if self._use_v2:
            self._global_builder.load_pca_state(model_dir / "global_state.pkl")

        # 验证 meta.json 中的 feature_version 与当前配置一致
        meta_path = model_dir / "meta.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("feature_version") != self._cfg.feature_version:
                raise ValueError(
                    f"Model feature_version={meta['feature_version']} "
                    f"!= config feature_version={self._cfg.feature_version}"
                )
```

### 3.5 离线训练脚本

新建 `scripts/train_layer1.py`。

#### CLI 接口

```
usage: train_layer1.py [-h] --data-dir DIR --config YAML
                       [--train-end DATE] [--output-dir DIR]
                       [--validate] [--no-validate]

参数：
  --data-dir      训练数据目录（Parquet 文件）
  --config        配置文件路径（configs/app.yaml）
  --train-end     训练数据截止日期（默认=昨天，格式 YYYY-MM-DD）
  --output-dir    模型输出目录（默认 models/layer1/）
  --validate      运行 Walk-Forward OOS 验证（默认）
  --no-validate   跳过 OOS 验证
```

#### 流程

```
1. load_config(args.config) + InstrumentRegistry
2. 校验 pca_universe_symbols 非空（否则 raise ValueError）
3. 加载 data-dir 中的 Parquet 文件 → dict[str, list[Bar]]
4. 根据 train-end 和 retrain_window_bars 计算训练窗口 [start, end]
5. 构建 GlobalStateBuilder（非冻结模式）
6. 调用 Layer1._fit_pooled()（构建特征 → 训练 LightGBM）
7. [可选] Walk-Forward OOS 验证（见下方指标要求）
8. 保存产出物：
   a. model.pkl（Layer1Model.save()）
   b. global_state.pkl（GlobalStateBuilder.save_pca_state()）
   c. meta.json（训练参数 + OOS 指标 + lgb_best_iteration）
9. 打印摘要并退出
```

#### OOS 验证指标要求

当 `--validate` 启用时（默认），训练完成后在 `[train_end, train_end + oos_window]` 的未见数据上进行 Walk-Forward 评估。

**必须评估的指标**：

| 指标 | 定义 | 最低门槛 |
|------|------|----------|
| IC mean | Spearman rank corr(预测值, 实际前向收益率) 的均值 | > 0.02 |
| IC IR | IC mean / IC std（信息比率） | > 0.3 |
| OOS Sharpe | 基于模型预测值构建的多空组合的 Sharpe ratio | 仅记录，无门槛 |

**可选评估的指标**（记录到 meta.json，不设门槛）：

- 分 regime 的 IC（TRENDING_STRONG / TRENDING_WEAK / RANGING / EXHAUSTED 分别计算）
- LightGBM 特征重要度 top-5

**门槛行为**：若 IC mean <= 0.02 或 IC IR <= 0.3，打印 WARNING 日志，但**不阻止模型保存**。用户自行决定是否使用该模型版本。meta.json 中记录所有指标供事后审计。

### 3.6 main.py CLI 变更

#### 新增参数

```python
parser.add_argument(
    "--model",
    default=None,
    help="预训练 Layer1 模型目录路径（用于回测和实盘推理）",
)
```

#### run_backtest() 变更

```python
def run_backtest(cfg, registry, data_path, model_path=None):
    layer1 = Layer1(cfg.strategy.layer1)

    # 如果提供了预训练模型，加载之
    if model_path is not None:
        layer1.load_model(model_path)

    engine = BacktestEngine(
        config=cfg, registry=registry, layer1=layer1,
        ...  # 其余不变
    )
    result = engine.run({symbol: bars})  # 不再传入 training_universe
```

#### run_live() 变更

```python
async def run_live(cfg, registry, model_path=None):
    layer1 = Layer1(cfg.strategy.layer1)
    if model_path is not None:
        layer1.load_model(model_path)
    # ... 其余不变
```

#### 向后兼容

- `--training-data` 参数从 `main.py` 中移除（迁移到 `scripts/train_layer1.py`）。
- 当 `--model` 未指定时，Layer1 使用未训练的模型（`predict()` 返回 `(0.0, 0.0)`），与当前不提供 `--training-data` 时行为一致。

---

## 4. 测试影响分析

### 需要重写的测试（2 个）

| 测试 | 当前行为 | 重构后 |
|------|----------|--------|
| `test_walk_forward::test_layer1_fit_called_at_warmup` | 验证 BacktestEngine 在 warmup 后触发 `fit_window()` | 迁移至测试 `scripts/train_layer1.py` 的离线流水线 |
| `test_walk_forward::test_layer1_fit_called_periodically` | 验证 BacktestEngine 按周期调用 `fit_window()` | 同上 |

### 不受影响的测试（~110 个）

| 测试文件 | 理由 |
|----------|------|
| `test_walk_forward::test_forward_horizon_from_config` | 测试 `Layer1.fit()` 方法本身，不涉及 BacktestEngine |
| `test_engine_backtest.py` (2 个) | 使用 Mock layer1，不调用 `fit_window()` |
| `test_pooled_training.py` | 测试 `Layer1._fit_pooled()` 内部逻辑 |
| `test_layer1_features.py`, `test_universal_features.py` | 测试 FeatureBuilder / GlobalState |
| `test_pca_factor.py` | 测试 PCA 计算 |
| 其余所有测试 | 与 BacktestEngine 训练逻辑无关 |

### 新增测试

| 测试 | 覆盖内容 |
|------|----------|
| `test_global_state_frozen` | GlobalStateBuilder frozen 模式：加载 PCA 状态后不重拟合 |
| `test_global_state_save_load` | `save_pca_state()` / `load_pca_state()` 往返一致性 |
| `test_layer1_load_model` | `Layer1.load_model()` 加载模型 + PCA + feature_version 校验 |
| `test_train_pipeline_e2e` | `scripts/train_layer1.py` 端到端测试（小数据集） |
| `test_backtest_with_pretrained` | BacktestEngine 使用预训练模型运行回测 |

---

## 5. 实施计划

### Phase 1：序列化基础设施（最小改动，不破坏现有功能）

1. `GlobalStateBuilder` 新增 `frozen` 参数 + `save_pca_state()` / `load_pca_state()`
2. `Layer1` 新增 `load_model()` 方法
3. 新增对应的单元测试
4. **不修改 BacktestEngine**——现有功能完全保留

### Phase 2：离线训练脚本

1. 新建 `scripts/train_layer1.py`
2. 实现 CLI 参数解析 + 数据加载 + 训练 + OOS 验证 + 保存
3. 新增端到端测试
4. **不修改 BacktestEngine**——离线训练与现有在线训练并存

### Phase 3：推理路径解耦

1. `main.py` 新增 `--model` 参数
2. `BacktestEngine` 删除训练触发逻辑（步骤 4b）
3. `BacktestEngine.run()` 移除 `training_universe` 参数
4. 重写 2 个受影响的测试
5. 移除 `main.py` 中的 `--training-data` 参数

### Phase 4：验证与清理

1. 运行完整测试套件（112 个测试）
2. 对比重构前后的回测结果（使用相同数据 + 相同模型参数）
3. 更新 CLAUDE.md 和项目文档

### 分阶段的理由

- Phase 1-2 是**纯新增**，不修改任何现有代码，零回归风险。
- Phase 3 是**删除 + 修改**，但此时离线训练已经可用，有安全网。
- 任何阶段完成后系统都处于可工作状态（渐进式重构）。

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| PCA 品种域变化导致维度不匹配 | 中 | 推理失败 | 强制显式配置 `pca_universe_symbols`；`load_pca_state()` 中校验品种列表 |
| 冻结 PCA 在极端市场中失效 | 低 | 信号质量下降 | 每周重训练足以覆盖；加入 PCA 解释方差监控告警 |
| pickle 反序列化安全风险 | 极低 | 代码执行 | 模型文件本地生产，不从网络加载；未来可迁移至 safetensors |
| 训练/推理特征数值微小差异 | 中 | 信号轻微偏移 | 滑窗统计从零积累（warmup 期间填充），PCA 组件冻结保证一致性 |
| 回测结果与重构前不完全一致 | 确定 | 需要重新基线 | Phase 4 中对比并建立新基线；差异来源是"移除在线重训练" |

---

## 7. 附录：变更文件清单

| 文件 | 变更类型 | Phase |
|------|----------|-------|
| `src/quant/strategy/layer1/features.py` | 修改（GlobalStateBuilder frozen 模式） | 1 |
| `src/quant/strategy/layer1/layer1.py` | 修改（新增 `load_model()`） | 1 |
| `src/quant/strategy/layer1/model.py` | 不变 | - |
| `src/quant/strategy/layer1/walk_forward.py` | 不变（保留供离线流水线使用） | - |
| `scripts/train_layer1.py` | **新建** | 2 |
| `main.py` | 修改（`--model` 参数，移除 `--training-data`） | 3 |
| `src/quant/engine/backtest.py` | 修改（删除步骤 4b + training_universe） | 3 |
| `tests/test_walk_forward.py` | 修改（重写 2 个测试） | 3 |
| `tests/test_global_state_frozen.py` | **新建** | 1 |
| `tests/test_train_pipeline.py` | **新建** | 2 |
| `tests/test_backtest_pretrained.py` | **新建** | 3 |
