# Sprint Log — Route B: 多后端 Layer1 架构

> **日期：** 2026-04-08
> **状态：** 完成 ✅
> **测试：** 167 passed / 0 failed

---

## 背景

上一轮已完成 Layer1 V2（Pooled 训练 + 37 维特征 + 12,021 品种训练宇宙）。本轮解决两个架构瓶颈：

1. **训练/推理紧耦合**：`BacktestEngine` 每 130 bar 触发在线重训练，阻塞推理循环、回测不可复现、模型不可审计
2. **单一 LightGBM 后端**：37 维快照特征存在信息瓶颈，无法验证时序模型的理论优势

用户选择 **Route B**（一次性架构重设计），同时支持 tabular 和 sequential 两类后端。

---

## 设计阶段（2026-04-08 上午）

### 参与者
- **strategist-b**：策略架构设计
- **engineer-b**：工程实现规划
- 两轮交叉互答（14 个设计决策点），产出 `docs/design_route_b_final.md`（744 行）

### 关键设计决策

| # | 决策点 | 结论 |
|---|--------|------|
| 1 | Sequential 特征维度 | **D=14**（纯价量，不含 PCA — 逐时步重建 GlobalState 开销 100x 不可接受） |
| 2 | Lookback window | **T=120 bars**（约 2 个完整交易日） |
| 3 | DLinear channel 模式 | **channel-independent**（individual=True，D 小时不易过拟合） |
| 4 | PatchTST channel 模式 | **channel-independent**（小样本场景参数共享更鲁棒） |
| 5 | PyTorch 依赖方式 | **可选依赖** `pip install .[torch-cpu]`，factory 做 ImportError 兜底 |
| 6 | 实盘推理设备 | **强制 CPU**（batch=1 时 GPU kernel launch 反而更慢） |
| 7 | 模型文件路由 | meta.json 新增 `model_file` 字段，消除硬编码扩展名 |
| 8 | confidence_temperature | **推理时可调**（后处理参数，不属于冻结范围） |
| 9 | 训练 shuffle | Sequential 后端 **shuffle=False**（时序顺序保持） |
| 10 | LGBMBackend 继承 | **独立实现**，保留原生 lgb.train() API + categorical_feature |
| 11 | 模型版本管理 | **文件系统目录**，不引入 MLflow/DVC |
| 12 | Winsorize 位置 | 由调用方（_fit_pooled）执行，SequenceFeatureBuilder 保持纯粹 |
| 13 | ONNX 导出 | 推迟，当前 Python 环境直接用 torch-cpu 足够 |
| 14 | 实施优先级 | LGBM → XGBoost → DLinear → PatchTST → GRU → Ridge/ElasticNet/RF |

---

## 实施阶段（2026-04-08 下午）

三个 Track 并行执行，Track C 等待 A+B 完成后启动。

---

### Track A — Tabular Backend 基础设施

**执行者：** engineer-tabular
**新增测试：** 23 个

#### 新建文件

| 文件 | 说明 |
|------|------|
| `src/quant/strategy/layer1/backend_protocol.py` | `ModelBackend` Protocol（`@runtime_checkable`）+ `BackendMeta` frozen dataclass |
| `src/quant/strategy/layer1/backends/__init__.py` | `BACKEND_REGISTRY` + `register_backend()` 装饰器 + `create_backend()` 工厂 |
| `src/quant/strategy/layer1/backends/lgbm_backend.py` | `LGBMBackend`：重构自 `model.py`，保留原生 `lgb.train()` + early_stopping + categorical_feature |
| `src/quant/strategy/layer1/backends/_sklearn_base.py` | `SklearnTabularBase`：fit/predict/save/load 共享基类，P90 confidence_scale |
| `src/quant/strategy/layer1/backends/xgb_backend.py` | `XGBBackend`（可选依赖，ordinal 编码类别特征） |
| `src/quant/strategy/layer1/backends/ridge_backend.py` | `RidgeBackend`（Pipeline: StandardScaler + Ridge） |
| `src/quant/strategy/layer1/backends/elasticnet_backend.py` | `ElasticNetBackend`（Pipeline: StandardScaler + ElasticNet） |
| `src/quant/strategy/layer1/backends/rf_backend.py` | `RandomForestBackend` |

#### 修改文件

| 文件 | 改动 |
|------|------|
| `src/quant/strategy/layer1/model.py` | 改为 backward-compat shim：`from .backends.lgbm_backend import LGBMBackend as Layer1Model` |
| `src/quant/config/schema.py` | `Layer1Config` 新增 18 个 Route B 字段（model_type、seq_lookback_T、DLinear/PatchTST/LSTM 超参、confidence_temperature 等）+ `field_validator` |

#### 测试文件

- `tests/test_backend_registry.py` — 5 tests：工厂函数、未知类型 ValueError、Protocol isinstance 检查
- `tests/test_lgbm_backend.py` — 6 tests：fit/predict、未训练返回零、save/load 往返
- `tests/test_sklearn_backends.py` — 12 tests（parametrize × 3 backends）：fit/predict、save/load

---

### Track B — Sequential Backend 基础设施

**执行者：** engineer-sequential
**新增测试：** 18 个

#### 新建文件

| 文件 | 说明 |
|------|------|
| `src/quant/strategy/layer1/sequence_features.py` | `SequenceFeatureBuilder`：14 维逐资产特征，`build_sequence()` → (T,14)，`build_training_sequences()` → (N,T,14) sliding window，严格 no-lookahead，NaN/Inf 替换为 0.0 |
| `src/quant/strategy/layer1/backends/_seq_base.py` | `SequentialBackendBase`：AdamW + CosineAnnealingLR + Huber Loss + 梯度裁剪(max_norm=1.0) + 早停；训练用 GPU（若可用），推理强制 CPU；所有 `import torch` 为 lazy import |
| `src/quant/strategy/layer1/backends/dlinear_backend.py` | `DLinearBackend`：moving_avg 趋势分解，`individual=True`（每维独立线性层），kernel_size=25 |
| `src/quant/strategy/layer1/backends/patchtst_backend.py` | `PatchTSTBackend`：CI 模式，patch_len=16，stride=8，d_model=64，n_heads=4，n_layers=2 |
| `src/quant/strategy/layer1/backends/lstm_backend.py` | `LSTMBackend` + `GRUBackend`：hidden_dim=64，num_layers=2，取最后时步隐状态 → 线性头 |

**Sequential 特征维度（14 维）：**
`ret_1`, `ret_5`, `ret_20`, `ret_60`, `ret_120`, `vol_20`, `vol_60`, `vol_ratio`, `volume_ratio`, `ret_vol_scaled_20`, `ret_vol_scaled_60`, `trend_consistency_20`, `staleness_weight`, `market_beta_20`

#### 测试文件

- `tests/test_seq_feature_builder.py` — 7 tests：形状校验、bars 不足返回 None、no-lookahead 验证、NaN/Inf 清洁
- `tests/test_seq_backends.py` — 11 tests（`pytest.importorskip("torch")`）：DLinear/PatchTST/LSTM fit/predict/save/load，未训练返回零，CPU 模式 N=200,T=32,D=14

---

### Track C — 集成：Layer1 路由 + 训练/推理分离

**执行者：** engineer-integration
**新增测试：** 14 个

#### 修改文件

**`src/quant/strategy/layer1/features.py`** — GlobalStateBuilder frozen 模式
- 新增 `frozen: bool = False` 参数
- `update()` 中 PCA 重拟合在 `frozen=True` 时跳过
- 新增 `save_pca_state(path)` / `load_pca_state(path)`（pickle 序列化 PCA components + syms_fitted）

**`src/quant/strategy/layer1/layer1.py`** — 后端路由
- `__init__`：根据 `cfg.model_type` 通过 `create_backend()` 初始化 `_backend`
- 新增 `load_model(model_dir)`：读 meta.json → create_backend → backend.load() → load_pca_state
- `_compute_v2()`：根据 `backend.input_type` 路由到 `FeatureBuilderV2`（tabular）或 `SequenceFeatureBuilder`（sequential）
- 新增 `_compute_v2_sequential()`：sequential 推理路径

**`src/quant/engine/backtest.py`** — 纯推理引擎（只做减法）
- 删除 `WalkForwardScheduler` import
- 删除 `self._scheduler`、`self._training_universe`
- 删除 `_run_loop()` 中步骤 4b 训练触发块（~42 行）
- 删除 `run()` 签名中 `training_universe` 参数

**`main.py`** — CLI 调整
- 新增 `--model MODEL_DIR`（预训练模型目录）
- 删除 `--training-data`
- `run_backtest()` / `run_live()` 接受 `model_path`，调用 `layer1.load_model()`

**`tests/test_walk_forward.py`** — 重写 2 个已失效测试
- `test_layer1_fit_called_at_warmup` → `test_layer1_fit_not_called_by_engine`
- `test_layer1_fit_called_periodically` → `test_layer1_fit_never_called_regardless_of_bar_count`

#### 新建文件

**`scripts/train_layer1.py`** — 离线训练脚本（415 行）
```
CLI: --data-dir --config [--train-end] [--output-dir] [--validate/--no-validate]

流程：
1. load_config + 校验 pca_universe_symbols 非空
2. 加载 Parquet → dict[str, list[Bar]]
3. create_backend(cfg.model_type)
4. 根据 input_type 选特征构建器（tabular: FeatureBuilderV2，sequential: SequenceFeatureBuilder）
5. 构建 X, y → Winsorize → 逆频率权重
6. backend.fit(X, y, sample_weights)
7. OOS 验证（IC mean>0.02 / IC IR>0.3 为 WARNING，不阻止保存）
8. backend.save(output_dir/version_tag/) + save_pca_state() + meta.json
```

#### 测试文件

| 测试文件 | 测试数 | 覆盖内容 |
|---------|--------|---------|
| `tests/test_global_state_frozen.py` | 4 | frozen 模式跳过 PCA 重拟合，save/load 往返，load 后 frozen=True |
| `tests/test_layer1_routing.py` | 4 | tabular 路径路由，load_model() 加载，sequential 路径路由，feature_version 不匹配 |
| `tests/test_model_persistence.py` | 3 | meta.json 含 model_file 字段，save/load predict 一致，confidence_scale 存储 |
| `tests/test_backtest_pretrained.py` | 3 | BacktestEngine 无 training_universe 参数，预训练模型回测，不触发 fit |

---

### 环境修复

**问题：** macOS LightGBM + OpenMP 在 pytest 多次训练后 segfault

**根因：** macOS 动态库加载顺序问题，多个 OpenMP 实例共存导致内存损坏

**修复：** `tests/conftest.py` 在 pytest 启动时用 `ctypes.CDLL` 预加载 libomp
```python
import ctypes
_LIBOMP = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
if os.path.exists(_LIBOMP):
    ctypes.CDLL(_LIBOMP)
```

---

## 测试结果

```
167 passed in 13.84s
```

| 模块 | 测试数 | 状态 |
|------|--------|------|
| 原有测试（Phase 1-8 + Layer1 V2） | 112 | ✅ 零回归 |
| Track A 新增（tabular backends） | 23 | ✅ |
| Track B 新增（sequential backends） | 18 | ✅ |
| Track C 新增（集成测试） | 14 | ✅ |
| **合计** | **167** | **✅** |

---

## 变更文件汇总

**新建（22 个）：**
```
docs/design_route_b_strategy.md
docs/design_route_b_engineering.md
docs/design_route_b_final.md
scripts/train_layer1.py
src/quant/strategy/layer1/backend_protocol.py
src/quant/strategy/layer1/backends/__init__.py
src/quant/strategy/layer1/backends/_seq_base.py
src/quant/strategy/layer1/backends/_sklearn_base.py
src/quant/strategy/layer1/backends/dlinear_backend.py
src/quant/strategy/layer1/backends/elasticnet_backend.py
src/quant/strategy/layer1/backends/lgbm_backend.py
src/quant/strategy/layer1/backends/lstm_backend.py
src/quant/strategy/layer1/backends/patchtst_backend.py
src/quant/strategy/layer1/backends/rf_backend.py
src/quant/strategy/layer1/backends/ridge_backend.py
src/quant/strategy/layer1/backends/xgb_backend.py
src/quant/strategy/layer1/sequence_features.py
tests/test_backend_registry.py
tests/test_backtest_pretrained.py
tests/test_global_state_frozen.py
tests/test_layer1_routing.py
tests/test_lgbm_backend.py
tests/test_model_persistence.py
tests/test_seq_backends.py
tests/test_seq_feature_builder.py
tests/test_sklearn_backends.py
```

**修改（7 个）：**
```
main.py                                          -- --model 参数，删除 --training-data
pyproject.toml                                   -- pytest-env 配置
src/quant/config/schema.py                       -- Layer1Config +18 字段
src/quant/engine/backtest.py                     -- 删除训练触发逻辑
src/quant/strategy/layer1/features.py            -- GlobalStateBuilder frozen 模式
src/quant/strategy/layer1/layer1.py              -- 后端路由 + load_model()
src/quant/strategy/layer1/model.py               -- 改为 shim
tests/conftest.py                                -- libomp 预加载
tests/test_integration_layer1_v2.py              -- 修复 training_universe 引用
tests/test_walk_forward.py                       -- 重写 2 个测试
```

---

## 后续计划

1. **运行基线训练：** `python scripts/train_layer1.py --data-dir data/training_universe/... --config configs/app.yaml --validate`
2. **对比实验：** 依次训练 lgbm → dlinear → patchtst → gru，对比 OOS IC 和回测 Sharpe
3. **确定重训练频率：** 根据 OOS 指标衰减速度决定（目前未设定，用户待验证）
