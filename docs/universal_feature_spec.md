# Universal Feature Specification — Layer1 v2

> **文档版本**: 1.1
> **作者**: strategist-l1
> **日期**: 2026-04-02
> **状态**: 设计稿，待 team-lead 审批
> **依赖文献**: LITERATURE_REVIEW.md（59 篇）

---

## 0. 设计哲学

本规格文档遵循以下原则（均有文献支撑）：

1. **通用性优先** — 所有特征必须同时适用于股票、ETF、期货、crypto（Cakici et al. 2023: 跨 46 国的 dominant features 是 momentum/reversal/volatility/liquidity）
2. **越大越好** — 训练 universe 不设人为上限，目标 10,000+（Kelly et al. 2024: 理论证明效果随数据规模单调提升）
3. **Vol-scaling 标准化** — 所有收益类特征和标签均做波动率标准化（Moskowitz et al. 2012; Barroso & Santa-Clara 2015）
4. **标签回归化** — 废弃 3 类分类标签，改为 vol-scaled continuous regression（Gu et al. 2020: regression 目标在 pooled training 中一致优于 classification）
5. **最少自由度** — 特征设计不引入需要优化的超参（Harvey et al. 2016: 每个自由度都是过拟合风险）

---

## 1. 通用价量特征集

### 1.1 特征列表

每个资产贡献 **14 个特征**（从当前 10 个扩展到 14 个）。所有特征仅依赖 OHLCV 数据，不依赖任何资产类别特有信息。

| # | 特征名 | 公式 | 类别 | 文献依据 | 备注 |
|---|--------|------|------|---------|------|
| 1 | `ret_1` | `(close[t] / close[t-1]) - 1` | 短期动量 | Gu et al. 2020: momentum 是 #1 重要特征 | 原始 bar return |
| 2 | `ret_5` | `(close[t] / close[t-5]) - 1` | 短期动量 | 同上 | ~1.25 小时 |
| 3 | `ret_20` | `(close[t] / close[t-20]) - 1` | 中期动量 | Moskowitz et al. 2012: 1-12 月动量均显著 | ~1 交易日 |
| 4 | `ret_60` | `(close[t] / close[t-60]) - 1` | 中期动量 | 同上 | ~3 交易日 |
| 5 | `ret_120` | `(close[t] / close[t-120]) - 1` | 长期动量 | 同上 | ~1 周 |
| 6 | `vol_20` | `std(log_return[-20:])` | 波动率 | Gu et al. 2020: volatility 是 top-3 重要特征 | 20-bar realized vol |
| 7 | `vol_60` | `std(log_return[-60:])` | 波动率 | Barroso & Santa-Clara 2015 | 60-bar realized vol (新增) |
| 8 | `vol_ratio` | `vol_20 / vol_60` | 波动率比 | Baltas & Kosowski 2020: vol regime 变化有预测力 | 短期/长期 vol 比值 (新增) |
| 9 | `volume_ratio` | `volume[t] / mean(volume[-20:])` | 流动性 | Gu et al. 2020: liquidity 是 top-3 重要特征 | 相对成交量 |
| 10 | `ret_vol_scaled_20` | `ret_20 / (vol_20 + eps)` | 标准化动量 | Moskowitz et al. 2012: vol-scaled return 是 TSMOM 标准 | 风险调整动量 (新增) |
| 11 | `ret_vol_scaled_60` | `ret_60 / (vol_60 + eps)` | 标准化动量 | 同上 | 中期风险调整动量 (新增) |
| 12 | `trend_consistency_20` | `fraction(sign(bar_ret) == sign(overall_trend))` over 20 bars | 趋势质量 | Baz et al. 2015: TS momentum 与 CS momentum 提供不同信息 | 保留自 v1 |
| 13 | `staleness_weight` | `exp(-hours_since_last_bar / halflife)` | 数据时效 | Phase 1 设计: 24h 全球交易, 休市资产降权 | 保留自 v1 |
| 14 | `market_beta_20` | `cov(ret_asset, ret_composite) / var(ret_composite)` over 20 bars | 系统性风险 | Frazzini & Pedersen 2014: BAB 因子跨资产有效 | 替换 v1 的 cross_corr_20（见 Q2） |

### 1.2 与 v1 的差异

| 变更 | 说明 | 理由 |
|------|------|------|
| 新增 `vol_60` | 长期波动率 | 需要与 `vol_20` 配合计算 `vol_ratio` |
| 新增 `vol_ratio` | 波动率 regime 指标 | Baltas & Kosowski 2020: vol clustering 有预测力 |
| 新增 `ret_vol_scaled_20` | 20-bar vol-scaled momentum | Moskowitz 2012 的核心信号，之前缺失 |
| 新增 `ret_vol_scaled_60` | 60-bar vol-scaled momentum | 同上，多时间尺度 |
| 移除 `ret_1` 的 LOO 排除 | 在 10,000+ universe 下，LOO 排除粒度需要重新设计 | 见第 7 节 |
| `FEATURES_PER_SYMBOL` | 10 → 14 | 新增 4 个特征 |

### 1.3 特征计算的通用性保障

所有 14 个特征仅依赖：
- `close` 价格序列
- `volume` 成交量序列
- `timestamp` 时间戳

**不依赖**任何以下资产类别特有数据：
- 市盈率/市净率（股票特有）
- 期限结构/basis/carry（期货特有）
- 区块链数据（crypto 特有）
- 利率曲线（债券特有）

这确保了同一套代码可无修改地应用于所有资产类别。

### 1.4 数据不足时的安全填充

| 情况 | 处理 |
|------|------|
| `len(closes) < n + 1` for `ret_n` | 返回 `0.0` |
| `len(closes) < 21` for `vol_20` | 返回 `0.0` |
| `len(closes) < 61` for `vol_60` | 返回 `0.0` |
| `vol_20 < eps` or `vol_60 < eps` for ratio features | 返回 `0.0` (避免除零) |
| `cross_corr` 标准差为零 | 返回 `0.0` |

`eps` 取 `1e-9`，与现有代码一致。

---

## 2. Vol-Scaling 规格

### 2.1 目的

使不同波动率的资产（ES vol ~0.5%/bar vs BTC vol ~3%/bar）在训练中具有可比性。

### 2.2 公式

对于资产 i 在时间 t 的 raw return `r_{i,t}`：

```
r_scaled_{i,t} = r_{i,t} / sigma_{i,t}
```

其中 `sigma_{i,t}` 是 **ex-ante** realized volatility：

```
sigma_{i,t} = std(log_return_{i, [t-L:t]})   其中 L = vol_lookback
```

### 2.3 参数

| 参数 | 值 | 理由 |
|------|-----|------|
| `vol_lookback` | 60 bars | Moskowitz et al. 2012 使用 60-day for daily; 按比例 60 bars for 15-min（约 3 交易日） |
| `eps` (floor) | `1e-6` | 防止极低波动率时除零爆炸 |
| `sigma_cap` | `None` (不设上限) | Barroso & Santa-Clara 2015 不做 cap |
| `sigma_target` | `None` (不做 rescale) | 直接除以 sigma 即可，不需要乘回目标 vol |

### 2.4 特征 vs 标签的 vol-scaling

| 用途 | 是否 vol-scale | 说明 |
|------|--------------|------|
| `ret_vol_scaled_20` / `ret_vol_scaled_60` (特征) | 是 | 用自身的 vol_20 / vol_60 分别标准化 |
| `ret_1`, `ret_5`, `ret_20`, `ret_60`, `ret_120` (特征) | **否** | 保留原始 return 作为独立特征，让模型自行学习 raw vs scaled 的信息 |
| 训练标签 `y` | **是** | 见第 6 节 |

**设计理由**: 同时提供 raw return 和 vol-scaled return 两组特征（#1-5 和 #10-11），让模型自行决定哪组更有用。Gu et al. 2020 的特征集也同时包含 raw 和 standardized 版本。

---

## 3. PCA 宏观因子规格

### 3.1 目的

从跨资产收益中提取少量 latent macro factors（如 risk-on/risk-off），作为额外特征输入模型。

**PCA 仅作为特征，不用于标签构建**（Kelly et al. 2019 IPCA; Macrosynergy Research: PCA 降维丢失 weak factors）。

### 3.2 PCA 输入宇宙

| 选项 | 方案 | 理由 |
|------|------|------|
| ~~全部 10,000+~~ | 不采用 | 10,000+ 的 PCA 计算量大，且包含大量噪声小盘股 |
| **核心代表子集** | **采用** | 选取流动性最高的 50-100 个品种，覆盖主要资产类别 |

#### 核心 PCA 宇宙建议（~80 品种）

| 资产类别 | 品种数 | 示例 |
|---------|--------|------|
| 美股指数 ETF | 10 | SPY, QQQ, IWM, DIA, XLF, XLE, XLK, XLV, XLC, XLI |
| 全球股指 ETF | 10 | EFA, EEM, FXI, EWJ, EWZ, EWG, EWH, VGK, INDA, MCHI |
| 债券 ETF | 8 | TLT, IEF, SHY, HYG, LQD, TIP, BND, AGG |
| 商品 ETF/期货 | 10 | GLD, SLV, USO, UNG, DBA, CORN, WEAT, CPER, GDX, XME |
| 外汇 ETF | 6 | UUP, FXE, FXY, FXB, FXA, FXC |
| 股指期货（交易标的） | 7 | ES, NQ, YM, RTY, HSI, MHI, HHI |
| VIX/波动率 | 3 | VIX (index), VIXY, UVXY |
| Crypto | 6 | BTC, ETH, SOL, BNB, XRP, ADA |

总计 ~60-80 品种。具体名单由数据可获取性决定。

### 3.3 Rolling PCA 参数

| 参数 | 值 | 理由 |
|------|-----|------|
| `pca_window` | 500 bars (~25 交易日) | 需要足够样本估计协方差；太长则对 regime 变化不敏感 |
| `pca_update_freq` | 20 bars (~1 交易日) | 非每 bar 更新——PCA 变化慢，节省计算 |
| `n_components` | 5 | 前 5 个 PC 通常解释 60-80% 方差 |
| `pca_input` | 各品种的 `ret_vol_scaled_20` (vol-scaled 20-bar return) | 标准化后的收益做 PCA 更稳定 |

### 3.4 Sign Correction (符号修正)

PCA 的特征向量方向有 ±1 的不确定性（数学上等价）。跨时间窗口，PC 方向可能翻转。

**修正规则**：
```
if loading_of_ES_on_PC1 < 0:
    PC1 *= -1       # 翻转 PC1
    loadings_PC1 *= -1
```

- 固定 **ES**（S&P 500 E-mini）在 PC1 上的 loading 为正
- 经济含义：PC1 > 0 → "risk-on"（全球风险偏好上升）
- 其他 PC 无需修正（高阶 PC 经济含义不明确，模型自行学习）

### 3.5 PCA 输出特征

每个 PCA 主成分生成 **3 个特征**，共 `n_components * 3 = 15` 个全局特征：

| # | 特征名 | 公式 | 说明 |
|---|--------|------|------|
| 1 | `pc{k}_score` | 当前资产在 PC_k 上的投影值 | 当前 regime 定位 |
| 2 | `pc{k}_momentum_5` | `pc{k}_score[t] - pc{k}_score[t-5]` | PC 的短期变化方向 |
| 3 | `pc{k}_momentum_20` | `pc{k}_score[t] - pc{k}_score[t-20]` | PC 的中期变化趋势 |

k = 1, 2, 3, 4, 5。

**注意**: PCA 特征是**全局的**（对所有资产相同的 PCA 空间），但每个资产的 `pc{k}_score` 不同（因为 loading 不同）。

---

## 4. Composite 宏观特征

### 4.1 目的

捕获全市场整体动量方向，作为特征（非标签）输入模型。

**文献依据**: Rapach et al. 2013 (aggregate signal predicts international returns); BIS/ECB/Goldman Global Risk Appetite Index。

### 4.2 Equal-Weight Composite Return

```
composite_ew_ret_N = mean_over_all_assets( ret_vol_scaled_N )
```

- 对训练 universe 中所有活跃资产的 vol-scaled N-bar return 取简单平均
- N = 5, 20（两个时间尺度）
- **不排除** target symbol（因为在万级 universe 中单个品种的权重可忽略）

生成 2 个特征：`composite_ew_ret_5`, `composite_ew_ret_20`

### 4.3 Inverse-Volatility Composite Return

```
w_i = 1 / sigma_{i,t}    (inverse vol weight, 然后归一化使 sum(w) = 1)
composite_iv_ret_N = sum_over_all_assets( w_i * ret_N_i )
```

- 波动率低的资产获得更高权重
- N = 5, 20（两个时间尺度）

生成 2 个特征：`composite_iv_ret_5`, `composite_iv_ret_20`

### 4.4 Composite 特征汇总

| # | 特征名 | 说明 |
|---|--------|------|
| 1 | `composite_ew_ret_5` | 全市场等权 vol-scaled 5-bar return 均值 |
| 2 | `composite_ew_ret_20` | 全市场等权 vol-scaled 20-bar return 均值 |
| 3 | `composite_iv_ret_5` | 全市场 inverse-vol 加权 5-bar return |
| 4 | `composite_iv_ret_20` | 全市场 inverse-vol 加权 20-bar return |

这 4 个特征对所有资产是**相同的**（全局特征），与 PCA 特征类似。

---

## 5. Asset ID Embedding 规格

### 5.1 目的

让模型知道"当前正在预测哪个资产"以及"这个资产属于什么类别"，从而学习 asset-specific 和 class-specific 的规律。

**文献依据**: Gu et al. 2020 使用 firm characteristics 隐含了 asset identity；Cakici et al. 2023 发现不同国家的 dominant features 不同，说明需要 asset/group identity。

### 5.2 Categorical 特征定义

| # | 特征名 | 类型 | 取值范围 | 编码方式 |
|---|--------|------|---------|---------|
| 1 | `asset_class` | categorical | `stock`, `etf`, `futures`, `crypto`, `bond`, `fx` | LightGBM native categorical |
| 2 | `exchange` | categorical | `CME`, `CBOT`, `HKEX`, `NYSE`, `NASDAQ`, `CRYPTO`, ... | LightGBM native categorical |
| 3 | `currency` | categorical | `USD`, `HKD`, `EUR`, `JPY`, `CNY`, ... | LightGBM native categorical |
| 4 | `region` | categorical | `US`, `HK`, `EU`, `JP`, `CN`, `GLOBAL` | LightGBM native categorical |

### 5.3 LightGBM Categorical Handling

LightGBM 原生支持 categorical features，**不需要 one-hot encoding**。使用方式：

```python
# 在 lgb.Dataset 中指定 categorical_feature
train_data = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=[col_idx_asset_class, col_idx_exchange, col_idx_currency, col_idx_region],
)
```

Integer encoding 规则：每个 categorical 值映射为连续整数（从 0 开始），具体映射表在训练时固定，推理时查表。

### 5.4 未来扩展：Learned Embedding

当模型从 LightGBM 迁移到 Neural Network 时，categorical features 应替换为 **learned embedding**：

```python
# PyTorch style
asset_class_emb = nn.Embedding(num_classes=6, embedding_dim=8)
exchange_emb = nn.Embedding(num_classes=20, embedding_dim=8)
```

FASCL (2025) 的 contrastive embedding 是更高级的替代——用对比学习从收益相关性中自动学习 asset embedding，无需预定义类别。

---

## 6. 训练标签规格

### 6.1 废弃 3 类分类标签

**当前实现** (`model.py:32-33`):
```python
threshold = 0.0005  # ±0.05%
labels = np.where(y > threshold, 1, np.where(y < -threshold, -1, 0)) + 1  # 0/1/2
```

**问题**:
1. 分类阈值 ±0.05% 是人为设定的，不同 vol 的资产适用不同的阈值
2. 分类标签丢失了收益大小信息（涨 0.1% 和涨 5% 被等同对待）
3. Gu et al. 2020 在 pooled training 中使用 continuous return regression，效果更好

### 6.2 新标签：Vol-Scaled Forward Return Regression

```
y_{i,t} = r_{i, [t, t+H]} / sigma_{i,t}
```

其中：
- `r_{i, [t, t+H]} = (close_{i, t+H} / close_{i, t}) - 1` — H-bar forward return
- `sigma_{i,t} = std(log_return_{i, [t-60:t]})` — 60-bar ex-ante realized vol（与第 2 节 vol-scaling 一致）
- `H = forward_horizon_bars`（默认 26，从 `Layer1Config` 读取）

### 6.3 标签参数

| 参数 | 值 | 来源 | 说明 |
|------|-----|------|------|
| `forward_horizon_bars` | 26 | `Layer1Config` (已有) | ~6.5 小时 @15min |
| `vol_lookback` | 60 | 新增参数 | 与特征的 `vol_60` 一致 |
| `label_winsorize_sigma` | 5.0 | 新增参数 | vol-scaled label 超过 ±5σ 时截断，防止极端值 |
| `eps` | 1e-6 | 硬编码 | sigma floor |

### 6.4 Winsorization

Vol-scaled return 理论上应近似标准正态分布，但尾部事件（flash crash、circuit breaker）会产生极端值。

```python
y_clipped = np.clip(y, -label_winsorize_sigma, +label_winsorize_sigma)
```

`label_winsorize_sigma = 5.0` 意味着截断在 ±5 标准差处。Gu et al. 2020 使用类似的 winsorization。

### 6.5 LightGBM Objective 变更

| 参数 | 当前值 | 新值 | 理由 |
|------|--------|------|------|
| `objective` | `multiclass` | **`huber`** | Huber loss 对异常值更鲁棒，兼具 MSE 的效率和 MAE 的鲁棒性 |
| `num_class` | 3 | 移除 | 不再是分类问题 |
| `alpha` (huber delta) | N/A | `1.0` | Huber loss 参数：\|residual\| < delta 用 MSE, > delta 用 MAE |

### 6.6 模型输出变更

| 维度 | 当前 | 新 | 说明 |
|------|------|-----|------|
| `beta_score` | `P(up) - P(down)` ∈ [-1, +1] | 模型回归预测值 (vol-scaled expected return) | 正值=看多，负值=看空 |
| `confidence` | `max(P(class))` ∈ [0, 1] | `abs(beta_score)` 或模型 uncertainty 估计 | 用于 MII 和仓位调整 |

---

## 7. LOO (Leave-One-Out) 排除规格——更新

### 7.1 当前设计的局限

当前 LOO 仅排除 7 个期货中的相关品种（如 ES 模型排除 NQ/YM/RTY），在 7 品种的 universe 中合理。

### 7.2 10,000+ Universe 下的 LOO

在万级 universe 下，LOO 需要重新设计：

| 层级 | 排除规则 | 说明 |
|------|---------|------|
| **自身排除** | 排除 target symbol 自身 | 不变 |
| **紧密替代品排除** | 排除与 target 相关性 > 0.95 的品种 | 如 SPY 和 VOO 几乎完全相同 |
| ~~相关组排除~~ | ~~排除 loo_correlation_groups 中的品种~~ | 废弃硬编码的 loo_groups |

**理由**: 在 10,000+ 品种中，ES 和 NQ 的信息贡献占比极小（~0.01%），排不排除对模型影响可忽略。但仍需排除近乎完全相同的品种（如同一指数的不同 ETF）以避免信息泄漏。

### 7.3 动态相关性排除

```python
# 每次 retrain 时重新计算
for target in trading_universe:
    corr_with_all = compute_rolling_corr(target, all_assets, window=500)
    exclude_set = {a for a in all_assets if abs(corr_with_all[a]) > 0.95}
    exclude_set.add(target)
```

`0.95` 阈值可配置，加入 `Layer1Config`。

---

## 8. 完整特征向量结构

### 8.1 Per-Asset 特征（14 维 x N_universe 个资产）

对于 target symbol 的预测，来自 universe 中每个其他活跃资产的 14 维特征向量：

```
[ret_1, ret_5, ret_20, ret_60, ret_120,
 vol_20, vol_60, vol_ratio,
 volume_ratio,
 ret_vol_scaled_20, ret_vol_scaled_60,
 trend_consistency_20, staleness_weight, market_beta_20]
```

### 8.2 Global 特征（19 维，对所有资产相同）

```
PCA 特征:     pc1_score, pc1_mom5, pc1_mom20, pc2_score, pc2_mom5, pc2_mom20,
              pc3_score, pc3_mom5, pc3_mom20, pc4_score, pc4_mom5, pc4_mom20,
              pc5_score, pc5_mom5, pc5_mom20                               (15 维)
Composite:    composite_ew_ret_5, composite_ew_ret_20,
              composite_iv_ret_5, composite_iv_ret_20                       (4 维)
```

### 8.3 Categorical 特征（4 维）

```
asset_class, exchange, currency, region
```

### 8.4 总特征维度

**Pooled Training 模式**（每个 training sample 是一个 `(asset, time)` pair）：

```
feature_dim = 14 (per-asset self features)
            + 19 (global PCA + composite)
            + 4  (categorical)
            = 37 维
```

**注意**: 这与当前的 `14 * N_universe` 维度模式完全不同。

### 8.5 架构变更：从 "Per-Target Cross-Asset" 到 "Pooled Sample"

| 维度 | 当前 (v1) | 新 (v2) |
|------|----------|---------|
| 训练样本 | 每个 target symbol 有独立模型，特征 = 其他所有资产的 10 维拼接 | **所有 (asset, time) pairs 共享一个模型**，每个 sample 是 37 维 |
| 特征维度 | `10 * (N_universe - 1)` ≈ 60 维 (7 品种) | **固定 37 维**（不随 universe 大小变化） |
| 模型数量 | N 个独立 LightGBM（每品种一个） | **1 个全局 LightGBM** |
| 跨资产信息 | 通过 cross_corr 和 LOO 间接传递 | 通过 PCA, Composite, asset_class categorical 传递 |
| 可扩展性 | O(N^2) 特征维度 | **O(1) 特征维度** |

**这是最关键的架构变更**——从 N 个 per-asset 模型变为 1 个 pooled global 模型。文献一致支持 pooled training 优于 per-asset training（Gu et al. 2020; Cakici et al. 2023; Kelly & Xiu 2023）。

---

## 9. Walk-Forward 训练规格——更新

### 9.1 保留参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `retrain_interval_bars` | 130 | 每 130 bars (~2 交易日) 触发重训练 |
| `retrain_window_bars` | 13000 | 训练窗口 13000 bars (~100 交易日) |
| `forward_horizon_bars` | 26 | 标签前瞻跨度 26 bars (~6.5 小时) |
| `cv_embargo_bars` | 5 | Train/Val gap 防止信息泄漏 |

### 9.2 新增参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `vol_lookback` | 60 | Vol-scaling 使用的回看窗口 |
| `label_winsorize_sigma` | 5.0 | 标签 winsorization 阈值 |
| `pca_window` | 500 | Rolling PCA 计算窗口 |
| `pca_update_freq` | 20 | PCA 更新频率（bars） |
| `pca_n_components` | 5 | PCA 主成分数量 |
| `loo_corr_threshold` | 0.95 | 动态 LOO 排除相关性阈值 |
| `lgb_objective` | `huber` | LightGBM 目标函数 |
| `lgb_huber_delta` | 1.0 | Huber loss 参数 |

### 9.3 Pooled Training 数据构建

每次 retrain 时：

```
1. 对 window [start_idx : end_idx] 中的每个 bar t：
   2. 对 universe 中的每个 asset i（排除 LOO）：
      3. 构建 37 维特征向量 X_{i,t}
      4. 构建标签 y_{i,t} = r_{i,[t,t+H]} / sigma_{i,t}
      5. (X_{i,t}, y_{i,t}) 加入训练集

训练集规模：~N_assets * N_bars_in_window
```

对于 10,000 品种 * 13,000 bars = 1.3 亿样本。LightGBM 可以处理这个规模（设置 `subsample` 和 `bagging_freq` 参数）。

### 9.4 未来演进：自适应 Retrain

当前 `retrain_interval_bars = 130` 是固定间隔。基于 Concept Drift 检测文献（#58），未来应改为自适应：

```
if concept_drift_detected(recent_predictions, recent_actuals):
    trigger_retrain()  # 检测到 drift 时立即重训练
else:
    retrain at fixed interval  # 保底
```

这不在 v2 初始范围内，记录为后续演进方向。

---

## 10. 配置 Schema 变更清单

以下字段需要添加到 `Layer1Config`（`schema.py`）：

```python
# Vol-scaling & label
vol_lookback: int = Field(60, ge=10)
label_winsorize_sigma: float = Field(5.0, gt=0)
lgb_objective: str = "huber"
lgb_huber_delta: float = Field(1.0, gt=0)

# PCA
pca_window: int = Field(500, ge=50)
pca_update_freq: int = Field(20, ge=1)
pca_n_components: int = Field(5, ge=1, le=20)
pca_universe_symbols: list[str] = []  # 空 = 自动选择 top-N liquid

# LOO
loo_corr_threshold: float = Field(0.95, gt=0, le=1.0)

# Feature version flag
feature_version: int = Field(2, ge=1)  # 1 = v1 (10 features per symbol), 2 = v2 (37 dim pooled)
```

`feature_version` 字段允许在过渡期内 v1 和 v2 共存。

---

## 11. 开放问题

| # | 问题 | 当前建议 | 需要验证 |
|---|------|---------|---------|
| 1 | `forward_horizon_bars = 26` 对所有资产类别是否最优？ | 保持 26，回测验证 | 不同 vol 的资产可能需要不同 horizon |
| 2 | `pca_n_components = 5` 是否足够？ | 5 作为起点，观察 explained_variance_ratio | 可能需要 3-8 的 grid search |
| 3 | Huber delta = 1.0 是否最优？ | 1.0 作为起点 | 可能需要 {0.5, 1.0, 2.0} 的比较 |
| 4 | 10,000+ 品种的 PCA 子集选择标准？ | 流动性 top 80 | 需要数据验证流动性排序 |
| 5 | `cross_corr_20` 在 pooled 模式下是否仍需要？ | 保留（作为 pair-level 信息） | 可能被 PCA 特征替代 |
| 6 | RL 的 position sizing 何时引入？ | 中期计划（Task #11 范围外） | 需要 Offline RL (CQL) 的 PoC |

---

## 12. 与下游模块的接口

### 12.1 FeatureBuilder v2 接口

```python
class FeatureBuilderV2:
    FEATURE_DIM: int = 37  # 固定维度

    def build(
        self,
        asset_symbol: str,
        asset_bars: list[Bar],
        global_state: GlobalState,  # 包含 PCA scores, composite returns
        as_of: datetime,
    ) -> np.ndarray:
        """返回 37 维特征向量。"""
        ...
```

### 12.2 GlobalState 数据结构

```python
@dataclass
class GlobalState:
    """全局状态，每 bar 更新一次（或每 pca_update_freq bars 更新 PCA）。"""
    pca_scores: dict[str, np.ndarray]  # symbol -> [pc1, pc2, ..., pc5] scores
    pca_momentum_5: dict[str, np.ndarray]   # symbol -> 5-bar PC momentum
    pca_momentum_20: dict[str, np.ndarray]  # symbol -> 20-bar PC momentum
    composite_ew_ret_5: float
    composite_ew_ret_20: float
    composite_iv_ret_5: float
    composite_iv_ret_20: float
```

### 12.3 Layer1 v2 接口

```python
class Layer1V2:
    def __init__(self, cfg: Layer1Config) -> None:
        self._model: lgb.Booster  # 单一全局模型
        self._feature_builder = FeatureBuilderV2(...)
        ...

    def fit(self, all_bars: dict[str, list[Bar]], global_states: list[GlobalState]) -> None:
        """Pooled training: 所有 (asset, time) pairs 训练一个模型。"""
        ...

    def predict(self, asset_symbol: str, bars: list[Bar], global_state: GlobalState) -> tuple[float, float]:
        """返回 (beta_score, confidence) for a single asset at current time."""
        ...
```

---

---

## 13. Engineer 决策点澄清（v1.1 补充）

以下 5 个问题由 @engineer 在 Task #13 预研中提出，此处给出明确答案。

### Q1: Symbol Identifier 编码方式——Integer Label

**决定：使用 LightGBM native integer label，不用 one-hot。**

理由：
- LightGBM 原生支持 categorical feature 的 integer 编码，内部会自动做最优分裂（比 one-hot 更高效且信息量更大）
- One-hot 在 10,000+ 品种时会产生 10,000 维的稀疏特征列，极度浪费内存且降低树模型效率
- 4 个 categorical 特征（`asset_class`, `exchange`, `currency`, `region`）的 cardinality 分别约为 6, 20, 10, 6，远低于 LightGBM 的 categorical 上限

**实现规格：**

```python
# 编码映射表（在训练前构建，推理时复用）
ASSET_CLASS_MAP = {"stock": 0, "etf": 1, "futures": 2, "crypto": 3, "bond": 4, "fx": 5}
EXCHANGE_MAP = {"CME": 0, "CBOT": 1, "HKEX": 2, "NYSE": 3, "NASDAQ": 4, "CRYPTO": 5, ...}
CURRENCY_MAP = {"USD": 0, "HKD": 1, "EUR": 2, "JPY": 3, "CNY": 4, ...}
REGION_MAP = {"US": 0, "HK": 1, "EU": 2, "JP": 3, "CN": 4, "GLOBAL": 5}

# LightGBM Dataset 创建时指定
categorical_feature_indices = [33, 34, 35, 36]  # 特征向量中 4 个 categorical 的列索引
train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_feature_indices)
```

**注意：不需要单独的 symbol-level identifier**（如 "AAPL"=0, "MSFT"=1, ...）。原因：
1. 万级 cardinality 的 categorical 会导致 LightGBM 分裂时遍历过多子集
2. Asset identity 信息通过 4 个粗粒度 categorical + 14 维 per-asset 数值特征隐式传递
3. Gu et al. 2020 也没有使用 firm_id 作为特征

### Q2: cross_corr_20 在 Pooled 模式中的处理——替换为 market_beta

**决定：废弃原始 `cross_corr_20`（自身与 target 的相关系数），替换为 `market_beta_20`。**

**问题分析**：在 v1 中，`cross_corr_20` 计算的是"其他品种与 target 品种的 Pearson 相关系数"。在 v2 pooled 模式下，每个 sample 是自身的特征，不存在 "target vs other" 的概念——自身与自身的相关系数恒为 1，没有信息量。

**替代方案：`market_beta_20`**

```python
# market_beta_20: 资产 i 相对于 equal-weight composite 的 beta
# 衡量该资产对整体市场风险的暴露度
composite_ret = composite_ew_ret  # 全市场等权 return（已在 GlobalState 中计算）
asset_ret = diff(log(closes[-21:]))
market_ret = composite_ret_series[-20:]  # 需要 GlobalState 保存最近 20 bar 的 composite

cov = np.cov(asset_ret, market_ret)[0, 1]
var_market = np.var(market_ret)
market_beta_20 = cov / (var_market + eps)
```

**特征列表更新**：第 14 号特征从 `cross_corr_20` 变为 `market_beta_20`。

| # | 原特征 | 新特征 | 说明 |
|---|--------|--------|------|
| 14 | `cross_corr_20` | **`market_beta_20`** | 对 EW composite 的 20-bar rolling beta；衡量系统性风险暴露 |

**GlobalState 需额外保存**：最近 20 bar 的 `composite_ew_ret` 序列（用于 beta 计算）。

```python
@dataclass
class GlobalState:
    # ... 已有字段 ...
    composite_ew_ret_history: np.ndarray  # shape (20,), 最近 20 bar 的 EW composite bar return
```

### Q3: Pooled 模型 predict() 的 confidence 定义——|predicted_value| 归一化

**决定：`confidence = min(abs(beta_score) / confidence_scale, 1.0)`**

**分析**：
- 回归模型的输出是 vol-scaled expected return（连续值，无界限），不像分类模型有天然的概率输出
- 直接用 `abs(beta_score)` 的问题：值域不固定，不同训练周期的尺度可能不同
- 设为常量 1.0 的问题：MII 和 downstream 依赖 confidence 做信号过滤

**归一化方案**：

```python
# confidence_scale: 训练集上 abs(predicted) 的 90th percentile
# 在 fit() 结束时计算并保存
y_pred_train = model.predict(X_train)
confidence_scale = np.percentile(np.abs(y_pred_train), 90)

# predict() 时
raw_pred = model.predict(X.reshape(1, -1))[0]
beta_score = float(raw_pred)
confidence = float(min(abs(raw_pred) / (confidence_scale + 1e-9), 1.0))
```

**语义**：
- `confidence ≈ 0`：模型预测接近零（无方向性观点）
- `confidence ≈ 0.5`：模型有中等强度的预测
- `confidence ≈ 1.0`：模型预测值超过训练集 90% 分位（极端预测）

**新增需要持久化的参数**：`confidence_scale: float`，在 `model.save()` 时与模型一起保存。

### Q4: LOO 策略在 Pooled 设定下——保留但范围缩小

**决定：LOO 保留，但仅在标签构建时排除，不在特征构建时排除。**

**详细说明**：

| 环节 | 是否排除 | 说明 |
|------|---------|------|
| **特征构建** (`build()`) | **不排除** | v2 的每个 sample 只使用自身 14 维特征 + 全局特征，不拼接其他品种特征。没有跨品种特征泄露的问题。 |
| **标签构建** | **不排除** | 每个 sample 的标签是自身的 vol-scaled forward return，与其他品种无关。 |
| **Composite / PCA 计算** | **排除近乎完全替代品** | 计算 `composite_ew_ret` 和 PCA 时，排除与 target 相关性 > 0.95 的品种（防止 SPY/VOO 这类近乎相同的品种双重计入）。但在万级 universe 下这个影响极小。 |
| **训练样本选择** | **不排除** | ES 的训练样本和 NQ 的训练样本都进入同一个 pooled 训练集，这是正确的——模型需要学习"当 NQ 涨时 ES 大概率也涨"这种规律。 |

**为什么不再需要 v1 的 LOO**：v1 中 ES 模型的特征直接包含 NQ 的 return/vol 等 10 维特征，存在"用 NQ 的 return 直接预测 ES 的 return"的信息泄露风险。v2 中每个 sample 只看自身的特征，跨资产信息通过 PCA/Composite 间接传递（这些是全局聚合后的信息，单个品种的权重 < 0.01%，不构成泄露）。

**配置变更**：
- `loo_correlation_groups` 字段保留（用于 composite/PCA 排除），但默认值简化
- 新增 `loo_corr_threshold = 0.95`（动态排除阈值）
- 原有的 `{"ES": ["NQ", "YM", "RTY"], ...}` 硬编码在 pooled 模式下不再使用

### Q5: 样本权重均衡——使用 Inverse Frequency Weighting

**决定：按品种数据量做 inverse frequency weighting。**

**问题**：如果 AAPL 有 13,000 bars 而某小盘股只有 2,000 bars，不加权时 AAPL 贡献 6.5x 的训练样本，模型可能过度学习大盘股/高流动性资产的模式。

**方案**：

```python
# 计算每个品种的样本数
symbol_counts = Counter(symbol for symbol, _ in training_samples)
total_samples = sum(symbol_counts.values())
n_symbols = len(symbol_counts)

# 每个品种的 target weight = (1/n_symbols)，即等权
# 每个 sample 的 weight = target_weight / actual_weight = (total / n_symbols) / count_for_this_symbol
sample_weights = np.array([
    total_samples / (n_symbols * symbol_counts[sym])
    for sym, _ in training_samples
])

# LightGBM 支持 sample_weight
train_data = lgb.Dataset(X, label=y, weight=sample_weights, categorical_feature=cat_indices)
```

**效果**：
- 数据量多的品种的每个 sample 权重被降低
- 数据量少的品种的每个 sample 权重被提升
- 整体效果等价于：每个品种对 loss 的总贡献相等

**文献支持**：Gu et al. 2020 在 value-weighted vs equal-weighted portfolio 评估中发现 equal-weighted 在截面预测中更稳健。Inverse frequency weighting 是让训练等效于 equal-weighted 的标准做法。

**可选的替代方案**（不在 v2 初始范围内）：
- Cakici et al. 2023 发现 ML 在小盘股中表现更好——如果这成立，可以反而给小盘股/低流动性品种**更高**的权重
- 这需要回测验证，暂不实施

---

> **本文档定义了 Layer1 v2 的完整特征规格。Engineer 可据此实现 FeatureBuilderV2、GlobalState、Label 构建、LightGBM 训练 pipeline。所有设计决策均有文献支撑（见 LITERATURE_REVIEW.md #1-59）。**
