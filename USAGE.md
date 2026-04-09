# IBKR Signal Assistant — 使用说明

本文档覆盖环境搭建、三种运行模式（信号/回测/实盘）、配置文件详解以及常见操作示例。

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [目录结构概览](#2-目录结构概览)
3. [信号模式（Signal Mode）](#3-信号模式signal-mode)
4. [回测模式（Backtest Mode）](#4-回测模式backtest-mode)
5. [实盘模式（Live Trading Mode）](#5-实盘模式live-trading-mode)
6. [配置文件详解](#6-配置文件详解)
7. [交易标的规格](#7-交易标的规格)
8. [风险管理参数调优](#8-风险管理参数调优)
9. [数据格式要求](#9-数据格式要求)
10. [运维与日志](#10-运维与日志)
11. [常见问题](#11-常见问题)

---

## 1. 环境搭建

### 1.1 系统依赖

```bash
# macOS — 安装 Python 3.12
brew install python@3.12

# TA-Lib C 库（部分指标使用）
brew install ta-lib
```

### 1.2 创建虚拟环境并安装依赖

```bash
# 进入 new/ 目录（所有命令均在此目录执行）
cd new/

# 创建虚拟环境
python3.12 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 以可编辑模式安装项目
pip install -e ".[dev]"
```

> **注意**：如果已有 `.venv`，直接执行 `source .venv/bin/activate` 即可，无需重建。

### 1.3 配置环境变量（可选）

```bash
# 复制示例环境变量文件
cp .env.example .env

# 编辑 .env，填入必要信息
# FRED_API_KEY=your_fred_api_key_here   # 宏观数据源（可选）
# IBKR_HOST=127.0.0.1
# IBKR_PORT=7497
```

### 1.4 验证安装

```bash
# 运行测试套件（67 个测试全部应通过）
.venv/bin/python -m pytest tests/ -v
```

预期输出：
```
==================== 67 passed in X.XXs ====================
```

---

## 2. 目录结构概览

```
new/
├── main.py                    # 统一入口点（信号/实盘/回测模式）
├── configs/
│   ├── app.yaml               # 主配置文件（运行时参数）
│   └── instruments.yaml       # 7 个期货合约规格
├── src/quant/
│   ├── config/                # Pydantic 配置 schema + YAML 加载器
│   ├── core/                  # Protocol 抽象层（类型/事件/接口）
│   ├── data/                  # 数据源（Yahoo/FRED/IBKR）+ Parquet 存储
│   ├── engine/                # 回测引擎（BaseEngine / BacktestEngine）
│   ├── execution/             # 执行器（SimExecutor / IBKRExecutor）
│   ├── instrument/            # InstrumentRegistry
│   ├── portfolio/             # CostModel / PortfolioBook / Metrics
│   ├── risk/                  # ATRPositionSizer / RiskEngine / TieredExit
│   └── strategy/
│       ├── layer1/            # 跨资产宏观动量（LightGBM + MII）
│       └── layer2/            # 8 个 Alpha 因子 + IC 权重
├── tests/                     # 67 个单元测试
└── logs/                      # 运行日志（自动创建）
```

---

## 3. 信号模式（Signal Mode）

信号模式下，系统计算并打印交易信号，**不执行任何下单操作**。适合上线前验证信号质量。

### 启动命令

```bash
cd new/

# 使用默认配置（auto_trade=false）
.venv/bin/python main.py

# 显式指定配置文件
.venv/bin/python main.py --config configs/app.yaml
```

### 示例输出

```
2026-04-01 09:15:00 INFO quant.main — 启动 信号模式，品种：['ES', 'NQ', 'YM', 'RTY', 'HSI', 'MHI', 'HHI']
2026-04-01 09:15:01 INFO quant.main — 主循环就绪（接入实时数据源后完整运行）
```

### 配置要点

确认 `configs/app.yaml` 中：
```yaml
strategy:
  auto_trade: false   # 信号模式：保持 false
```

---

## 4. 回测模式（Backtest Mode）

回测模式从 CSV 或 Parquet 文件加载历史 K 线，运行完整的信号 → 仓位 → 成本 → 风控流程，最后输出绩效报告。

### 启动命令

```bash
cd new/

# 基本用法（CSV 文件，文件名前缀决定交易品种）
.venv/bin/python main.py --backtest --data ../data/ES_15min.csv

# 使用 Parquet 文件
.venv/bin/python main.py --backtest --data ../data/NQ_15min.parquet

# 使用自定义配置 + 数据
.venv/bin/python main.py --config configs/app.yaml --backtest --data ../data/HSI_15min.csv
```

### 数据文件命名规则

文件名**前缀**（下划线前的部分）必须与 `configs/instruments.yaml` 中的 symbol 匹配：

| 文件名 | 解析 symbol |
|--------|------------|
| `ES_15min.csv` | ES |
| `NQ_daily.parquet` | NQ |
| `HSI_1hour.csv` | HSI |
| `MHI_15min.csv` | MHI |

### 示例输出

```
==================================================
回测结果 — ES
  总收益率：12.34%
  年化收益：18.56%
  Sharpe：1.42
  Sortino：1.87
  Calmar：0.93
  最大回撤：-8.21%
  胜率：54.32%
  总交易次数：128
==================================================
```

### 热身期说明

回测引擎默认需要 **210 根 K 线**（`backtest.warmup_bars`）作为指标热身期。数据文件建议包含至少 **700 根以上**的 K 线，以保证有效回测时间段足够长。

---

## 5. 实盘模式（Live Trading Mode）

实盘模式通过 IBKR TWS / IB Gateway 接入，自动执行交易。**启用前请务必在模拟账户测试充分。**

### 前置条件

1. **安装并登录 IBKR TWS 或 IB Gateway**
2. **开启 API 连接**：
   - TWS：`Edit → Global Configuration → API → Settings`，勾选 "Enable ActiveX and Socket Clients"
   - 端口设置：模拟账户 `7497`，实盘账户 `7496`，Gateway 默认 `4001`
3. **修改配置**，将 `auto_trade` 改为 `true`

### 修改配置文件

```yaml
# configs/app.yaml 或自定义 configs/live.yaml
ibkr:
  host: "127.0.0.1"
  port: 7497          # 模拟盘; 实盘改为 7496
  client_id: 1

strategy:
  auto_trade: true    # 关键：开启实盘下单
```

### 启动命令

```bash
cd new/

# 模拟盘实盘模式
.venv/bin/python main.py --config configs/app.yaml

# 生产实盘（建议单独一份配置文件）
.venv/bin/python main.py --config configs/live.yaml
```

### 安全提示

- 首次上线**务必用模拟账户**（Paper Trading）验证连接正常
- 建议为实盘账户设置 IBKR 侧的每日最大亏损限额
- 检查 `risk.max_daily_loss_usd` 和 `risk.circuit_breaker_drawdown` 已配置合理值

---

## 6. 配置文件详解

主配置文件位于 `configs/app.yaml`。各字段说明如下：

### 6.1 IBKR 连接配置（`ibkr`）

```yaml
ibkr:
  host: "127.0.0.1"       # TWS/Gateway IP，本机运行填 127.0.0.1
  port: 7497              # 模拟:7497 / 实盘:7496 / Gateway:4001
  client_id: 1            # 同一账户可开多个连接，需各自不同的 client_id
  symbols: [ES, NQ, YM, RTY, HSI, MHI, HHI]  # 交易品种列表
  bar_size: "15 min"      # K 线周期（影响信号频率）
  history_bars: 500       # 初始化时拉取的历史 K 线数量
```

### 6.2 策略配置（`strategy`）

#### Layer 1（跨资产宏观动量）

```yaml
strategy:
  layer1:
    weight: 0.35                    # L1 权重（与 L2 之和建议为 1.0）
    daily_lookback_bars: 252        # 日频回看周期（约1年）
    bar_lookback_bars: 500          # Bar 级回看窗口
    mii_amplitude_lookback: 60      # MII 振幅分位数计算窗口
    mii_extreme_percentile: 95.0    # 动量极端状态阈值（高于此为衰竭）
    mii_duration_halflife: 10       # 动量持续时间半衰期（单位：bar）
    mii_volume_confirm: true        # 启用成交量确认（降低假突破）
    staleness_halflife_hours: 8.0   # 特征时效性半衰期（小时）
```

#### Layer 2（Alpha 因子合成）

```yaml
  layer2:
    weight: 0.65                  # L2 权重
    use_ic_weights: true          # 启用 IC 加权（false 则等权）
    ic_lookback_bars: 500         # IC 历史回看窗口
    ic_forward_horizon: 10        # IC 计算的前瞻收益期（bar 数）
    ic_shrinkage: 0.5             # IC 权重收缩强度（0=无收缩, 1=等权）
    confirm_bars: 3               # 信号确认所需连续 bar 数（防抖）
    entry_threshold: 0.3          # 入场信号阈值（|S| > 0.3 触发开仓）
    exit_threshold: 0.05          # 离场信号阈值（|S| < 0.05 触发平仓）
```

#### 通用策略参数

```yaml
  open_filter_minutes: 15   # 开盘后跳过前 N 分钟（避开开盘波动）
  auto_trade: false         # false=信号模式，true=实盘下单
```

### 6.3 风控配置（`risk`）

```yaml
risk:
  risk_per_trade: 0.02             # 单笔风险占净值比（2%）
  stop_loss_atr: 2.0               # 止损距离（2 倍 ATR）
  take_profit_atr: 4.0             # 止盈距离（4 倍 ATR，盈亏比 2:1）
  trailing_stop_atr: 1.5           # 跟踪止损收紧距离（1.5 倍 ATR）
  max_leverage: 2.0                # 最大杠杆倍数
  max_contracts_per_instrument: 10 # 单品种最大持仓手数
  max_portfolio_positions: 5       # 同时最大持仓品种数
  max_drawdown_pct: 0.10           # 最大回撤警戒线（10%）
  max_daily_loss_usd: 5000.0       # 每日最大亏损（USD）
  max_daily_trades: 20             # 每日最大交易次数
  circuit_breaker_drawdown: 0.08   # 熔断触发阈值（日内回撤 8%）
  circuit_breaker_cooldown_bars: 40 # 熔断后冷却 bar 数（40 bar ≈ 10小时）
  tiered_exit_aggressive_ticks: 3  # 分批平仓报价超出对手价的 tick 数
  max_holding_bars: 0              # 最大持仓时间（0=不限制）
  default_hkd_usd_rate: 0.128      # HKD/USD 汇率（用于港股期货保证金换算）
```

### 6.4 交易成本配置（`cost`）

系统按交易所分别计算佣金，单位为每手：

```yaml
cost:
  CME:                              # ES、NQ、RTY
    commission_per_contract: 0.85   # 经纪商佣金（USD/手）
    exchange_fee_per_contract: 1.28 # 交易所费用（USD/手）
    nfa_fee_per_contract: 0.02      # NFA 监管费（USD/手）
    slippage_ticks: 1.0             # 预估滑点（tick 数）
  CBOT:                             # YM
    commission_per_contract: 0.85
    exchange_fee_per_contract: 1.28
    nfa_fee_per_contract: 0.02
    slippage_ticks: 1.0
  HKEX:                             # HSI、MHI、HHI（单位：HKD）
    commission_per_contract: 20.0
    exchange_fee_per_contract: 10.0
    nfa_fee_per_contract: 0.0
    slippage_ticks: 1.0
```

### 6.5 回测配置（`backtest`）

```yaml
backtest:
  warmup_bars: 210           # 指标热身所需 bar 数（不计入统计）
  window_size: 500           # 策略滑动窗口大小
  initial_capital: 100000.0  # 初始资金（USD）
  data_dir: "data/"          # 数据目录
  train_months: 6            # Walk-Forward 训练期（月）
  test_months: 2             # Walk-Forward 测试期（月）
  cv_embargo_bars: 5         # 训练/测试之间的 gap（防止信息泄漏）
```

### 6.6 全局参数

```yaml
instruments_path: "configs/instruments.yaml"  # 合约规格文件路径
log_file: "logs/signals.log"                   # 日志文件路径
log_level: "INFO"                              # 日志级别（DEBUG/INFO/WARNING/ERROR）
fred_api_key: ""                               # FRED API Key（宏观数据，可空）
refresh_interval: 15                           # 实盘主循环刷新间隔（秒）
```

---

## 7. 交易标的规格

系统内置 7 个期货合约（`configs/instruments.yaml`），均已包含完整规格：

| Symbol | 名称 | 交易所 | 货币 | 乘数 | 最小跳动 | 初始保证金 |
|--------|------|--------|------|------|----------|------------|
| ES | 标普500期货 | CME | USD | 50 | 0.25 | $15,200 |
| NQ | 纳斯达克100期货 | CME | USD | 20 | 0.25 | $21,000 |
| YM | 道琼斯期货 | CBOT | USD | 5 | 1.0 | $11,000 |
| RTY | 罗素2000期货 | CME | USD | 50 | 0.10 | $8,800 |
| HSI | 恒生指数期货 | HKEX | HKD | 50 | 1.0 | HKD 132,180 |
| MHI | 小型恒指期货 | HKEX | HKD | 10 | 1.0 | HKD 26,436 |
| HHI | 国企指数期货 | HKEX | HKD | 50 | 1.0 | HKD 56,000 |

### 交易时段

- **美股期货（ES/NQ/YM/RTY）**：周日 17:00 CT — 周五 16:00 CT（近24小时），每天 16:00-16:15 CT 维护中断
- **港股期货（HSI/MHI/HHI）**：
  - 早盘：09:15–12:00 HKT
  - 午盘：13:00–16:30 HKT
  - 夜盘（T+1）：17:15–次日 03:00 HKT

---

## 8. 风险管理参数调优

### 8.1 单笔风险与仓位计算

系统使用 ATR 动态确定持仓手数：

```
止损距离（点）= stop_loss_atr × ATR(14)
风险金额（USD）= 净值 × risk_per_trade
手数 = floor(风险金额 / (止损距离 × multiplier))
```

**示例**（ES，净值 $100,000）：
- ATR(14) = 20 points
- 止损距离 = 2.0 × 20 = 40 points
- 风险金额 = $100,000 × 2% = $2,000
- 手数 = floor($2,000 / (40 × $50)) = floor(1.0) = **1 手**

### 8.2 熔断机制

当日内回撤超过 `circuit_breaker_drawdown`（默认 8%）时自动停止交易，并在 `circuit_breaker_cooldown_bars` 个 bar（默认 40 bar ≈ 10 小时）后自动恢复。

### 8.3 分批平仓（TieredExit）

持仓信号反转时，系统分三批平仓：
- **第一批**：33%（距当前价 +3 tick 的 LIMIT_AGGRESSIVE 单）
- **第二批**：33%
- **第三批**：34%

---

## 9. 数据格式要求

### 9.1 CSV 格式

回测 CSV 文件须包含以下列（列名大小写不敏感）：

```csv
timestamp,open,high,low,close,volume
2024-01-02 09:30:00,4750.25,4762.50,4748.75,4758.00,45231
2024-01-02 09:45:00,4758.00,4771.25,4755.50,4769.75,38902
...
```

支持的时间戳列名：`timestamp`、`date`、`datetime`

### 9.2 Parquet 格式

与 CSV 相同的 schema，列名一致即可。Polars 原生读取：

```python
import polars as pl
df = pl.read_parquet("data/ES_15min.parquet")
```

### 9.3 从 Yahoo Finance 下载数据（示例）

```python
import yfinance as yf
import polars as pl

# 下载 ES 期货日频数据（仅作演示，实盘请使用 IBKR 历史数据）
ticker = yf.Ticker("ES=F")
df = ticker.history(period="2y", interval="1h")
df.reset_index(inplace=True)
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"datetime": "timestamp"})
pl.from_pandas(df[["timestamp","open","high","low","close","volume"]]).write_csv("data/ES_1hour.csv")
```

---

## 10. 运维与日志

### 10.1 日志文件

运行后日志写入 `logs/signals.log`（目录自动创建）。实时查看：

```bash
tail -f logs/signals.log
```

调整日志级别（`configs/app.yaml`）：
```yaml
log_level: "DEBUG"   # 开发调试时使用 DEBUG
```

### 10.2 测试套件

```bash
cd new/

# 运行所有测试
.venv/bin/python -m pytest tests/ -v

# 仅运行某类测试
.venv/bin/python -m pytest tests/test_risk_engine.py -v

# 带覆盖率报告
.venv/bin/python -m pytest tests/ --cov=src/quant --cov-report=term-missing
```

### 10.3 代码质量检查

```bash
cd new/

# Ruff 代码风格检查
.venv/bin/ruff check src/ tests/

# Mypy 类型检查
.venv/bin/mypy src/ --ignore-missing-imports
```

### 10.4 GitHub Actions CI

Push 到 `main` 分支后，`.github/workflows/ci.yml` 自动运行 ruff + mypy + pytest（Ubuntu 最新版）。

---

## 11. 常见问题

### Q1：连接 TWS 失败，报 `ConnectionRefusedError`

**检查步骤**：
1. 确认 TWS/Gateway 正在运行且已登录
2. `Edit → Global Configuration → API → Settings`，确认已勾选 "Enable ActiveX and Socket Clients"
3. 确认端口号与配置一致（模拟盘 7497，实盘 7496）
4. 检查防火墙未阻止 `127.0.0.1:7497`

### Q2：回测输出 `Unknown symbol in filename`

文件名前缀必须与 `instruments.yaml` 中定义的 symbol 完全匹配（大小写），例如 `ES_15min.csv` 而非 `es_15min.csv`。

### Q3：如何只回测单一品种且不下单到 IBKR？

回测模式本身不连接 IBKR，直接运行：
```bash
.venv/bin/python main.py --backtest --data data/ES_15min.csv
```

### Q4：如何调整信号灵敏度？

- 降低 `entry_threshold`（如 0.2）：信号更灵敏，交易次数增多
- 提高 `entry_threshold`（如 0.4）：仅在高置信度时交易，次数减少
- 调整 `confirm_bars`（1–5）：防抖力度，越大越保守

### Q5：港股期货保证金换算问题

系统自动使用 `risk.default_hkd_usd_rate` 换算 HKD 保证金为 USD。若汇率有变，更新该配置即可：
```yaml
risk:
  default_hkd_usd_rate: 0.128   # 根据实时汇率更新
```

### Q6：如何仅对部分品种运行？

修改 `configs/app.yaml` 中的 `symbols` 列表：
```yaml
ibkr:
  symbols: [ES, NQ]   # 只交易美股期货
```

### Q7：FRED 宏观数据获取失败

Layer1 宏观特征可选依赖 FRED。若无 API Key，系统将使用零值填充（信号退化为纯技术面）。申请 Key：https://fred.stlouisfed.org/docs/api/api_key.html，填入后：
```yaml
fred_api_key: "your_key_here"
```

---

*最后更新：2026-04-01*
