# IBKR Signal Assistant — 使用说明

## 目录
1. [快速开始](#快速开始)
2. [实盘/模拟盘监控](#实盘模拟盘监控)
3. [回测系统](#回测系统)
4. [参数优化（Walk-Forward）](#参数优化walk-forward)
5. [配置参考](#配置参考)
6. [策略说明](#策略说明)
7. [自动交易](#自动交易)

---

## 快速开始

### 环境准备

```bash
# 1. 克隆/进入项目
cd ibkr-signal-assistant

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 TA-Lib（macOS）
brew install ta-lib

# 5. 复制配置文件
cp .env.example .env
```

### 配置 .env

用文本编辑器打开 `.env`，根据实际情况修改：

```env
IBKR_HOST=127.0.0.1
IBKR_PORT=7497        # 7497=TWS模拟盘, 7496=TWS实盘, 4002=IB Gateway模拟盘
IBKR_CLIENT_ID=1

SYMBOLS=AAPL,MSFT,TSLA,SPY,QQQ   # 监控的标的

BAR_SIZE=15 min       # 15分钟K线（推荐）
CONFIRM_BARS=5        # 连续5根K线信号一致才触发（75分钟确认窗口）
```

### 推荐工作流

**在启用自动交易前，必须先完成以下步骤：**

```
1. 准备历史数据（CSV）
       ↓
2. 运行 Walk-Forward 优化，找到稳定参数
       ↓
3. 用最优参数运行回测，验证收益曲线
       ↓
4. 更新 .env（BUY_THRESHOLD / CONFIRM_BARS）
       ↓
5. 模拟盘运行监控，观察信号质量
       ↓
6. 确认后再启用 AUTO_TRADE=true
```

---

## 实盘/模拟盘监控

### 前提条件

1. **TWS 已运行**，并在 TWS 设置中开启 API：
   - `File → Global Configuration → API → Settings`
   - 勾选 `Enable ActiveX and Socket Clients`
   - Socket port 填写与 `.env` 中 `IBKR_PORT` 一致的端口

2. 使用**模拟账户**测试（强烈建议）：
   - TWS 登录时选择 "Paper Trading" 账户
   - 端口默认 `7497`

### 启动监控

```bash
.venv/bin/python main.py
```

### 终端看板说明

启动后会显示 Rich 终端看板，每 15 秒按定时器刷新（不阻塞事件循环）：

```
┌─────────┬───────┬──────────────────┬──────────────────────┬────────┬───┐
│ Symbol  │ Price │ Regime           │ Score                │ Signal │ ✓ │
├─────────┼───────┼──────────────────┼──────────────────────┼────────┼───┤
│ AAPL    │150.23 │ ↗ TREND          │ ·····|████·····  +0.612 │ BUY ▲  │ ✓ │
│ MSFT    │380.10 │ ↔ RANGE          │ ·····|·····  +0.121  │ HOLD ─ │ · │
│ SPY     │470.55 │ ↗ NOISY          │ ···██|·····  -0.451  │ SELL ▼ │ ✓ │
└─────────┴───────┴──────────────────┴──────────────────────┴────────┴───┘
```

| 字段 | 含义 |
|------|------|
| Score | 加权综合得分，范围 [-1, +1]；图形直观显示方向和强度 |
| Signal | BUY / SELL / HOLD |
| Regime | 市场状态（见策略说明） |
| ✓ | 已通过连续 5 根 K 线一致性确认 |

### 信号日志

确认信号会以 JSON Lines 格式写入 `logs/signals.log`：

```bash
# 查看最新信号
tail -f logs/signals.log | python3 -m json.tool
```

---

## 回测系统

### 获取数据

**方法一：从 IBKR 直接下载（推荐）**

无需手动准备 CSV，程序自动从 TWS 拉取数据并缓存为 Parquet：

```bash
# 默认：2年、15分钟K线
.venv/bin/python run_backtest.py --symbol AAPL --download

# 自定义时长和粒度
.venv/bin/python run_backtest.py --symbol AAPL --download --duration "1 Y" --bar-size "5 min"
```

数据自动缓存到 `data/AAPL_15_min_2Y.parquet`，下次运行直接使用缓存（删除文件可重新下载）。

**方法二：使用本地 CSV/Parquet 文件**

支持 Yahoo Finance 导出格式，需包含 `Date, Open, High, Low, Close, Volume` 列：

```bash
.venv/bin/python run_backtest.py --symbol AAPL --data data/AAPL_15min.csv
```

### 运行回测

```bash
# 从 IBKR 下载并回测（最常用）
.venv/bin/python run_backtest.py --symbol AAPL --download

# 自定义参数
.venv/bin/python run_backtest.py --symbol SPY --download \
    --capital 50000 \
    --position-size 200 \
    --slippage 3 \
    --output my_report.html
```

### 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | 必填 | 股票代码 |
| `--download` | 否 | 从 IBKR 下载数据（与 `--data` 二选一） |
| `--data` | — | CSV 或 Parquet 文件路径（与 `--download` 二选一） |
| `--duration` | `2 Y` | IBKR 下载时长（`"1 Y"`, `"6 M"`, `"2 Y"` 等） |
| `--bar-size` | `15 min` | K线粒度（`"1 min"`, `"5 min"`, `"15 min"`, `"1 hour"`, `"1 day"`） |
| `--capital` | 100000 | 初始资金（美元） |
| `--position-size` | 100 | 每笔交易股数 |
| `--slippage` | 5 | 滑点（基点，5bps=0.05%） |
| `--output` | `backtest_report_<SYMBOL>.html` | 报告输出路径 |

### 回测报告

回测完成后在浏览器中打开 HTML 报告：

```bash
open backtest_report_AAPL.html   # macOS
```

**报告包含：**
- 关键指标卡片（Total Return / CAGR / Sharpe / Sortino / Max DD / Win Rate 等）
- K 线图 + 交易标记（绿色三角=买入，红色三角=卖出，×=平仓）+ 市场状态底色
- 权益曲线 + 高水位线
- 回撤深度图
- 信号得分走势（含阈值线）
- 成交量
- 月度收益热力图
- 交易明细表（含费用细项）
- 各市场状态下的胜率/平均盈亏分析

### 费用模型

回测内置 IBKR 美股 Tiered 费率模型：

| 费用项 | 计算方式 |
|--------|----------|
| 佣金 | $0.005/股，最低 $1，最高交易额的 1% |
| SEC 费（仅卖出） | $8/百万美元交易额 |
| TAF 费（仅卖出） | $0.000166/股，最高 $8.30 |
| FINRA 费（仅卖出） | $0.00278/股 |
| 滑点 | 可配置基点数（默认 5bps） |
| 融资利率 | 年化 6.83%，按日计息（仅空仓） |

---

## 参数优化（Walk-Forward）

> 这是在启用自动交易前的**必要步骤**。手工设定的参数（如 0.4 阈值）未经验证，不应直接用于实盘。

### 优化原理

**Walk-Forward 验证流程：**

```
完整历史数据（例如 2 年）
├─ 窗口 1: 训练 Jan–Jun → 测试 Jul–Aug
├─ 窗口 2: 训练 Mar–Aug → 测试 Sep–Oct
├─ 窗口 3: 训练 May–Oct → 测试 Nov–Dec
└─ ...
```

每个训练窗口网格搜索 **405 种参数组合**，取最优参数在**样本外（OOS）** 测试集上验证。通过 IS vs OOS Sharpe 对比检测过拟合。

### 参数网格（默认）

| 参数 | 候选值 | 含义 |
|------|--------|------|
| `buy_threshold` | 0.30, 0.35, 0.40, 0.45, 0.50 | 买入信号得分阈值 |
| `confirm_bars` | 3, 5, 7 | 确认所需的连续 K 线数 |
| `trend_weight_mult` | 0.8, 1.0, 1.5 | 趋势类指标权重缩放 |
| `momentum_weight_mult` | 0.5, 1.0, 1.2 | 动量类指标权重缩放 |
| `volume_weight_mult` | 0.8, 1.0, 1.2 | 成交量类指标权重缩放 |
| **总计** | | **5×3×3×3×3 = 405 组合** |

### 运行优化

```bash
# 从 IBKR 下载 + 优化（最简单，推荐）
.venv/bin/python run_optimize.py --symbol AAPL --download

# 自定义下载参数
.venv/bin/python run_optimize.py --symbol AAPL --download --duration "2 Y" --bar-size "15 min"

# 从本地文件
.venv/bin/python run_optimize.py --symbol AAPL --data data/AAPL_15min.csv

# 自定义窗口和目标函数
.venv/bin/python run_optimize.py --symbol AAPL --download \
    --train-months 6 \
    --test-months 2 \
    --objective sharpe_trades \
    --min-trades 10 \
    --output optimization_AAPL.html
```

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | 必填 | 股票代码 |
| `--download` | 否 | 从 IBKR 下载数据（与 `--data` 二选一） |
| `--data` | — | CSV 或 Parquet 文件路径（与 `--download` 二选一） |
| `--duration` | `2 Y` | IBKR 下载时长 |
| `--bar-size` | `15 min` | K线粒度 |
| `--train-months` | 6 | 训练窗口长度（月） |
| `--test-months` | 2 | 测试窗口长度（月） |
| `--step-months` | 同 test | 滚动步进（月） |
| `--objective` | `sharpe` | 目标函数：`sharpe` 或 `sharpe_trades` |
| `--min-trades` | 5 | 有效参数组所需的最少交易次数 |
| `--capital` | 100000 | 初始资金 |
| `--position-size` | 100 | 每笔股数 |
| `--slippage` | 5 | 滑点（基点） |
| `--workers` | CPU-1 | 并行进程数 |
| `--output` | `optimization_report_<SYMBOL>.html` | 报告路径 |

### 目标函数说明

| 函数 | 公式 | 适用场景 |
|------|------|----------|
| `sharpe` | Sharpe 比率（交易数 < min_trades 则惩罚为 -999） | 优先追求风险调整收益 |
| `sharpe_trades` | Sharpe × √(交易数) | 兼顾质量与频率，避免"零交易高 Sharpe"陷阱 |

### 优化报告

```bash
open optimization_report_AAPL.html   # macOS
```

**报告包含：**
1. **摘要卡片** — 聚合 OOS Sharpe、最稳定参数
2. **窗口明细表** — 每个窗口 IS vs OOS Sharpe、交易数、收益率、Sharpe 衰减百分比
3. **参数频率柱状图** — 每个参数值被选为"最优"的次数（稳定性指标）
4. **OOS 权益曲线** — 所有测试窗口的权益曲线拼接
5. **过拟合检测散点图** — IS Sharpe vs OOS Sharpe（偏离对角线越远 = 过拟合越严重）

### 解读结果与更新参数

```
优化报告 → "Most Stable Parameters" → 更新 .env
```

例如，优化结果显示 `buy_threshold=0.45, confirm_bars=5`，则：

```env
BUY_THRESHOLD=0.45
SELL_THRESHOLD=-0.45
CONFIRM_BARS=5
```

然后用最优参数重跑回测验证：

```bash
# 数据已缓存在 data/ 目录（首次下载后自动保存），无需重新下载
.venv/bin/python run_backtest.py --symbol AAPL --download
```

---

## 配置参考

`.env` 完整参数说明：

```env
# ── IBKR 连接 ──
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # 7497=TWS模拟, 7496=TWS实盘, 4002=Gateway模拟
IBKR_CLIENT_ID=1

# ── 监控标的 ──
SYMBOLS=AAPL,MSFT,TSLA,SPY,QQQ

# ── K线设置 ──
BAR_SIZE=15 min         # 推荐15分钟，策略针对此周期校准
HISTORY_BARS=500        # 历史K线数（500×15min ≈ 3周）

# ── 信号阈值（建议通过 run_optimize.py 确定，勿直接手填）──
BUY_THRESHOLD=0.4       # 综合得分超过此值触发买入信号
SELL_THRESHOLD=-0.4     # 综合得分低于此值触发卖出信号

# ── 防抖动 ──
CONFIRM_BARS=5          # 需要连续5根K线信号一致才确认（75分钟）

# ── 界面刷新 ──
REFRESH_INTERVAL=15     # 看板刷新间隔（秒），渲染解耦于数据更新

# ── 开盘过滤 ──
OPEN_FILTER_MINUTES=30  # 跳过NYSE 09:30后的前30分钟，避免开盘噪音

# ── 自动交易（默认关闭）──
AUTO_TRADE=false
ATR_LIMIT_OFFSET=0.5    # 限价单偏移 = ATR × 此系数
MAX_POSITION_SIZE=100   # 每个标的最大持仓股数
DAILY_LOSS_LIMIT=500    # 日亏损熔断（美元），超过后停止当日交易
ORDER_TIMEOUT=60        # 未成交订单取消等待时间（秒）
MAX_ORDERS_PER_DAY=20   # 每日最大下单次数（防信号风暴）

# ── 日志 ──
LOG_FILE=logs/signals.log
LOG_LEVEL=INFO
```

---

## 策略说明

### 综合得分机制

系统运行 11 个去相关指标，每个指标输出 +1（看涨）、0（中性）、-1（看跌），经 Regime 动态加权后得到综合得分：

| 类别 | 指标 | 基础权重 |
|------|------|----------|
| 趋势 | EMA 多线排列 | 1.5 |
| 趋势 | MACD 柱状图 | 1.5 |
| 趋势 | ADX + DI 交叉 | 1.2 |
| 动量 | RSI(14) 极值区 | 1.2 |
| 动量 | KDJ/Stoch | 1.0 |
| 动量 | CCI(20) 极值区 | 0.8 |
| 动量 | ROC(12) | 0.8 |
| 波动 | 布林带位置 | 1.2 |
| 成交量 | OBV 趋势 | 1.0 |
| 成交量 | VWAP 锚点 | 1.2 |
| 支撑压力 | 经典 Pivot 点位 | 1.0 |

> 基础权重可通过 Walk-Forward 优化按类别整体缩放，无需手工调参。

### 市场状态（Regime）

系统实时检测市场状态，动态调整各类指标权重和信号阈值：

| 状态 | 判断条件 | 趋势权重 | 动量权重 | 阈值乘数 |
|------|----------|----------|----------|----------|
| TRENDING_CALM | ADX>25, ATR正常 | ×1.3 | ×0.8 | ×1.0（基准） |
| TRENDING_VOLATILE | ADX>25, ATR偏高 | ×1.5 | ×0.5 | ×1.3（更严格） |
| RANGING | ADX≤25, ATR正常 | ×0.5 | ×1.5 | ×0.85（略宽松） |
| CHOPPY_VOLATILE | ADX≤25, ATR偏高 | ×0.3 | ×0.4 | ×2.0（极难触发） |

---

## 自动交易

> **警告：** 自动交易有资金风险。**必须先通过 Walk-Forward 优化确认参数稳定性**，再在模拟账户充分验证后启用。

### 启用步骤

1. 完成 Walk-Forward 优化，将最优参数写入 `.env`
2. 在 `.env` 中设置 `AUTO_TRADE=true`
3. 确认 TWS/Gateway 已允许 API 下单
4. 启动程序，**终端会显示醒目警告**

### 交易逻辑

- 收到确认的 BUY 信号（连续 5 根 K 线）→ 以 `当前价 - ATR×0.5` 挂限价买单
- 收到确认的 SELL 信号 → 以 `当前价 + ATR×0.5` 挂限价卖单
- 60 秒未成交自动撤单
- 日亏损超过 `DAILY_LOSS_LIMIT` 自动熔断，停止当日所有新订单
- 每日下单上限 `MAX_ORDERS_PER_DAY`（默认 20 次）

### 风控建议

| 参数 | 保守设置 | 说明 |
|------|---------|------|
| `MAX_POSITION_SIZE` | 50-100 股 | 限制单笔头寸规模 |
| `DAILY_LOSS_LIMIT` | 200-500 美元 | 日亏损熔断 |
| `CONFIRM_BARS` | 5（默认）| 不要低于3，避免假信号 |
| `BUY_THRESHOLD` | 由优化器确定 | 不要手工设定，用 run_optimize.py 找最优值 |

---

## 常见问题

**Q: 连接 TWS 失败**
- 检查 TWS 是否运行，端口是否正确（模拟盘 7497）
- TWS → Global Configuration → API → Settings → 确认已开启 Socket 连接
- 检查防火墙是否阻止了 127.0.0.1 的连接

**Q: 历史数据请求失败**
- IBKR 对 API 请求有频率限制，稍等后重试
- 检查账户是否有相应市场数据订阅

**Q: 数据怎么获取？**
- **推荐**：使用 `--download` 直接从 IBKR 拉取，自动缓存为 Parquet，无需手动准备
- 也可使用本地 CSV（需含 Date/Open/High/Low/Close/Volume 列）
- IBKR 支持的 bar_size：`"1 min"`, `"5 min"`, `"15 min"`, `"1 hour"`, `"1 day"`
- IBKR 支持的 duration：`"1 D"`, `"1 W"`, `"1 M"`, `"6 M"`, `"1 Y"`, `"2 Y"`

**Q: 优化耗时估算**
- 默认 405 组合，2年数据，8核机器：约 4 分钟/窗口
- 总时长 ≈ 窗口数 × 4 分钟（2年数据约 6 个窗口 ≈ 24 分钟）
- 使用 `--workers` 调整并行度；使用小网格快速验证

**Q: 运行测试**
```bash
.venv/bin/python -m pytest tests/ -v
# 预期：176 tests passed
```
