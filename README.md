# IBKR Signal Assistant

A real-time trading signal system and factor-based quantitative trading pipeline for Interactive Brokers (IBKR). Supports live monitoring, backtesting, walk-forward optimization, and a full multi-factor alpha engine.

## Features

- **Dual-mode pipeline** — classic indicator mode (discrete signals) or factor mode (continuous alpha score)
- **11 technical indicators** with regime-adaptive weighting (EMA, MACD, ADX, RSI, KDJ, CCI, ROC, Bollinger Bands, OBV, VWAP, Pivot Points)
- **20 factors** across 4 categories: technical, sentiment (VIX, put/call), macro (yield curve, DXY, fed rate), fundamental (earnings yield, P/B, revenue growth)
- **AlphaCombiner** — z-score normalization → weighted sum → regime adjustment → tanh → score ∈ [-1, +1]
- **ATR-based position sizing** proportional to conviction (alpha score)
- **Risk management** — per-trade stop-loss / take-profit / trailing stop; portfolio circuit breaker
- **Walk-forward optimizer** — 405-combination grid search with IS vs OOS Sharpe validation
- **Interactive Plotly reports** — backtest, optimization, and factor analytics
- **Auto-trader** — limit orders via IBKR API (disabled by default)

## Architecture

```
TWS API → DataFeed → [Indicator Mode]  SignalAggregator → AggregateResult
                   → [Factor Mode]     FactorRegistry
                                         → 20 factors (continuous floats)
                                         → AlphaCombiner (z-score + tanh)
                                         → PositionSizer (ATR-based)
                                         → RiskManager (SL/TP/trailing)
                                       → Dashboard + AutoTrader
```

## Requirements

- Python 3.10+
- [TA-Lib](https://ta-lib.org/) C library (`brew install ta-lib` on macOS)
- Interactive Brokers TWS or IB Gateway (for live data)
- FRED API key (optional, for macro factors)

## Installation

```bash
git clone https://github.com/gluttonyray/ibkr-assistant.git
cd ibkr-assistant

python3 -m venv .venv
source .venv/bin/activate

brew install ta-lib          # macOS only
pip install -r requirements.txt

cp .env.example .env         # edit with your settings
```

## Quick Start

### Live monitoring

```bash
# Requires TWS/Gateway running on configured port
.venv/bin/python main.py
```

### Backtest

```bash
# Download from IBKR and backtest
.venv/bin/python run_backtest.py --symbol AAPL --download

# From local CSV
.venv/bin/python run_backtest.py --symbol AAPL --data data/AAPL_15min.csv
```

### Factor-based backtest

```bash
# Uses AlphaCombiner + PositionSizer + RiskManager
.venv/bin/python run_factor_backtest.py --symbol AAPL --data data/AAPL_15min.csv
```

### Walk-forward optimization

```bash
.venv/bin/python run_optimize.py --symbol AAPL --download \
    --train-months 6 --test-months 2 --objective sharpe_trades
```

## Configuration

Copy `.env.example` to `.env` and set your parameters. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `IBKR_HOST` | `127.0.0.1` | TWS host |
| `IBKR_PORT` | `7497` | 7497=TWS paper, 7496=TWS live, 4002=Gateway paper |
| `SYMBOLS` | `AAPL,MSFT,TSLA,SPY,QQQ` | Symbols to monitor |
| `BAR_SIZE` | `15 min` | Bar size |
| `CONFIRM_BARS` | `5` | Consecutive bars required to confirm a signal |
| `BUY_THRESHOLD` | `0.4` | Alpha score threshold for BUY |
| `SELL_THRESHOLD` | `-0.4` | Alpha score threshold for SELL |
| `AUTO_TRADE` | `false` | Enable order submission |
| `FACTOR_MODE` | `false` | Use factor-based pipeline |
| `FRED_API_KEY` | — | [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) for macro factors |
| `RISK_PER_TRADE` | `0.02` | Max risk per trade (fraction of equity) |
| `STOP_LOSS_ATR` | `2.0` | Stop-loss distance in ATR multiples |
| `TAKE_PROFIT_ATR` | `4.0` | Take-profit distance in ATR multiples |
| `MAX_DRAWDOWN_PCT` | `0.10` | Portfolio circuit breaker threshold |

## Recommended Workflow

```
1. Download historical data (--download)
2. Run walk-forward optimization  → find stable parameters
3. Update .env with optimal thresholds
4. Backtest with optimal params   → verify equity curve
5. Run in paper trading mode      → validate signal quality
6. Enable AUTO_TRADE=true         → go live
```

## Market Regimes

The system detects 4 regimes via ADX + ATR, dynamically adjusting weights and thresholds:

| Regime | Condition | Trend wt | Momentum wt | Threshold |
|--------|-----------|----------|-------------|-----------|
| `TRENDING_CALM` | ADX > 25, normal ATR | ×1.3 | ×0.8 | ×1.0 |
| `TRENDING_VOLATILE` | ADX > 25, high ATR | ×1.5 | ×0.5 | ×1.3 |
| `RANGING` | ADX ≤ 25, normal ATR | ×0.5 | ×1.5 | ×0.85 |
| `CHOPPY_VOLATILE` | ADX ≤ 25, high ATR | ×0.3 | ×0.4 | ×2.0 |

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
# 465 tests passing
```

## Project Structure

```
src/
├── indicators/          # 11 indicators + RegimeDetector + SignalAggregator
├── factors/             # 20 factors (technical/sentiment/macro/fundamental)
│   ├── base.py          # BaseFactor ABC, FactorData, FactorResult
│   ├── registry.py      # @FactorRegistry.register decorator
│   └── technical/sentiment/macro/fundamental/
├── alpha/combiner.py    # AlphaCombiner: z-score + regime + tanh
├── risk/
│   ├── position_sizer.py  # ATR-based conviction sizing
│   └── risk_manager.py    # SL / TP / trailing stop / circuit breaker
├── analytics/           # IC, factor attribution, Plotly report
├── backtest/            # BacktestEngine, FactorBacktestEngine, optimizer
├── data/                # Yahoo Finance, FRED, DataCache, DataValidator
├── trading/             # DataFeed (IBKR), AutoTrader
├── ui/                  # Rich terminal dashboard
└── config.py            # Env-var config singleton
```

## Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Use at your own risk. Always test thoroughly in paper trading before using real capital.
