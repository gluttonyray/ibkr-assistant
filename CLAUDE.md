# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run live trading assistant (requires TWS/Gateway running)
.venv/bin/python main.py

# Backtesting
.venv/bin/python run_backtest.py --symbol AAPL --data data/AAPL_15min.csv
.venv/bin/python run_backtest.py --symbol AAPL --download  # fetch from IBKR

# Walk-Forward optimization
.venv/bin/python run_optimize.py --symbol AAPL --data data/AAPL_15min.csv
.venv/bin/python run_optimize.py --symbol AAPL --download --train-months 6 --test-months 2

# Tests (176 tests expected to pass)
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m pytest tests/test_indicators.py -v  # single module

# Dependencies
pip install -r requirements.txt
brew install ta-lib  # required C library (macOS)
```

## Architecture

### Live Trading Data Flow
```
TWS API → DataFeed.on_bar()
  → SignalAggregator.process() [thread pool, non-blocking]
    → RegimeDetector (ADX + ATR → 4 regimes)
    → 11 indicators with regime-adjusted weights
    → Anti-flicker: confirm_bars consecutive consistent signals
    → AggregateResult (score ∈ [-1,+1], regime, signal)
      → Dashboard.update()      [Rich terminal UI]
      → AutoTrader.on_signal()  [disabled by default]
      → logs/signals.log        [JSON Lines]
```

### Indicator System (`src/indicators/`)
- `BaseIndicator` returns `SignalResult(signal: int, value: float, label: str)` where signal ∈ {-1, 0, +1}
- 11 indicators across 5 categories: trend (EMAcross/MACD/ADX), momentum (RSI/KDJ/CCI/ROC), volatility (BB), volume (OBV/VWAP), support-resistance (PivotPoints)
- `RegimeDetector` classifies into: `TRENDING_CALM`, `TRENDING_VOLATILE`, `RANGING`, `CHOPPY_VOLATILE`
- `SignalAggregator` applies per-regime category multipliers (e.g., `CHOPPY_VOLATILE` uses 2.0× threshold, suppressing most signals)
- Anti-flicker via `deque(maxlen=confirm_bars)` — signal only fires if last N bars agree

### Backtest/Optimization Parity with Live
- Same 500-bar rolling window and 210-bar warmup (matches EMAcross `min_bars=205`)
- Same `SignalAggregator` logic — optimizer tunes 5 category-level weight multipliers (not per-indicator) to keep search space to 405 combinations
- `CostModel` mirrors real IBKR US tiered pricing: commission, SEC/TAF/FINRA fees, slippage (bps), margin interest
- `WalkForwardOptimizer` uses `ProcessPoolExecutor` for parallel grid evaluation

### Key Files
- `src/config.py` — singleton `cfg` loaded from `.env`; `BAR_SIZE`, `CONFIRM_BARS`, `BUY_THRESHOLD`, `AUTO_TRADE`, `MAX_ORDERS_PER_DAY` etc.
- `src/trading/data_feed.py` — IBKR connection; futures detected by "main" suffix (e.g., "ESmain")
- `src/backtest/engine.py` — bar-by-bar simulation; limit orders checked against next bar range
- `src/backtest/report.py` — Plotly HTML report with 9 sections
- `tests/conftest.py` — shared synthetic OHLCV fixtures (bullish/bearish/flat/volatile)

### Async Architecture
- Entry: `asyncio.run(_main())` with ib_insync event loop (`util.startLoop()`)
- DataFeed bar callbacks offload indicator computation to thread pool to avoid blocking market data
- AutoTrader coroutines scheduled back via `run_coroutine_threadsafe()`
