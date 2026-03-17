"""Tests for the backtest engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, WARMUP_BARS
from src.backtest.portfolio import Portfolio
from src.backtest.cost_model import CostModel


def _make_backtest_df(n=500, trend=0.001, seed=42):
    """Generate a synthetic OHLCV DataFrame for backtesting."""
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for _ in range(n - 1):
        ret = trend + 0.008 * rng.standard_normal()
        prices.append(prices[-1] * (1 + ret))

    closes = np.array(prices)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = closes * (1 + abs(rng.normal(0, 0.005, n)))
    lows = closes * (1 - abs(rng.normal(0, 0.005, n)))
    volumes = rng.integers(100_000, 5_000_000, size=n).astype(float)

    # Use intraday timestamps (15-min bars during NYSE hours)
    dates = []
    current = pd.Timestamp("2024-01-02 10:00")
    for i in range(n):
        dates.append(current)
        current += pd.Timedelta(minutes=15)
        # Skip to next day's 10:00 after 16:00
        if current.hour >= 16:
            current = current.normalize() + pd.Timedelta(days=1, hours=10)
            # Skip weekends
            while current.weekday() >= 5:
                current += pd.Timedelta(days=1)

    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


class TestEngineBasic:
    def test_engine_runs(self):
        df = _make_backtest_df(n=500, trend=0.002)
        engine = BacktestEngine("TEST", df, initial_capital=100000, position_size=100)
        result = engine.run()
        assert result.symbol == "TEST"
        assert result.initial_capital == 100000
        assert len(result.equity_curve) == len(df)

    def test_insufficient_data(self):
        df = _make_backtest_df(n=100)
        with pytest.raises(ValueError, match="Insufficient data"):
            engine = BacktestEngine("TEST", df)
            engine.run()

    def test_equity_starts_at_capital(self):
        df = _make_backtest_df(n=500)
        engine = BacktestEngine("TEST", df, initial_capital=50000)
        result = engine.run()
        # First equity point should be near initial capital
        assert abs(result.equity_curve.iloc[0] - 50000) < 100

    def test_warmup_no_trades(self):
        """No trades should occur during warmup period."""
        df = _make_backtest_df(n=300, trend=0.0)  # flat market, few signals
        engine = BacktestEngine("TEST", df, initial_capital=100000)
        result = engine.run()
        # All trades should have entry time after warmup
        for trade in result.trades:
            trade_idx = df.index.get_loc(trade.entry_time, method="nearest")
            assert trade_idx >= WARMUP_BARS


class TestPortfolioIntegration:
    def test_trades_have_costs(self):
        df = _make_backtest_df(n=500, trend=0.003)
        engine = BacktestEngine("TEST", df, slippage_bps=5)
        result = engine.run()
        if result.trades:
            t = result.trades[0]
            assert t.entry_costs.total > 0
            assert t.exit_costs.total > 0

    def test_final_equity_reflects_pnl(self):
        df = _make_backtest_df(n=500, trend=0.002)
        engine = BacktestEngine("TEST", df, initial_capital=100000)
        result = engine.run()
        # Final equity should differ from initial if any trades occurred
        if result.trades:
            assert result.final_equity != result.initial_capital

    def test_symbol_set_on_trades(self):
        df = _make_backtest_df(n=500, trend=0.003)
        engine = BacktestEngine("AAPL", df)
        result = engine.run()
        for trade in result.trades:
            assert trade.symbol == "AAPL"


class TestEngineSignals:
    def test_signals_recorded(self):
        df = _make_backtest_df(n=500, trend=0.002)
        engine = BacktestEngine("TEST", df)
        result = engine.run()
        assert len(result.signals) > 0

    def test_signal_fields(self):
        df = _make_backtest_df(n=500)
        engine = BacktestEngine("TEST", df)
        result = engine.run()
        if result.signals:
            sig = result.signals[0]
            assert "score" in sig
            assert "signal" in sig
            assert "confirmed" in sig
            assert "regime" in sig
            assert "price" in sig


class TestCostModelIntegration:
    def test_slippage_affects_pnl(self):
        df = _make_backtest_df(n=500, trend=0.003, seed=1)
        # Run with different slippage
        e1 = BacktestEngine("TEST", df, slippage_bps=0)
        r1 = e1.run()
        e2 = BacktestEngine("TEST", df, slippage_bps=20)
        r2 = e2.run()
        # Higher slippage should reduce final equity (if trades occur)
        if r1.trades and r2.trades:
            assert r2.final_equity <= r1.final_equity + 100  # allow small tolerance
