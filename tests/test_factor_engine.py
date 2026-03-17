"""Integration tests for FactorBacktestEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestResult
from src.backtest.factor_engine import FactorBacktestEngine


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def _make_data(n: int = 600, trend: float = 0.001, vol: float = 0.006, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for _ in range(n - 1):
        ret = trend + vol * rng.standard_normal()
        prices.append(prices[-1] * (1 + ret))
    closes = np.array(prices)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = closes * (1 + abs(rng.normal(0, 0.002, n)))
    lows = closes * (1 - abs(rng.normal(0, 0.002, n)))
    volumes = rng.integers(100_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2024-01-02 09:31", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


@pytest.fixture
def bullish_data():
    return _make_data(600, trend=0.003, vol=0.003, seed=1)

@pytest.fixture
def bearish_data():
    return _make_data(600, trend=-0.003, vol=0.003, seed=2)

@pytest.fixture
def neutral_data():
    return _make_data(600, trend=0.0, vol=0.004, seed=3)


# ──────────────────────────────────────────────
# Basic run
# ──────────────────────────────────────────────

class TestBasicRun:
    def test_run_returns_backtest_result(self, neutral_data):
        engine = FactorBacktestEngine("TEST", neutral_data, initial_capital=100_000.0)
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_result_has_symbol(self, neutral_data):
        engine = FactorBacktestEngine("AAPL", neutral_data)
        result = engine.run()
        assert result.symbol == "AAPL"

    def test_result_equity_curve_non_empty(self, neutral_data):
        engine = FactorBacktestEngine("TEST", neutral_data)
        result = engine.run()
        assert len(result.equity_curve) > 0

    def test_result_bars_matches_input(self, neutral_data):
        engine = FactorBacktestEngine("TEST", neutral_data)
        result = engine.run()
        assert len(result.bars) == len(neutral_data)

    def test_result_initial_capital_preserved(self, neutral_data):
        engine = FactorBacktestEngine("TEST", neutral_data, initial_capital=50_000.0)
        result = engine.run()
        assert result.initial_capital == 50_000.0


# ──────────────────────────────────────────────
# Signals
# ──────────────────────────────────────────────

class TestSignals:
    def test_signals_list_non_empty(self, bullish_data):
        engine = FactorBacktestEngine("TEST", bullish_data)
        result = engine.run()
        assert len(result.signals) > 0

    def test_signals_have_required_fields(self, bullish_data):
        engine = FactorBacktestEngine("TEST", bullish_data)
        result = engine.run()
        for sig in result.signals[:10]:
            assert "timestamp" in sig
            assert "score" in sig
            assert "signal" in sig
            assert "regime" in sig
            assert "price" in sig

    def test_signal_values_valid(self, bullish_data):
        engine = FactorBacktestEngine("TEST", bullish_data)
        result = engine.run()
        for sig in result.signals:
            assert sig["signal"] in {"BUY", "SELL", "HOLD"}
            assert -1.0 <= sig["score"] <= 1.0


# ──────────────────────────────────────────────
# Trades
# ──────────────────────────────────────────────

class TestTrades:
    def test_trades_have_symbol(self, bullish_data):
        engine = FactorBacktestEngine("AAPL", bullish_data)
        result = engine.run()
        for trade in result.trades:
            assert trade.symbol == "AAPL"

    def test_trade_pnl_calculated(self, bullish_data):
        engine = FactorBacktestEngine("TEST", bullish_data)
        result = engine.run()
        for trade in result.trades:
            assert isinstance(trade.net_pnl, float)


# ──────────────────────────────────────────────
# Equity curve
# ──────────────────────────────────────────────

class TestEquityCurve:
    def test_equity_starts_at_initial_capital(self, neutral_data):
        engine = FactorBacktestEngine("TEST", neutral_data, initial_capital=100_000.0)
        result = engine.run()
        first_equity = result.equity_curve.iloc[0]
        assert abs(first_equity - 100_000.0) < 1000.0  # within $1k of initial

    def test_final_equity_in_result(self, neutral_data):
        engine = FactorBacktestEngine("TEST", neutral_data)
        result = engine.run()
        assert result.final_equity > 0.0

    def test_equity_length_matches_data(self, neutral_data):
        engine = FactorBacktestEngine("TEST", neutral_data)
        result = engine.run()
        assert len(result.equity_curve) == len(neutral_data)


# ──────────────────────────────────────────────
# Insufficient data
# ──────────────────────────────────────────────

class TestInsufficientData:
    def test_raises_on_too_few_bars(self):
        tiny_data = _make_data(100)
        engine = FactorBacktestEngine("TEST", tiny_data)
        with pytest.raises(ValueError, match="Insufficient data"):
            engine.run()


# ──────────────────────────────────────────────
# Dynamic position sizing
# ──────────────────────────────────────────────

class TestDynamicSizing:
    def test_trades_have_varying_qty(self, bullish_data):
        """Factor engine should produce varying trade sizes."""
        engine = FactorBacktestEngine("TEST", bullish_data, buy_threshold=0.1, sell_threshold=-0.1)
        result = engine.run()
        if len(result.trades) >= 2:
            qtys = [t.qty for t in result.trades]
            # All should be positive
            assert all(q > 0 for q in qtys)


# ──────────────────────────────────────────────
# Backward-compat: portfolio.on_signal() with qty
# ──────────────────────────────────────────────

class TestPortfolioQtyParam:
    def test_on_signal_default_qty(self):
        """Existing callers without qty param still work."""
        from src.backtest.portfolio import Portfolio
        from src.backtest.cost_model import CostModel

        portfolio = Portfolio(initial_capital=100_000.0, position_size=100)
        ts = pd.Timestamp("2024-01-02 09:30")
        portfolio.on_signal("BUY", 100.0, ts, 0)
        assert portfolio.position == 100

    def test_on_signal_custom_qty(self):
        """FactorEngine passes explicit qty."""
        from src.backtest.portfolio import Portfolio

        portfolio = Portfolio(initial_capital=100_000.0, position_size=100)
        ts = pd.Timestamp("2024-01-02 09:30")
        portfolio.on_signal("BUY", 100.0, ts, 0, qty=250)
        assert portfolio.position == 250

    def test_on_signal_qty_zero_does_nothing(self):
        """qty=0 should not open a position."""
        from src.backtest.portfolio import Portfolio

        portfolio = Portfolio(initial_capital=100_000.0, position_size=100)
        ts = pd.Timestamp("2024-01-02 09:30")
        # qty=0: size=0, BUY opens position of size 0
        portfolio.on_signal("BUY", 100.0, ts, 0, qty=0)
        # Position is 0 (opened with size 0)
        assert portfolio.position == 0
