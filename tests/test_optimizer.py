"""
Tests for the Walk-Forward optimizer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.optimizer import (
    DEFAULT_GRID,
    WalkForwardOptimizer,
    build_param_grid,
    params_to_aggregator_kwargs,
    objective_sharpe,
    objective_sharpe_trades,
    PENALTY,
)
from src.backtest.metrics import Metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_long_df(months: int = 12, bars_per_day: int = 26, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic 15-min OHLCV data spanning `months` months.
    ~26 bars/day × 21 trading days/month.
    """
    rng = np.random.default_rng(seed)
    n_days = int(months * 21)
    n = n_days * bars_per_day

    # Create business-day-aware 15min index
    days = pd.bdate_range("2023-01-03", periods=n_days, freq="B")
    timestamps = []
    for day in days:
        for bar in range(bars_per_day):
            minutes = 9 * 60 + 30 + bar * 15
            ts = day.replace(hour=minutes // 60, minute=minutes % 60)
            timestamps.append(ts)
    idx = pd.DatetimeIndex(timestamps[:n])

    prices = [150.0]
    for _ in range(n - 1):
        ret = 0.0005 + 0.008 * rng.standard_normal()
        prices.append(prices[-1] * (1 + ret))

    closes = np.array(prices)
    highs = closes * (1 + abs(rng.normal(0, 0.003, n)))
    lows = closes * (1 - abs(rng.normal(0, 0.003, n)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = rng.integers(100_000, 5_000_000, size=n).astype(float)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grid tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParamGrid:
    def test_default_grid_size(self):
        grid = build_param_grid()
        # 5 × 3 × 3 × 3 × 3 = 405
        assert len(grid) == 405

    def test_custom_grid(self):
        grid = build_param_grid({"a": [1, 2], "b": [3, 4, 5]})
        assert len(grid) == 6
        assert all("a" in g and "b" in g for g in grid)

    def test_params_to_aggregator_kwargs(self):
        params = {
            "buy_threshold": 0.35,
            "confirm_bars": 3,
            "trend_weight_mult": 1.5,
            "momentum_weight_mult": 0.5,
            "volume_weight_mult": 1.2,
        }
        kw = params_to_aggregator_kwargs(params)
        assert kw["buy_threshold"] == 0.35
        assert kw["sell_threshold"] == -0.35
        assert kw["confirm_bars"] == 3
        assert kw["disable_logging"] is True
        assert isinstance(kw["weight_overrides"], dict)
        assert len(kw["weight_overrides"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Objective function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectives:
    def test_sharpe_penalizes_few_trades(self):
        m = Metrics(sharpe_ratio=2.0, total_trades=2)
        assert objective_sharpe(m, min_trades=5) == PENALTY

    def test_sharpe_returns_ratio(self):
        m = Metrics(sharpe_ratio=1.5, total_trades=10)
        assert objective_sharpe(m, min_trades=5) == 1.5

    def test_sharpe_trades_scales_by_sqrt(self):
        m = Metrics(sharpe_ratio=1.0, total_trades=16)
        result = objective_sharpe_trades(m, min_trades=5)
        assert result == pytest.approx(4.0)  # 1.0 * sqrt(16) = 4

    def test_sharpe_trades_penalizes_few(self):
        m = Metrics(sharpe_ratio=2.0, total_trades=1)
        assert objective_sharpe_trades(m, min_trades=5) == PENALTY


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward window tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardWindows:
    def test_makes_windows(self):
        data = _make_long_df(months=12)
        opt = WalkForwardOptimizer(
            symbol="TEST", data=data,
            train_months=4, test_months=2,
        )
        windows = opt._make_windows()
        assert len(windows) >= 2
        # Each window: (train_start, train_end, test_start, test_end)
        for tr_s, tr_e, te_s, te_e in windows:
            assert tr_e == te_s  # test starts right after train ends

    def test_insufficient_data_raises(self):
        data = _make_long_df(months=3)
        opt = WalkForwardOptimizer(
            symbol="TEST", data=data,
            train_months=6, test_months=2,
        )
        with pytest.raises(ValueError, match="Insufficient data"):
            opt.run()


# ─────────────────────────────────────────────────────────────────────────────
# Integration: small end-to-end run
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    @pytest.mark.slow
    def test_small_grid_runs(self):
        """End-to-end with a tiny grid (4 combos) and short data."""
        data = _make_long_df(months=12, seed=7)
        small_grid = {
            "buy_threshold": [0.35, 0.45],
            "confirm_bars": [3, 5],
            "trend_weight_mult": [1.0],
            "momentum_weight_mult": [1.0],
            "volume_weight_mult": [1.0],
        }
        opt = WalkForwardOptimizer(
            symbol="TEST", data=data,
            train_months=4, test_months=2,
            grid=small_grid,
            max_workers=2,
            min_trades=1,
        )
        result = opt.run()

        assert result.total_windows >= 2
        assert result.param_grid_size == 4
        assert len(result.windows) == result.total_windows
        assert result.stable_params  # should have found params

        # Each window should have best_params
        for w in result.windows:
            assert "buy_threshold" in w.best_params
            assert "confirm_bars" in w.best_params
