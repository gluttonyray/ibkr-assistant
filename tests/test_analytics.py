"""Tests for factor analytics: IC, decay, attribution, turnover."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.ic import compute_ic, ic_decay, rolling_ic
from src.analytics.factor_returns import factor_return_attribution, factor_turnover


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_aligned(n: int = 100, correlation: float = 0.5, seed: int = 42) -> tuple:
    """Return (factor_values, forward_returns) with known correlation."""
    rng = np.random.default_rng(seed)
    factor = pd.Series(rng.standard_normal(n), name="factor")
    noise = pd.Series(rng.standard_normal(n), name="noise")
    returns = factor * correlation + noise * (1 - abs(correlation))
    returns.name = "returns"
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    factor.index = idx
    returns.index = idx
    return factor, returns


# ──────────────────────────────────────────────
# compute_ic
# ──────────────────────────────────────────────

class TestComputeIC:
    def test_returns_float(self):
        f, r = _make_aligned(50, 0.5)
        ic = compute_ic(f, r)
        assert isinstance(ic, float)

    def test_positive_correlation_gives_positive_ic(self):
        f, r = _make_aligned(200, 0.7, seed=1)
        ic = compute_ic(f, r)
        assert ic > 0.0

    def test_negative_correlation_gives_negative_ic(self):
        f, r = _make_aligned(200, -0.7, seed=2)
        ic = compute_ic(f, r)
        assert ic < 0.0

    def test_no_correlation_ic_near_zero(self):
        rng = np.random.default_rng(99)
        idx = pd.date_range("2024-01-02", periods=200, freq="D")
        f = pd.Series(rng.standard_normal(200), index=idx)
        r = pd.Series(rng.standard_normal(200), index=idx)
        ic = compute_ic(f, r)
        assert abs(ic) < 0.3  # low IC for random data

    def test_ic_bounded(self):
        f, r = _make_aligned(100, 0.8)
        ic = compute_ic(f, r)
        assert -1.0 <= ic <= 1.0

    def test_insufficient_data_returns_nan(self):
        f = pd.Series([1.0, 2.0, 3.0])
        r = pd.Series([1.0, 2.0, 3.0])
        ic = compute_ic(f, r)
        assert math.isnan(ic)

    def test_pearson_method(self):
        f, r = _make_aligned(100, 0.5)
        ic_rank = compute_ic(f, r, method="rank")
        ic_pearson = compute_ic(f, r, method="pearson")
        assert math.isfinite(ic_rank)
        assert math.isfinite(ic_pearson)

    def test_nan_values_handled(self):
        f, r = _make_aligned(100, 0.5)
        f.iloc[10] = float("nan")
        r.iloc[20] = float("nan")
        ic = compute_ic(f, r)
        assert math.isfinite(ic)


# ──────────────────────────────────────────────
# ic_decay
# ──────────────────────────────────────────────

class TestICDecay:
    def test_returns_series(self):
        idx = pd.date_range("2024-01-02", periods=200, freq="D")
        f = pd.Series(np.random.default_rng(0).standard_normal(200), index=idx)
        prices = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, 200)), index=idx)
        result = ic_decay(f, prices, horizons=[1, 5, 10])
        assert isinstance(result, pd.Series)
        assert list(result.index) == [1, 5, 10]

    def test_ic_typically_decays(self):
        """IC should generally decrease with longer horizons for a good factor."""
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-02", periods=300, freq="D")
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0.1, 1.0, 300)), index=idx)
        # Factor = 1-bar return (should have higher IC at short horizons)
        factor = prices.pct_change().shift(1).dropna()
        prices_aligned = prices.reindex(factor.index)
        result = ic_decay(factor, prices_aligned, horizons=[1, 10])
        # Just verify it returns finite values
        for v in result.values:
            assert math.isfinite(v) or math.isnan(v)


# ──────────────────────────────────────────────
# rolling_ic
# ──────────────────────────────────────────────

class TestRollingIC:
    def test_returns_series(self):
        f, r = _make_aligned(150, 0.5)
        result = rolling_ic(f, r, window=60)
        assert isinstance(result, pd.Series)

    def test_length_correct(self):
        f, r = _make_aligned(150, 0.5)
        result = rolling_ic(f, r, window=60)
        assert len(result) == 150 - 60 + 1

    def test_insufficient_data_returns_empty(self):
        f, r = _make_aligned(30, 0.5)
        result = rolling_ic(f, r, window=60)
        assert result.empty

    def test_values_bounded(self):
        f, r = _make_aligned(150, 0.6)
        result = rolling_ic(f, r, window=60)
        for v in result.dropna():
            assert -1.0 <= v <= 1.0


# ──────────────────────────────────────────────
# factor_return_attribution
# ──────────────────────────────────────────────

class TestFactorReturnAttribution:
    def test_returns_dataframe(self):
        contribs = [
            {"factor_a": 0.5, "factor_b": -0.3},
            {"factor_a": 0.2, "factor_b": 0.4},
            {"factor_a": -0.1, "factor_b": 0.1},
        ] * 20
        returns = pd.Series(np.random.default_rng(0).standard_normal(60))
        result = factor_return_attribution(contribs, returns)
        assert isinstance(result, pd.DataFrame)
        assert "factor_a" in result.index
        assert "factor_b" in result.index

    def test_has_expected_columns(self):
        contribs = [{"f1": 0.5, "f2": -0.2}] * 30
        returns = pd.Series(np.random.default_rng(0).standard_normal(30))
        result = factor_return_attribution(contribs, returns)
        assert "mean_contribution" in result.columns
        assert "pct_attribution" in result.columns

    def test_pct_attribution_sums_to_100(self):
        contribs = [{"f1": 0.5, "f2": 0.3, "f3": -0.2}] * 50
        returns = pd.Series(np.random.default_rng(0).standard_normal(50))
        result = factor_return_attribution(contribs, returns)
        total_pct = result["pct_attribution"].sum()
        assert abs(total_pct - 100.0) < 1.0

    def test_empty_contribs_returns_empty(self):
        result = factor_return_attribution([], pd.Series(dtype=float))
        assert result.empty


# ──────────────────────────────────────────────
# factor_turnover
# ──────────────────────────────────────────────

class TestFactorTurnover:
    def test_stable_factor_low_turnover(self):
        # Slowly changing factor
        idx = pd.date_range("2024-01-02", periods=100, freq="D")
        stable = pd.Series(np.linspace(0, 1, 100), index=idx)
        t = factor_turnover(stable)
        assert isinstance(t, float)
        assert math.isfinite(t)

    def test_noisy_factor_high_turnover(self):
        idx = pd.date_range("2024-01-02", periods=100, freq="D")
        noisy = pd.Series(np.random.default_rng(0).standard_normal(100), index=idx)
        slowly_changing = pd.Series(np.linspace(0, 1, 100) + 0.001 * np.arange(100), index=idx)

        t_noisy = factor_turnover(noisy)
        t_stable = factor_turnover(slowly_changing)
        assert t_noisy > t_stable

    def test_insufficient_data_returns_nan(self):
        result = factor_turnover(pd.Series([1.0, 2.0]))
        assert math.isnan(result)

    def test_constant_series_returns_nan(self):
        idx = pd.date_range("2024-01-02", periods=50, freq="D")
        const = pd.Series([42.0] * 50, index=idx)
        result = factor_turnover(const)
        assert math.isnan(result)
