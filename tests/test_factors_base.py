"""Tests for BaseFactor, FactorResult, and FactorData."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.factors.base import BaseFactor, FactorData, FactorResult


# ──────────────────────────────────────────────
# Concrete test factor (simple: returns close[-1])
# ──────────────────────────────────────────────

class LastCloseFactorForTest(BaseFactor):
    name = "last_close_test"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 5
    weight = 1.0
    z_window = 10

    def _compute(self, data: FactorData) -> float:
        return float(data.ohlcv["close"].iloc[-1])


class ErrorFactorForTest(BaseFactor):
    name = "error_factor_test"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 5
    weight = 1.0
    z_window = 10

    def _compute(self, data: FactorData) -> float:
        raise RuntimeError("deliberate error")


class NanFactorForTest(BaseFactor):
    name = "nan_factor_test"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 5
    weight = 1.0
    z_window = 10

    def _compute(self, data: FactorData) -> float:
        return float("nan")


def _make_data(n: int = 20, seed: int = 0) -> FactorData:
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=idx,
    )
    return FactorData(ohlcv=df, symbol="TEST")


# ──────────────────────────────────────────────
# FactorResult dataclass
# ──────────────────────────────────────────────

class TestFactorResult:
    def test_construction(self):
        ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
        r = FactorResult(
            name="test",
            raw_value=42.0,
            z_score=1.5,
            category="technical",
            frequency="intraday",
            timestamp=ts,
            is_stale=False,
            label="test=42.0000",
            weight=1.0,
        )
        assert r.name == "test"
        assert r.raw_value == 42.0
        assert r.z_score == 1.5
        assert r.is_stale is False

    def test_stale_flag(self):
        ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
        r = FactorResult(
            name="test", raw_value=float("nan"), z_score=0.0,
            category="technical", frequency="intraday", timestamp=ts,
            is_stale=True, label="stale",
        )
        assert r.is_stale is True
        assert math.isnan(r.raw_value)


# ──────────────────────────────────────────────
# BaseFactor: insufficient data
# ──────────────────────────────────────────────

class TestBaseFactorInsufficientData:
    def test_returns_stale_when_too_few_bars(self):
        f = LastCloseFactorForTest()
        tiny_df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1e6]},
            index=pd.date_range("2024-01-02 09:30", periods=1, freq="1min"),
        )
        data = FactorData(ohlcv=tiny_df, symbol="TEST")
        result = f.compute(data)
        assert result.is_stale is True
        assert math.isnan(result.raw_value)


# ──────────────────────────────────────────────
# BaseFactor: error handling
# ──────────────────────────────────────────────

class TestBaseFactorErrorHandling:
    def test_exception_in_compute_returns_stale(self):
        f = ErrorFactorForTest()
        data = _make_data(20)
        result = f.compute(data)
        assert result.is_stale is True
        assert math.isnan(result.raw_value)

    def test_nan_return_gives_stale(self):
        f = NanFactorForTest()
        data = _make_data(20)
        result = f.compute(data)
        assert result.is_stale is True


# ──────────────────────────────────────────────
# BaseFactor: z-score math
# ──────────────────────────────────────────────

class TestBaseFactorZScore:
    def test_z_score_converges(self):
        """After enough samples, z-score should be non-zero for extreme values."""
        f = LastCloseFactorForTest()
        # Feed a slowly rising close series
        for i in range(15):
            data = _make_data(20, seed=i)
            result = f.compute(data)

        # After building history, z-score should be computed
        assert result.z_score == pytest.approx(result.z_score, abs=1e-3)

    def test_z_score_zero_with_constant(self):
        """Constant values → std=0 → z-score = 0."""
        class ConstFactor(BaseFactor):
            name = "const_test"
            lookback_bars = 5
            z_window = 10
            def _compute(self, data: FactorData) -> float:
                return 42.0

        f = ConstFactor()
        data = _make_data(20)
        for _ in range(15):
            result = f.compute(data)
        assert result.z_score == 0.0

    def test_z_score_zero_when_insufficient_history(self):
        f = LastCloseFactorForTest()
        # Only compute once — not enough history for z-score
        data = _make_data(20)
        result = f.compute(data)
        assert result.z_score == 0.0  # only 1 sample

    def test_z_score_positive_for_high_value(self):
        """A value much higher than recent mean should have positive z-score."""
        class ControlledFactor(BaseFactor):
            name = "controlled_test"
            lookback_bars = 5
            z_window = 20
            _values: list = []

            def _compute(self, data: FactorData) -> float:
                idx = len(self._history)
                if idx < 19:
                    return 50.0
                return 100.0  # high outlier at the end

        f = ControlledFactor()
        data = _make_data(20)
        for _ in range(25):
            result = f.compute(data)
        assert result.z_score > 0

    def test_reset_clears_history(self):
        f = LastCloseFactorForTest()
        data = _make_data(20)
        for _ in range(10):
            f.compute(data)
        assert len(f._history) > 0
        f.reset()
        assert len(f._history) == 0


# ──────────────────────────────────────────────
# BaseFactor: metadata fields
# ──────────────────────────────────────────────

class TestBaseFactorMetadata:
    def test_category_preserved(self):
        f = LastCloseFactorForTest()
        data = _make_data(20)
        result = f.compute(data)
        assert result.category == "technical"

    def test_frequency_preserved(self):
        f = LastCloseFactorForTest()
        data = _make_data(20)
        result = f.compute(data)
        assert result.frequency == "intraday"

    def test_weight_preserved(self):
        f = LastCloseFactorForTest()
        data = _make_data(20)
        result = f.compute(data)
        assert result.weight == 1.0

    def test_name_preserved(self):
        f = LastCloseFactorForTest()
        data = _make_data(20)
        result = f.compute(data)
        assert result.name == "last_close_test"
