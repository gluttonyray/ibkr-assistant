"""Tests for macro factors (yield curve, fed rate, DXY)."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.factors.base import FactorData
import src.factors.macro  # noqa: F401
from src.factors.macro.rates import MacroYieldCurve, MacroFedRate
from src.factors.macro.dollar import MacroDXY


def _make_ohlcv(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min")
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [1e6] * n,
    }, index=idx)


def _data(macro: dict | None = None) -> FactorData:
    return FactorData(ohlcv=_make_ohlcv(), symbol="TEST", macro=macro)


# ──────────────────────────────────────────────
# MacroYieldCurve
# ──────────────────────────────────────────────

class TestMacroYieldCurve:
    def test_stale_when_no_macro(self):
        f = MacroYieldCurve()
        result = f.compute(_data(macro=None))
        assert result.is_stale is True

    def test_stale_when_spread_missing(self):
        f = MacroYieldCurve()
        result = f.compute(_data(macro={"fed_rate": 5.0}))
        assert result.is_stale is True

    def test_positive_spread_gives_positive_raw(self):
        f = MacroYieldCurve()
        result = f.compute(_data(macro={"spread": 50.0}))
        assert result.raw_value == pytest.approx(50.0)

    def test_negative_spread_inverted(self):
        f = MacroYieldCurve()
        result = f.compute(_data(macro={"spread": -30.0}))
        assert result.raw_value == pytest.approx(-30.0)

    def test_category_is_macro(self):
        f = MacroYieldCurve()
        result = f.compute(_data(macro={"spread": 20.0}))
        assert result.category == "macro"

    def test_frequency_is_daily(self):
        f = MacroYieldCurve()
        assert f.frequency == "daily"


# ──────────────────────────────────────────────
# MacroFedRate
# ──────────────────────────────────────────────

class TestMacroFedRate:
    def test_stale_when_no_macro(self):
        f = MacroFedRate()
        result = f.compute(_data(macro=None))
        assert result.is_stale is True

    def test_high_rate_gives_negative_raw(self):
        """High fed rate = bearish = negative raw (inverted)."""
        f = MacroFedRate()
        result = f.compute(_data(macro={"fed_rate": 5.5}))
        assert result.raw_value == pytest.approx(-5.5)

    def test_low_rate_gives_smaller_negative(self):
        f = MacroFedRate()
        r_high = f.compute(_data(macro={"fed_rate": 5.5}))
        r_low = f.compute(_data(macro={"fed_rate": 1.0}))
        assert r_high.raw_value < r_low.raw_value


# ──────────────────────────────────────────────
# MacroDXY
# ──────────────────────────────────────────────

class TestMacroDXY:
    def test_stale_when_no_macro(self):
        f = MacroDXY()
        result = f.compute(_data(macro=None))
        assert result.is_stale is True

    def test_high_dxy_gives_negative_raw(self):
        """Strong dollar = bearish for equities = negative raw."""
        f = MacroDXY()
        result = f.compute(_data(macro={"dxy": 105.0}))
        assert result.raw_value == pytest.approx(-105.0)

    def test_low_dxy_less_negative(self):
        f = MacroDXY()
        r_high = f.compute(_data(macro={"dxy": 105.0}))
        r_low = f.compute(_data(macro={"dxy": 90.0}))
        assert r_high.raw_value < r_low.raw_value

    def test_category_is_macro(self):
        f = MacroDXY()
        result = f.compute(_data(macro={"dxy": 100.0}))
        assert result.category == "macro"
