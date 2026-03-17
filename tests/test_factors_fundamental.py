"""Tests for fundamental factors: earnings yield, P/B, revenue growth."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.factors.base import FactorData
import src.factors.fundamental  # noqa: F401
from src.factors.fundamental.valuation import FundamentalEarningsYield, FundamentalPB
from src.factors.fundamental.growth import FundamentalRevenueGrowth


def _make_ohlcv(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [1e6] * n,
    }, index=idx)


def _data(fundamental: dict | None = None) -> FactorData:
    return FactorData(ohlcv=_make_ohlcv(), symbol="TEST", fundamental=fundamental)


# ──────────────────────────────────────────────
# FundamentalEarningsYield
# ──────────────────────────────────────────────

class TestEarningsYield:
    def test_stale_when_no_fundamental(self):
        f = FundamentalEarningsYield()
        result = f.compute(_data(fundamental=None))
        assert result.is_stale is True

    def test_stale_when_pe_missing(self):
        f = FundamentalEarningsYield()
        result = f.compute(_data(fundamental={"price_to_book": 3.0}))
        assert result.is_stale is True

    def test_pe_25_gives_4pct_yield(self):
        f = FundamentalEarningsYield()
        result = f.compute(_data(fundamental={"trailing_pe": 25.0}))
        assert result.raw_value == pytest.approx(4.0)

    def test_negative_pe_gives_stale(self):
        f = FundamentalEarningsYield()
        result = f.compute(_data(fundamental={"trailing_pe": -10.0}))
        assert result.is_stale is True

    def test_zero_pe_gives_stale(self):
        f = FundamentalEarningsYield()
        result = f.compute(_data(fundamental={"trailing_pe": 0.0}))
        assert result.is_stale is True

    def test_lower_pe_higher_yield(self):
        f = FundamentalEarningsYield()
        r_cheap = f.compute(_data(fundamental={"trailing_pe": 10.0}))  # 10% yield
        r_expensive = f.compute(_data(fundamental={"trailing_pe": 50.0}))  # 2% yield
        assert r_cheap.raw_value > r_expensive.raw_value

    def test_category_is_fundamental(self):
        f = FundamentalEarningsYield()
        result = f.compute(_data(fundamental={"trailing_pe": 20.0}))
        assert result.category == "fundamental"

    def test_label_contains_ey(self):
        f = FundamentalEarningsYield()
        result = f.compute(_data(fundamental={"trailing_pe": 20.0}))
        assert "EY" in result.label or "%" in result.label


# ──────────────────────────────────────────────
# FundamentalPB
# ──────────────────────────────────────────────

class TestFundamentalPB:
    def test_stale_when_no_fundamental(self):
        f = FundamentalPB()
        result = f.compute(_data(fundamental=None))
        assert result.is_stale is True

    def test_pb_2_gives_0pt5_inverse(self):
        f = FundamentalPB()
        result = f.compute(_data(fundamental={"price_to_book": 2.0}))
        assert result.raw_value == pytest.approx(0.5)

    def test_negative_pb_gives_stale(self):
        f = FundamentalPB()
        result = f.compute(_data(fundamental={"price_to_book": -1.0}))
        assert result.is_stale is True

    def test_low_pb_higher_inverse(self):
        """Cheap stock (low PB) → higher inverse PB → bullish."""
        f = FundamentalPB()
        r_value = f.compute(_data(fundamental={"price_to_book": 1.0}))  # value
        r_growth = f.compute(_data(fundamental={"price_to_book": 10.0}))  # growth
        assert r_value.raw_value > r_growth.raw_value


# ──────────────────────────────────────────────
# FundamentalRevenueGrowth
# ──────────────────────────────────────────────

class TestFundamentalRevenueGrowth:
    def test_stale_when_no_fundamental(self):
        f = FundamentalRevenueGrowth()
        result = f.compute(_data(fundamental=None))
        assert result.is_stale is True

    def test_growth_8pct_decimal_converts(self):
        f = FundamentalRevenueGrowth()
        result = f.compute(_data(fundamental={"revenue_growth": 0.08}))
        assert result.raw_value == pytest.approx(8.0)

    def test_negative_growth_bearish(self):
        f = FundamentalRevenueGrowth()
        result = f.compute(_data(fundamental={"revenue_growth": -0.05}))
        assert result.raw_value == pytest.approx(-5.0)

    def test_frequency_is_quarterly(self):
        f = FundamentalRevenueGrowth()
        assert f.frequency == "quarterly"

    def test_label_contains_growth(self):
        f = FundamentalRevenueGrowth()
        result = f.compute(_data(fundamental={"revenue_growth": 0.10}))
        assert "%" in result.label or "Grow" in result.label
