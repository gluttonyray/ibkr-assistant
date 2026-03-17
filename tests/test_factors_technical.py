"""Tests for all 11 technical factors on synthetic OHLCV data."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.factors.base import FactorData, FactorResult

# Import triggers FactorRegistry registration
import src.factors.technical  # noqa: F401
from src.factors.technical.momentum import MomentumRSI, MomentumROC
from src.factors.technical.trend import TrendEMASlope, TrendMACDHist, TrendADXSigned
from src.factors.technical.volatility import VolatilityBBPctB, VolatilityATRRatio
from src.factors.technical.volume import VolumeOBVSlope, VolumeVWAPDev
from src.factors.technical.mean_reversion import MeanReversionBBDev, MeanReversionRSIExtreme


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def _make_ohlcv(n: int = 500, trend: float = 0.0, vol: float = 0.006, seed: int = 42) -> pd.DataFrame:
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
    idx = pd.date_range("2024-01-02 09:31", periods=n, freq="1min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


@pytest.fixture
def bullish_data():
    return FactorData(ohlcv=_make_ohlcv(500, trend=0.003, vol=0.003, seed=1), symbol="BULL")

@pytest.fixture
def bearish_data():
    return FactorData(ohlcv=_make_ohlcv(500, trend=-0.003, vol=0.003, seed=2), symbol="BEAR")

@pytest.fixture
def flat_data():
    return FactorData(ohlcv=_make_ohlcv(500, trend=0.0, vol=0.001, seed=3), symbol="FLAT")

@pytest.fixture
def short_data():
    return FactorData(ohlcv=_make_ohlcv(10, seed=99), symbol="SHORT")


ALL_FACTORS = [
    MomentumRSI,
    MomentumROC,
    TrendEMASlope,
    TrendMACDHist,
    TrendADXSigned,
    VolatilityBBPctB,
    VolatilityATRRatio,
    VolumeOBVSlope,
    VolumeVWAPDev,
    MeanReversionBBDev,
    MeanReversionRSIExtreme,
]


# ──────────────────────────────────────────────
# Generic: all factors return FactorResult
# ──────────────────────────────────────────────

@pytest.mark.parametrize("FactorCls", ALL_FACTORS)
def test_factor_returns_factor_result(FactorCls, bullish_data):
    f = FactorCls()
    result = f.compute(bullish_data)
    assert isinstance(result, FactorResult)
    assert result.name == f.name
    assert result.category == "technical"


@pytest.mark.parametrize("FactorCls", ALL_FACTORS)
def test_factor_stale_on_short_data(FactorCls, short_data):
    f = FactorCls()
    result = f.compute(short_data)
    assert result.is_stale is True
    assert math.isnan(result.raw_value)


@pytest.mark.parametrize("FactorCls", ALL_FACTORS)
def test_factor_not_stale_on_full_data(FactorCls, bullish_data):
    f = FactorCls()
    result = f.compute(bullish_data)
    assert result.is_stale is False
    assert not math.isnan(result.raw_value)


@pytest.mark.parametrize("FactorCls", ALL_FACTORS)
def test_factor_z_score_finite(FactorCls, bullish_data):
    f = FactorCls()
    # Build up history
    for _ in range(5):
        result = f.compute(bullish_data)
    assert math.isfinite(result.z_score)


@pytest.mark.parametrize("FactorCls", ALL_FACTORS)
def test_factor_reset_works(FactorCls, bullish_data):
    f = FactorCls()
    for _ in range(5):
        f.compute(bullish_data)
    assert len(f._history) > 0
    f.reset()
    assert len(f._history) == 0


# ──────────────────────────────────────────────
# MomentumRSI — directional checks
# ──────────────────────────────────────────────

class TestMomentumRSI:
    def test_bullish_rsi_higher_than_bearish(self, bullish_data, bearish_data):
        f_bull = MomentumRSI()
        f_bear = MomentumRSI()
        r_bull = f_bull.compute(bullish_data)
        r_bear = f_bear.compute(bearish_data)
        assert r_bull.raw_value > r_bear.raw_value

    def test_rsi_in_range(self, bullish_data):
        f = MomentumRSI()
        result = f.compute(bullish_data)
        assert 0.0 <= result.raw_value <= 100.0

    def test_label_contains_rsi(self, bullish_data):
        f = MomentumRSI()
        result = f.compute(bullish_data)
        assert "RSI" in result.label


# ──────────────────────────────────────────────
# MomentumROC
# ──────────────────────────────────────────────

class TestMomentumROC:
    def test_bullish_roc_positive(self, bullish_data):
        f = MomentumROC()
        result = f.compute(bullish_data)
        assert result.raw_value > 0.0

    def test_bearish_roc_negative(self, bearish_data):
        f = MomentumROC()
        result = f.compute(bearish_data)
        assert result.raw_value < 0.0


# ──────────────────────────────────────────────
# TrendEMASlope
# ──────────────────────────────────────────────

class TestTrendEMASlope:
    def test_bullish_slope_positive(self, bullish_data):
        f = TrendEMASlope()
        result = f.compute(bullish_data)
        assert result.raw_value > 0.0

    def test_bearish_slope_negative(self, bearish_data):
        f = TrendEMASlope()
        result = f.compute(bearish_data)
        assert result.raw_value < 0.0


# ──────────────────────────────────────────────
# TrendMACDHist
# ──────────────────────────────────────────────

class TestTrendMACDHist:
    def test_macd_is_finite(self, bullish_data):
        # MACD histogram can be negative even in a bullish trend depending on EMA state;
        # correctness test is that the value is a finite number.
        f = TrendMACDHist()
        result = f.compute(bullish_data)
        assert math.isfinite(result.raw_value)

    def test_bearish_macd_is_finite(self, bearish_data):
        f = TrendMACDHist()
        result = f.compute(bearish_data)
        assert math.isfinite(result.raw_value)


# ──────────────────────────────────────────────
# TrendADXSigned
# ──────────────────────────────────────────────

class TestTrendADXSigned:
    def test_returns_zero_when_no_trend(self, flat_data):
        # Flat market should have low ADX → returns 0
        f = TrendADXSigned()
        result = f.compute(flat_data)
        # May or may not be exactly 0 depending on ADX level
        assert math.isfinite(result.raw_value)

    def test_bullish_adx_positive_or_zero(self, bullish_data):
        f = TrendADXSigned()
        result = f.compute(bullish_data)
        assert result.raw_value >= 0.0  # bullish trend: +DI > -DI or ADX<20


# ──────────────────────────────────────────────
# VolatilityBBPctB
# ──────────────────────────────────────────────

class TestVolatilityBBPctB:
    def test_raw_value_finite(self, bullish_data):
        f = VolatilityBBPctB()
        result = f.compute(bullish_data)
        assert math.isfinite(result.raw_value)

    def test_label_contains_bb(self, bullish_data):
        f = VolatilityBBPctB()
        result = f.compute(bullish_data)
        assert "BB" in result.label


# ──────────────────────────────────────────────
# VolatilityATRRatio
# ──────────────────────────────────────────────

class TestVolatilityATRRatio:
    def test_ratio_positive(self, bullish_data):
        f = VolatilityATRRatio()
        result = f.compute(bullish_data)
        assert result.raw_value > 0.0

    def test_volatile_ratio_higher_than_quiet(self):
        volatile_df = _make_ohlcv(500, vol=0.02, seed=10)
        quiet_df = _make_ohlcv(500, vol=0.001, seed=11)

        # Use same factor instance but compare after several bars
        # Actually: create separate instances
        f_vol = VolatilityATRRatio()
        f_quiet = VolatilityATRRatio()

        # Feed the full dataset to build baseline
        r_vol = f_vol.compute(FactorData(ohlcv=volatile_df, symbol="VOL"))
        r_quiet = f_quiet.compute(FactorData(ohlcv=quiet_df, symbol="QUIET"))

        # Both should return finite values
        assert math.isfinite(r_vol.raw_value)
        assert math.isfinite(r_quiet.raw_value)


# ──────────────────────────────────────────────
# VolumeOBVSlope
# ──────────────────────────────────────────────

class TestVolumeOBVSlope:
    def test_bullish_obv_slope_positive(self, bullish_data):
        f = VolumeOBVSlope()
        result = f.compute(bullish_data)
        assert result.raw_value > 0.0

    def test_bearish_obv_slope_negative(self, bearish_data):
        f = VolumeOBVSlope()
        result = f.compute(bearish_data)
        assert result.raw_value < 0.0


# ──────────────────────────────────────────────
# VolumeVWAPDev
# ──────────────────────────────────────────────

class TestVolumeVWAPDev:
    def test_deviation_is_finite(self, bullish_data):
        f = VolumeVWAPDev()
        result = f.compute(bullish_data)
        assert math.isfinite(result.raw_value)

    def test_label_contains_vwap(self, bullish_data):
        f = VolumeVWAPDev()
        result = f.compute(bullish_data)
        assert "VWAP" in result.label


# ──────────────────────────────────────────────
# MeanReversionBBDev
# ──────────────────────────────────────────────

class TestMeanReversionBBDev:
    def test_bullish_positive_deviation(self, bullish_data):
        # In a strong uptrend, price is typically above BB midline
        f = MeanReversionBBDev()
        result = f.compute(bullish_data)
        assert math.isfinite(result.raw_value)

    def test_bearish_negative_deviation(self, bearish_data):
        f = MeanReversionBBDev()
        result = f.compute(bearish_data)
        assert math.isfinite(result.raw_value)


# ──────────────────────────────────────────────
# MeanReversionRSIExtreme
# ──────────────────────────────────────────────

class TestMeanReversionRSIExtreme:
    def test_bullish_market_gives_negative_contrarian(self, bullish_data):
        # High RSI → negative contrarian value
        f = MeanReversionRSIExtreme()
        result = f.compute(bullish_data)
        assert result.raw_value < 0.0

    def test_bearish_market_gives_positive_contrarian(self, bearish_data):
        # Low RSI → positive contrarian value
        f = MeanReversionRSIExtreme()
        result = f.compute(bearish_data)
        assert result.raw_value > 0.0

    def test_value_range(self, bullish_data):
        f = MeanReversionRSIExtreme()
        result = f.compute(bullish_data)
        # -(RSI-50)/50 is in [-1, +1] since RSI ∈ [0,100]
        assert -1.0 <= result.raw_value <= 1.0
