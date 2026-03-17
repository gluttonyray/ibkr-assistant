"""Trend technical factors: EMA slope, MACD histogram, ADX signed."""
from __future__ import annotations

import math

import pandas_ta as ta

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class TrendEMASlope(BaseFactor):
    """Composite EMA slope: weighted average of 9/21/55 EMA % changes.

    Returns the weighted-average 5-bar % slope of three EMAs.
    Shorter EMAs get higher weight (3:2:1).
    Positive = uptrend, negative = downtrend.
    """

    name = "trend_ema_slope"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 60
    weight = 1.2
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        close = data.ohlcv["close"]
        if len(close) < 60:
            return float("nan")

        ema9 = ta.ema(close, length=9)
        ema21 = ta.ema(close, length=21)
        ema55 = ta.ema(close, length=55)

        if ema9 is None or ema21 is None or ema55 is None:
            return float("nan")

        e9 = ema9.dropna()
        e21 = ema21.dropna()
        e55 = ema55.dropna()

        if len(e9) < 6 or len(e21) < 6 or len(e55) < 6:
            return float("nan")

        def pct_slope(series) -> float:
            prev = float(series.iloc[-6])
            if abs(prev) < 1e-10:
                return float("nan")
            return (float(series.iloc[-1]) - prev) / prev * 100.0

        s9 = pct_slope(e9)
        s21 = pct_slope(e21)
        s55 = pct_slope(e55)

        if any(math.isnan(v) for v in [s9, s21, s55]):
            return float("nan")

        # Weighted average: shorter EMAs get more weight
        return (s9 * 3.0 + s21 * 2.0 + s55 * 1.0) / 6.0

    def _label(self, raw: float) -> str:
        return f"EMA_slope={raw:.3f}%"


@FactorRegistry.register
class TrendMACDHist(BaseFactor):
    """MACD histogram: difference between MACD line and signal line.

    Positive/growing = bullish momentum building.
    Negative/falling = bearish momentum building.
    Unbounded — z-score normalizes to recent volatility.
    """

    name = "trend_macd_hist"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 35
    weight = 1.2
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        close = data.ohlcv["close"]
        result = ta.macd(close, fast=12, slow=26, signal=9)
        if result is None:
            return float("nan")

        # Find histogram column (MACDh_12_26_9)
        hist_cols = [c for c in result.columns if "MACDh" in c]
        if not hist_cols:
            return float("nan")

        val = result[hist_cols[0]].iloc[-1]
        return float("nan") if math.isnan(val) else float(val)

    def _label(self, raw: float) -> str:
        return f"MACD_hist={raw:.4f}"


@FactorRegistry.register
class TrendADXSigned(BaseFactor):
    """ADX-gated directional signal: (+DI - -DI) when ADX > 20, else 0.

    Combines trend strength (ADX) with direction (+DI vs -DI):
    - ADX < 20: no trend → returns 0
    - ADX >= 20: returns +DI - -DI (positive = bullish, negative = bearish)
    """

    name = "trend_adx_signed"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 30
    weight = 1.0
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        high = data.ohlcv["high"]
        low = data.ohlcv["low"]
        close = data.ohlcv["close"]

        result = ta.adx(high, low, close, length=14)
        if result is None:
            return float("nan")

        adx_cols = [c for c in result.columns if c.startswith("ADX_")]
        dmp_cols = [c for c in result.columns if c.startswith("DMP_")]
        dmn_cols = [c for c in result.columns if c.startswith("DMN_")]

        if not adx_cols or not dmp_cols or not dmn_cols:
            return float("nan")

        adx = float(result[adx_cols[0]].iloc[-1])
        dmp = float(result[dmp_cols[0]].iloc[-1])
        dmn = float(result[dmn_cols[0]].iloc[-1])

        if any(math.isnan(v) for v in [adx, dmp, dmn]):
            return float("nan")

        if adx < 20.0:
            return 0.0

        return dmp - dmn

    def _label(self, raw: float) -> str:
        return f"ADX_dir={raw:.2f}"
