"""Volume technical factors: OBV slope, VWAP deviation."""
from __future__ import annotations

import math

import pandas_ta as ta

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class VolumeOBVSlope(BaseFactor):
    """OBV slope: EMA(10)-smoothed OBV change normalized by level.

    Positive = volume-confirmed accumulation (rising OBV).
    Negative = distribution (falling OBV).
    Normalized by |OBV| level to make comparable across symbols.
    """

    name = "volume_obv_slope"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 25
    weight = 1.0
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        close = data.ohlcv["close"]
        volume = data.ohlcv["volume"]

        obv = ta.obv(close, volume)
        if obv is None:
            return float("nan")

        obv_clean = obv.dropna()
        if len(obv_clean) < 12:
            return float("nan")

        # Smooth with EMA(10)
        obv_ema = obv_clean.ewm(span=10, min_periods=5).mean()
        if len(obv_ema) < 12:
            return float("nan")

        current = float(obv_ema.iloc[-1])
        prev = float(obv_ema.iloc[-11])

        if math.isnan(current) or math.isnan(prev):
            return float("nan")

        denominator = abs(current) + 1.0
        return (current - prev) / denominator * 100.0

    def _label(self, raw: float) -> str:
        return f"OBV_slope={raw:.3f}"


@FactorRegistry.register
class VolumeVWAPDev(BaseFactor):
    """VWAP deviation: (price - VWAP) / VWAP * 100 in %.

    Positive = price above VWAP (intraday strength).
    Negative = price below VWAP (intraday weakness).
    Uses 20-bar rolling VWAP for intraday consistency.
    """

    name = "volume_vwap_dev"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 25
    weight = 1.0
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        high = data.ohlcv["high"]
        low = data.ohlcv["low"]
        close = data.ohlcv["close"]
        volume = data.ohlcv["volume"]

        # 20-bar rolling VWAP
        typical_price = (high + low + close) / 3.0
        window = 20

        tp_vol = (typical_price * volume).rolling(window, min_periods=1).sum()
        vol_sum = volume.rolling(window, min_periods=1).sum()
        vol_sum = vol_sum.where(vol_sum > 0)  # avoid div-by-zero
        vwap = tp_vol / vol_sum

        current_price = float(close.iloc[-1])
        current_vwap = float(vwap.iloc[-1])

        if math.isnan(current_vwap) or current_vwap < 1e-10:
            return float("nan")

        return (current_price - current_vwap) / current_vwap * 100.0

    def _label(self, raw: float) -> str:
        return f"VWAP_dev={raw:.2f}%"
