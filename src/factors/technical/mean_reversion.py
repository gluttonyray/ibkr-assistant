"""Mean-reversion technical factors: BB deviation, RSI extreme distance."""
from __future__ import annotations

import math

import pandas_ta as ta

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class MeanReversionBBDev(BaseFactor):
    """Bollinger Band deviation: (price - BB_mid) / BB_std.

    Returns the z-score of price relative to the Bollinger middle band:
    Positive = above mean (overextended upward in ranging markets).
    Negative = below mean (overextended downward).
    Note: this is the raw BB deviation, which BaseFactor will *also* z-score.
    """

    name = "mean_reversion_bb_dev"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 25
    weight = 1.0
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        close = data.ohlcv["close"]
        result = ta.bbands(close, length=20, std=2)
        if result is None:
            return float("nan")

        mid_cols = [c for c in result.columns if "BBM" in c]
        upper_cols = [c for c in result.columns if "BBU" in c]
        lower_cols = [c for c in result.columns if "BBL" in c]

        if not mid_cols or not upper_cols or not lower_cols:
            return float("nan")

        mid = float(result[mid_cols[0]].iloc[-1])
        upper = float(result[upper_cols[0]].iloc[-1])
        lower = float(result[lower_cols[0]].iloc[-1])
        current = float(close.iloc[-1])

        if any(math.isnan(v) for v in [mid, upper, lower]):
            return float("nan")

        # BB std = (upper - lower) / 4 (2-sigma bands)
        bb_std = (upper - lower) / 4.0
        if bb_std < 1e-10:
            return float("nan")

        return (current - mid) / bb_std

    def _label(self, raw: float) -> str:
        return f"BB_dev={raw:.3f}σ"


@FactorRegistry.register
class MeanReversionRSIExtreme(BaseFactor):
    """RSI extreme distance: -(RSI - 50) / 50 (contrarian).

    Flips RSI into a contrarian signal:
    - RSI=70 → -0.40 (overbought → bearish contrarian)
    - RSI=30 → +0.40 (oversold → bullish contrarian)
    - RSI=50 → 0.0  (neutral)
    """

    name = "mean_reversion_rsi_extreme"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 20
    weight = 0.8
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        close = data.ohlcv["close"]
        result = ta.rsi(close, length=14)
        if result is None:
            return float("nan")

        val = result.iloc[-1]
        if math.isnan(val):
            return float("nan")

        rsi = float(val)
        return -(rsi - 50.0) / 50.0

    def _label(self, raw: float) -> str:
        return f"RSI_cntr={raw:.3f}"
