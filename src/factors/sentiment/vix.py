"""VIX-based sentiment factors.

Reads from FactorData.sentiment dict, which is populated by the engine
via DataCache + yahoo.py. Returns NaN (stale) when sentiment data is absent.
"""
from __future__ import annotations

import math

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class SentimentVIX(BaseFactor):
    """VIX level (inverted): low VIX = bullish environment.

    Raw value = -VIX so that:
      - Low VIX (calm market) → negative raw, positive z-score → bullish
      - High VIX (fear)       → negative raw, negative z-score → bearish

    Reads: FactorData.sentiment["vix"] (daily float).
    """

    name = "sentiment_vix"
    category = "sentiment"
    frequency = "daily"
    lookback_bars = 1
    weight = 1.0
    z_window = 60
    stale_seconds = 86400  # daily data — stale after 24h

    def _compute(self, data: FactorData) -> float:
        if data.sentiment is None:
            return float("nan")
        vix = data.sentiment.get("vix")
        if vix is None or math.isnan(float(vix)):
            return float("nan")
        return -float(vix)  # inverted: high VIX = bearish

    def _label(self, raw: float) -> str:
        return f"VIX={-raw:.1f}"


@FactorRegistry.register
class SentimentVIXTerm(BaseFactor):
    """VIX term structure: VIX / VIX3M ratio.

    > 1 = backwardation (VIX > VIX3M = short-term fear spike) → bearish.
    < 1 = contango (normal, low fear) → bullish.

    Raw value = -(VIX/VIX3M - 1) * 100 so positive = bullish.

    Reads: FactorData.sentiment["vix"] and ["vix3m"].
    """

    name = "sentiment_vix_term"
    category = "sentiment"
    frequency = "daily"
    lookback_bars = 1
    weight = 0.8
    z_window = 60
    stale_seconds = 86400

    def _compute(self, data: FactorData) -> float:
        if data.sentiment is None:
            return float("nan")
        vix = data.sentiment.get("vix")
        vix3m = data.sentiment.get("vix3m")
        if vix is None or vix3m is None:
            return float("nan")
        vix = float(vix)
        vix3m = float(vix3m)
        if math.isnan(vix) or math.isnan(vix3m) or vix3m < 1e-3:
            return float("nan")
        # Negative of ratio deviation: contango (+) = bullish, backwardation (-) = bearish
        return -(vix / vix3m - 1.0) * 100.0

    def _label(self, raw: float) -> str:
        return f"VIX_term={raw:.2f}"
