"""DXY (US Dollar Index) macro factor.

A stronger dollar is generally a headwind for US equities and commodities.
The factor is inverted: rising DXY → negative raw → bearish.

Reads: FactorData.macro["dxy"].
"""
from __future__ import annotations

import math

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class MacroDXY(BaseFactor):
    """DXY index level (inverted for equities).

    High DXY (strong dollar) → headwind for equities → bearish.
    Low DXY  (weak dollar)   → tailwind for equities → bullish.

    Raw value = -dxy so positive z-score = weak dollar = bullish.

    Reads: FactorData.macro["dxy"].
    """

    name = "macro_dxy"
    category = "macro"
    frequency = "daily"
    lookback_bars = 1
    weight = 0.8
    z_window = 120
    stale_seconds = 86400 * 3

    def _compute(self, data: FactorData) -> float:
        if data.macro is None:
            return float("nan")
        dxy = data.macro.get("dxy")
        if dxy is None or math.isnan(float(dxy)):
            return float("nan")
        return -float(dxy)  # inverted: high DXY = bearish

    def _label(self, raw: float) -> str:
        return f"DXY={-raw:.2f}"
