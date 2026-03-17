"""Macro rate factors: yield curve spread and fed funds rate.

Reads from FactorData.macro dict. Returns NaN (stale) when macro data absent.
"""
from __future__ import annotations

import math

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class MacroYieldCurve(BaseFactor):
    """10Y-2Y Treasury yield spread (basis points).

    Positive spread (normal yield curve) → growth expectations → bullish.
    Negative spread (inverted) → recession risk → bearish.

    Reads: FactorData.macro["spread"] (pre-computed 10Y-2Y in bps).
    """

    name = "macro_yield_curve"
    category = "macro"
    frequency = "daily"
    lookback_bars = 1
    weight = 1.0
    z_window = 120  # longer window for macro data
    stale_seconds = 86400 * 3  # tolerate weekends

    def _compute(self, data: FactorData) -> float:
        if data.macro is None:
            return float("nan")
        spread = data.macro.get("spread")
        if spread is None or math.isnan(float(spread)):
            return float("nan")
        return float(spread)

    def _label(self, raw: float) -> str:
        return f"YldSpread={raw:.1f}bps"


@FactorRegistry.register
class MacroFedRate(BaseFactor):
    """Effective federal funds rate (%).

    Higher rate → tighter financial conditions → headwind for equities.
    Raw value is the rate itself; z-score captures whether it's unusually
    high or low relative to recent history.

    Reads: FactorData.macro["fed_rate"].
    """

    name = "macro_fed_rate"
    category = "macro"
    frequency = "daily"
    lookback_bars = 1
    weight = 0.8
    z_window = 120
    stale_seconds = 86400 * 3

    def _compute(self, data: FactorData) -> float:
        if data.macro is None:
            return float("nan")
        rate = data.macro.get("fed_rate")
        if rate is None or math.isnan(float(rate)):
            return float("nan")
        return -float(rate)  # inverted: higher rate = bearish for equities

    def _label(self, raw: float) -> str:
        return f"FedRate={-raw:.2f}%"
