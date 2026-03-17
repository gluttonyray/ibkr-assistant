"""Growth fundamental factors: revenue growth, EPS growth.

Reads from FactorData.fundamental dict.
"""
from __future__ import annotations

import math

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class FundamentalRevenueGrowth(BaseFactor):
    """YoY revenue growth rate (%).

    Positive = growing revenues → bullish.
    Source: FactorData.fundamental["revenue_growth"] (decimal: 0.08 = 8%).

    Reads: FactorData.fundamental["revenue_growth"].
    """

    name = "fundamental_revenue_growth"
    category = "fundamental"
    frequency = "quarterly"
    lookback_bars = 1
    weight = 1.0
    z_window = 60
    stale_seconds = 86400 * 90  # quarterly data

    def _compute(self, data: FactorData) -> float:
        if data.fundamental is None:
            return float("nan")
        growth = data.fundamental.get("revenue_growth")
        if growth is None:
            return float("nan")
        growth = float(growth)
        if math.isnan(growth):
            return float("nan")
        return growth * 100.0  # convert to percentage

    def _label(self, raw: float) -> str:
        return f"RevGrow={raw:.1f}%"
