"""Valuation fundamental factors: earnings yield, P/B.

Reads from FactorData.fundamental dict. Updated daily in live mode.
Returns NaN (stale) when fundamental data is absent.
"""
from __future__ import annotations

import math

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class FundamentalEarningsYield(BaseFactor):
    """Earnings yield: 1 / P/E ratio (%).

    Higher earnings yield → cheaper stock → bullish.
    Negative P/E (losses) → returns NaN.

    Reads: FactorData.fundamental["trailing_pe"].
    """

    name = "fundamental_earnings_yield"
    category = "fundamental"
    frequency = "daily"
    lookback_bars = 1
    weight = 1.0
    z_window = 120
    stale_seconds = 86400 * 7  # weekly update tolerance

    def _compute(self, data: FactorData) -> float:
        if data.fundamental is None:
            return float("nan")
        pe = data.fundamental.get("trailing_pe")
        if pe is None:
            return float("nan")
        pe = float(pe)
        if math.isnan(pe) or pe <= 0.0:
            return float("nan")
        return (1.0 / pe) * 100.0  # as percentage

    def _label(self, raw: float) -> str:
        return f"EY={raw:.2f}%"


@FactorRegistry.register
class FundamentalPB(BaseFactor):
    """Inverse P/B ratio: 1 / P/B.

    Higher inverse P/B → lower price-to-book → value stock → bullish.
    Negative book value → returns NaN.

    Reads: FactorData.fundamental["price_to_book"].
    """

    name = "fundamental_pb"
    category = "fundamental"
    frequency = "daily"
    lookback_bars = 1
    weight = 0.8
    z_window = 120
    stale_seconds = 86400 * 7

    def _compute(self, data: FactorData) -> float:
        if data.fundamental is None:
            return float("nan")
        pb = data.fundamental.get("price_to_book")
        if pb is None:
            return float("nan")
        pb = float(pb)
        if math.isnan(pb) or pb <= 0.0:
            return float("nan")
        return 1.0 / pb

    def _label(self, raw: float) -> str:
        return f"1/PB={raw:.3f}"
