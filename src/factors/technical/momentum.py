"""Momentum technical factors: RSI continuous, ROC continuous."""
from __future__ import annotations

import math

import pandas_ta as ta

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class MomentumRSI(BaseFactor):
    """RSI(14) raw value: 0–100.

    High RSI (>70) = overbought/strong momentum.
    Low RSI (<30)  = oversold/weak momentum.
    Used as a continuous value — z-score expresses whether current
    RSI is high/low relative to its recent history.
    """

    name = "momentum_rsi"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 20
    weight = 1.0
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        close = data.ohlcv["close"]
        result = ta.rsi(close, length=14)
        if result is None:
            return float("nan")
        val = result.iloc[-1]
        return float("nan") if math.isnan(val) else float(val)

    def _label(self, raw: float) -> str:
        return f"RSI={raw:.1f}"


@FactorRegistry.register
class MomentumROC(BaseFactor):
    """ROC(12): rate of change in %.

    Positive = bullish momentum.  Negative = bearish.
    Unbounded — z-score normalizes to recent volatility regime.
    """

    name = "momentum_roc"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 20
    weight = 1.0
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        close = data.ohlcv["close"]
        result = ta.roc(close, length=12)
        if result is None:
            return float("nan")
        val = result.iloc[-1]
        return float("nan") if math.isnan(val) else float(val)

    def _label(self, raw: float) -> str:
        return f"ROC={raw:.2f}%"
