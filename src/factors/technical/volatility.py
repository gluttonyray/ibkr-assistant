"""Volatility technical factors: Bollinger %B, ATR ratio."""
from __future__ import annotations

import math

import pandas_ta as ta

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class VolatilityBBPctB(BaseFactor):
    """Bollinger %B: (price - lower_band) / (upper_band - lower_band).

    ~0.0 = at lower band (oversold in ranging markets).
    ~0.5 = at middle band (neutral).
    ~1.0 = at upper band (overbought in ranging markets).
    Can exceed [0,1] during strong breakouts.
    """

    name = "volatility_bb_pctb"
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

        pctb_cols = [c for c in result.columns if "BBP" in c]
        if not pctb_cols:
            return float("nan")

        val = result[pctb_cols[0]].iloc[-1]
        return float("nan") if math.isnan(val) else float(val)

    def _label(self, raw: float) -> str:
        return f"BB_%B={raw:.3f}"


@FactorRegistry.register
class VolatilityATRRatio(BaseFactor):
    """ATR(14) / SMA(ATR, 50): current volatility vs. its 50-bar average.

    > 1.0 = current volatility is above average (high vol regime).
    < 1.0 = current volatility is below average (quiet regime).
    Used as a volatility regime signal and for risk scaling.
    """

    name = "volatility_atr_ratio"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 60
    weight = 0.8
    z_window = 60

    def _compute(self, data: FactorData) -> float:
        high = data.ohlcv["high"]
        low = data.ohlcv["low"]
        close = data.ohlcv["close"]

        atr = ta.atr(high, low, close, length=14)
        if atr is None:
            return float("nan")

        atr_clean = atr.dropna()
        if len(atr_clean) < 5:
            return float("nan")

        current_atr = float(atr_clean.iloc[-1])
        if math.isnan(current_atr):
            return float("nan")

        window = min(50, len(atr_clean))
        atr_sma = float(atr_clean.iloc[-window:].mean())

        if atr_sma < 1e-10:
            return float("nan")

        return current_atr / atr_sma

    def _label(self, raw: float) -> str:
        return f"ATR_ratio={raw:.3f}"
