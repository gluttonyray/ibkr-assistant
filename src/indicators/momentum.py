"""
Momentum indicators:
  - RSI(14)
  - KDJ / Stochastic(14,3,3)
  - CCI(20)
  - Williams %R(14)
  - ROC(12)
  - Ultimate Oscillator(7,14,28)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    import talib
    _TALIB = True
except ImportError:
    _TALIB = False

from src.indicators.base import BaseIndicator, SignalResult


class RSI(BaseIndicator):
    """
    RSI(14): only uses overbought/oversold extreme zones.
    <30 → +1 (oversold), >70 → -1 (overbought), 30-70 → 0 (neutral/no signal).

    The previous midline logic (< 50 = buy, > 50 = sell) was internally
    contradictory: it mixed contrarian (mean-reversion) and momentum
    (trend-following) philosophies in a single indicator, causing conflicting
    signals across market regimes.
    """
    weight = 1.2
    name = "RSI(14)"
    min_bars = 20

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        rsi = ta.rsi(df["close"], length=14)
        val = self._last(rsi)
        if np.isnan(val):
            return SignalResult(0, float("nan"), "no data")

        if val < 30:
            return SignalResult(1, round(val, 2), f"oversold ({val:.1f})")
        elif val > 70:
            return SignalResult(-1, round(val, 2), f"overbought ({val:.1f})")
        else:
            return SignalResult(0, round(val, 2), f"neutral ({val:.1f})")


class StochKDJ(BaseIndicator):
    """
    Stochastic / KDJ (14,3,3).
    %K < 20 → oversold (+1), %K > 80 → overbought (-1).
    %K cross above %D → +1 in oversold zone.
    """
    weight = 1.0
    name = "KDJ/Stoch"
    min_bars = 25

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        if stoch is None or stoch.empty:
            return SignalResult(0, float("nan"), "no data")

        k_col = stoch.filter(like="STOCHk_")
        d_col = stoch.filter(like="STOCHd_")

        k = self._last(k_col.iloc[:, 0]) if not k_col.empty else float("nan")
        d = self._last(d_col.iloc[:, 0]) if not d_col.empty else float("nan")

        if np.isnan(k):
            return SignalResult(0, float("nan"), "no data")

        if k < 20 and k > d:
            return SignalResult(1, round(k, 2), f"oversold+cross K={k:.1f}")
        elif k < 20:
            return SignalResult(1, round(k, 2), f"oversold K={k:.1f}")
        elif k > 80 and k < d:
            return SignalResult(-1, round(k, 2), f"overbought+cross K={k:.1f}")
        elif k > 80:
            return SignalResult(-1, round(k, 2), f"overbought K={k:.1f}")
        elif k > d:
            return SignalResult(1, round(k, 2), f"K>D ({k:.1f}>{d:.1f})")
        elif k < d:
            return SignalResult(-1, round(k, 2), f"K<D ({k:.1f}<{d:.1f})")
        else:
            return SignalResult(0, round(k, 2), "neutral")


class CCI(BaseIndicator):
    """
    CCI(20): only uses the extreme ±100 zones.
    < -100 → +1 (oversold), > +100 → -1 (overbought), -100 to +100 → 0 (neutral).

    The zero-line cross logic (< 0 = buy, > 0 = sell) was removed because it
    fires on nearly every bar and, like the RSI midline, conflates contrarian
    and momentum signals without context.
    """
    weight = 0.8
    name = "CCI(20)"
    min_bars = 25

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        cci = ta.cci(df["high"], df["low"], df["close"], length=20)
        val = self._last(cci)

        if np.isnan(val):
            return SignalResult(0, float("nan"), "no data")

        if val < -100:
            return SignalResult(1, round(val, 2), f"oversold ({val:.1f})")
        elif val > 100:
            return SignalResult(-1, round(val, 2), f"overbought ({val:.1f})")
        else:
            return SignalResult(0, round(val, 2), f"neutral ({val:.1f})")


class WilliamsR(BaseIndicator):
    """Williams %R(14): below -80 → +1, above -20 → -1."""
    weight = 0.8
    name = "Williams %R"
    min_bars = 20

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        wr = ta.willr(df["high"], df["low"], df["close"], length=14)
        val = self._last(wr)

        if np.isnan(val):
            return SignalResult(0, float("nan"), "no data")

        if val < -80:
            return SignalResult(1, round(val, 2), f"oversold ({val:.1f})")
        elif val > -20:
            return SignalResult(-1, round(val, 2), f"overbought ({val:.1f})")
        elif val < -50:
            return SignalResult(1, round(val, 2), f"below -50 ({val:.1f})")
        elif val > -50:
            return SignalResult(-1, round(val, 2), f"above -50 ({val:.1f})")
        else:
            return SignalResult(0, round(val, 2), "neutral")


class ROC(BaseIndicator):
    """Rate of Change(12): positive momentum → +1, negative → -1."""
    weight = 0.8
    name = "ROC(12)"
    min_bars = 15

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        roc = ta.roc(df["close"], length=12)
        val = self._last(roc)

        if np.isnan(val):
            return SignalResult(0, float("nan"), "no data")

        if val > 2.0:
            return SignalResult(1, round(val, 4), f"strong up ({val:.2f}%)")
        elif val < -2.0:
            return SignalResult(-1, round(val, 4), f"strong down ({val:.2f}%)")
        elif val > 0:
            return SignalResult(1, round(val, 4), f"up ({val:.2f}%)")
        elif val < 0:
            return SignalResult(-1, round(val, 4), f"down ({val:.2f}%)")
        else:
            return SignalResult(0, round(val, 4), "flat")


class UltimateOscillator(BaseIndicator):
    """Ultimate Oscillator(7,14,28): <30 → +1, >70 → -1."""
    weight = 0.8
    name = "UltOsc"
    min_bars = 35

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        uo = ta.uo(df["high"], df["low"], df["close"], fast=7, medium=14, slow=28)
        val = self._last(uo)

        if np.isnan(val):
            return SignalResult(0, float("nan"), "no data")

        if val < 30:
            return SignalResult(1, round(val, 2), f"oversold ({val:.1f})")
        elif val > 70:
            return SignalResult(-1, round(val, 2), f"overbought ({val:.1f})")
        elif val < 50:
            return SignalResult(1, round(val, 2), f"below 50 ({val:.1f})")
        elif val > 50:
            return SignalResult(-1, round(val, 2), f"above 50 ({val:.1f})")
        else:
            return SignalResult(0, round(val, 2), "neutral")
