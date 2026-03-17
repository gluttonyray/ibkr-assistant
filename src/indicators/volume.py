"""
Volume indicators:
  - OBV trend
  - VWAP (intraday rolling)
  - MFI(14)
  - CMF(20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.indicators.base import BaseIndicator, SignalResult


class OBVTrend(BaseIndicator):
    """
    OBV trend using a short EMA of OBV.
    OBV rising (EMA slope positive) → +1, falling → -1.
    """
    weight = 1.0
    name = "OBV"
    min_bars = 30

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        obv = ta.obv(df["close"], df["volume"])
        if obv is None or obv.empty or len(obv.dropna()) < 10:
            return SignalResult(0, float("nan"), "no data")

        # Short EMA of OBV to determine trend
        obv_ema = ta.ema(obv, 10)
        current_ema = self._last(obv_ema)
        # Compare current vs 5 bars ago
        valid = obv_ema.dropna()
        if len(valid) < 5:
            return SignalResult(0, float("nan"), "insufficient OBV data")

        prev_ema = float(valid.iloc[-5])
        obv_val = self._last(obv)

        if current_ema > prev_ema * 1.001:
            return SignalResult(1, round(obv_val, 0), f"OBV rising ({obv_val:.0f})")
        elif current_ema < prev_ema * 0.999:
            return SignalResult(-1, round(obv_val, 0), f"OBV falling ({obv_val:.0f})")
        else:
            return SignalResult(0, round(obv_val, 0), f"OBV flat ({obv_val:.0f})")


class VWAPSignal(BaseIndicator):
    """
    Intraday rolling VWAP (resets based on cumulative within the DataFrame window).
    Price above VWAP → +1 (bullish), below → -1 (bearish).
    """
    weight = 1.2
    name = "VWAP"
    min_bars = 10

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        # pandas-ta VWAP needs datetime index
        try:
            vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        except Exception:
            # Manual VWAP calculation as fallback
            typical = (df["high"] + df["low"] + df["close"]) / 3
            cumvol = df["volume"].cumsum()
            cumtp = (typical * df["volume"]).cumsum()
            vwap = cumtp / cumvol

        val = self._last(vwap)
        price = df["close"].iloc[-1]

        if np.isnan(val) or val == 0:
            return SignalResult(0, float("nan"), "no VWAP data")

        diff_pct = (price - val) / val * 100

        if price > val:
            return SignalResult(1, round(val, 4), f"above VWAP +{diff_pct:.2f}%")
        elif price < val:
            return SignalResult(-1, round(val, 4), f"below VWAP {diff_pct:.2f}%")
        else:
            return SignalResult(0, round(val, 4), "at VWAP")


class MFI(BaseIndicator):
    """Money Flow Index(14): <20 → +1, >80 → -1."""
    weight = 1.0
    name = "MFI(14)"
    min_bars = 20

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        val = self._last(mfi)

        if np.isnan(val):
            return SignalResult(0, float("nan"), "no data")

        if val < 20:
            return SignalResult(1, round(val, 2), f"oversold ({val:.1f})")
        elif val > 80:
            return SignalResult(-1, round(val, 2), f"overbought ({val:.1f})")
        elif val < 50:
            return SignalResult(1, round(val, 2), f"below 50 ({val:.1f})")
        elif val > 50:
            return SignalResult(-1, round(val, 2), f"above 50 ({val:.1f})")
        else:
            return SignalResult(0, round(val, 2), "neutral")


class CMF(BaseIndicator):
    """Chaikin Money Flow(20): >0.05 → +1, <-0.05 → -1."""
    weight = 0.8
    name = "CMF(20)"
    min_bars = 25

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
        val = self._last(cmf)

        if np.isnan(val):
            return SignalResult(0, float("nan"), "no data")

        if val > 0.1:
            return SignalResult(1, round(val, 4), f"strong buy flow ({val:.3f})")
        elif val < -0.1:
            return SignalResult(-1, round(val, 4), f"strong sell flow ({val:.3f})")
        elif val > 0.05:
            return SignalResult(1, round(val, 4), f"buy flow ({val:.3f})")
        elif val < -0.05:
            return SignalResult(-1, round(val, 4), f"sell flow ({val:.3f})")
        else:
            return SignalResult(0, round(val, 4), f"neutral ({val:.3f})")
