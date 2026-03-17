"""
Volatility indicators:
  - Bollinger Bands(20,2)
  - Keltner Channel(20,2)
  - Donchian Channel(20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.indicators.base import BaseIndicator, SignalResult


class BollingerBands(BaseIndicator):
    """
    BB(20,2):
      price < lower → +1 (bounce expected)
      price > upper → -1 (reversal expected)
      %B used for partial signals
    """
    weight = 1.2
    name = "Bollinger"
    min_bars = 25

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is None or bb.empty:
            return SignalResult(0, float("nan"), "no data")

        upper = self._last(bb.filter(like="BBU_").iloc[:, 0])
        lower = self._last(bb.filter(like="BBL_").iloc[:, 0])
        mid = self._last(bb.filter(like="BBM_").iloc[:, 0])
        price = df["close"].iloc[-1]

        if np.isnan(upper) or np.isnan(lower):
            return SignalResult(0, float("nan"), "no data")

        band_width = upper - lower
        pct_b = (price - lower) / band_width if band_width != 0 else 0.5

        if price <= lower:
            return SignalResult(1, round(pct_b, 4), f"below lower ({price:.2f}<={lower:.2f})")
        elif price >= upper:
            return SignalResult(-1, round(pct_b, 4), f"above upper ({price:.2f}>={upper:.2f})")
        elif pct_b < 0.2:
            return SignalResult(1, round(pct_b, 4), f"%B low ({pct_b:.2%})")
        elif pct_b > 0.8:
            return SignalResult(-1, round(pct_b, 4), f"%B high ({pct_b:.2%})")
        elif pct_b < 0.5:
            return SignalResult(1, round(pct_b, 4), f"below mid ({pct_b:.2%})")
        else:
            return SignalResult(-1, round(pct_b, 4), f"above mid ({pct_b:.2%})")


class KeltnerChannel(BaseIndicator):
    """
    Keltner Channel(20,2):
      price < lower KC → +1, price > upper KC → -1
    """
    weight = 1.0
    name = "Keltner"
    min_bars = 25

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2)
        if kc is None or kc.empty:
            return SignalResult(0, float("nan"), "no data")

        upper = self._last(kc.filter(like="KCUe_").iloc[:, 0]) if not kc.filter(like="KCUe_").empty else float("nan")
        lower = self._last(kc.filter(like="KCLe_").iloc[:, 0]) if not kc.filter(like="KCLe_").empty else float("nan")

        # Fallback column names
        if np.isnan(upper):
            upper_cols = [c for c in kc.columns if "U" in c.upper() and "KC" in c.upper()]
            lower_cols = [c for c in kc.columns if "L" in c.upper() and "KC" in c.upper()]
            upper = self._last(kc[upper_cols[0]]) if upper_cols else float("nan")
            lower = self._last(kc[lower_cols[0]]) if lower_cols else float("nan")

        price = df["close"].iloc[-1]

        if np.isnan(upper) or np.isnan(lower):
            return SignalResult(0, float("nan"), "no data")

        if price < lower:
            return SignalResult(1, round(lower, 4), f"below KC lower ({price:.2f}<{lower:.2f})")
        elif price > upper:
            return SignalResult(-1, round(upper, 4), f"above KC upper ({price:.2f}>{upper:.2f})")
        else:
            mid = (upper + lower) / 2
            if price < mid:
                return SignalResult(1, round(mid, 4), f"below KC mid ({price:.2f}<{mid:.2f})")
            else:
                return SignalResult(-1, round(mid, 4), f"above KC mid ({price:.2f}>{mid:.2f})")


class DonchianChannel(BaseIndicator):
    """
    Donchian Channel(20):
      price == upper → bearish breakout (-1)
      price == lower → bullish breakout (+1)
      else position within channel
    """
    weight = 0.8
    name = "Donchian"
    min_bars = 25

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        dc = ta.donchian(df["high"], df["low"], lower_length=20, upper_length=20)
        if dc is None or dc.empty:
            return SignalResult(0, float("nan"), "no data")

        upper = self._last(dc.filter(like="DCU_").iloc[:, 0]) if not dc.filter(like="DCU_").empty else float("nan")
        lower = self._last(dc.filter(like="DCL_").iloc[:, 0]) if not dc.filter(like="DCL_").empty else float("nan")
        mid = self._last(dc.filter(like="DCM_").iloc[:, 0]) if not dc.filter(like="DCM_").empty else float("nan")

        price = df["close"].iloc[-1]

        if np.isnan(upper) or np.isnan(lower):
            return SignalResult(0, float("nan"), "no data")

        if price >= upper:
            return SignalResult(-1, round(upper, 4), f"at upper ({price:.2f}>={upper:.2f})")
        elif price <= lower:
            return SignalResult(1, round(lower, 4), f"at lower ({price:.2f}<={lower:.2f})")
        else:
            channel = upper - lower
            pos = (price - lower) / channel if channel != 0 else 0.5
            if pos < 0.3:
                return SignalResult(1, round(pos, 4), f"near lower ({pos:.0%})")
            elif pos > 0.7:
                return SignalResult(-1, round(pos, 4), f"near upper ({pos:.0%})")
            else:
                return SignalResult(0, round(pos, 4), f"mid channel ({pos:.0%})")
