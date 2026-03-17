"""
Support / Resistance indicators:
  - Daily Pivot Points (Classic)
  - Fibonacci Retracement on recent swing high/low
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.base import BaseIndicator, SignalResult


class PivotPoints(BaseIndicator):
    """
    Classic daily pivot points computed from the *actual* previous trading
    session's High / Low / Close, detected from the DatetimeIndex.

    Pivot Points derive their effectiveness from market participants
    collectively watching the same prior-day levels.  Using a fixed
    rolling window of intraday bars (the previous implementation's approach)
    produced levels that no one else in the market was referencing,
    eliminating the self-fulfilling property of the indicator.
    """
    weight = 1.0
    name = "Pivot Points"
    min_bars = 2   # need at least 2 bars with a datetime index

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        if not isinstance(df.index, pd.DatetimeIndex):
            return SignalResult(0, float("nan"), "no datetime index")

        # Identify unique trading dates present in the data
        dates = df.index.normalize().unique().sort_values()
        if len(dates) < 2:
            return SignalResult(0, float("nan"), "need ≥2 trading days in history")

        # Previous complete session = second-to-last unique date
        prev_date = dates[-2]
        prior_mask = df.index.normalize() == prev_date
        prior = df[prior_mask]

        if len(prior) == 0:
            return SignalResult(0, float("nan"), "no prior session bars found")

        prev_high = float(prior["high"].max())
        prev_low = float(prior["low"].min())
        prev_close = float(prior["close"].iloc[-1])

        pp = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pp - prev_low
        s1 = 2 * pp - prev_high
        r2 = pp + (prev_high - prev_low)
        s2 = pp - (prev_high - prev_low)

        price = float(df["close"].iloc[-1])

        if price > r2:
            sig, lbl = -1, f"above R2 ({price:.2f}>R2={r2:.2f})"
        elif price > r1:
            sig, lbl = -1, f"above R1 ({price:.2f}>R1={r1:.2f})"
        elif price > pp:
            sig, lbl = 1, f"above PP ({price:.2f}>PP={pp:.2f})"
        elif price < s2:
            sig, lbl = 1, f"below S2 ({price:.2f}<S2={s2:.2f})"
        elif price < s1:
            sig, lbl = 1, f"below S1 ({price:.2f}<S1={s1:.2f})"
        else:
            sig, lbl = -1, f"below PP ({price:.2f}<PP={pp:.2f})"

        return SignalResult(sig, round(pp, 4), lbl)


class FibonacciRetracement(BaseIndicator):
    """
    Fibonacci retracement on the most recent swing high/low (last 50 bars).
    Price near a key Fib level (38.2%, 50%, 61.8%):
      - Near support (from above, in uptrend context) → +1
      - Near resistance (from below, in downtrend context) → -1
    """
    weight = 1.0
    name = "Fibonacci"
    min_bars = 55

    LOOKBACK = 50
    PROXIMITY = 0.005   # within 0.5% of a level = "at the level"

    # Key Fibonacci levels
    LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        window = df.tail(self.LOOKBACK)
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        price = df["close"].iloc[-1]

        if swing_high == swing_low:
            return SignalResult(0, float("nan"), "no range")

        rng = swing_high - swing_low
        # uptrend: retrace from high; downtrend: retrace from low
        # Determine trend direction from 20-bar SMA slope
        sma_now = float(df["close"].tail(20).mean())
        sma_prev = float(df["close"].iloc[-30:-10].mean()) if len(df) >= 30 else sma_now

        uptrend = sma_now > sma_prev

        # Calculate Fib levels
        levels = {}
        for lvl in self.LEVELS:
            if uptrend:
                # retracement from top down
                levels[lvl] = swing_high - lvl * rng
            else:
                # retracement from bottom up
                levels[lvl] = swing_low + lvl * rng

        # Find nearest level
        nearest_level = min(levels, key=lambda k: abs(levels[k] - price))
        nearest_price = levels[nearest_level]
        dist_pct = abs(price - nearest_price) / nearest_price

        if dist_pct > self.PROXIMITY * 3:
            # Not near any level — use position in range
            pos = (price - swing_low) / rng
            if pos < 0.3:
                return SignalResult(1, round(swing_low, 4), f"near swing low ({price:.2f})")
            elif pos > 0.7:
                return SignalResult(-1, round(swing_high, 4), f"near swing high ({price:.2f})")
            else:
                return SignalResult(0, round(pos, 4), f"mid range ({pos:.0%})")

        # At a key Fibonacci level
        level_name = f"Fib {nearest_level:.1%}"
        if dist_pct <= self.PROXIMITY:
            # Right at the level — wait for confirmation
            return SignalResult(0, round(nearest_price, 4), f"at {level_name} ({nearest_price:.2f})")

        if uptrend and price > nearest_price:
            # Bounced above support level
            return SignalResult(1, round(nearest_price, 4), f"above {level_name} support")
        elif not uptrend and price < nearest_price:
            # Rejected below resistance level
            return SignalResult(-1, round(nearest_price, 4), f"below {level_name} resistance")
        else:
            return SignalResult(0, round(nearest_price, 4), f"near {level_name}")
