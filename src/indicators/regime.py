"""
Market regime detector.

Classifies the current market into one of four states by combining
ADX trend strength with an ATR volatility ratio:

  ┌──────────────────────┬──────────────────────────────────────────────────┐
  │ Regime               │ Description                                      │
  ├──────────────────────┼──────────────────────────────────────────────────┤
  │ TRENDING_CALM        │ ADX > 25, ATR normal — cleanest trend signals    │
  │ TRENDING_VOLATILE    │ ADX > 25, ATR elevated — trend + noise           │
  │ RANGING              │ ADX ≤ 25, ATR normal — mean-reversion favoured   │
  │ CHOPPY_VOLATILE      │ ADX ≤ 25, ATR elevated — avoid most signals      │
  └──────────────────────┴──────────────────────────────────────────────────┘

Used by SignalAggregator to:
  1. Scale per-category indicator weights (trend vs. momentum emphasis).
  2. Adjust the score threshold (tighter in choppy/volatile markets).
"""
from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regime from an OHLCV DataFrame.

    Returns one of: "TRENDING_CALM", "TRENDING_VOLATILE",
                    "RANGING", "CHOPPY_VOLATILE", "UNKNOWN"
    """

    ADX_THRESHOLD: float = 25.0         # above = trending
    ATR_RATIO_THRESHOLD: float = 1.3    # current/baseline ATR above this = elevated vol
    ATR_BASELINE_PERIOD: int = 50       # bars used to compute baseline ATR
    MIN_BARS: int = 30                  # minimum bars needed for reliable detection

    def detect(self, df: pd.DataFrame) -> str:
        """Compute and return the current market regime label."""
        if len(df) < self.MIN_BARS:
            return "UNKNOWN"

        adx = self._adx(df)
        atr_ratio = self._atr_ratio(df)

        trending = adx > self.ADX_THRESHOLD
        volatile = atr_ratio > self.ATR_RATIO_THRESHOLD

        if trending and volatile:
            return "TRENDING_VOLATILE"
        elif trending:
            return "TRENDING_CALM"
        elif volatile:
            return "CHOPPY_VOLATILE"
        else:
            return "RANGING"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adx(df: pd.DataFrame) -> float:
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx_df is None or adx_df.empty:
                return 20.0
            col = adx_df.filter(like="ADX_").iloc[:, 0].dropna()
            return float(col.iloc[-1]) if len(col) else 20.0
        except Exception:
            return 20.0

    @classmethod
    def _atr_ratio(cls, df: pd.DataFrame) -> float:
        try:
            atr = ta.atr(df["high"], df["low"], df["close"], length=14)
            if atr is None or atr.empty:
                return 1.0
            atr_valid = atr.dropna()
            if len(atr_valid) < 2:
                return 1.0
            current = float(atr_valid.iloc[-1])
            baseline_window = min(cls.ATR_BASELINE_PERIOD, len(atr_valid))
            baseline = float(atr_valid.tail(baseline_window).mean())
            return current / baseline if baseline > 0 else 1.0
        except Exception:
            return 1.0
