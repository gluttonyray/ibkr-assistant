"""
Base class and helpers for all indicators.

Each indicator:
  - Receives a OHLCV DataFrame (index=datetime, cols: open/high/low/close/volume)
  - Returns a scalar int: +1 (buy), 0 (hold), -1 (sell)
  - Has a *weight* used by the aggregator
  - Has a *name* for display
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
import pandas as pd


class SignalResult(NamedTuple):
    signal: int       # +1, 0, -1
    value: float      # raw indicator value(s) for display (primary value)
    label: str        # short human-readable description of current state


class BaseIndicator(ABC):
    weight: float = 1.0
    name: str = "indicator"

    def compute(self, df: pd.DataFrame) -> SignalResult:
        """
        Public entry point.  Returns SignalResult or (0, nan, 'insufficient data')
        if there are not enough bars.
        """
        if len(df) < self.min_bars:
            return SignalResult(0, float("nan"), "insufficient data")
        try:
            return self._compute(df)
        except Exception as exc:
            return SignalResult(0, float("nan"), f"error: {exc}")

    @property
    @abstractmethod
    def min_bars(self) -> int: ...

    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> SignalResult: ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last(series: pd.Series) -> float:
        """Return last non-NaN value or NaN."""
        vals = series.dropna()
        return float(vals.iloc[-1]) if len(vals) else float("nan")

    @staticmethod
    def _sign(val: float, threshold: float = 0.0) -> int:
        """Map a continuous value to +1/0/-1."""
        if np.isnan(val):
            return 0
        if val > threshold:
            return 1
        if val < -threshold:
            return -1
        return 0
