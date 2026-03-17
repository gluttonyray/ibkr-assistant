"""Core abstractions for the factor system.

Every factor:
  - Implements _compute() returning a continuous float
  - Maintains a rolling deque for z-score normalization
  - Returns FactorResult with raw value, z-score, and metadata
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Optional

import pandas as pd


@dataclass
class FactorResult:
    """Output of a single factor computation."""
    name: str
    raw_value: float
    z_score: float
    category: str           # technical / fundamental / sentiment / macro
    frequency: str          # intraday / daily / weekly / monthly
    timestamp: datetime
    is_stale: bool
    label: str
    weight: float = 1.0


@dataclass
class FactorData:
    """Unified data container passed to all factors."""
    ohlcv: pd.DataFrame             # OHLCV with DatetimeIndex
    symbol: str
    fundamental: Optional[dict] = None
    sentiment: Optional[dict] = None
    macro: Optional[dict] = None
    timestamp: Optional[datetime] = None


class BaseFactor(ABC):
    """Abstract base class for all quantitative factors.

    Subclasses must set class attributes (name, category, etc.) and
    implement _compute() returning a raw continuous float.

    Z-score is computed automatically from a rolling deque of past raw
    values so factors from different scales are comparable.
    """

    name: str = "factor"
    category: str = "technical"
    frequency: str = "intraday"
    lookback_bars: int = 20
    weight: float = 1.0
    z_window: int = 60          # rolling window for z-score normalization
    stale_seconds: int = 3600   # mark stale if last bar older than this

    def __init__(self) -> None:
        self._history: Deque[float] = deque(maxlen=self.z_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, data: FactorData) -> FactorResult:
        """Compute factor value, normalize to z-score, return FactorResult."""
        ts = data.timestamp or datetime.now(timezone.utc)

        if len(data.ohlcv) < self.lookback_bars:
            return self._stale_result(ts, "insufficient data")

        try:
            raw = self._compute(data)
        except Exception:
            raw = float("nan")

        if raw is None or math.isnan(raw):
            return self._stale_result(ts, "compute error")

        # Update rolling history for z-score
        self._history.append(raw)

        # Compute z-score from rolling history
        z_score = self._z_score(raw)

        # Staleness check
        is_stale = self._check_stale(data, ts)

        return FactorResult(
            name=self.name,
            raw_value=round(raw, 6),
            z_score=round(z_score, 6),
            category=self.category,
            frequency=self.frequency,
            timestamp=ts,
            is_stale=is_stale,
            label=self._label(raw),
            weight=self.weight,
        )

    def reset(self) -> None:
        """Clear rolling history (call between independent backtests)."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Abstract / overridable
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute(self, data: FactorData) -> float:
        """Return continuous raw value. Subclasses must implement."""
        ...

    def _label(self, raw: float) -> str:
        return f"{self.name}={raw:.4f}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _z_score(self, raw: float) -> float:
        if len(self._history) < 3:
            return 0.0
        vals = list(self._history)
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        if std < 1e-10:
            return 0.0
        return (raw - mean) / std

    def _stale_result(self, ts: datetime, reason: str) -> FactorResult:
        return FactorResult(
            name=self.name,
            raw_value=float("nan"),
            z_score=0.0,
            category=self.category,
            frequency=self.frequency,
            timestamp=ts,
            is_stale=True,
            label=reason,
            weight=self.weight,
        )

    def _check_stale(self, data: FactorData, ts: datetime) -> bool:
        if data.timestamp is None:
            return False
        try:
            bar_ts = data.ohlcv.index[-1]
            if hasattr(bar_ts, "to_pydatetime"):
                bar_dt = bar_ts.to_pydatetime()
                if bar_dt.tzinfo is None:
                    bar_dt = bar_dt.replace(tzinfo=timezone.utc)
                now = ts
                if now.tzinfo is None:
                    now = now.replace(tzinfo=timezone.utc)
                age = (now - bar_dt).total_seconds()
                return age > self.stale_seconds
        except Exception:
            pass
        return False
