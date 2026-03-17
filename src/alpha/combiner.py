"""Alpha combiner: aggregates factor z-scores into a single alpha signal.

Pipeline:
  1. Filter stale factors
  2. Apply regime-based category weight multipliers
  3. Weighted sum of z-scores
  4. Normalize by total weight → raw_alpha
  5. tanh(raw_alpha) → score ∈ [-1, +1]

Reuses REGIME_WEIGHT_MULT and REGIME_THRESHOLD_MULT constants from the
existing SignalAggregator to keep factor-system behaviour consistent with
the indicator-based system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.factors.base import FactorResult
from src.indicators.aggregator import REGIME_WEIGHT_MULT, REGIME_THRESHOLD_MULT


@dataclass
class AlphaSignal:
    """Output of AlphaCombiner.combine()."""
    score: float                            # tanh-bounded [-1, +1]
    confidence: float                       # fraction of active factors agreeing on direction
    regime: str
    factor_contributions: Dict[str, float]  # per-factor normalised contribution
    timestamp: datetime
    active_factor_count: int
    raw_weighted_sum: float                 # pre-tanh score (for diagnostics)


class AlphaCombiner:
    """Combines factor z-scores into a bounded alpha signal.

    Parameters
    ----------
    regime_weight_mult : dict, optional
        Override for regime × category weight multipliers.
        Defaults to REGIME_WEIGHT_MULT from aggregator.py.
    tanh_scale : float
        Scales raw_alpha before tanh. Higher values → faster saturation.
        Default 1.0 gives meaningful curvature around ±1 std of z-scores.
    """

    def __init__(
        self,
        regime_weight_mult: Optional[Dict[str, Dict[str, float]]] = None,
        tanh_scale: float = 1.0,
    ) -> None:
        self._regime_mult = regime_weight_mult if regime_weight_mult is not None else REGIME_WEIGHT_MULT
        self._tanh_scale = tanh_scale

    def combine(
        self,
        results: List[FactorResult],
        regime: str = "UNKNOWN",
        timestamp: Optional[datetime] = None,
    ) -> AlphaSignal:
        """Combine a list of FactorResults into a single AlphaSignal.

        Parameters
        ----------
        results : list[FactorResult]
            One result per factor. Stale results are filtered out.
        regime : str
            Current market regime (from RegimeDetector). Used to select
            category weight multipliers.
        timestamp : datetime, optional
            Signal timestamp. Defaults to now.
        """
        ts = timestamp or datetime.now(timezone.utc)

        # Filter stale factors and NaN z-scores
        active = [
            r for r in results
            if not r.is_stale and not math.isnan(r.z_score)
        ]

        if not active:
            return AlphaSignal(
                score=0.0,
                confidence=0.0,
                regime=regime,
                factor_contributions={},
                timestamp=ts,
                active_factor_count=0,
                raw_weighted_sum=0.0,
            )

        weight_mult_map = self._regime_mult.get(regime, self._regime_mult.get("UNKNOWN", {}))

        weighted_sum = 0.0
        total_weight = 0.0
        contributions: Dict[str, float] = {}

        for r in active:
            regime_mult = weight_mult_map.get(r.category, 1.0)
            effective_weight = r.weight * regime_mult
            contribution = r.z_score * effective_weight
            weighted_sum += contribution
            total_weight += abs(effective_weight)
            contributions[r.name] = contribution

        if total_weight < 1e-10:
            return AlphaSignal(
                score=0.0,
                confidence=0.0,
                regime=regime,
                factor_contributions={k: round(v, 6) for k, v in contributions.items()},
                timestamp=ts,
                active_factor_count=len(active),
                raw_weighted_sum=weighted_sum,
            )

        raw_alpha = weighted_sum / total_weight
        score = math.tanh(raw_alpha * self._tanh_scale)

        # Normalize individual contributions
        contributions = {k: round(v / total_weight, 6) for k, v in contributions.items()}

        # Confidence: fraction of active factors agreeing with signal direction
        if abs(score) < 1e-6:
            confidence = 0.0
        else:
            direction = 1 if score > 0 else -1
            agreeing = sum(1 for r in active if r.z_score * direction > 0)
            confidence = agreeing / len(active)

        return AlphaSignal(
            score=round(score, 6),
            confidence=round(confidence, 4),
            regime=regime,
            factor_contributions=contributions,
            timestamp=ts,
            active_factor_count=len(active),
            raw_weighted_sum=round(raw_alpha, 6),
        )
