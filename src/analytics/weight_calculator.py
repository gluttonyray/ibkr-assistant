"""IC-based factor weight calculator.

Computes Information Coefficient (Spearman rank correlation) between each
factor's z-score and forward returns, then assigns weights proportional to
max(0, IC). Factors with negative or near-zero IC receive zero weight.

Reference: Grinold & Kahn, "Active Portfolio Management" (2000).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List

import pandas as pd

from src.analytics.ic import compute_ic

logger = logging.getLogger(__name__)


class ICWeightCalculator:
    """Compute per-factor weights from Information Coefficients.

    Parameters
    ----------
    forward_horizon : int
        Forward return horizon in bars (default 10 ≈ 2.5 hours at 15-min).
    min_ic : float
        Minimum IC to receive any weight. Factors below this threshold
        are zeroed out (default 0.02).
    """

    def __init__(self, forward_horizon: int = 10, min_ic: float = 0.02) -> None:
        self.forward_horizon = forward_horizon
        self.min_ic = min_ic

    def compute(
        self,
        factor_scores: Dict[str, List[float]],
        prices: pd.Series,
    ) -> Dict[str, float]:
        """Compute IC-based weights.

        Parameters
        ----------
        factor_scores : dict
            {factor_name: [z_score_t0, z_score_t1, ...]}.
            Lists may contain NaN for stale bars.
        prices : pd.Series
            Close prices aligned with factor_scores (same length).

        Returns
        -------
        dict
            {factor_name: weight}. Normalized so mean active weight = 1.0.
            Factors with IC < min_ic receive weight 0.0.
        """
        n = len(prices)
        fwd_returns = prices.pct_change(self.forward_horizon).shift(-self.forward_horizon)

        raw: Dict[str, float] = {}
        ic_log: Dict[str, float] = {}

        for name, scores in factor_scores.items():
            length = min(len(scores), n)
            if length < 30:
                raw[name] = 1.0
                ic_log[name] = float("nan")
                continue

            factor_series = pd.Series(scores[:length], index=prices.index[:length])
            ic = compute_ic(factor_series, fwd_returns.iloc[:length], method="rank")
            ic_log[name] = ic

            if math.isnan(ic):
                raw[name] = 1.0          # insufficient data → default
            elif ic >= self.min_ic:
                raw[name] = ic           # positive IC → proportional weight
            else:
                raw[name] = 0.0         # low or negative IC → zero out

        # Log IC summary
        active = {k: v for k, v in ic_log.items() if not math.isnan(v)}
        if active:
            sorted_ic = sorted(active.items(), key=lambda x: x[1], reverse=True)
            logger.info(
                "IC weights computed (horizon=%d bars): top=%s | zeroed=%d/%d",
                self.forward_horizon,
                ", ".join(f"{k}={v:.3f}" for k, v in sorted_ic[:3]),
                sum(1 for v in raw.values() if v == 0.0),
                len(raw),
            )

        # Normalize: if all zeros, fall back to equal weights
        active_weights = {k: v for k, v in raw.items() if v > 0}
        if not active_weights:
            logger.warning("All factor ICs below threshold — using equal weights")
            return {name: 1.0 for name in factor_scores}

        # Scale so mean active weight = 1.0 (preserves overall signal strength)
        mean_active = sum(active_weights.values()) / len(active_weights)
        scale = 1.0 / mean_active
        return {name: w * scale for name, w in raw.items()}
