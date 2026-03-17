"""Information Coefficient (IC) computation.

IC measures the correlation between a factor's predictions and actual
forward returns. Rank IC (Spearman) is preferred to avoid sensitivity
to outliers in raw factor values.

References:
  - Grinold & Kahn "Active Portfolio Management" (2000)
  - Quantopian/FactSet factor IC framework
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    method: str = "rank",
) -> float:
    """Compute Information Coefficient between factor values and forward returns.

    Parameters
    ----------
    factor_values : pd.Series
        Factor values indexed by timestamp.
    forward_returns : pd.Series
        Forward returns indexed by the same timestamps.
    method : str
        "rank" (Spearman correlation, default) or "pearson".

    Returns
    -------
    float
        IC value in [-1, +1]. NaN if insufficient data.
    """
    # Align on common index, drop NaN
    aligned = pd.DataFrame({"factor": factor_values, "returns": forward_returns}).dropna()
    if len(aligned) < 10:
        return float("nan")

    if method == "rank":
        ic, _ = spearmanr(aligned["factor"], aligned["returns"])
        return float(ic) if not np.isnan(ic) else float("nan")
    else:
        corr = aligned["factor"].corr(aligned["returns"])
        return float(corr) if not np.isnan(corr) else float("nan")


def ic_decay(
    factor_values: pd.Series,
    prices: pd.Series,
    horizons: Optional[List[int]] = None,
    method: str = "rank",
) -> pd.Series:
    """Compute IC at multiple forward-return horizons (IC decay curve).

    Parameters
    ----------
    factor_values : pd.Series
        Factor values indexed by timestamp.
    prices : pd.Series
        Price series indexed by the same timestamps.
    horizons : list[int], optional
        Forward bar horizons to evaluate (default: [1, 5, 10, 20, 40]).
    method : str
        IC computation method (default "rank").

    Returns
    -------
    pd.Series
        IC value at each horizon, indexed by horizon.
    """
    horizons = horizons or [1, 5, 10, 20, 40]
    results = {}
    for h in horizons:
        fwd_returns = prices.pct_change(h).shift(-h)
        results[h] = compute_ic(factor_values, fwd_returns, method=method)
    return pd.Series(results, name="ic_at_horizon")


def rolling_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    window: int = 60,
    method: str = "rank",
) -> pd.Series:
    """Compute rolling IC over time.

    Parameters
    ----------
    factor_values : pd.Series
        Factor values indexed by timestamp.
    forward_returns : pd.Series
        Forward returns indexed by the same timestamps.
    window : int
        Rolling window size in bars.
    method : str
        IC computation method.

    Returns
    -------
    pd.Series
        Rolling IC values indexed by timestamp.
    """
    aligned = pd.DataFrame({"factor": factor_values, "returns": forward_returns}).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    ic_values = {}
    for i in range(window, len(aligned) + 1):
        chunk = aligned.iloc[i - window:i]
        ts = chunk.index[-1]
        ic_values[ts] = compute_ic(chunk["factor"], chunk["returns"], method=method)

    return pd.Series(ic_values, name="rolling_ic")
