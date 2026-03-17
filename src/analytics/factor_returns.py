"""Factor return attribution and turnover metrics.

Factor attribution: decompose portfolio P&L into per-factor contributions.
Factor turnover: stability metric (low turnover = factor changes slowly).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def factor_return_attribution(
    factor_contributions: List[Dict[str, float]],
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """Attribute portfolio returns to individual factors.

    Parameters
    ----------
    factor_contributions : list[dict]
        Per-bar factor contribution dicts from AlphaSignal.factor_contributions.
        Each dict maps factor_name → contribution (normalized z-score weight).
    portfolio_returns : pd.Series
        Period portfolio returns.

    Returns
    -------
    pd.DataFrame
        Columns = factor names, rows = attribution metrics:
        - mean_contribution: average contribution across all bars
        - std_contribution:  variability
        - corr_with_returns: Spearman correlation with portfolio returns
        - pct_attribution:   fraction of total absolute contribution
    """
    if not factor_contributions:
        return pd.DataFrame()

    contrib_df = pd.DataFrame(factor_contributions).fillna(0.0)

    # Align lengths
    min_len = min(len(contrib_df), len(portfolio_returns))
    contrib_df = contrib_df.iloc[:min_len]
    ret = portfolio_returns.iloc[:min_len].values

    metrics: Dict[str, Dict] = {}
    for col in contrib_df.columns:
        vals = contrib_df[col].values
        mean_c = float(np.mean(vals))
        std_c = float(np.std(vals))
        if len(vals) >= 5 and np.std(vals) > 1e-10 and np.std(ret) > 1e-10:
            try:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(vals, ret)
            except Exception:
                corr = float("nan")
        else:
            corr = float("nan")

        metrics[col] = {
            "mean_contribution": round(mean_c, 6),
            "std_contribution": round(std_c, 6),
            "corr_with_returns": round(float(corr), 4) if not np.isnan(corr) else float("nan"),
        }

    result = pd.DataFrame(metrics).T

    # Percentage attribution (share of total absolute contribution)
    total_abs = result["mean_contribution"].abs().sum()
    if total_abs > 1e-10:
        result["pct_attribution"] = (result["mean_contribution"].abs() / total_abs * 100).round(2)
    else:
        result["pct_attribution"] = 0.0

    return result


def factor_turnover(factor_values: pd.Series, periods: int = 1) -> float:
    """Compute factor turnover: average absolute change normalized by std.

    Low turnover (< 0.5) means the factor is stable.
    High turnover (> 2.0) means the factor is noisy.

    Parameters
    ----------
    factor_values : pd.Series
        Time series of a single factor's raw or z-scored values.
    periods : int
        Lag for change computation (default 1 bar).

    Returns
    -------
    float
        Mean absolute change / std of the series. NaN if insufficient data.
    """
    if len(factor_values) < periods + 5:
        return float("nan")

    changes = factor_values.diff(periods).abs().dropna()
    std = factor_values.std()

    if std < 1e-10:
        return float("nan")

    return float(changes.mean() / std)
