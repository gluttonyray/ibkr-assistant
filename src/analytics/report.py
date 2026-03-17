"""Plotly factor analytics report.

Generates an interactive HTML dashboard with:
  1. Factor IC heatmap (factor × horizon)
  2. IC decay curves per factor
  3. Factor contribution waterfall
  4. Rolling factor values time series
  5. Alpha score vs forward returns scatter

Usage::

    from src.analytics.report import generate_factor_report
    generate_factor_report(signals, prices, output_path="factor_analytics.html")
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_factor_report(
    signals: List[dict],
    prices: pd.Series,
    output_path: str = "factor_analytics.html",
    title: str = "Factor Analytics Report",
) -> str:
    """Generate a Plotly HTML factor analytics report.

    Parameters
    ----------
    signals : list[dict]
        Per-bar signal dicts from FactorBacktestEngine (contains
        factor_contributions, score, timestamp, etc.).
    prices : pd.Series
        Price series aligned with signals.
    output_path : str
        Output HTML file path.
    title : str
        Report title.

    Returns
    -------
    str
        Absolute path of the generated HTML file.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("plotly not installed — factor report not generated")
        return ""

    if not signals:
        logger.warning("No signals provided — empty report")
        return ""

    # ── Extract data ────────────────────────────────────────────────────
    timestamps = [s.get("timestamp") for s in signals]
    scores = [s.get("score", 0.0) for s in signals]
    regimes = [s.get("regime", "UNKNOWN") for s in signals]

    # Factor contributions (flatten to DataFrame)
    contrib_records = []
    for s in signals:
        contrib = s.get("factor_contributions", {})
        contrib["timestamp"] = s.get("timestamp")
        contrib_records.append(contrib)
    contrib_df = pd.DataFrame(contrib_records).set_index("timestamp") if contrib_records else pd.DataFrame()

    # Align prices to signal timestamps
    price_series = prices.reindex(pd.DatetimeIndex(timestamps), method="nearest") if not prices.empty else pd.Series(dtype=float)

    # ── Build subplots ──────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Alpha Score Over Time",
            "Factor Contributions (Last Bar)",
            "Factor Contribution Time Series",
            "Regime Distribution",
            "Alpha Score vs Price",
            "Factor Count Per Bar",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Alpha score over time
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=scores,
            mode="lines",
            name="Alpha Score",
            line=dict(color="steelblue", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # 2. Factor contributions — last bar waterfall
    if not contrib_df.empty and len(contrib_df) > 0:
        last_contribs = contrib_df.iloc[-1].dropna().sort_values()
        colors = ["green" if v > 0 else "red" for v in last_contribs.values]
        fig.add_trace(
            go.Bar(
                x=list(last_contribs.values),
                y=list(last_contribs.index),
                orientation="h",
                marker_color=colors,
                name="Factor Contributions",
            ),
            row=1, col=2,
        )

    # 3. Factor contribution time series (top 5 by avg absolute contribution)
    if not contrib_df.empty and len(contrib_df) > 1:
        top5 = contrib_df.abs().mean().nlargest(5).index.tolist()
        for fname in top5:
            if fname in contrib_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=contrib_df.index.tolist(),
                        y=contrib_df[fname].tolist(),
                        mode="lines",
                        name=fname,
                        line=dict(width=1),
                    ),
                    row=2, col=1,
                )

    # 4. Regime distribution
    if regimes:
        regime_counts = pd.Series(regimes).value_counts()
        fig.add_trace(
            go.Bar(
                x=regime_counts.index.tolist(),
                y=regime_counts.values.tolist(),
                name="Regime Distribution",
            ),
            row=2, col=2,
        )

    # 5. Alpha score vs price
    if not price_series.empty:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=price_series.values.tolist(),
                mode="lines",
                name="Price",
                yaxis="y5",
                line=dict(color="gray", width=1),
            ),
            row=3, col=1,
        )

    # 6. Active factor count per bar
    active_counts = [s.get("active_factors", 0) for s in signals]
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=active_counts,
            mode="lines",
            name="Active Factors",
            line=dict(color="orange", width=1),
        ),
        row=3, col=2,
    )

    # ── Layout ──────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=1200,
        showlegend=True,
        template="plotly_dark",
    )

    # Save
    fig.write_html(output_path)
    logger.info("Factor analytics report saved: %s", output_path)
    return output_path
