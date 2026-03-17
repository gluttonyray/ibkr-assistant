"""
Plotly HTML report for Walk-Forward optimization results.

Sections:
  1. Summary cards (aggregate OOS metrics, most stable params)
  2. Window detail table (IS vs OOS Sharpe, trades, return)
  3. Parameter frequency histogram
  4. OOS equity curve overlay
  5. Overfitting scatter (IS Sharpe vs OOS Sharpe)
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.backtest.optimizer import OptimizationResult, WindowResult

logger = logging.getLogger(__name__)


def generate_optimization_report(
    result: OptimizationResult,
    output_path: str = "optimization_report.html",
) -> str:
    """Generate an interactive Plotly HTML report and return the file path."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    html_parts = []

    # 1. Summary cards
    html_parts.append(_summary_html(result))

    # 2. Window detail table
    html_parts.append(_window_table_html(result.windows))

    # 3. Parameter frequency charts
    html_parts.append(_param_frequency_html(result.windows))

    # 4. OOS equity curve
    if result.oos_equity is not None and len(result.oos_equity) > 0:
        html_parts.append(_oos_equity_html(result))

    # 5. Overfitting detection scatter
    html_parts.append(_overfit_scatter_html(result.windows))

    full_html = _wrap_html(result.symbol, "\n".join(html_parts))
    path = Path(output_path)
    path.write_text(full_html, encoding="utf-8")
    logger.info("Optimization report written to %s", path.absolute())
    return str(path.absolute())


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _summary_html(r: OptimizationResult) -> str:
    """Summary cards with aggregate metrics and stable params."""
    cards = [
        ("Symbol", r.symbol),
        ("Objective", r.objective),
        ("Windows", str(r.total_windows)),
        ("Grid Size", str(r.param_grid_size)),
        ("Avg OOS Sharpe", f"{r.agg_oos_sharpe:.3f}"),
        ("Avg OOS Return", f"{r.agg_oos_return_pct:+.2f}%"),
        ("Total OOS Trades", str(r.agg_oos_trades)),
    ]
    # Stable params
    for k, v in r.stable_params.items():
        label = k.replace("_", " ").title()
        cards.append((f"Stable: {label}", f"{v}"))

    items = "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in cards
    )
    return f'<h2>Walk-Forward Optimization Summary</h2><div class="metrics-panel">{items}</div>'


def _window_table_html(windows: List[WindowResult]) -> str:
    """IS vs OOS comparison table."""
    rows = []
    for w in windows:
        sharpe_decay = ""
        if w.is_sharpe != 0:
            decay_pct = (w.oos_sharpe - w.is_sharpe) / abs(w.is_sharpe) * 100 if w.is_sharpe != 0 else 0
            color = "#c62828" if decay_pct < -50 else "#f57f17" if decay_pct < -20 else "#2e7d32"
            sharpe_decay = f'<span style="color:{color}">{decay_pct:+.0f}%</span>'

        params_str = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
        rows.append(
            f"<tr>"
            f"<td>{w.window_idx + 1}</td>"
            f"<td>{w.train_start[:10]}</td>"
            f"<td>{w.train_end[:10]}</td>"
            f"<td>{w.test_start[:10]}</td>"
            f"<td>{w.test_end[:10]}</td>"
            f"<td>{w.is_sharpe:.3f}</td>"
            f"<td>{w.is_return_pct:+.2f}%</td>"
            f"<td>{w.is_trades}</td>"
            f"<td>{w.oos_sharpe:.3f}</td>"
            f"<td>{w.oos_return_pct:+.2f}%</td>"
            f"<td>{w.oos_trades}</td>"
            f"<td>{sharpe_decay}</td>"
            f"<td style='font-size:11px'>{params_str}</td>"
            f"</tr>"
        )
    return f"""
    <h3>Window Detail</h3>
    <table>
    <thead><tr>
        <th>#</th><th>Train Start</th><th>Train End</th>
        <th>Test Start</th><th>Test End</th>
        <th>IS Sharpe</th><th>IS Return</th><th>IS Trades</th>
        <th>OOS Sharpe</th><th>OOS Return</th><th>OOS Trades</th>
        <th>Decay</th><th>Best Params</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>
    """


def _param_frequency_html(windows: List[WindowResult]) -> str:
    """Parameter frequency bar charts using Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not windows or not windows[0].best_params:
        return ""

    param_keys = list(windows[0].best_params.keys())
    n = len(param_keys)
    fig = make_subplots(rows=1, cols=n, subplot_titles=param_keys)

    for col_idx, key in enumerate(param_keys, 1):
        values = [w.best_params[key] for w in windows]
        counter = Counter(values)
        sorted_items = sorted(counter.items())
        labels = [str(x[0]) for x in sorted_items]
        counts = [x[1] for x in sorted_items]

        fig.add_trace(
            go.Bar(x=labels, y=counts, name=key, marker_color="#1976d2"),
            row=1, col=col_idx,
        )

    fig.update_layout(
        height=300,
        showlegend=False,
        title_text="Parameter Selection Frequency (times chosen as best)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return f"<h3>Parameter Stability</h3>{fig.to_html(full_html=False, include_plotlyjs=False)}"


def _oos_equity_html(result: OptimizationResult) -> str:
    """OOS equity curve chart."""
    import plotly.graph_objects as go

    eq = result.oos_equity
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        name="OOS Equity",
        line=dict(color="#1976d2", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(25,118,210,0.1)",
    ))
    fig.update_layout(
        height=350,
        title="Concatenated Out-of-Sample Equity Curve",
        template="plotly_white",
        yaxis_title="Equity ($)",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return f"<h3>OOS Equity</h3>{fig.to_html(full_html=False, include_plotlyjs=False)}"


def _overfit_scatter_html(windows: List[WindowResult]) -> str:
    """IS vs OOS Sharpe scatter plot for overfitting detection."""
    import plotly.graph_objects as go

    is_sharpes = [w.is_sharpe for w in windows]
    oos_sharpes = [w.oos_sharpe for w in windows]
    labels = [f"Window {w.window_idx + 1}" for w in windows]

    fig = go.Figure()

    # 45-degree reference line
    all_vals = is_sharpes + oos_sharpes
    min_val = min(all_vals) - 0.5 if all_vals else -1
    max_val = max(all_vals) + 0.5 if all_vals else 1
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color="#bdbdbd", dash="dash"),
        name="No Decay (IS = OOS)",
    ))

    fig.add_trace(go.Scatter(
        x=is_sharpes, y=oos_sharpes,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=10, color="#e53935"),
        name="Windows",
    ))

    fig.update_layout(
        height=400,
        title="Overfitting Detection: IS vs OOS Sharpe",
        xaxis_title="In-Sample Sharpe",
        yaxis_title="Out-of-Sample Sharpe",
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return f"<h3>Overfitting Analysis</h3>{fig.to_html(full_html=False, include_plotlyjs=False)}"


def _wrap_html(symbol: str, body: str) -> str:
    """Full HTML page wrapper."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{symbol} Walk-Forward Optimization Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 20px; background: #fafafa; color: #333; }}
h2 {{ color: #1565c0; }}
h3 {{ margin-top: 30px; color: #555; }}
.metrics-panel {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
.card {{ background: white; border-radius: 8px; padding: 12px 16px; min-width: 120px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card .label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
.card .value {{ font-size: 18px; font-weight: 600; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
th, td {{ padding: 6px 10px; border: 1px solid #e0e0e0; text-align: right; }}
th {{ background: #f5f5f5; font-weight: 600; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
