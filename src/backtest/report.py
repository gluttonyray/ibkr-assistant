"""
Plotly interactive HTML report generator.

Produces a single self-contained HTML file with:
  1. Metrics summary panel
  2. Price chart with trade markers + regime background
  3. Equity curve + high-water mark
  4. Drawdown chart
  5. Signal score line
  6. Volume bars
  7. Monthly returns heatmap
  8. Trade detail table
  9. Regime analysis chart
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.metrics import Metrics
from src.backtest.portfolio import Trade

logger = logging.getLogger(__name__)

# Regime colors
REGIME_COLORS = {
    "TRENDING_CALM": "rgba(76,175,80,0.08)",
    "TRENDING_VOLATILE": "rgba(255,152,0,0.08)",
    "RANGING": "rgba(33,150,243,0.08)",
    "CHOPPY_VOLATILE": "rgba(244,67,54,0.08)",
    "UNKNOWN": "rgba(158,158,158,0.05)",
}


def generate_report(
    result: BacktestResult,
    metrics: Metrics,
    output_path: str = "backtest_report.html",
) -> str:
    """Generate an interactive Plotly HTML report and return the file path."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly is required for report generation: pip install plotly")

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.40, 0.20, 0.10, 0.15, 0.15],
        subplot_titles=[
            f"{result.symbol} — Price & Trades",
            "Equity Curve",
            "Drawdown",
            "Signal Score",
            "Volume",
        ],
    )

    bars = result.bars
    equity = result.equity_curve
    signals_df = pd.DataFrame(result.signals) if result.signals else pd.DataFrame()

    # ── Row 1: Candlestick + trade markers + regime background ──
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Regime background bands
    if not signals_df.empty and "regime" in signals_df.columns:
        _add_regime_bands(fig, signals_df, row=1)

    # Trade markers
    _add_trade_markers(fig, result.trades, row=1)

    # ── Row 2: Equity curve + high-water mark ──
    if not equity.empty:
        fig.add_trace(
            go.Scatter(
                x=equity.index, y=equity.values,
                name="Equity",
                line=dict(color="#1976d2", width=1.5),
            ),
            row=2, col=1,
        )
        hwm = equity.cummax()
        fig.add_trace(
            go.Scatter(
                x=hwm.index, y=hwm.values,
                name="High Water Mark",
                line=dict(color="#bdbdbd", width=1, dash="dot"),
            ),
            row=2, col=1,
        )

    # ── Row 3: Drawdown ──
    if not equity.empty:
        cummax = equity.cummax()
        dd = (equity - cummax) / cummax * 100
        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd.values,
                name="Drawdown %",
                fill="tozeroy",
                line=dict(color="#ef5350", width=1),
                fillcolor="rgba(239,83,80,0.3)",
            ),
            row=3, col=1,
        )

    # ── Row 4: Signal score ──
    if not signals_df.empty:
        fig.add_trace(
            go.Scatter(
                x=signals_df["timestamp"],
                y=signals_df["score"],
                name="Score",
                line=dict(color="#7e57c2", width=1),
            ),
            row=4, col=1,
        )
        # Threshold lines
        fig.add_hline(y=0.4, line_dash="dash", line_color="green",
                      annotation_text="Buy", row=4, col=1)
        fig.add_hline(y=-0.4, line_dash="dash", line_color="red",
                      annotation_text="Sell", row=4, col=1)

    # ── Row 5: Volume ──
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(bars["close"], bars["open"])]
    fig.add_trace(
        go.Bar(
            x=bars.index, y=bars["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
        ),
        row=5, col=1,
    )

    # Layout
    fig.update_layout(
        height=1400,
        title=dict(
            text=_build_title_html(result, metrics),
            font=dict(size=14),
        ),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=60, r=30, t=120, b=30),
    )

    # Build full HTML with metrics panel, heatmap, trade table, regime chart
    html_parts = []
    html_parts.append(_metrics_panel_html(metrics))
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append(_monthly_heatmap_html(metrics))
    html_parts.append(_trade_table_html(result.trades))
    html_parts.append(_regime_chart_html(metrics))

    full_html = _wrap_html(result.symbol, "\n".join(html_parts))

    path = Path(output_path)
    path.write_text(full_html, encoding="utf-8")
    logger.info("Report written to %s", path.absolute())
    return str(path.absolute())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_regime_bands(fig, signals_df: pd.DataFrame, row: int) -> None:
    """Add colored background rectangles for regime changes."""
    import plotly.graph_objects as go
    regimes = signals_df[["timestamp", "regime"]].copy()
    regimes["group"] = (regimes["regime"] != regimes["regime"].shift()).cumsum()

    for _, grp in regimes.groupby("group"):
        regime = grp["regime"].iloc[0]
        color = REGIME_COLORS.get(regime, REGIME_COLORS["UNKNOWN"])
        fig.add_vrect(
            x0=grp["timestamp"].iloc[0],
            x1=grp["timestamp"].iloc[-1],
            fillcolor=color,
            layer="below",
            line_width=0,
            row=row, col=1,
        )


def _add_trade_markers(fig, trades: List[Trade], row: int) -> None:
    """Add entry/exit markers to the price chart."""
    import plotly.graph_objects as go
    for t in trades:
        # Entry
        entry_color = "green" if t.side == "LONG" else "red"
        entry_symbol = "triangle-up" if t.side == "LONG" else "triangle-down"
        fig.add_trace(
            go.Scatter(
                x=[t.entry_time], y=[t.entry_price],
                mode="markers",
                marker=dict(symbol=entry_symbol, size=10, color=entry_color),
                name=f"{t.side} Entry",
                hovertext=f"{t.side} @ {t.entry_price:.2f}<br>Score: {t.entry_score:.3f}<br>Regime: {t.entry_regime}",
                hoverinfo="text",
            ),
            row=row, col=1,
        )
        # Exit
        exit_color = "#ff9800" if t.net_pnl > 0 else "#9e9e9e"
        fig.add_trace(
            go.Scatter(
                x=[t.exit_time], y=[t.exit_price],
                mode="markers",
                marker=dict(symbol="x", size=8, color=exit_color),
                name=f"Exit",
                hovertext=f"Exit @ {t.exit_price:.2f}<br>PnL: ${t.net_pnl:.2f}",
                hoverinfo="text",
            ),
            row=row, col=1,
        )


def _build_title_html(result: BacktestResult, metrics: Metrics) -> str:
    """Build a compact title string."""
    return (
        f"{result.symbol} Backtest | "
        f"Return: {metrics.total_return_pct:+.2f}% | "
        f"Sharpe: {metrics.sharpe_ratio:.2f} | "
        f"MaxDD: {metrics.drawdown.max_dd_pct:.2f}% | "
        f"Trades: {metrics.total_trades} | "
        f"Win Rate: {metrics.win_rate:.1f}%"
    )


def _metrics_panel_html(m: Metrics) -> str:
    """HTML metrics summary cards."""
    cards = [
        ("Total Return", f"{m.total_return_pct:+.2f}%"),
        ("CAGR", f"{m.cagr_pct:+.2f}%"),
        ("TWR", f"{m.twr_pct:+.2f}%"),
        ("MWR (IRR)", f"{m.mwr_pct:+.2f}%"),
        ("Sharpe", f"{m.sharpe_ratio:.2f}"),
        ("Sortino", f"{m.sortino_ratio:.2f}"),
        ("Calmar", f"{m.calmar_ratio:.2f}"),
        ("Max Drawdown", f"{m.drawdown.max_dd_pct:.2f}%"),
        ("Win Rate", f"{m.win_rate:.1f}%"),
        ("Profit Factor", f"{m.profit_factor:.2f}"),
        ("Expectancy", f"${m.expectancy:.2f}"),
        ("Total Trades", f"{m.total_trades}"),
        ("Total Costs", f"${m.costs.total_costs:.2f}"),
        ("Cost Drag", f"{m.costs.cost_drag_pct:.2f}%"),
    ]
    items = "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in cards
    )
    return f'<div class="metrics-panel">{items}</div>'


def _monthly_heatmap_html(m: Metrics) -> str:
    """Render monthly returns as an HTML heatmap table."""
    if m.monthly_returns is None or m.monthly_returns.empty:
        return ""

    df = m.monthly_returns
    rows_html = []
    for year in df.index:
        cells = f"<td><strong>{year}</strong></td>"
        for col in df.columns:
            val = df.loc[year, col]
            if pd.isna(val):
                cells += "<td>—</td>"
            else:
                color = "#c8e6c9" if val > 0 else "#ffcdd2" if val < 0 else "#fff"
                cells += f'<td style="background:{color}">{val:+.1f}%</td>'
        rows_html.append(f"<tr>{cells}</tr>")

    headers = "<th>Year</th>" + "".join(f"<th>{c}</th>" for c in df.columns)

    return f"""
    <h3>Monthly Returns (%)</h3>
    <table class="heatmap"><thead><tr>{headers}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody></table>
    """


def _trade_table_html(trades: List[Trade]) -> str:
    """Render sortable trade details table."""
    if not trades:
        return "<h3>No Trades</h3>"

    rows = []
    for i, t in enumerate(trades, 1):
        pnl_class = "win" if t.net_pnl > 0 else "loss"
        total_fees = t.entry_costs.total + t.exit_costs.total + t.margin_interest
        rows.append(
            f"<tr class='{pnl_class}'>"
            f"<td>{i}</td>"
            f"<td>{t.side}</td>"
            f"<td>{t.entry_time}</td>"
            f"<td>${t.entry_price:.2f}</td>"
            f"<td>{t.exit_time}</td>"
            f"<td>${t.exit_price:.2f}</td>"
            f"<td>{t.qty}</td>"
            f"<td>${t.gross_pnl:.2f}</td>"
            f"<td>${total_fees:.2f}</td>"
            f"<td><strong>${t.net_pnl:.2f}</strong></td>"
            f"<td>{t.holding_bars}</td>"
            f"<td>{t.entry_score:.3f}</td>"
            f"<td>{t.entry_regime}</td>"
            f"</tr>"
        )

    return f"""
    <h3>Trade Details</h3>
    <table class="trades">
    <thead><tr>
        <th>#</th><th>Side</th><th>Entry Time</th><th>Entry $</th>
        <th>Exit Time</th><th>Exit $</th><th>Qty</th>
        <th>Gross PnL</th><th>Fees</th><th>Net PnL</th>
        <th>Bars</th><th>Score</th><th>Regime</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>
    """


def _regime_chart_html(m: Metrics) -> str:
    """Render regime stats as a simple HTML bar chart."""
    if not m.regime_stats:
        return ""

    bars = []
    for rs in m.regime_stats:
        width = min(abs(rs.avg_pnl) * 2, 200)
        color = "#4caf50" if rs.avg_pnl > 0 else "#ef5350"
        bars.append(
            f'<div class="regime-row">'
            f'<span class="regime-label">{rs.regime}</span>'
            f'<div class="regime-bar" style="width:{width}px;background:{color}">'
            f'${rs.avg_pnl:.2f}</div>'
            f'<span class="regime-meta">{rs.num_trades} trades | WR: {rs.win_rate:.0f}%</span>'
            f'</div>'
        )

    return f'<h3>Regime Analysis</h3><div class="regime-chart">{"".join(bars)}</div>'


def _wrap_html(symbol: str, body: str) -> str:
    """Wrap content in a full HTML page with CSS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{symbol} Backtest Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 20px; background: #fafafa; color: #333; }}
.metrics-panel {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
.card {{ background: white; border-radius: 8px; padding: 12px 16px; min-width: 120px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card .label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
.card .value {{ font-size: 18px; font-weight: 600; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
th, td {{ padding: 6px 10px; border: 1px solid #e0e0e0; text-align: right; }}
th {{ background: #f5f5f5; font-weight: 600; }}
.heatmap td {{ text-align: center; min-width: 60px; }}
.trades .win td:nth-child(10) {{ color: #2e7d32; }}
.trades .loss td:nth-child(10) {{ color: #c62828; }}
h3 {{ margin-top: 30px; color: #555; }}
.regime-chart {{ margin: 10px 0; }}
.regime-row {{ display: flex; align-items: center; margin: 6px 0; }}
.regime-label {{ width: 160px; font-weight: 600; font-size: 13px; }}
.regime-bar {{ height: 24px; border-radius: 4px; color: white; font-size: 12px;
               display: flex; align-items: center; padding: 0 8px; min-width: 60px; }}
.regime-meta {{ margin-left: 12px; font-size: 12px; color: #888; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
