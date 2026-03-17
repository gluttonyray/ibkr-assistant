"""
Performance metrics for backtesting.

Computes return metrics (Total Return, CAGR, TWR, MWR), risk metrics
(Sharpe, Sortino, Calmar, Max Drawdown), and trade-level statistics.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.portfolio import Trade


@dataclass
class DrawdownInfo:
    """Maximum drawdown details."""
    max_dd_pct: float           # depth as negative percentage
    peak_date: Optional[object] = None
    trough_date: Optional[object] = None
    recovery_date: Optional[object] = None
    duration_bars: int = 0      # peak to trough
    recovery_bars: int = 0      # trough to recovery (0 if not recovered)


@dataclass
class CostSummary:
    """Aggregate cost breakdown."""
    total_commissions: float = 0.0
    total_slippage: float = 0.0
    total_sec_fees: float = 0.0
    total_taf_fees: float = 0.0
    total_finra_fees: float = 0.0
    total_margin_interest: float = 0.0
    total_costs: float = 0.0
    cost_drag_pct: float = 0.0   # total_costs / initial_capital


@dataclass
class RegimeStats:
    """Per-regime trade statistics."""
    regime: str
    num_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float


@dataclass
class Metrics:
    """Full performance metrics."""
    # Return
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    twr_pct: float = 0.0           # time-weighted return
    mwr_pct: float = 0.0           # money-weighted return (IRR)

    # Risk
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    drawdown: DrawdownInfo = field(default_factory=lambda: DrawdownInfo(0.0))

    # Trade
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    winner_loser_ratio: float = 0.0
    max_consec_wins: int = 0
    max_consec_losses: int = 0

    # Costs
    costs: CostSummary = field(default_factory=CostSummary)

    # Regime breakdown
    regime_stats: List[RegimeStats] = field(default_factory=list)

    # Monthly returns
    monthly_returns: Optional[pd.DataFrame] = None


def compute_metrics(
    trades: List[Trade],
    equity_curve: pd.Series,
    initial_capital: float,
    cash_flows: Optional[List[tuple]] = None,
    risk_free_rate: float = 0.05,
) -> Metrics:
    """Compute all performance metrics from backtest results."""
    m = Metrics()

    if equity_curve.empty:
        return m

    final = equity_curve.iloc[-1]

    # ---- Return metrics ----
    m.total_return_pct = round((final / initial_capital - 1) * 100, 4)

    # CAGR
    days = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400
    years = days / 365.25 if days > 0 else 1
    if initial_capital > 0 and final > 0 and years > 0:
        try:
            log_ratio = np.log(final / initial_capital) / years
            if abs(log_ratio) < 500:  # prevent overflow
                m.cagr_pct = round((np.exp(log_ratio) - 1) * 100, 4)
        except (OverflowError, FloatingPointError, ValueError):
            m.cagr_pct = 0.0

    # TWR (Time-Weighted Return) — geometric linking of periodic returns
    m.twr_pct = _compute_twr(equity_curve)

    # MWR (Money-Weighted Return / IRR)
    m.mwr_pct = _compute_mwr(initial_capital, final, cash_flows, years)

    # ---- Risk metrics ----
    returns = equity_curve.pct_change().dropna()

    if len(returns) > 1:
        # Sharpe
        excess = returns - risk_free_rate / 252
        std = returns.std()
        m.sharpe_ratio = round(float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0, 4)

        # Sortino
        downside = returns[returns < 0]
        down_std = downside.std() if len(downside) > 1 else 0.0
        m.sortino_ratio = round(float(excess.mean() / down_std * np.sqrt(252)) if down_std > 0 else 0.0, 4)

    # Drawdown
    m.drawdown = _compute_drawdown(equity_curve)

    # Calmar
    if m.drawdown.max_dd_pct < 0 and years > 0:
        m.calmar_ratio = round(m.cagr_pct / abs(m.drawdown.max_dd_pct), 4)

    # ---- Trade metrics ----
    m.total_trades = len(trades)
    if trades:
        _compute_trade_metrics(m, trades)

    # ---- Costs ----
    m.costs = _compute_costs(trades, initial_capital)

    # ---- Regime breakdown ----
    m.regime_stats = _compute_regime_stats(trades)

    # ---- Monthly returns ----
    m.monthly_returns = _compute_monthly_returns(equity_curve)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_twr(equity: pd.Series) -> float:
    """Time-Weighted Return via geometric linking of daily returns."""
    returns = equity.pct_change().dropna()
    if returns.empty:
        return 0.0
    twr = float(np.prod(1 + returns) - 1)
    return round(twr * 100, 4)


def _compute_mwr(
    initial_capital: float,
    final_value: float,
    cash_flows: Optional[List[tuple]],
    years: float,
) -> float:
    """Money-Weighted Return using scipy brentq (IRR solver)."""
    try:
        from scipy.optimize import brentq
    except ImportError:
        return 0.0

    if not cash_flows or years <= 0:
        # Simple return when no intermediate flows
        return round((final_value / initial_capital - 1) * 100, 4) if initial_capital > 0 else 0.0

    # Build cash flow series: initial outflow, intermediate flows, final inflow
    flows = [(-initial_capital, 0.0)]  # (amount, time_in_years)

    if cash_flows:
        t0 = cash_flows[0][0] if cash_flows else None
        for ts, amount in cash_flows:
            dt = (ts - flows[0][1]) if isinstance(ts, (int, float)) else 0
            # Approximate time fraction
            flows.append((amount, dt))

    flows.append((final_value, years))

    def npv(r: float) -> float:
        return sum(cf / (1 + r) ** t for cf, t in flows)

    try:
        irr = brentq(npv, -0.99, 10.0, maxiter=1000)
        return round(irr * 100, 4)
    except (ValueError, RuntimeError):
        # Fallback to simple return
        return round((final_value / initial_capital - 1) * 100, 4) if initial_capital > 0 else 0.0


def _compute_drawdown(equity: pd.Series) -> DrawdownInfo:
    """Calculate maximum drawdown with dates and duration."""
    if equity.empty:
        return DrawdownInfo(0.0)

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax

    if drawdown.min() == 0:
        return DrawdownInfo(0.0)

    trough_idx = drawdown.idxmin()
    trough_loc = equity.index.get_loc(trough_idx)
    peak_idx = equity.iloc[:trough_loc + 1].idxmax()
    peak_loc = equity.index.get_loc(peak_idx)

    # Recovery: first time equity >= peak after trough
    peak_value = equity[peak_idx]
    recovery_idx = None
    recovery_bars = 0
    after_trough = equity.iloc[trough_loc:]
    recovered = after_trough[after_trough >= peak_value]
    if len(recovered) > 0:
        recovery_idx = recovered.index[0]
        recovery_bars = equity.index.get_loc(recovery_idx) - trough_loc

    return DrawdownInfo(
        max_dd_pct=round(float(drawdown.min()) * 100, 4),
        peak_date=peak_idx,
        trough_date=trough_idx,
        recovery_date=recovery_idx,
        duration_bars=trough_loc - peak_loc,
        recovery_bars=recovery_bars,
    )


def _compute_trade_metrics(m: Metrics, trades: List[Trade]) -> None:
    """Populate trade-level metrics."""
    pnls = [t.net_pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    m.win_rate = round(len(winners) / len(pnls) * 100, 2) if pnls else 0.0

    total_profit = sum(winners)
    total_loss = abs(sum(losers))
    m.profit_factor = round(total_profit / total_loss, 4) if total_loss > 0 else float("inf") if total_profit > 0 else 0.0

    m.expectancy = round(sum(pnls) / len(pnls), 2) if pnls else 0.0
    m.avg_winner = round(sum(winners) / len(winners), 2) if winners else 0.0
    m.avg_loser = round(sum(losers) / len(losers), 2) if losers else 0.0
    m.winner_loser_ratio = round(m.avg_winner / abs(m.avg_loser), 4) if m.avg_loser != 0 else float("inf")

    # Consecutive wins/losses
    m.max_consec_wins = _max_consecutive(pnls, positive=True)
    m.max_consec_losses = _max_consecutive(pnls, positive=False)


def _max_consecutive(pnls: List[float], positive: bool) -> int:
    """Count max consecutive positive or non-positive PnLs."""
    max_streak = current = 0
    for p in pnls:
        if (positive and p > 0) or (not positive and p <= 0):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _compute_costs(trades: List[Trade], initial_capital: float) -> CostSummary:
    """Aggregate all trading costs."""
    cs = CostSummary()
    for t in trades:
        for costs in (t.entry_costs, t.exit_costs):
            cs.total_commissions += costs.commission
            cs.total_slippage += costs.slippage
            cs.total_sec_fees += costs.sec_fee
            cs.total_taf_fees += costs.taf_fee
            cs.total_finra_fees += costs.finra_fee
        cs.total_margin_interest += t.margin_interest

    cs.total_costs = (
        cs.total_commissions + cs.total_slippage + cs.total_sec_fees
        + cs.total_taf_fees + cs.total_finra_fees + cs.total_margin_interest
    )
    cs.cost_drag_pct = round(cs.total_costs / initial_capital * 100, 4) if initial_capital > 0 else 0.0

    # Round all
    for attr in ("total_commissions", "total_slippage", "total_sec_fees",
                 "total_taf_fees", "total_finra_fees", "total_margin_interest", "total_costs"):
        setattr(cs, attr, round(getattr(cs, attr), 2))

    return cs


def _compute_regime_stats(trades: List[Trade]) -> List[RegimeStats]:
    """Group trades by entry regime."""
    by_regime: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        by_regime[t.entry_regime or "UNKNOWN"].append(t)

    stats = []
    for regime, regime_trades in sorted(by_regime.items()):
        pnls = [t.net_pnl for t in regime_trades]
        wins = sum(1 for p in pnls if p > 0)
        stats.append(RegimeStats(
            regime=regime,
            num_trades=len(regime_trades),
            win_rate=round(wins / len(regime_trades) * 100, 2) if regime_trades else 0.0,
            avg_pnl=round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
            total_pnl=round(sum(pnls), 2),
        ))
    return stats


def _compute_monthly_returns(equity: pd.Series) -> Optional[pd.DataFrame]:
    """Compute a year × month pivot table of returns."""
    if equity.empty or len(equity) < 2:
        return None

    # Resample to month-end equity
    monthly = equity.resample("ME").last().dropna()
    if len(monthly) < 2:
        return None

    returns = monthly.pct_change().dropna() * 100  # percentages
    df = pd.DataFrame({
        "year": returns.index.year,
        "month": returns.index.month,
        "return": returns.values,
    })

    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    pivot.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]
        for m in pivot.columns
    ]

    # Add YTD column
    pivot["YTD"] = pivot.sum(axis=1)

    return pivot
