"""回测绩效指标。

根据权益曲线和交易记录，
计算 Sharpe、Sortino、Calmar、回撤、交易统计和状态分层分析。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quant.portfolio.book import TradeRecord


@dataclass
class BacktestMetrics:
    """已完成回测的汇总统计。"""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_bars: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl_usd: float
    total_trades: int
    avg_holding_bars: float
    regime_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)


def compute_metrics(
    equity_curve: list[float],
    trades: list[TradeRecord],
    bars_per_year: int = 252 * 26,
    risk_free_rate: float = 0.04,
) -> BacktestMetrics:
    """计算回测绩效指标。

    Parameters
    ----------
    equity_curve : list[float]
        组合权益的时间序列。
    trades : list[TradeRecord]
        已完成的往返交易记录。
    bars_per_year : int
        一个交易年度的 K 线数量（默认：252 天 × 26 根/天，即 15 分钟级别）。
    risk_free_rate : float
        计算超额收益时使用的年化无风险利率。
    """
    # 边缘情况：数据不足
    if len(equity_curve) < 2:
        return _zero_metrics()

    eq = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(eq) / eq[:-1]

    rf_per_bar = risk_free_rate / bars_per_year
    excess = returns - rf_per_bar

    # Sharpe 比率
    sharpe = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(bars_per_year))

    # Sortino 比率（下行偏差）
    downside = excess[excess < 0]
    if len(downside) > 0:
        sortino = float(
            excess.mean() / (downside.std() + 1e-9) * np.sqrt(bars_per_year)
        )
    else:
        sortino = sharpe  # 无下行 K 线

    # 总收益与年化收益
    total_return = float(eq[-1] / eq[0] - 1.0)
    n_years = len(eq) / bars_per_year
    if n_years > 0:
        annualized_return = float((1.0 + total_return) ** (1.0 / n_years) - 1.0)
    else:
        annualized_return = 0.0

    # 最大回撤
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / peak
    max_drawdown = float(drawdown.max())

    # 最大回撤持续时间（K 线数）
    max_dd_duration = _max_drawdown_duration(eq, peak)

    # Calmar 比率
    calmar = float(annualized_return / (max_drawdown + 1e-9))

    # 交易统计
    total_trades = len(trades)
    if total_trades > 0:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = float(len(wins) / total_trades)

        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0.0
        profit_factor = float(gross_profit / (gross_loss + 1e-9))

        avg_pnl = float(sum(t.pnl_usd for t in trades) / total_trades)
        avg_holding = float(sum(t.holding_bars for t in trades) / total_trades)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_pnl = 0.0
        avg_holding = 0.0

    # 状态分层分析
    regime_breakdown = _compute_regime_breakdown(trades)

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_drawdown,
        max_drawdown_duration_bars=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl_usd=avg_pnl,
        total_trades=total_trades,
        avg_holding_bars=avg_holding,
        regime_breakdown=regime_breakdown,
    )


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _zero_metrics() -> BacktestMetrics:
    """数据不足时返回全零指标。"""
    return BacktestMetrics(
        total_return=0.0,
        annualized_return=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown=0.0,
        max_drawdown_duration_bars=0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_trade_pnl_usd=0.0,
        total_trades=0,
        avg_holding_bars=0.0,
    )


def _max_drawdown_duration(eq: np.ndarray, peak: np.ndarray) -> int:
    """计算最长回撤持续时间（K 线数）。"""
    in_drawdown = eq < peak
    max_duration = 0
    current = 0
    for v in in_drawdown:
        if v:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_duration


def _compute_regime_breakdown(trades: list[TradeRecord]) -> dict[str, dict[str, float]]:
    """按入场时状态对交易统计进行分组。"""
    if not trades:
        return {}

    from collections import defaultdict

    by_regime: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        regime = t.regime_at_entry or "UNKNOWN"
        by_regime[regime].append(t)

    breakdown: dict[str, dict[str, float]] = {}
    for regime, regime_trades in by_regime.items():
        n = len(regime_trades)
        wins = sum(1 for t in regime_trades if t.pnl_usd > 0)
        total_pnl = sum(t.pnl_usd for t in regime_trades)
        breakdown[regime] = {
            "trades": float(n),
            "win_rate": wins / n if n > 0 else 0.0,
            "total_pnl_usd": total_pnl,
            "avg_pnl_usd": total_pnl / n if n > 0 else 0.0,
        }

    return breakdown
