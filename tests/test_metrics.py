"""Tests for backtest metrics calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.cost_model import FillCosts
from src.backtest.metrics import (
    Metrics,
    compute_metrics,
    _compute_drawdown,
    _compute_twr,
    _max_consecutive,
)
from src.backtest.portfolio import Trade


def _make_equity(values, start="2024-01-02 09:30", freq="15min"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name="equity")


def _make_trade(net_pnl, regime="UNKNOWN", side="LONG"):
    return Trade(
        symbol="TEST",
        side=side,
        entry_price=100.0,
        exit_price=100.0 + net_pnl,
        qty=100,
        entry_time=pd.Timestamp("2024-01-02 10:00"),
        exit_time=pd.Timestamp("2024-01-02 11:00"),
        gross_pnl=net_pnl + 5,
        entry_costs=FillCosts(1.0, 0.0, 0.0, 0.0, 1.0, 2.0),
        exit_costs=FillCosts(1.0, 0.08, 0.02, 0.28, 1.0, 2.38),
        margin_interest=0.0,
        net_pnl=net_pnl,
        holding_bars=10,
        entry_score=0.5,
        entry_regime=regime,
    )


class TestTotalReturn:
    def test_positive_return(self):
        equity = _make_equity([100000, 100500, 101000, 110000])
        m = compute_metrics([], equity, 100000)
        assert m.total_return_pct == pytest.approx(10.0, abs=0.01)

    def test_negative_return(self):
        equity = _make_equity([100000, 99500, 99000, 95000])
        m = compute_metrics([], equity, 100000)
        assert m.total_return_pct == pytest.approx(-5.0, abs=0.01)

    def test_flat_return(self):
        equity = _make_equity([100000, 100000, 100000])
        m = compute_metrics([], equity, 100000)
        assert m.total_return_pct == pytest.approx(0.0, abs=0.01)


class TestDrawdown:
    def test_no_drawdown(self):
        equity = _make_equity([100, 101, 102, 103, 104])
        dd = _compute_drawdown(equity)
        assert dd.max_dd_pct == 0.0

    def test_simple_drawdown(self):
        equity = _make_equity([100, 110, 100, 105])
        dd = _compute_drawdown(equity)
        # Drop from 110 to 100 = -9.09%
        assert dd.max_dd_pct == pytest.approx(-9.0909, abs=0.01)

    def test_drawdown_with_recovery(self):
        equity = _make_equity([100, 110, 100, 110, 115])
        dd = _compute_drawdown(equity)
        assert dd.recovery_date is not None


class TestTWR:
    def test_twr_matches_total_return(self):
        # Without cash flows, TWR should match total return
        equity = _make_equity([100000, 105000, 110000])
        twr = _compute_twr(equity)
        assert twr == pytest.approx(10.0, abs=0.1)


class TestTradeMetrics:
    def test_win_rate(self):
        trades = [_make_trade(100), _make_trade(50), _make_trade(-30)]
        equity = _make_equity([100000, 100100, 100150, 100120])
        m = compute_metrics(trades, equity, 100000)
        assert m.win_rate == pytest.approx(66.67, abs=0.01)

    def test_profit_factor(self):
        trades = [_make_trade(200), _make_trade(-100)]
        equity = _make_equity([100000, 100200, 100100])
        m = compute_metrics(trades, equity, 100000)
        assert m.profit_factor == pytest.approx(2.0, abs=0.01)

    def test_expectancy(self):
        trades = [_make_trade(100), _make_trade(-50)]
        equity = _make_equity([100000, 100100, 100050])
        m = compute_metrics(trades, equity, 100000)
        assert m.expectancy == pytest.approx(25.0, abs=0.01)

    def test_no_trades(self):
        equity = _make_equity([100000, 100000])
        m = compute_metrics([], equity, 100000)
        assert m.total_trades == 0
        assert m.win_rate == 0.0


class TestConsecutive:
    def test_max_wins(self):
        assert _max_consecutive([10, 20, -5, 15, 25, 30], positive=True) == 3

    def test_max_losses(self):
        assert _max_consecutive([10, -5, -10, -3, 20], positive=False) == 3

    def test_all_wins(self):
        assert _max_consecutive([10, 20, 30], positive=True) == 3

    def test_empty(self):
        assert _max_consecutive([], positive=True) == 0


class TestCosts:
    def test_cost_aggregation(self):
        trades = [_make_trade(100), _make_trade(-50)]
        equity = _make_equity([100000, 100100, 100050])
        m = compute_metrics(trades, equity, 100000)
        assert m.costs.total_commissions > 0
        assert m.costs.total_slippage > 0
        assert m.costs.total_costs > 0

    def test_cost_drag(self):
        trades = [_make_trade(100)]
        equity = _make_equity([100000, 100100])
        m = compute_metrics(trades, equity, 100000)
        assert m.costs.cost_drag_pct > 0


class TestRegimeStats:
    def test_regime_grouping(self):
        trades = [
            _make_trade(100, regime="TRENDING_CALM"),
            _make_trade(-50, regime="TRENDING_CALM"),
            _make_trade(200, regime="RANGING"),
        ]
        equity = _make_equity([100000, 100100, 100050, 100250])
        m = compute_metrics(trades, equity, 100000)
        assert len(m.regime_stats) == 2
        regimes = {rs.regime for rs in m.regime_stats}
        assert "TRENDING_CALM" in regimes
        assert "RANGING" in regimes


class TestSharpe:
    def test_positive_sharpe(self):
        # Steadily increasing equity → positive Sharpe
        values = [100000 + i * 100 for i in range(100)]
        equity = _make_equity(values)
        m = compute_metrics([], equity, 100000)
        assert m.sharpe_ratio > 0

    def test_empty_equity(self):
        equity = pd.Series(dtype=float)
        m = compute_metrics([], equity, 100000)
        assert m.sharpe_ratio == 0.0
