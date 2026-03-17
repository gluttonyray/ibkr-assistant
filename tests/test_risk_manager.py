"""Tests for RiskManager: stop-loss, take-profit, trailing stop, circuit breaker."""
from __future__ import annotations

import pytest

from src.risk.risk_manager import PositionInfo, RiskManager


@pytest.fixture
def rm():
    return RiskManager(
        stop_loss_atr=2.0,
        take_profit_atr=4.0,
        trailing_stop_atr=1.5,
        max_drawdown_pct=0.10,
    )


def _long_pos(entry=100.0, current=100.0, atr=1.0, high=100.0, low=99.0) -> PositionInfo:
    return PositionInfo(
        entry_price=entry,
        current_price=current,
        direction=1,
        atr_at_entry=atr,
        highest_price=high,
        lowest_price=low,
    )


def _short_pos(entry=100.0, current=100.0, atr=1.0, high=101.0, low=100.0) -> PositionInfo:
    return PositionInfo(
        entry_price=entry,
        current_price=current,
        direction=-1,
        atr_at_entry=atr,
        highest_price=high,
        lowest_price=low,
    )


# ──────────────────────────────────────────────
# No exit — healthy positions
# ──────────────────────────────────────────────

class TestNoExit:
    def test_no_exit_when_neutral(self, rm):
        pos = _long_pos(entry=100.0, current=100.5, high=100.5)
        assert rm.check_exit(pos) is None

    def test_no_exit_short_neutral(self, rm):
        pos = _short_pos(entry=100.0, current=99.5, low=99.5)
        assert rm.check_exit(pos) is None

    def test_zero_atr_returns_none(self, rm):
        pos = _long_pos(entry=100.0, current=105.0, atr=0.0)
        assert rm.check_exit(pos) is None


# ──────────────────────────────────────────────
# Stop-loss
# ──────────────────────────────────────────────

class TestStopLoss:
    def test_long_stop_loss_triggers(self, rm):
        # ATR=1, stop=2 → stop at 98. current=97.5 → stop loss
        pos = _long_pos(entry=100.0, current=97.5, atr=1.0, high=100.5)
        assert rm.check_exit(pos) == "STOP_LOSS"

    def test_long_stop_loss_exact_boundary(self, rm):
        # loss = 2.0 ATR → exactly at stop
        pos = _long_pos(entry=100.0, current=98.0, atr=1.0, high=100.0)
        assert rm.check_exit(pos) == "STOP_LOSS"

    def test_long_stop_loss_not_triggered_above(self, rm):
        # loss = 1.9 ATR → just above stop. Use high=98.2 so trail_level=96.7 < 98.1
        pos = _long_pos(entry=100.0, current=98.1, atr=1.0, high=98.2)
        assert rm.check_exit(pos) is None

    def test_short_stop_loss_triggers(self, rm):
        # Short: entry=100, ATR=1, stop=2 → stop at 102. current=102.5
        pos = _short_pos(entry=100.0, current=102.5, atr=1.0, low=99.5)
        assert rm.check_exit(pos) == "STOP_LOSS"


# ──────────────────────────────────────────────
# Take-profit
# ──────────────────────────────────────────────

class TestTakeProfit:
    def test_long_take_profit_triggers(self, rm):
        # ATR=1, TP=4 → TP at 104. current=104.5
        pos = _long_pos(entry=100.0, current=104.5, atr=1.0, high=104.5)
        assert rm.check_exit(pos) == "TAKE_PROFIT"

    def test_long_take_profit_exact(self, rm):
        pos = _long_pos(entry=100.0, current=104.0, atr=1.0, high=104.0)
        assert rm.check_exit(pos) == "TAKE_PROFIT"

    def test_long_take_profit_not_triggered(self, rm):
        pos = _long_pos(entry=100.0, current=103.9, atr=1.0, high=103.9)
        assert rm.check_exit(pos) is None

    def test_short_take_profit_triggers(self, rm):
        # Short: profit when price falls. TP at 96. current=95.5
        pos = _short_pos(entry=100.0, current=95.5, atr=1.0, low=95.5)
        assert rm.check_exit(pos) == "TAKE_PROFIT"


# ──────────────────────────────────────────────
# Trailing stop
# ──────────────────────────────────────────────

class TestTrailingStop:
    def test_long_trailing_stop_triggers(self, rm):
        # Long: high=105, ATR=1, trail=1.5 → trail_level=103.5. current=103.0
        pos = _long_pos(entry=100.0, current=103.0, atr=1.0, high=105.0)
        assert rm.check_exit(pos) == "TRAILING_STOP"

    def test_long_trailing_stop_not_triggered(self, rm):
        # Long: high=105, ATR=1, trail=1.5 → trail_level=103.5. current=103.8
        # TP at 104.0 not triggered (pnl_atr=3.8 < 4.0). Trail not triggered (103.8 > 103.5).
        pos = _long_pos(entry=100.0, current=103.8, atr=1.0, high=105.0)
        assert rm.check_exit(pos) is None

    def test_short_trailing_stop_triggers(self, rm):
        # Short: low=95, ATR=1, trail=1.5 → trail_level=96.5. current=97.0
        pos = _short_pos(entry=100.0, current=97.0, atr=1.0, low=95.0)
        assert rm.check_exit(pos) == "TRAILING_STOP"

    def test_short_trailing_stop_not_triggered(self, rm):
        # Short: low=95, ATR=1, trail_level=96.5, current=96.2
        # TP at 96.0 not triggered (pnl_atr=3.8 < 4.0). Trail not triggered (96.2 < 96.5).
        pos = _short_pos(entry=100.0, current=96.2, atr=1.0, low=95.0)
        assert rm.check_exit(pos) is None


# ──────────────────────────────────────────────
# Priority: TP > SL > trailing
# ──────────────────────────────────────────────

class TestPriority:
    def test_take_profit_takes_priority_over_trailing(self, rm):
        # Both TP and trailing triggered: TP should win
        pos = _long_pos(entry=100.0, current=104.5, atr=1.0, high=104.5)
        result = rm.check_exit(pos)
        assert result == "TAKE_PROFIT"


# ──────────────────────────────────────────────
# Portfolio circuit breaker
# ──────────────────────────────────────────────

class TestPortfolioRisk:
    def test_no_trigger_within_drawdown(self, rm):
        # 5% drawdown, limit=10%
        assert rm.check_portfolio_risk(95_000.0, 100_000.0) is False

    def test_trigger_at_exact_limit(self, rm):
        assert rm.check_portfolio_risk(90_000.0, 100_000.0) is True

    def test_trigger_above_limit(self, rm):
        assert rm.check_portfolio_risk(85_000.0, 100_000.0) is True

    def test_no_trigger_with_zero_peak(self, rm):
        assert rm.check_portfolio_risk(100_000.0, 0.0) is False

    def test_no_drawdown_no_trigger(self, rm):
        assert rm.check_portfolio_risk(100_000.0, 100_000.0) is False

    def test_custom_threshold(self):
        rm_strict = RiskManager(max_drawdown_pct=0.05)
        # 6% drawdown (94k/100k) > 5% limit → should trigger
        assert rm_strict.check_portfolio_risk(94_000.0, 100_000.0) is True
        # 4% drawdown (96k/100k) < 5% limit → should NOT trigger
        assert rm_strict.check_portfolio_risk(96_000.0, 100_000.0) is False
        # 10% drawdown → definitely triggers
        assert rm_strict.check_portfolio_risk(90_000.0, 100_000.0) is True
