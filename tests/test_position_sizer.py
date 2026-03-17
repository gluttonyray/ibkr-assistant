"""Tests for PositionSizer."""
from __future__ import annotations

import pytest

from src.risk.position_sizer import PositionSizer


@pytest.fixture
def sizer():
    return PositionSizer(
        max_risk_per_trade=0.02,
        max_position_pct=0.20,
        stop_atr_multiple=2.0,
        min_shares=1,
    )


class TestBasicSizing:
    def test_returns_integer(self, sizer):
        qty = sizer.compute_size(alpha_score=0.5, price=100.0, atr=1.0, equity=100_000.0)
        assert isinstance(qty, int)

    def test_positive_score_returns_nonzero(self, sizer):
        qty = sizer.compute_size(0.5, 100.0, 1.0, 100_000.0)
        assert qty > 0

    def test_zero_score_returns_zero(self, sizer):
        qty = sizer.compute_size(0.0, 100.0, 1.0, 100_000.0)
        assert qty == 0

    def test_negative_score_same_as_positive(self, sizer):
        qty_pos = sizer.compute_size(0.5, 100.0, 1.0, 100_000.0)
        qty_neg = sizer.compute_size(-0.5, 100.0, 1.0, 100_000.0)
        assert qty_pos == qty_neg  # direction is handled by signal, not sizer


class TestConvictionScaling:
    def test_higher_score_more_shares(self):
        # Use a sizer where the max_position cap doesn't bind
        sizer = PositionSizer(max_risk_per_trade=0.01, max_position_pct=0.50, stop_atr_multiple=2.0)
        qty_low = sizer.compute_size(0.2, 100.0, 1.0, 100_000.0)
        qty_high = sizer.compute_size(0.8, 100.0, 1.0, 100_000.0)
        assert qty_high > qty_low

    def test_full_conviction_gives_max_size(self):
        sizer = PositionSizer(max_risk_per_trade=0.01, max_position_pct=0.50, stop_atr_multiple=2.0)
        qty_max = sizer.compute_size(1.0, 100.0, 1.0, 100_000.0)
        qty_half = sizer.compute_size(0.5, 100.0, 1.0, 100_000.0)
        assert qty_max > qty_half


class TestRiskBudget:
    def test_size_proportional_to_equity(self, sizer):
        qty_small = sizer.compute_size(0.5, 100.0, 1.0, 50_000.0)
        qty_large = sizer.compute_size(0.5, 100.0, 1.0, 100_000.0)
        assert qty_large > qty_small

    def test_size_inversely_proportional_to_atr(self):
        # Use uncapped sizer so ATR effect isn't hidden by position cap
        sizer = PositionSizer(max_risk_per_trade=0.01, max_position_pct=0.50, stop_atr_multiple=2.0)
        qty_low_vol = sizer.compute_size(0.5, 100.0, 0.5, 100_000.0)
        qty_high_vol = sizer.compute_size(0.5, 100.0, 2.0, 100_000.0)
        assert qty_low_vol > qty_high_vol

    def test_formula_correctness(self):
        """Manual verification of sizing formula."""
        sizer = PositionSizer(
            max_risk_per_trade=0.02,
            max_position_pct=0.50,  # high cap so it doesn't bind
            stop_atr_multiple=2.0,
        )
        # risk_budget = 100_000 * 0.02 = 2000
        # risk_per_share = 2.0 * 1.0 = 2.0
        # base_shares = 2000 / 2.0 = 1000
        # scaled_shares = 1000 * 0.5 = 500
        qty = sizer.compute_size(0.5, 100.0, 1.0, 100_000.0)
        assert qty == 500


class TestPositionCap:
    def test_max_position_pct_cap_applied(self):
        sizer = PositionSizer(
            max_risk_per_trade=0.50,  # very large to force cap binding
            max_position_pct=0.10,
            stop_atr_multiple=2.0,
        )
        # max_shares = 100_000 * 0.10 / 100.0 = 100
        qty = sizer.compute_size(1.0, 100.0, 1.0, 100_000.0)
        assert qty <= 100


class TestEdgeCases:
    def test_zero_price_returns_min_shares(self, sizer):
        qty = sizer.compute_size(0.5, 0.0, 1.0, 100_000.0)
        assert qty == sizer.min_shares

    def test_zero_atr_returns_min_shares(self, sizer):
        qty = sizer.compute_size(0.5, 100.0, 0.0, 100_000.0)
        assert qty == sizer.min_shares

    def test_zero_equity_returns_min_shares(self, sizer):
        qty = sizer.compute_size(0.5, 100.0, 1.0, 0.0)
        assert qty == sizer.min_shares

    def test_min_shares_floor(self):
        sizer = PositionSizer(min_shares=5)
        # Very low conviction should still get at least min_shares
        qty = sizer.compute_size(0.001, 100.0, 1.0, 100_000.0)
        assert qty >= 5 or qty == 0  # 0 only if score < 1e-6
