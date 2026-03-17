"""Tests for IBKR cost model."""
from __future__ import annotations

import pytest
from src.backtest.cost_model import CostModel, FillCosts


class TestCommission:
    def test_basic_commission(self):
        cm = CostModel(slippage_bps=0)
        costs = cm.fill_cost(100.0, 200, "BUY")
        # 200 * 0.005 = 1.00 (equals min)
        assert costs.commission == 1.0

    def test_commission_above_min(self):
        cm = CostModel(slippage_bps=0)
        costs = cm.fill_cost(100.0, 500, "BUY")
        # 500 * 0.005 = 2.50
        assert costs.commission == 2.5

    def test_commission_min_floor(self):
        cm = CostModel(slippage_bps=0)
        costs = cm.fill_cost(100.0, 10, "BUY")
        # 10 * 0.005 = 0.05, but min = 1.00
        assert costs.commission == 1.0

    def test_commission_max_cap(self):
        cm = CostModel(slippage_bps=0)
        # 100 shares @ $0.50 = $50 notional, 1% = $0.50
        # raw comm = 100 * 0.005 = 0.50
        # max = 0.01 * 50 = 0.50
        costs = cm.fill_cost(0.50, 100, "BUY")
        assert costs.commission == 1.0  # still hits min floor

    def test_commission_max_cap_large(self):
        cm = CostModel(slippage_bps=0)
        # Penny stock: 10000 shares @ $0.10 = $1000, 1% = $10
        # raw = 10000 * 0.005 = $50 → capped at $10
        costs = cm.fill_cost(0.10, 10000, "BUY")
        assert costs.commission == 10.0


class TestRegulatoryFees:
    def test_buy_no_reg_fees(self):
        cm = CostModel(slippage_bps=0)
        costs = cm.fill_cost(100.0, 100, "BUY")
        assert costs.sec_fee == 0.0
        assert costs.taf_fee == 0.0
        assert costs.finra_fee == 0.0

    def test_sell_has_reg_fees(self):
        cm = CostModel(slippage_bps=0)
        costs = cm.fill_cost(100.0, 100, "SELL")
        assert costs.sec_fee > 0
        assert costs.taf_fee > 0
        assert costs.finra_fee > 0

    def test_sec_fee_calculation(self):
        cm = CostModel(slippage_bps=0)
        costs = cm.fill_cost(100.0, 100, "SELL")
        # notional = 10000, SEC rate = 8/1M
        expected_sec = round(10000 * 8 / 1_000_000, 2)
        assert costs.sec_fee == expected_sec

    def test_taf_fee_calculation(self):
        cm = CostModel(slippage_bps=0)
        costs = cm.fill_cost(100.0, 100, "SELL")
        expected_taf = 100 * 0.000166
        assert abs(costs.taf_fee - expected_taf) < 0.001


class TestSlippage:
    def test_slippage_calculation(self):
        cm = CostModel(slippage_bps=5)
        costs = cm.fill_cost(100.0, 100, "BUY")
        # notional = 10000, 5bps = 0.05% = $5.00
        assert costs.slippage == 5.0

    def test_zero_slippage(self):
        cm = CostModel(slippage_bps=0)
        costs = cm.fill_cost(100.0, 100, "BUY")
        assert costs.slippage == 0.0

    def test_total_includes_slippage(self):
        cm = CostModel(slippage_bps=10)
        costs = cm.fill_cost(100.0, 100, "BUY")
        assert costs.slippage > 0
        assert costs.total >= costs.commission + costs.slippage


class TestMarginInterest:
    def test_basic_margin(self):
        cm = CostModel(margin_rate=0.0683)
        interest = cm.margin_interest(10000, 30)
        # 10000 * 0.0683 / 360 * 30 = 56.917
        assert abs(interest - 56.9167) < 0.01

    def test_zero_days(self):
        cm = CostModel()
        assert cm.margin_interest(10000, 0) == 0.0

    def test_zero_notional(self):
        cm = CostModel()
        assert cm.margin_interest(0, 30) == 0.0


class TestLimitOrderFill:
    def test_buy_limit_fills(self):
        cm = CostModel()
        # Buy limit at 99, next bar low = 98.5 → fills
        assert cm.limit_order_fills(99.0, "BUY", 101.0, 98.5) is True

    def test_buy_limit_no_fill(self):
        cm = CostModel()
        # Buy limit at 99, next bar low = 99.5 → no fill
        assert cm.limit_order_fills(99.0, "BUY", 101.0, 99.5) is False

    def test_sell_limit_fills(self):
        cm = CostModel()
        # Sell limit at 101, next bar high = 101.5 → fills
        assert cm.limit_order_fills(101.0, "SELL", 101.5, 99.0) is True

    def test_sell_limit_no_fill(self):
        cm = CostModel()
        # Sell limit at 101, next bar high = 100.5 → no fill
        assert cm.limit_order_fills(101.0, "SELL", 100.5, 99.0) is False


class TestFillCosts:
    def test_zero_factory(self):
        z = FillCosts.zero()
        assert z.total == 0.0
        assert z.commission == 0.0
