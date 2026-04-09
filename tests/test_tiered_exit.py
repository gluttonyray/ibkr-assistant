import pytest
from quant.risk.tiered_exit import TieredExitManager
from quant.config.schema import RiskConfig
from quant.core.types import Position, Side
from datetime import datetime, timezone

UTC = timezone.utc


def make_position(instrument, qty=6):
    return Position(
        instrument=instrument, qty=qty,
        avg_entry_price=4000.0,
        entry_time=datetime.now(UTC),
    )


def test_three_batches(es_instrument):
    """分层退出应生成 3 批订单，数量之和等于总持仓。"""
    manager = TieredExitManager(RiskConfig())
    pos = make_position(es_instrument, qty=6)
    orders = manager.get_exit_orders(es_instrument, pos, reason="test")
    assert len(orders) == 3
    total = sum(o.qty for o in orders)
    assert total == 6


def test_all_limit_aggressive(es_instrument):
    """所有分层退出订单类型应为 LIMIT_AGGRESSIVE。"""
    manager = TieredExitManager(RiskConfig())
    pos = make_position(es_instrument, qty=3)
    orders = manager.get_exit_orders(es_instrument, pos, reason="test")
    for order in orders:
        assert order.order_type == "LIMIT_AGGRESSIVE"


def test_zero_position_returns_empty(es_instrument):
    """持仓为 0 时，应返回空订单列表。"""
    manager = TieredExitManager(RiskConfig())
    pos = make_position(es_instrument, qty=0)
    orders = manager.get_exit_orders(es_instrument, pos, reason="test")
    assert orders == []


def test_short_position_sells_buy_side(es_instrument):
    """空头仓位退出时，订单方向应为买入（BUY）。"""
    manager = TieredExitManager(RiskConfig())
    pos = make_position(es_instrument, qty=-3)  # 空头持仓
    orders = manager.get_exit_orders(es_instrument, pos, reason="test")
    for order in orders:
        assert order.side == Side.BUY  # 平空头 -> 买入
