import pytest
import asyncio
from datetime import datetime, timezone
from quant.execution.sim_executor import SimExecutor
from quant.core.types import Side
from quant.core.events import OrderEvent

UTC = timezone.utc


def test_queue_and_pop(es_instrument):
    """入队一条订单后，pop_pending 应返回该订单，再次 pop 应为空。"""
    executor = SimExecutor()
    order = OrderEvent(
        instrument=es_instrument, side=Side.BUY, qty=2,
        order_type="MARKET", timestamp=datetime.now(UTC),
    )
    executor.queue_order(order)
    pending = executor.pop_pending()
    assert len(pending) == 1
    assert pending[0].instrument.symbol == "ES"
    assert pending[0].qty == 2
    # 再次 pop 应为空
    assert executor.pop_pending() == []


def test_pop_clears_queue(es_instrument):
    """pop_pending 应一次性清空所有待处理订单。"""
    executor = SimExecutor()
    for _ in range(3):
        executor.queue_order(OrderEvent(
            instrument=es_instrument, side=Side.BUY, qty=1,
            order_type="MARKET",
        ))
    assert len(executor.pop_pending()) == 3
    assert len(executor.pop_pending()) == 0


@pytest.mark.asyncio
async def test_submit_returns_order_id(es_instrument):
    """submit 应返回以 'sim-ES-' 开头的订单 ID，且订单应进入待处理队列。"""
    executor = SimExecutor()
    order_id = await executor.submit(es_instrument, Side.BUY, 1, "MARKET")
    assert order_id.startswith("sim-ES-")
    # 订单应已进入待处理队列
    pending = executor.pop_pending()
    assert len(pending) == 1
    assert pending[0].order_id == order_id


@pytest.mark.asyncio
async def test_cancel_removes_order(es_instrument):
    """cancel 应从待处理队列中移除指定订单，其余订单保持不变。"""
    executor = SimExecutor()
    oid1 = await executor.submit(es_instrument, Side.BUY, 1)
    # 通过 queue_order 显式指定 order_id，确保 ID 唯一
    order2 = OrderEvent(
        instrument=es_instrument, side=Side.SELL, qty=2,
        order_type="MARKET", timestamp=datetime.now(UTC),
        order_id="manual-order-2",
    )
    executor.queue_order(order2)
    cancelled = await executor.cancel(oid1)
    assert cancelled is True
    pending = executor.pop_pending()
    assert len(pending) == 1
    assert pending[0].order_id == "manual-order-2"


@pytest.mark.asyncio
async def test_cancel_nonexistent_returns_false(es_instrument):
    """取消不存在的订单应返回 False。"""
    executor = SimExecutor()
    result = await executor.cancel("nonexistent-id")
    assert result is False


@pytest.mark.asyncio
async def test_get_fills_empty():
    """get_fills 在模拟器中应始终返回空列表。"""
    executor = SimExecutor()
    fills = await executor.get_fills()
    assert fills == []
