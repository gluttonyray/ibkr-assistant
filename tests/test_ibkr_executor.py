import pytest
from quant.execution.ibkr_executor import IBKRExecutor
from quant.config.schema import IBKRConfig
from quant.instrument.registry import InstrumentRegistry
from quant.core.types import Side


def test_import_and_init(es_instrument):
    """IBKRExecutor 应可在无实时 TWS 连接的情况下正常实例化。"""
    registry = InstrumentRegistry({"ES": es_instrument})
    executor = IBKRExecutor(IBKRConfig(), registry)
    assert executor._ib is None
    assert executor._daily_order_count == 0


@pytest.mark.asyncio
async def test_submit_without_connect_raises(es_instrument):
    """未连接时调用 submit 应抛出 RuntimeError。"""
    registry = InstrumentRegistry({"ES": es_instrument})
    executor = IBKRExecutor(IBKRConfig(), registry)
    with pytest.raises(RuntimeError, match="Not connected"):
        await executor.submit(es_instrument, Side.BUY, 1, "MARKET")


@pytest.mark.asyncio
async def test_cancel_without_connect_returns_false(es_instrument):
    """未连接时调用 cancel 应返回 False。"""
    registry = InstrumentRegistry({"ES": es_instrument})
    executor = IBKRExecutor(IBKRConfig(), registry)
    result = await executor.cancel("fake-order-id")
    assert result is False


@pytest.mark.asyncio
async def test_get_fills_empty(es_instrument):
    """未连接时 get_fills 应返回空列表。"""
    registry = InstrumentRegistry({"ES": es_instrument})
    executor = IBKRExecutor(IBKRConfig(), registry)
    fills = await executor.get_fills()
    assert fills == []


def test_reset_daily(es_instrument):
    """reset_daily 应将每日订单计数归零。"""
    registry = InstrumentRegistry({"ES": es_instrument})
    executor = IBKRExecutor(IBKRConfig(), registry)
    executor._daily_order_count = 10
    executor.reset_daily()
    assert executor._daily_order_count == 0


@pytest.mark.asyncio
async def test_disconnect_when_not_connected(es_instrument):
    """未连接时调用 disconnect 不应抛出异常（应为空操作）。"""
    registry = InstrumentRegistry({"ES": es_instrument})
    executor = IBKRExecutor(IBKRConfig(), registry)
    await executor.disconnect()  # 应为空操作
    assert executor._ib is None
