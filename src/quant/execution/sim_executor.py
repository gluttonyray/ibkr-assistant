from __future__ import annotations
import logging
from collections import deque
from datetime import datetime, timezone
from quant.core.types import Fill, Instrument, Side
from quant.core.events import OrderEvent

logger = logging.getLogger(__name__)

class SimExecutor:
    """回测用模拟执行器，实现 Executor Protocol。

    pending_orders 在当前 bar 收集，下一 bar 开盘时由 BacktestEngine 处理成交。
    SimExecutor 本身不做成交决策，只存储订单供引擎读取。
    """

    def __init__(self) -> None:
        self._pending: deque[OrderEvent] = deque()
        self._fills: deque[Fill] = deque()

    def queue_order(self, order: OrderEvent) -> None:
        """引擎调用此方法排队订单。"""
        self._pending.append(order)

    def pop_pending(self) -> list[OrderEvent]:
        """引擎每 bar 开始时调用，取出所有待处理订单。"""
        orders = list(self._pending)
        self._pending.clear()
        return orders

    async def submit(
        self,
        instrument: Instrument,
        side: Side,
        qty: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
    ) -> str:
        order_id = f"sim-{instrument.symbol}-{datetime.now(timezone.utc).timestamp():.0f}"
        order = OrderEvent(
            instrument=instrument,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            timestamp=datetime.now(timezone.utc),
            order_id=order_id,
        )
        self._pending.append(order)
        return order_id

    async def cancel(self, order_id: str) -> bool:
        new_pending: deque[OrderEvent] = deque()
        removed = False
        for o in self._pending:
            if o.order_id == order_id and not removed:
                removed = True  # 只移除第一个匹配项
            else:
                new_pending.append(o)
        self._pending = new_pending
        return removed

    async def get_fills(self) -> list[Fill]:
        fills = list(self._fills)
        self._fills.clear()
        return fills
