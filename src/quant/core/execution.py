"""Executor 协议 —— 订单管理抽象层。

实盘：通过 ib_insync 向 IBKR TWS/Gateway 提交订单。
回测：SimExecutor，使用下一根 K 线的开盘价模拟成交。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from quant.core.types import Fill, Instrument, Side


@runtime_checkable
class Executor(Protocol):
    """提交、取消并查询订单成交。"""

    async def submit(
        self,
        instrument: Instrument,
        side: Side,
        qty: int,
        order_type: str = "LIMIT",
        limit_price: float | None = None,
    ) -> str:
        """提交订单；返回订单 ID 字符串。"""
        ...

    async def cancel(self, order_id: str) -> bool:
        """取消待成交订单；成功取消返回 True。"""
        ...

    async def get_fills(self) -> list[Fill]:
        """返回自上次调用以来收到的所有成交记录（清空缓冲区）。"""
        ...
