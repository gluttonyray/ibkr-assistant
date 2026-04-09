"""DataFeed 协议 —— 行情数据源抽象层。

实盘：通过 ib_insync 流式接收 IBKR TWS/Gateway 数据。
回测：历史 CSV/Parquet 回放。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from quant.core.types import Bar, Instrument


@runtime_checkable
class DataFeed(Protocol):
    """行情数据提供方。"""

    async def start(self, instruments: list[Instrument]) -> None:
        """连接并开始为 *instruments* 流式推送 K 线数据。"""
        ...

    async def stop(self) -> None:
        """断开连接并释放资源。"""
        ...

    def get_window(self, instrument: Instrument) -> list[Bar] | None:
        """返回 *instrument* 的当前滚动窗口，若不存在则返回 None。

        窗口长度由实现决定（通常为
        ``config.backtest.window_size`` 根 K 线）。
        """
        ...

    def on_bar(
        self,
        callback: Callable[[Instrument, Bar], Awaitable[None]],
    ) -> None:
        """注册回调，在任意合约产生新 K 线时被调用。"""
        ...
