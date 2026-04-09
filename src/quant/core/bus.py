"""EventBus 协议 —— 系统事件的类型安全发布/订阅机制。

默认实现为每种事件类型维护一个 ``asyncio.Queue``。
无外部依赖；对于 7 个合约的 K 线级别粒度完全够用。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventBus(Protocol):
    """发布/订阅事件总线。"""

    async def publish(self, event: Any) -> None:
        """向所有注册了该事件类型的处理器广播 *event*。"""
        ...

    def subscribe(
        self,
        event_type: type,
        handler: Callable[..., Awaitable[None]],
    ) -> None:
        """注册 *handler*，使其在 *event_type* 类型的事件发生时被调用。"""
        ...
