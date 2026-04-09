"""TradingClock 协议 —— 回测与实盘的时间抽象层。

实盘实现返回系统挂钟时间。回测实现根据 K 线数据推进时间，
确保所有时间相关逻辑（时段检查、持仓周期、冷却计时）
在回测和实盘中行为完全一致。
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from quant.core.types import Instrument


@runtime_checkable
class TradingClock(Protocol):
    """时间源抽象。"""

    def now(self) -> datetime:
        """当前 UTC 时间。"""
        ...

    def is_trading_hours(self, instrument: Instrument) -> bool:
        """判断 *instrument* 当前是否处于交易时段。"""
        ...

    async def wait_next_bar(self, instrument: Instrument) -> None:
        """阻塞直到 *instrument* 的下一根 K 线边界。

        回测模式下立即返回（时间由数据驱动）。
        实盘模式下休眠至下一根 K 线收盘。
        """
        ...
