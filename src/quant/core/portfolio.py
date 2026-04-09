"""PortfolioView 协议 —— 只读组合状态接口。

实际可变的 PositionBook 位于 ``src.quant.portfolio.book``。
本协议提供只读视图，确保策略和风控门无法意外修改账本。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from quant.core.types import (
    Instrument,
    PortfolioSnapshot,
    Position,
)


@runtime_checkable
class PortfolioView(Protocol):
    """组合状态的只读视图。"""

    def snapshot(self) -> PortfolioSnapshot:
        """返回完整组合的不可变时点快照。"""
        ...

    def get_position(self, instrument: Instrument) -> Position | None:
        """返回 *instrument* 的当前持仓，若无持仓则返回 None。"""
        ...

    def positions(self) -> dict[str, Position]:
        """返回所有未平仓持仓，以合约代码为键。"""
        ...

    @property
    def equity_usd(self) -> float:
        """当前 USD 总权益。"""
        ...

    @property
    def cash_usd(self) -> float:
        """可用 USD 现金。"""
        ...
