"""风控门协议 —— 交易前与交易后检查。

PreTradeRiskGate：在提交订单前执行检查。
PostTradeRiskGate：每根 K 线对持仓执行检查。
PositionSizerProtocol：根据风险预算计算合约数量。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from quant.core.types import (
    Bar,
    Instrument,
    PortfolioSnapshot,
    Position,
    Signal,
)


@runtime_checkable
class PreTradeRiskGate(Protocol):
    """检查拟议交易是否被允许。

    返回 (approved, adjusted_qty, reason)。
    若 approved 为 False，交易被拦截，reason 说明原因。
    若 approved 为 True 但 qty 被缩减，reason 可说明缩减原因。
    """

    def check(
        self,
        signal: Signal,
        proposed_qty: int,
        portfolio: PortfolioSnapshot,
    ) -> tuple[bool, int, str]:
        ...


@runtime_checkable
class PostTradeRiskGate(Protocol):
    """对持仓检查退出条件。

    返回 (instrument, reason) 对的列表，表示需立即平仓的持仓。
    """

    def check(
        self,
        positions: dict[str, Position],
        current_bars: dict[str, Bar],
        portfolio: PortfolioSnapshot,
    ) -> list[tuple[Instrument, str]]:
        ...


@runtime_checkable
class PositionSizerProtocol(Protocol):
    """计算应交易的合约数量。"""

    def compute_size(
        self,
        signal: Signal,
        instrument: Instrument,
        portfolio: PortfolioSnapshot,
        atr: float,
    ) -> int:
        """返回合约数量（始终 >= 0；0 表示不交易）。"""
        ...
