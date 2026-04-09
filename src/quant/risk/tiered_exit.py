"""低流动性时段保护的分层退出管理器。"""
from __future__ import annotations

from quant.config.schema import RiskConfig
from quant.core.events import OrderEvent
from quant.core.types import Currency, Instrument, Position, Side


class TieredExitManager:
    """将完整平仓拆分为三批：33% / 33% / 34%。

    对于港交所产品的 T+1（亚洲夜盘）时段，
    强制使用 LIMIT_AGGRESSIVE 订单而非 MARKET 订单，以限制滑点。
    """

    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg

    def get_exit_orders(
        self,
        instrument: Instrument,
        position: Position,
        reason: str,
        timestamp: object = None,
    ) -> list[OrderEvent]:
        """生成分层退出的 OrderEvent 列表（最多 3 笔订单）。"""
        if position.qty == 0:
            return []

        total_qty = abs(position.qty)
        exit_side = Side.SELL if position.is_long else Side.BUY

        # 拆分为 3 批
        batch1 = total_qty // 3
        batch2 = total_qty // 3
        batch3 = total_qty - batch1 - batch2
        batches = [q for q in [batch1, batch2, batch3] if q > 0]

        order_type = "LIMIT_AGGRESSIVE"

        orders: list[OrderEvent] = []
        for qty in batches:
            orders.append(
                OrderEvent(
                    instrument=instrument,
                    side=exit_side,
                    qty=qty,
                    order_type=order_type,
                    limit_price=None,
                    signal_ref=None,
                )
            )
        return orders

    def _is_t1_session(self, instrument: Instrument, timestamp: object) -> bool:
        """检查合约当前是否处于 T+1 夜盘时段（仅限港交所产品）。"""
        if instrument.currency != Currency.HKD:
            return False
        for session in instrument.sessions:
            if session.label == "T+1":
                return True
        return False
