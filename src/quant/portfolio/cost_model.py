"""期货交易成本模型。

使用 ``CostConfig`` 中的每交易所参数，
计算每笔交易的佣金、交易所费用和滑点成本。
"""
from __future__ import annotations

from quant.config.schema import CostConfig
from quant.core.types import Instrument, InstrumentType


class FuturesCostModel:
    """期货合约成本计算器。"""

    def __init__(self, cfg: CostConfig) -> None:
        self._cfg = cfg

    def calculate(
        self, instrument: Instrument, qty: int, fill_price: float
    ) -> tuple[float, float, float]:
        """返回 (佣金, 交易所费用, 滑点成本)，单位为 instrument.currency。"""
        if instrument.instrument_type != InstrumentType.FUTURE:
            raise ValueError(
                f"Only FUTURE supported, got {instrument.instrument_type}"
            )
        exc = self._cfg.for_exchange(instrument.exchange)
        commission = max(exc.commission_min, exc.commission_per_contract * qty)
        exchange_fees = (exc.exchange_fee_per_contract + exc.nfa_fee_per_contract) * qty
        slippage_cost = (
            instrument.tick_size * instrument.multiplier * exc.slippage_ticks * qty
        )
        return commission, exchange_fees, slippage_cost
