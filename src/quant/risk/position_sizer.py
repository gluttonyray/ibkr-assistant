"""基于 ATR 的期货仓位计算器，实现 PositionSizerProtocol。"""
from __future__ import annotations

import math

from quant.config.schema import RiskConfig
from quant.core.types import Currency, Instrument, PortfolioSnapshot, Signal


class ATRPositionSizer:
    """基于 ATR 的期货合约仓位计算器。

    核心公式::

        qty = floor(risk_amount / stop_distance_usd)

    其中：
        risk_amount = total_equity_usd * risk_per_trade
        stop_distance = atr * stop_loss_atr  （本地货币）
        stop_distance_usd = stop_distance * multiplier * hkd_rate  （若为港币）
    """

    def __init__(self, cfg: RiskConfig, hkd_usd_rate: float = 0.128) -> None:
        self._cfg = cfg
        self._hkd_usd_rate = hkd_usd_rate

    def compute_size(
        self,
        signal: Signal,
        instrument: Instrument,
        portfolio: PortfolioSnapshot,
        atr: float,
    ) -> int:
        """返回应交易的合约数量（>= 0；0 表示不交易）。"""
        if portfolio.total_equity_usd <= 0:
            return 0

        if atr <= 0:
            return 0

        risk_amount = portfolio.total_equity_usd * self._cfg.risk_per_trade
        stop_distance = atr * self._cfg.stop_loss_atr  # 本地货币

        # 换算为 USD
        if instrument.currency == Currency.HKD:
            stop_distance_usd = stop_distance * instrument.multiplier * self._hkd_usd_rate
        else:
            stop_distance_usd = stop_distance * instrument.multiplier

        if stop_distance_usd <= 0:
            return 0

        qty = math.floor(risk_amount / stop_distance_usd)
        qty = max(0, min(qty, self._cfg.max_contracts_per_instrument))
        return qty
