"""PortfolioBook —— 实现 PortfolioView 的可变持仓账本。

跟踪未平仓持仓、现金、权益曲线和交易记录。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from quant.core.types import (
    Bar,
    Currency,
    Fill,
    Instrument,
    Position,
    PortfolioSnapshot,
    Side,
)
from quant.portfolio.cost_model import FuturesCostModel


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """一笔完整往返交易的不可变记录。"""

    instrument: Instrument
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: int
    direction: int  # +1 多头，-1 空头
    pnl: float  # 单位：instrument.currency
    pnl_usd: float
    commission: float
    slippage_cost: float
    entry_score: float
    exit_reason: str
    regime_at_entry: str
    holding_bars: int


class PortfolioBook:
    """实现 PortfolioView 协议的可变组合账本。

    Parameters
    ----------
    initial_capital : float
        初始资金（USD）。
    """

    def __init__(self, initial_capital: float) -> None:
        self._cash_usd: float = initial_capital
        self._positions: dict[str, Position] = {}
        self._equity_curve: list[float] = []
        self._trade_records: list[TradeRecord] = []
        self._peak_equity_usd: float = initial_capital
        self._daily_pnl_usd: float = 0.0
        self._daily_trade_count: int = 0
        self._margin_held: dict[str, float] = {}  # symbol -> 保证金（USD）

    # -- PortfolioView 协议 ------------------------------------------------

    def snapshot(self) -> PortfolioSnapshot:
        """返回不可变的时点快照。"""
        return PortfolioSnapshot(
            total_equity_usd=self.equity_usd,
            cash_usd=self._cash_usd,
            positions=dict(self._positions),
            peak_equity_usd=self._peak_equity_usd,
            daily_pnl_usd=self._daily_pnl_usd,
            daily_trade_count=self._daily_trade_count,
            timestamp=datetime.now(timezone.utc),
        )

    def get_position(self, instrument: Instrument) -> Position | None:
        """返回 *instrument* 的当前持仓，若无持仓则返回 None。"""
        return self._positions.get(instrument.symbol)

    def positions(self) -> dict[str, Position]:
        """返回所有未平仓持仓，以合约代码为键。"""
        return dict(self._positions)

    @property
    def equity_usd(self) -> float:
        """当前 USD 总权益（现金 + 未实现盈亏）。"""
        unrealized = sum(p.unrealized_pnl_usd for p in self._positions.values())
        return self._cash_usd + unrealized + sum(self._margin_held.values())

    @property
    def cash_usd(self) -> float:
        """可用 USD 现金。"""
        return self._cash_usd

    @property
    def equity_curve(self) -> list[float]:
        """权益曲线历史数据。"""
        return self._equity_curve

    @property
    def trade_records(self) -> list[TradeRecord]:
        """所有已完成的交易记录。"""
        return self._trade_records

    # -- 变更操作 ---------------------------------------------------

    def on_fill(
        self,
        fill: Fill,
        cost_model: FuturesCostModel,
        hkd_usd_rate: float,
    ) -> None:
        """处理成交事件，更新持仓和现金。"""
        inst = fill.instrument
        sym = inst.symbol
        is_hkd = inst.currency == Currency.HKD
        rate = hkd_usd_rate if is_hkd else 1.0

        total_cost_local = fill.commission + fill.exchange_fees + fill.slippage_cost
        total_cost_usd = total_cost_local * rate

        existing = self._positions.get(sym)

        if existing is not None and not existing.is_flat:
            # 平仓交易：成交方向与持仓方向相反
            is_closing = (
                (existing.is_long and fill.side == Side.SELL)
                or (existing.is_short and fill.side == Side.BUY)
            )
            if is_closing:
                self._close_position(fill, existing, total_cost_usd, rate)
                return

        # 开仓或加仓
        self._open_position(fill, total_cost_usd, rate)

    def mark_to_market(
        self,
        bars: dict[str, Bar],
        hkd_usd_rate: float,
    ) -> None:
        """更新所有未平仓持仓的未实现盈亏。"""
        total_unrealized_usd = 0.0

        for sym, pos in self._positions.items():
            if sym not in bars:
                total_unrealized_usd += pos.unrealized_pnl_usd
                continue

            close = bars[sym].close
            direction = pos.direction
            inst = pos.instrument
            is_hkd = inst.currency == Currency.HKD
            rate = hkd_usd_rate if is_hkd else 1.0

            unrealized_pnl = (
                (close - pos.avg_entry_price)
                * inst.multiplier
                * abs(pos.qty)
                * direction
            )
            unrealized_pnl_usd = unrealized_pnl * rate

            pos.unrealized_pnl = unrealized_pnl
            pos.unrealized_pnl_usd = unrealized_pnl_usd

            # 追踪止损水位线
            if pos.is_long:
                pos.high_watermark = max(pos.high_watermark, close)
            elif pos.is_short:
                pos.low_watermark = min(pos.low_watermark, close)

            pos.holding_bars += 1
            total_unrealized_usd += unrealized_pnl_usd

        equity = self._cash_usd + total_unrealized_usd + sum(self._margin_held.values())
        self._equity_curve.append(equity)
        self._peak_equity_usd = max(self._peak_equity_usd, equity)

    def reset_daily(self) -> None:
        """重置每日计数器（在时段边界调用）。"""
        self._daily_pnl_usd = 0.0
        self._daily_trade_count = 0

    # -- 内部辅助方法 ------------------------------------------------------

    def _close_position(
        self,
        fill: Fill,
        pos: Position,
        total_cost_usd: float,
        rate: float,
    ) -> None:
        """平掉已有持仓并记录交易。"""
        inst = fill.instrument
        sym = inst.symbol
        direction = pos.direction

        pnl_local = (
            (fill.price - pos.avg_entry_price)
            * inst.multiplier
            * abs(pos.qty)
            * direction
        )
        pnl_usd = pnl_local * rate

        # 释放保证金
        margin_released_usd = self._margin_held.pop(sym, 0.0)
        self._cash_usd += margin_released_usd + pnl_usd - total_cost_usd

        # 记录交易
        record = TradeRecord(
            instrument=inst,
            entry_time=pos.entry_time,
            exit_time=fill.timestamp,
            entry_price=pos.avg_entry_price,
            exit_price=fill.price,
            qty=abs(pos.qty),
            direction=direction,
            pnl=pnl_local,
            pnl_usd=pnl_usd,
            commission=fill.commission + fill.exchange_fees,
            slippage_cost=fill.slippage_cost,
            entry_score=pos.entry_score,
            exit_reason="signal",
            regime_at_entry=pos.entry_regime,
            holding_bars=pos.holding_bars,
        )
        self._trade_records.append(record)

        self._daily_trade_count += 1
        self._daily_pnl_usd += pnl_usd - total_cost_usd

        # 删除持仓记录
        del self._positions[sym]

    def _open_position(
        self,
        fill: Fill,
        total_cost_usd: float,
        rate: float,
    ) -> None:
        """新开一个持仓。"""
        inst = fill.instrument
        sym = inst.symbol

        margin_local = inst.margin_initial * fill.qty
        margin_usd = margin_local * rate

        self._cash_usd -= margin_usd + total_cost_usd
        self._margin_held[sym] = margin_usd

        direction = 1 if fill.side == Side.BUY else -1
        qty = fill.qty * direction

        self._positions[sym] = Position(
            instrument=inst,
            qty=qty,
            avg_entry_price=fill.price,
            entry_time=fill.timestamp,
            high_watermark=fill.price,
            low_watermark=fill.price,
        )
