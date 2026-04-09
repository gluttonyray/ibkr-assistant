"""风控引擎，实现 PreTradeRiskGate 和 PostTradeRiskGate 协议。"""
from __future__ import annotations

import logging

from quant.config.schema import RiskConfig
from quant.core.types import (
    Bar,
    Currency,
    Instrument,
    Position,
    PortfolioSnapshot,
    Signal,
)

logger = logging.getLogger(__name__)


class RiskEngine:
    """核心风控规则引擎。

    内部状态：
        _circuit_breaker_open -- 熔断器是否已触发
        _cb_triggered_bar    -- 熔断器触发时的 K 线索引
        _current_bar_index   -- 当前 K 线索引（通过 tick() 更新）

    通过 .pre_trade 和 .post_trade 属性对外暴露两个符合协议的门面，
    使一个引擎实例同时满足 PreTradeRiskGate 和 PostTradeRiskGate 合约。
    """

    def __init__(self, cfg: RiskConfig, hkd_usd_rate: float = 0.128) -> None:
        self._cfg = cfg
        self._hkd_usd_rate = hkd_usd_rate
        self._circuit_breaker_open: bool = False
        self._cb_triggered_bar: int = -1
        self._current_bar_index: int = 0

    # -- K 线推进 ---------------------------------------------------------------

    def tick(self, bar_index: int) -> None:
        """在每根 K 线开始时调用，更新计数器并检查冷却期。"""
        self._current_bar_index = bar_index
        if self._circuit_breaker_open:
            bars_since = bar_index - self._cb_triggered_bar
            if bars_since >= self._cfg.circuit_breaker_cooldown_bars:
                self._circuit_breaker_open = False
                logger.info(
                    "Circuit breaker CLOSED after %d cooldown bars", bars_since
                )

    # -- 协议门面 -------------------------------------------------------

    @property
    def pre_trade(self) -> PreTradeRisk:
        """返回符合 PreTradeRiskGate 的门面。"""
        return PreTradeRisk(self)

    @property
    def post_trade(self) -> PostTradeRisk:
        """返回符合 PostTradeRiskGate 的门面。"""
        return PostTradeRisk(self)

    # -- 交易前逻辑 --------------------------------------------------------

    def pre_trade_check(
        self,
        signal: Signal,
        proposed_qty: int,
        portfolio: PortfolioSnapshot,
    ) -> tuple[bool, int, str]:
        """交易前风控检查，返回 (approved, adjusted_qty, reason)。"""
        qty = proposed_qty

        # 1. 熔断器检查
        if self._circuit_breaker_open:
            return False, 0, "Circuit breaker is OPEN"

        # 2. 当日亏损限额
        if portfolio.daily_pnl_usd < -self._cfg.max_daily_loss_usd:
            return (
                False,
                0,
                f"Daily loss limit hit: {portfolio.daily_pnl_usd:.0f} USD",
            )

        # 3. 当日交易次数上限
        if portfolio.daily_trade_count >= self._cfg.max_daily_trades:
            return (
                False,
                0,
                f"Daily trade count limit: {portfolio.daily_trade_count}",
            )

        # 4. 相关组限额
        sym = signal.instrument.symbol
        for grp in self._cfg.correlation_group_limits:
            if sym not in grp.symbols:
                continue

            # 互斥约束：同一时间只能持有组内一个合约
            if grp.mutually_exclusive:
                for other_sym, pos in portfolio.positions.items():
                    if (
                        other_sym != sym
                        and other_sym in grp.symbols
                        and pos.qty != 0
                    ):
                        return (
                            False,
                            0,
                            f"Mutually exclusive group: already hold {other_sym}",
                        )

            # 组内净合约数上限
            current_group_qty = sum(
                abs(pos.qty)
                for s, pos in portfolio.positions.items()
                if s in grp.symbols and pos.qty != 0
            )
            headroom = grp.max_net_contracts - current_group_qty
            if headroom <= 0:
                return (
                    False,
                    0,
                    f"Correlation group limit reached for {grp.symbols}",
                )
            qty = min(qty, headroom)

        # 5. 单品种合约数上限
        qty = min(qty, self._cfg.max_contracts_per_instrument)

        # 6. 保证金充足性检查
        inst = signal.instrument
        margin_per_contract = inst.margin_initial
        if inst.currency == Currency.HKD:
            margin_usd = margin_per_contract * qty * self._hkd_usd_rate
        else:
            margin_usd = margin_per_contract * qty

        if portfolio.cash_usd < margin_usd:
            # 尝试缩减至可承受的数量
            if inst.currency == Currency.HKD:
                max_affordable = int(
                    portfolio.cash_usd
                    / (inst.margin_initial * self._hkd_usd_rate + 1e-9)
                )
            else:
                max_affordable = int(
                    portfolio.cash_usd / (inst.margin_initial + 1e-9)
                )
            qty = min(qty, max_affordable)
            if qty <= 0:
                return (
                    False,
                    0,
                    f"Insufficient margin: need {margin_usd:.0f} USD, "
                    f"have {portfolio.cash_usd:.0f}",
                )

        if qty <= 0:
            return False, 0, "qty reduced to 0 after all checks"

        return True, qty, "OK"

    # -- 交易后逻辑 -------------------------------------------------------

    def post_trade_check(
        self,
        positions: dict[str, Position],
        current_bars: dict[str, Bar],
        portfolio: PortfolioSnapshot,
    ) -> list[tuple[Instrument, str]]:
        """交易后持仓检查，返回需要平仓的 (instrument, reason) 列表。"""
        to_close: list[tuple[Instrument, str]] = []

        # 熔断器：峰值到谷值回撤检查
        if portfolio.peak_equity_usd > 0:
            drawdown = 1.0 - portfolio.total_equity_usd / portfolio.peak_equity_usd
            if drawdown >= self._cfg.circuit_breaker_drawdown and not self._circuit_breaker_open:
                self._circuit_breaker_open = True
                self._cb_triggered_bar = self._current_bar_index
                logger.warning(
                    "Circuit breaker OPEN: drawdown=%.2f%%", drawdown * 100
                )
                for _sym, pos in positions.items():
                    if pos.qty != 0:
                        to_close.append(
                            (pos.instrument, f"Circuit breaker: drawdown={drawdown:.2%}")
                        )
                return to_close

        for sym, pos in positions.items():
            if pos.qty == 0 or sym not in current_bars:
                continue

            bar = current_bars[sym]
            atr = pos.atr_at_entry
            direction = pos.direction

            # ATR 止损
            if atr > 0:
                stop_dist = atr * self._cfg.stop_loss_atr
                if direction > 0 and bar.close < pos.avg_entry_price - stop_dist:
                    to_close.append(
                        (pos.instrument, f"ATR stop-loss long: close={bar.close:.2f}")
                    )
                    continue
                if direction < 0 and bar.close > pos.avg_entry_price + stop_dist:
                    to_close.append(
                        (pos.instrument, f"ATR stop-loss short: close={bar.close:.2f}")
                    )
                    continue

            # 追踪止损
            if atr > 0:
                trail_dist = atr * self._cfg.trailing_stop_atr
                if direction > 0 and pos.high_watermark > 0:
                    if bar.close < pos.high_watermark - trail_dist:
                        to_close.append(
                            (
                                pos.instrument,
                                f"Trailing stop long: hwm={pos.high_watermark:.2f}",
                            )
                        )
                        continue
                if direction < 0 and pos.low_watermark > 0:
                    if bar.close > pos.low_watermark + trail_dist:
                        to_close.append(
                            (
                                pos.instrument,
                                f"Trailing stop short: lwm={pos.low_watermark:.2f}",
                            )
                        )
                        continue

            # 最大持仓期检查
            if self._cfg.max_holding_bars > 0 and pos.holding_bars >= self._cfg.max_holding_bars:
                to_close.append(
                    (pos.instrument, f"Max holding bars: {pos.holding_bars}")
                )
                continue

        return to_close


class PreTradeRisk:
    """实现 PreTradeRiskGate 协议的门面类。"""

    def __init__(self, engine: RiskEngine) -> None:
        self._engine = engine

    def check(
        self,
        signal: Signal,
        proposed_qty: int,
        portfolio: PortfolioSnapshot,
    ) -> tuple[bool, int, str]:
        return self._engine.pre_trade_check(signal, proposed_qty, portfolio)


class PostTradeRisk:
    """实现 PostTradeRiskGate 协议的门面类。"""

    def __init__(self, engine: RiskEngine) -> None:
        self._engine = engine

    def check(
        self,
        positions: dict[str, Position],
        current_bars: dict[str, Bar],
        portfolio: PortfolioSnapshot,
    ) -> list[tuple[Instrument, str]]:
        return self._engine.post_trade_check(positions, current_bars, portfolio)
