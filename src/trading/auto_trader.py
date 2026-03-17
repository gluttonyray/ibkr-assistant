"""
Auto-trading module (disabled by default).

When AUTO_TRADE=true, sends limit orders based on confirmed signals:
  - Limit price = mid-price ± ATR(14) * cfg.atr_limit_offset
  - Tracks daily PnL; stops trading if loss exceeds cfg.daily_loss_limit
  - Cancels unfilled orders after cfg.order_timeout seconds
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from ib_insync import IB, LimitOrder, Order, Stock, Trade

from src.config import cfg

logger = logging.getLogger(__name__)


class CircuitBreakerTripped(Exception):
    """Raised when the daily loss limit is exceeded."""


class AutoTrader:
    """
    Submits and manages orders when auto-trading is enabled.

    Usage::

        trader = AutoTrader(ib)
        if cfg.auto_trade:
            await trader.on_signal(symbol, "BUY", df, result)
    """

    def __init__(self, ib: IB) -> None:
        self.ib = ib
        self._positions: Dict[str, int] = {}        # symbol → current shares
        self._open_trades: Dict[str, Trade] = {}    # symbol → pending Trade
        self._daily_realized_pnl: float = 0.0
        self._daily_order_count: int = 0
        self._trade_date: date = date.today()
        self._tripped: bool = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def on_signal(
        self,
        symbol: str,
        signal: str,
        df: pd.DataFrame,
        score: float,
        qty: Optional[int] = None,
    ) -> None:
        """
        Called by the main loop when a confirmed signal arrives.
        Submits a new order if conditions are met.

        Parameters
        ----------
        qty : int, optional
            Override default position size (from PositionSizer in factor mode).
            If None, uses cfg.max_position_size.
        """
        if not cfg.auto_trade:
            return

        self._reset_daily_if_needed()

        if self._tripped:
            logger.warning("[AutoTrader] Circuit breaker active — no new orders.")
            return

        if self._daily_order_count >= cfg.max_orders_per_day:
            logger.warning(
                "[AutoTrader] Daily order limit reached (%d) — no new orders.",
                cfg.max_orders_per_day,
            )
            return

        # Opening-hour filter: skip the first N minutes after NYSE open (09:30 ET).
        # This avoids the high-noise auction/gap-fill period.
        # NOTE: assumes bar timestamps are in US Eastern time (configure TWS accordingly).
        if self._is_in_open_filter(df):
            logger.debug(
                "[AutoTrader] Open filter active for %s — skipping signal.", symbol
            )
            return

        # Cancel any pending order for this symbol first
        await self._cancel_pending(symbol)

        position = self._positions.get(symbol, 0)
        atr = self._compute_atr(df)
        price = float(df["close"].iloc[-1])
        offset = atr * cfg.atr_limit_offset

        if signal == "BUY" and position <= 0:
            order_qty = qty if qty is not None else cfg.max_position_size
            limit_price = round(price - offset, 2)  # buy below current
            await self._submit_order(symbol, "BUY", order_qty, limit_price)

        elif signal == "SELL" and position >= 0:
            order_qty = qty if qty is not None else (abs(position) if position > 0 else cfg.max_position_size)
            limit_price = round(price + offset, 2)  # sell above current
            await self._submit_order(symbol, "SELL", order_qty, limit_price)

    def update_pnl(self, realized: float) -> None:
        """Called externally when a fill PnL is known."""
        self._daily_realized_pnl += realized
        if self._daily_realized_pnl < -abs(cfg.daily_loss_limit):
            logger.error(
                "[AutoTrader] Daily loss limit breached: %.2f < -%.2f. Circuit breaker ON.",
                self._daily_realized_pnl,
                cfg.daily_loss_limit,
            )
            self._tripped = True

    def get_status(self) -> dict:
        return {
            "tripped": self._tripped,
            "daily_pnl": round(self._daily_realized_pnl, 2),
            "positions": dict(self._positions),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _submit_order(
        self, symbol: str, action: str, qty: int, limit_price: float
    ) -> None:
        # Futures auto-trading requires margin/multiplier handling not yet
        # implemented.  Skip silently so signal logging still works.
        if symbol.lower().endswith("main"):
            logger.warning(
                "[AutoTrader] Futures auto-trading not yet supported (%s) — skipping.",
                symbol,
            )
            return

        contract = Stock(symbol, "SMART", "USD")
        try:
            await self.ib.qualifyContractsAsync(contract)
        except Exception as exc:
            logger.error("[AutoTrader] Could not qualify %s: %s", symbol, exc)
            return

        order = LimitOrder(action, qty, limit_price, tif="DAY", transmit=True)
        trade: Trade = self.ib.placeOrder(contract, order)
        self._open_trades[symbol] = trade
        self._daily_order_count += 1
        logger.info(
            "[AutoTrader] Placed %s %d %s @ %.2f (orderId=%s)",
            action, qty, symbol, limit_price, trade.order.orderId,
        )

        # Schedule timeout cancellation
        asyncio.create_task(self._watch_order(symbol, trade))

    async def _watch_order(self, symbol: str, trade: Trade) -> None:
        """Cancel order if not filled within timeout."""
        await asyncio.sleep(cfg.order_timeout)
        if symbol in self._open_trades and self._open_trades[symbol] is trade:
            status = trade.orderStatus.status
            if status not in ("Filled", "Cancelled", "Inactive"):
                logger.info(
                    "[AutoTrader] Order timeout for %s (status=%s) — cancelling.", symbol, status
                )
                self.ib.cancelOrder(trade.order)
            del self._open_trades[symbol]

    async def _cancel_pending(self, symbol: str) -> None:
        if symbol in self._open_trades:
            trade = self._open_trades.pop(symbol)
            status = trade.orderStatus.status
            if status not in ("Filled", "Cancelled", "Inactive"):
                self.ib.cancelOrder(trade.order)
                logger.info("[AutoTrader] Cancelled pending order for %s.", symbol)
            await asyncio.sleep(0.1)

    def _is_in_open_filter(self, df: pd.DataFrame) -> bool:
        """
        Return True if the last bar's timestamp falls within the opening
        filter window (first cfg.open_filter_minutes after 09:30 ET).

        NYSE opens at 09:30 ET.  Assumes bar timestamps are in ET; if TWS
        is configured for a different timezone adjust accordingly or set
        OPEN_FILTER_MINUTES=0 to disable.
        """
        if cfg.open_filter_minutes <= 0:
            return False
        try:
            last_ts = df.index[-1]
            # threshold: minutes elapsed since midnight
            threshold = 9 * 60 + 30 + cfg.open_filter_minutes
            bar_minutes = last_ts.hour * 60 + last_ts.minute
            return bar_minutes < threshold
        except (IndexError, AttributeError, TypeError):
            return False

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._trade_date:
            self._trade_date = today
            self._daily_realized_pnl = 0.0
            self._daily_order_count = 0
            self._tripped = False
            logger.info("[AutoTrader] New trading day — PnL and order count reset.")

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
        if atr_series is None or atr_series.empty:
            # Fallback: rough ATR from recent range
            return float(df["high"].tail(period).max() - df["low"].tail(period).min()) / period
        val = atr_series.dropna()
        return float(val.iloc[-1]) if len(val) else 0.01
