"""
Portfolio tracker: manages positions, records trades, and builds equity curve.

Mirrors the live AutoTrader logic:
  - BUY when flat/short → go long
  - SELL when flat/long → go short (or close long)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.backtest.cost_model import CostModel, FillCosts


@dataclass
class Trade:
    """A completed round-trip trade."""
    symbol: str
    side: str               # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    qty: int
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    entry_costs: FillCosts
    exit_costs: FillCosts
    margin_interest: float
    net_pnl: float
    holding_bars: int
    entry_score: float
    entry_regime: str


@dataclass
class EquitySnapshot:
    """Point-in-time equity record."""
    timestamp: datetime
    equity: float
    cash: float
    position_value: float


class Portfolio:
    """
    Tracks a single-symbol portfolio with cash, position, and equity curve.

    Parameters
    ----------
    initial_capital : float
        Starting cash.
    position_size : int
        Shares per trade.
    cost_model : CostModel
        Cost calculator.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_size: int = 100,
        cost_model: Optional[CostModel] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size = position_size
        self.cost_model = cost_model or CostModel()

        # Current position: +qty = long, -qty = short, 0 = flat
        self.position: int = 0
        self.entry_price: float = 0.0
        self.entry_time: Optional[datetime] = None
        self.entry_bar_idx: int = 0
        self.entry_score: float = 0.0
        self.entry_regime: str = ""
        self.entry_costs: FillCosts = FillCosts.zero()

        self.trades: List[Trade] = []
        self.equity_curve: List[EquitySnapshot] = []
        self.cash_flows: List[tuple] = []   # (datetime, amount) for MWR

    def on_signal(
        self,
        signal: str,
        price: float,
        timestamp: datetime,
        bar_idx: int,
        score: float = 0.0,
        regime: str = "",
        qty: Optional[int] = None,
    ) -> Optional[Trade]:
        """
        Process a confirmed signal. Returns a Trade if a position was closed.

        Logic mirrors AutoTrader:
          BUY + (flat or short) → open long
          SELL + (flat or long) → open short / close long

        Parameters
        ----------
        qty : int, optional
            Override default position_size. When provided (e.g. by
            FactorBacktestEngine for ATR-based sizing), this qty is used
            instead of self.position_size. Backward compatible — existing
            callers that omit qty continue to use self.position_size.
        """
        size = qty if qty is not None else self.position_size
        completed_trade = None

        if signal == "BUY" and self.position <= 0:
            # Close short if any
            if self.position < 0:
                completed_trade = self._close_position(price, timestamp, bar_idx)
            # Open long
            self._open_position(price, timestamp, bar_idx, size,
                                "BUY", score, regime)

        elif signal == "SELL" and self.position >= 0:
            # Close long if any
            if self.position > 0:
                completed_trade = self._close_position(price, timestamp, bar_idx)
            # Open short
            self._open_position(price, timestamp, bar_idx, -size,
                                "SELL", score, regime)

        return completed_trade

    def force_close(self, price: float, timestamp: datetime, bar_idx: int) -> Optional[Trade]:
        """Force-close any open position (end of backtest)."""
        if self.position != 0:
            return self._close_position(price, timestamp, bar_idx)
        return None

    def record_equity(self, timestamp: datetime, price: float) -> None:
        """Snapshot equity at current bar."""
        position_value = self.position * price
        equity = self.cash + position_value
        self.equity_curve.append(
            EquitySnapshot(
                timestamp=timestamp,
                equity=equity,
                cash=self.cash,
                position_value=position_value,
            )
        )

    def get_equity_series(self) -> pd.Series:
        """Return equity as a pandas Series indexed by timestamp."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        return pd.Series(
            [s.equity for s in self.equity_curve],
            index=pd.DatetimeIndex([s.timestamp for s in self.equity_curve]),
            name="equity",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_position(
        self,
        price: float,
        timestamp: datetime,
        bar_idx: int,
        qty: int,
        side: str,
        score: float,
        regime: str,
    ) -> None:
        costs = self.cost_model.fill_cost(price, abs(qty), side)
        self.position = qty
        self.entry_price = price
        self.entry_time = timestamp
        self.entry_bar_idx = bar_idx
        self.entry_score = score
        self.entry_regime = regime
        self.entry_costs = costs
        self.cash -= costs.total
        # Record cash outflow for long
        if qty > 0:
            outflow = price * abs(qty) + costs.total
            self.cash_flows.append((timestamp, -outflow))

    def _close_position(
        self, price: float, timestamp: datetime, bar_idx: int
    ) -> Trade:
        qty = abs(self.position)
        is_long = self.position > 0
        side_str = "LONG" if is_long else "SHORT"
        close_side = "SELL" if is_long else "BUY"

        exit_costs = self.cost_model.fill_cost(price, qty, close_side)
        holding_bars = bar_idx - self.entry_bar_idx

        # Margin interest (for short positions or leveraged longs)
        # Approximate days from bars (assuming 15-min bars, ~26 bars/day)
        approx_days = max(1, holding_bars // 26)
        notional = self.entry_price * qty
        margin_int = self.cost_model.margin_interest(notional, approx_days) if not is_long else 0.0

        if is_long:
            gross_pnl = (price - self.entry_price) * qty
        else:
            gross_pnl = (self.entry_price - price) * qty

        total_costs = self.entry_costs.total + exit_costs.total + margin_int
        net_pnl = gross_pnl - total_costs

        trade = Trade(
            symbol="",  # filled by engine
            side=side_str,
            entry_price=self.entry_price,
            exit_price=price,
            qty=qty,
            entry_time=self.entry_time,
            exit_time=timestamp,
            gross_pnl=round(gross_pnl, 2),
            entry_costs=self.entry_costs,
            exit_costs=exit_costs,
            margin_interest=round(margin_int, 4),
            net_pnl=round(net_pnl, 2),
            holding_bars=holding_bars,
            entry_score=self.entry_score,
            entry_regime=self.entry_regime,
        )
        self.trades.append(trade)

        # Update cash
        if is_long:
            inflow = price * qty - exit_costs.total
            self.cash += inflow
            self.cash_flows.append((timestamp, inflow))
        else:
            self.cash += gross_pnl - exit_costs.total - margin_int

        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None

        return trade
