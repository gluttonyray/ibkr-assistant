"""Per-trade and portfolio-level risk management.

Per-trade controls (checked every bar for open positions):
  - Stop-loss:     exit if loss >= stop_loss_atr × ATR
  - Take-profit:   exit if gain >= take_profit_atr × ATR
  - Trailing stop: exit if price retraces trailing_stop_atr × ATR from high/low

Portfolio controls:
  - Max drawdown:  circuit breaker if equity drawdown >= max_drawdown_pct
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionInfo:
    """Snapshot of current position state, passed to check_exit()."""
    entry_price: float
    current_price: float
    direction: int              # +1 = long, -1 = short
    atr_at_entry: float
    highest_price: float        # high watermark since entry (for long trailing stop)
    lowest_price: float         # low watermark since entry (for short trailing stop)


class RiskManager:
    """Checks per-trade and portfolio risk conditions.

    Parameters
    ----------
    stop_loss_atr : float
        Stop-loss distance in ATR units from entry (default 2.0).
    take_profit_atr : float
        Take-profit distance in ATR units from entry (default 4.0).
    trailing_stop_atr : float
        Trailing stop distance in ATR units from high/low watermark (default 1.5).
    max_drawdown_pct : float
        Portfolio circuit-breaker drawdown threshold (default 10%).
    """

    def __init__(
        self,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 4.0,
        trailing_stop_atr: float = 1.5,
        max_drawdown_pct: float = 0.10,
    ) -> None:
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.trailing_stop_atr = trailing_stop_atr
        self.max_drawdown_pct = max_drawdown_pct

    def check_exit(self, position: PositionInfo) -> Optional[str]:
        """Determine if an open position should be exited.

        Returns
        -------
        str or None
            "STOP_LOSS" / "TAKE_PROFIT" / "TRAILING_STOP" / None.
        """
        if position.atr_at_entry <= 0.0:
            return None

        entry = position.entry_price
        current = position.current_price
        direction = position.direction
        atr = position.atr_at_entry

        pnl_per_share = (current - entry) * direction
        pnl_atr = pnl_per_share / atr

        # Take-profit
        if pnl_atr >= self.take_profit_atr:
            return "TAKE_PROFIT"

        # Stop-loss
        if pnl_atr <= -self.stop_loss_atr:
            return "STOP_LOSS"

        # Trailing stop
        if direction == 1:      # Long: trail from high watermark
            trail_level = position.highest_price - self.trailing_stop_atr * atr
            if current < trail_level:
                return "TRAILING_STOP"
        else:                   # Short: trail from low watermark
            trail_level = position.lowest_price + self.trailing_stop_atr * atr
            if current > trail_level:
                return "TRAILING_STOP"

        return None

    def check_portfolio_risk(self, equity: float, peak_equity: float) -> bool:
        """Return True if portfolio circuit breaker should trigger.

        Parameters
        ----------
        equity : float
            Current portfolio equity.
        peak_equity : float
            Maximum equity achieved so far.

        Returns
        -------
        bool
            True = stop trading (drawdown limit breached).
        """
        if peak_equity <= 0.0:
            return False
        drawdown = (peak_equity - equity) / peak_equity
        return drawdown >= self.max_drawdown_pct
