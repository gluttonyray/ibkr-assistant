"""
Event-driven backtest engine.

Iterates bar-by-bar over historical OHLCV data, feeding rolling windows
to the SignalAggregator, then executing trades through Portfolio.

Key design:
  - Rolling window of 500 bars (matches live DataFeed behavior).
  - Warmup period: first 210 bars compute indicators but don't trade
    (EMAcross needs 205 bars minimum).
  - Opening-hour filter: mirrors AutoTrader._is_in_open_filter().
  - Fill simulation: signals generate limit orders checked against next bar.
  - End-of-backtest: force-close all open positions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.config import cfg
from src.indicators.aggregator import AggregateResult, SignalAggregator
from src.backtest.cost_model import CostModel
from src.backtest.portfolio import Portfolio, Trade

logger = logging.getLogger(__name__)

WARMUP_BARS = 210       # EMAcross needs ~205 bars
WINDOW_SIZE = 500       # Match live rolling buffer


@dataclass
class BacktestResult:
    """Complete backtest output."""
    symbol: str
    trades: List[Trade]
    equity_curve: pd.Series
    signals: List[dict]         # per-bar signal snapshots
    bars: pd.DataFrame          # full input data
    initial_capital: float
    final_equity: float


class BacktestEngine:
    """
    Run a backtest on historical data for a single symbol.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    data : pd.DataFrame
        Full OHLCV data with DatetimeIndex.
    initial_capital : float
        Starting cash.
    position_size : int
        Shares per trade.
    slippage_bps : float
        Slippage in basis points.
    """

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        position_size: int = 100,
        slippage_bps: float = 5.0,
        aggregator_kwargs: Optional[dict] = None,
    ) -> None:
        self.symbol = symbol
        self.data = data
        self.initial_capital = initial_capital

        self.cost_model = CostModel(slippage_bps=slippage_bps)
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            position_size=position_size,
            cost_model=self.cost_model,
        )

        # Aggregator — merge caller kwargs with defaults
        agg_kw = dict(aggregator_kwargs) if aggregator_kwargs else {}
        # Default to disable_logging in backtests to avoid temp files
        agg_kw.setdefault("disable_logging", True)
        self.aggregator = SignalAggregator(**agg_kw)

        self._signals: List[dict] = []

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        n = len(self.data)
        if n < WARMUP_BARS + 10:
            raise ValueError(
                f"Insufficient data: {n} bars (need at least {WARMUP_BARS + 10})"
            )

        logger.info(
            "Starting backtest: %s | %d bars | capital=$%.0f",
            self.symbol, n, self.initial_capital,
        )

        pending_signal: Optional[dict] = None  # {signal, price, bar_idx, score, regime}

        for i in range(n):
            ts = self.data.index[i]
            row = self.data.iloc[i]
            price = float(row["close"])

            # Build rolling window (same as live DataFeed)
            start = max(0, i + 1 - WINDOW_SIZE)
            window = self.data.iloc[start:i + 1]

            # Try to fill pending limit order using this bar's range
            if pending_signal is not None:
                filled = self.cost_model.limit_order_fills(
                    pending_signal["limit_price"],
                    pending_signal["signal"],
                    float(row["high"]),
                    float(row["low"]),
                )
                if filled:
                    trade = self.portfolio.on_signal(
                        signal=pending_signal["signal"],
                        price=pending_signal["limit_price"],
                        timestamp=ts,
                        bar_idx=i,
                        score=pending_signal["score"],
                        regime=pending_signal["regime"],
                    )
                    if trade is not None:
                        trade.symbol = self.symbol
                pending_signal = None

            # Record equity at every bar
            self.portfolio.record_equity(ts, price)

            # Skip warmup period — compute indicators but don't trade
            if i < WARMUP_BARS:
                continue

            # Opening filter: skip first N minutes after 09:30
            if self._is_in_open_filter(ts):
                continue

            # Run aggregator
            result: AggregateResult = self.aggregator.process(self.symbol, window)

            self._signals.append({
                "timestamp": ts,
                "score": result.score,
                "signal": result.final_signal,
                "confirmed": result.confirmed,
                "regime": result.regime,
                "price": price,
            })

            # Only act on confirmed non-HOLD signals
            if result.confirmed and result.final_signal != "HOLD":
                # Simulate limit order: offset from current price using ATR
                offset = self._compute_atr_offset(window)
                if result.final_signal == "BUY":
                    limit_price = round(price - offset, 2)
                else:
                    limit_price = round(price + offset, 2)

                pending_signal = {
                    "signal": result.final_signal,
                    "limit_price": limit_price,
                    "score": result.score,
                    "regime": result.regime,
                    "bar_idx": i,
                }

        # Force close any open position at last bar's close
        if self.portfolio.position != 0:
            last_price = float(self.data["close"].iloc[-1])
            last_ts = self.data.index[-1]
            trade = self.portfolio.force_close(last_price, last_ts, n - 1)
            if trade is not None:
                trade.symbol = self.symbol

        # Cleanup
        self.aggregator.close()

        equity = self.portfolio.get_equity_series()
        final_equity = equity.iloc[-1] if len(equity) else self.initial_capital

        logger.info(
            "Backtest complete: %d trades | final equity=$%.2f | return=%.2f%%",
            len(self.portfolio.trades),
            final_equity,
            (final_equity / self.initial_capital - 1) * 100,
        )

        return BacktestResult(
            symbol=self.symbol,
            trades=self.portfolio.trades,
            equity_curve=equity,
            signals=self._signals,
            bars=self.data,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
        )

    @staticmethod
    def _is_in_open_filter(ts: pd.Timestamp) -> bool:
        """Check if timestamp is within the opening filter window."""
        if cfg.open_filter_minutes <= 0:
            return False
        try:
            threshold = 9 * 60 + 30 + cfg.open_filter_minutes
            bar_minutes = ts.hour * 60 + ts.minute
            return bar_minutes < threshold
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _compute_atr_offset(df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR-based limit offset (mirrors AutoTrader logic)."""
        try:
            import pandas_ta as ta
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
            if atr_series is not None and not atr_series.empty:
                val = atr_series.dropna()
                if len(val):
                    return float(val.iloc[-1]) * cfg.atr_limit_offset
        except Exception:
            pass
        # Fallback
        return float(df["high"].tail(period).max() - df["low"].tail(period).min()) / period * cfg.atr_limit_offset
