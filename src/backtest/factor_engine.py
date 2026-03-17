"""Factor-based backtest engine.

Mirrors BacktestEngine but uses AlphaCombiner instead of SignalAggregator,
PositionSizer for dynamic sizing, and RiskManager for per-trade risk controls.

Key differences vs. BacktestEngine:
  - Position size is dynamic (ATR-based, score-proportional) not fixed
  - RiskManager checks stop-loss / take-profit / trailing-stop every bar
  - Score-based exit: close when alpha score reverses past exit threshold
  - Max holding period exit: close after max_holding_bars regardless of score
  - IC-based weighting: pre-pass computes per-factor IC on training data
  - No limit-order simulation (market-order fill at current bar's close)
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import pandas as pd
import pandas_ta as ta

from src.alpha.combiner import AlphaCombiner, AlphaSignal
from src.backtest.cost_model import CostModel
from src.backtest.engine import BacktestResult, WARMUP_BARS, WINDOW_SIZE
from src.backtest.portfolio import Portfolio, Trade
from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry
from src.indicators.regime import RegimeDetector
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import PositionInfo, RiskManager

logger = logging.getLogger(__name__)

# Bars used for IC weight pre-pass (after warmup)
IC_TRAINING_BARS = 500


class FactorBacktestEngine:
    """Run a factor-based backtest on historical OHLCV data.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    data : pd.DataFrame
        Full OHLCV data with DatetimeIndex.
    initial_capital : float
        Starting cash.
    slippage_bps : float
        Slippage in basis points.
    buy_threshold : float
        Minimum alpha score to enter a position (default 0.2).
    sell_threshold : float
        Maximum alpha score to enter a short (default -0.2).
    exit_long_threshold : float
        Exit a long position when score drops below this (default -0.05).
    exit_short_threshold : float
        Exit a short position when score rises above this (default 0.05).
    max_holding_bars : int
        Force-close any position held longer than this many bars (default 150).
    use_ic_weights : bool
        If True, run a pre-pass to compute IC-based factor weights before
        the main bar loop (default True).
    ic_forward_horizon : int
        Forward return horizon (in bars) for IC computation (default 10).
    factors : list[BaseFactor], optional
        Factor instances to use. Defaults to all registered factors.
    risk_kwargs : dict, optional
        Keyword args forwarded to RiskManager.
    sizer_kwargs : dict, optional
        Keyword args forwarded to PositionSizer.
    use_external_data : bool
        Fetch external data (VIX/macro/fundamentals) if True (default True).
    external_lookup : dict, optional
        Pre-loaded date→data dict. If provided, skips fetching.
    """

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        slippage_bps: float = 5.0,
        buy_threshold: float = 0.2,
        sell_threshold: float = -0.2,
        exit_long_threshold: float = -0.05,
        exit_short_threshold: float = 0.05,
        max_holding_bars: int = 150,
        use_ic_weights: bool = True,
        ic_forward_horizon: int = 10,
        factors: Optional[List[BaseFactor]] = None,
        risk_kwargs: Optional[dict] = None,
        sizer_kwargs: Optional[dict] = None,
        use_external_data: bool = True,
        external_lookup: Optional[dict] = None,
    ) -> None:
        self.symbol = symbol
        self.data = data
        self.initial_capital = initial_capital
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.max_holding_bars = max_holding_bars
        self.use_ic_weights = use_ic_weights
        self.ic_forward_horizon = ic_forward_horizon
        self.use_external_data = use_external_data
        self._external_lookup_override = external_lookup

        # Ensure all factor categories are registered
        import src.factors.technical    # noqa: F401
        import src.factors.sentiment    # noqa: F401
        import src.factors.macro        # noqa: F401
        import src.factors.fundamental  # noqa: F401

        self._factors: List[BaseFactor] = (
            factors if factors is not None else FactorRegistry.create_all()
        )
        self._combiner = AlphaCombiner()
        self._regime = RegimeDetector()
        self._sizer = PositionSizer(**(sizer_kwargs or {}))
        self._risk_mgr = RiskManager(**(risk_kwargs or {}))

        self.cost_model = CostModel(slippage_bps=slippage_bps)
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            position_size=100,  # fallback, overridden by sizer
            cost_model=self.cost_model,
        )

        self._signals: List[dict] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest bar-by-bar and return results."""
        n = len(self.data)
        if n < WARMUP_BARS + 10:
            raise ValueError(
                f"Insufficient data: {n} bars (need at least {WARMUP_BARS + 10})"
            )

        logger.info(
            "Starting factor backtest: %s | %d bars | capital=$%.0f | %d factors",
            self.symbol, n, self.initial_capital, len(self._factors),
        )

        # ── Load external data (sentiment / macro / fundamental) ──────
        external_lookup: dict = {}
        if self._external_lookup_override is not None:
            external_lookup = self._external_lookup_override
        elif self.use_external_data:
            try:
                from src.data.external_loader import ExternalDataLoader
                start_str = self.data.index[0].strftime("%Y-%m-%d")
                end_str = self.data.index[-1].strftime("%Y-%m-%d")
                external_lookup = ExternalDataLoader().load_aligned(
                    self.symbol, start_str, end_str
                )
            except Exception as exc:
                logger.warning(
                    "External data load failed — running with technical factors only: %s", exc
                )

        # ── IC weight pre-pass ────────────────────────────────────────
        if self.use_ic_weights:
            self._apply_ic_weights(external_lookup)

        # ── Reset factor history before actual run ────────────────────
        for f in self._factors:
            f.reset()

        # Position tracking state
        _entry_atr: float = 0.0
        _peak_price: float = 0.0
        _trough_price: float = 0.0
        _peak_equity: float = self.initial_capital
        _circuit_breaker: bool = False
        _bars_held: int = 0

        for i in range(n):
            ts = self.data.index[i]
            row = self.data.iloc[i]
            price = float(row["close"])

            # Rolling window (same as live DataFeed)
            start = max(0, i + 1 - WINDOW_SIZE)
            window = self.data.iloc[start : i + 1]

            # Current equity
            equity = self.portfolio.cash + self.portfolio.position * price
            _peak_equity = max(_peak_equity, equity)

            # Track holding bars
            if self.portfolio.position != 0:
                _bars_held += 1
            else:
                _bars_held = 0

            # Update watermarks for trailing stop
            if self.portfolio.position > 0:
                _peak_price = max(_peak_price, price)
            elif self.portfolio.position < 0:
                _trough_price = min(_trough_price, price)

            # ── Risk manager: SL / TP / trailing stop ────────────────
            if self.portfolio.position != 0 and _entry_atr > 0.0 and not _circuit_breaker:
                pos_info = PositionInfo(
                    entry_price=self.portfolio.entry_price,
                    current_price=price,
                    direction=1 if self.portfolio.position > 0 else -1,
                    atr_at_entry=_entry_atr,
                    highest_price=_peak_price,
                    lowest_price=_trough_price,
                )
                exit_reason = self._risk_mgr.check_exit(pos_info)
                if exit_reason:
                    trade = self.portfolio.force_close(price, ts, i)
                    if trade is not None:
                        trade.symbol = self.symbol
                    _entry_atr = 0.0
                    _bars_held = 0
                    logger.debug("Risk exit at bar %d: %s", i, exit_reason)

            # ── Circuit breaker ──────────────────────────────────────
            if self._risk_mgr.check_portfolio_risk(equity, _peak_equity):
                if not _circuit_breaker:
                    logger.warning(
                        "Portfolio circuit breaker at bar %d: equity=%.2f peak=%.2f",
                        i, equity, _peak_equity,
                    )
                _circuit_breaker = True

            # Record equity every bar
            self.portfolio.record_equity(ts, price)

            # ── Skip warmup and circuit-breaker state ─────────────────
            if i < WARMUP_BARS or _circuit_breaker:
                continue

            # ── Compute factors ───────────────────────────────────────
            bar_date = ts.date() if hasattr(ts, "date") else ts.to_pydatetime().date()
            ext = external_lookup.get(bar_date, {})
            factor_data = FactorData(
                ohlcv=window,
                symbol=self.symbol,
                sentiment=ext.get("sentiment"),
                macro=ext.get("macro"),
                fundamental=ext.get("fundamental"),
            )
            factor_results = [f.compute(factor_data) for f in self._factors]

            # ── Detect regime ─────────────────────────────────────────
            regime = self._regime.detect(window)

            # ── Combine into alpha signal ─────────────────────────────
            alpha: AlphaSignal = self._combiner.combine(
                factor_results, regime=regime, timestamp=ts
            )

            # ── Score-based exit on existing position ─────────────────
            if self.portfolio.position != 0 and not _circuit_breaker:
                is_long = self.portfolio.position > 0
                score_exit = (
                    (is_long and alpha.score < self.exit_long_threshold)
                    or (not is_long and alpha.score > self.exit_short_threshold)
                )
                time_exit = _bars_held >= self.max_holding_bars
                if score_exit or time_exit:
                    reason = "score_exit" if score_exit else "max_holding"
                    trade = self.portfolio.force_close(price, ts, i)
                    if trade is not None:
                        trade.symbol = self.symbol
                    _entry_atr = 0.0
                    _bars_held = 0
                    logger.debug("%s at bar %d (score=%.3f)", reason, i, alpha.score)

            # ── Classify signal ───────────────────────────────────────
            if alpha.score > self.buy_threshold:
                signal = "BUY"
            elif alpha.score < self.sell_threshold:
                signal = "SELL"
            else:
                signal = "HOLD"

            # ── Execute trade (enter or flip) ─────────────────────────
            if signal != "HOLD" and self.portfolio.position == 0:
                atr = self._compute_atr(window)
                qty = self._sizer.compute_size(alpha.score, price, atr, equity)
                if qty > 0:
                    trade = self.portfolio.on_signal(
                        signal=signal,
                        price=price,
                        timestamp=ts,
                        bar_idx=i,
                        score=alpha.score,
                        regime=regime,
                        qty=qty,
                    )
                    if trade is not None:
                        trade.symbol = self.symbol
                    _entry_atr = atr
                    _peak_price = price
                    _trough_price = price
                    _bars_held = 0

            elif signal != "HOLD" and self.portfolio.position != 0:
                is_long = self.portfolio.position > 0
                should_flip = (signal == "SELL" and is_long) or (
                    signal == "BUY" and not is_long
                )
                if should_flip:
                    trade = self.portfolio.force_close(price, ts, i)
                    if trade is not None:
                        trade.symbol = self.symbol
                    atr = self._compute_atr(window)
                    qty = self._sizer.compute_size(alpha.score, price, atr, equity)
                    if qty > 0:
                        trade = self.portfolio.on_signal(
                            signal=signal,
                            price=price,
                            timestamp=ts,
                            bar_idx=i,
                            score=alpha.score,
                            regime=regime,
                            qty=qty,
                        )
                        if trade is not None:
                            trade.symbol = self.symbol
                        _entry_atr = atr
                        _peak_price = price
                        _trough_price = price
                        _bars_held = 0

            # ── Record signal snapshot ────────────────────────────────
            self._signals.append({
                "timestamp": ts,
                "score": alpha.score,
                "signal": signal,
                "confirmed": signal != "HOLD",
                "regime": regime,
                "price": price,
                "confidence": alpha.confidence,
                "active_factors": alpha.active_factor_count,
                "factor_contributions": alpha.factor_contributions,
            })

        # ── Force close at end ────────────────────────────────────────
        if self.portfolio.position != 0:
            last_price = float(self.data["close"].iloc[-1])
            last_ts = self.data.index[-1]
            trade = self.portfolio.force_close(last_price, last_ts, n - 1)
            if trade is not None:
                trade.symbol = self.symbol

        equity_series = self.portfolio.get_equity_series()
        final_equity = (
            equity_series.iloc[-1] if len(equity_series) else self.initial_capital
        )

        logger.info(
            "Factor backtest complete: %d trades | final equity=$%.2f | return=%.2f%%",
            len(self.portfolio.trades),
            final_equity,
            (final_equity / self.initial_capital - 1) * 100,
        )

        return BacktestResult(
            symbol=self.symbol,
            trades=self.portfolio.trades,
            equity_curve=equity_series,
            signals=self._signals,
            bars=self.data,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_ic_weights(self, external_lookup: dict) -> None:
        """Pre-pass: compute IC for each factor on training data and set weights.

        Iterates through the first WARMUP_BARS + IC_TRAINING_BARS bars,
        collects each factor's z-scores, then computes Spearman IC against
        forward returns. Updates factor.weight proportional to max(0, IC).
        """
        from src.analytics.weight_calculator import ICWeightCalculator

        n = len(self.data)
        pre_pass_end = min(WARMUP_BARS + IC_TRAINING_BARS, n - self.ic_forward_horizon - 1)
        if pre_pass_end <= WARMUP_BARS + 30:
            logger.warning("Not enough data for IC pre-pass — using default weights")
            return

        logger.info(
            "IC pre-pass: bars 0→%d (collecting z-scores from bar %d onward)",
            pre_pass_end, WARMUP_BARS,
        )

        # Reset factors before pre-pass
        for f in self._factors:
            f.reset()

        factor_scores: Dict[str, List[float]] = {f.name: [] for f in self._factors}

        for i in range(pre_pass_end):
            start = max(0, i + 1 - WINDOW_SIZE)
            window = self.data.iloc[start : i + 1]

            bar_date = (
                self.data.index[i].date()
                if hasattr(self.data.index[i], "date")
                else self.data.index[i].to_pydatetime().date()
            )
            ext = external_lookup.get(bar_date, {})
            factor_data = FactorData(
                ohlcv=window,
                symbol=self.symbol,
                sentiment=ext.get("sentiment"),
                macro=ext.get("macro"),
                fundamental=ext.get("fundamental"),
            )

            for f in self._factors:
                result = f.compute(factor_data)
                if i >= WARMUP_BARS:
                    z = result.z_score
                    factor_scores[f.name].append(z if not result.is_stale else float("nan"))

        # Compute IC weights
        prices_slice = self.data["close"].iloc[WARMUP_BARS:pre_pass_end].reset_index(drop=True)
        prices_series = pd.Series(
            prices_slice.values,
            index=range(len(prices_slice)),
        )

        calc = ICWeightCalculator(
            forward_horizon=self.ic_forward_horizon,
            min_ic=0.02,
        )
        weights = calc.compute(factor_scores, prices_series)

        # Apply to factor instances
        for f in self._factors:
            old_w = f.weight
            new_w = weights.get(f.name, 1.0)
            f.weight = new_w
            if abs(new_w - old_w) > 0.01:
                logger.debug("  %s: weight %.2f → %.2f", f.name, old_w, new_w)

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Compute current ATR for position sizing."""
        try:
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
            if atr_series is not None and not atr_series.empty:
                val = atr_series.dropna()
                if len(val):
                    return float(val.iloc[-1])
        except Exception:
            pass
        recent = df.tail(period)
        return float(recent["high"].max() - recent["low"].min()) / period
