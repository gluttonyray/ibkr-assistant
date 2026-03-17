"""
IBKR Real-Time Stock Trading Signal Assistant
=============================================

Entry point.  Run::

    python main.py

Environment variables are loaded from .env (see .env.example).
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Dict

import pandas as pd
from ib_insync import util

from src.config import cfg
from src.indicators.aggregator import AggregateResult, SignalAggregator
from src.trading.data_feed import DataFeed
from src.trading.auto_trader import AutoTrader
from src.ui.dashboard import Dashboard

# Factor-mode imports (lazy — only loaded when FACTOR_MODE=true)
if cfg.factor_mode:
    import src.factors.technical  # registers all 11 technical factors
    from src.alpha.combiner import AlphaCombiner, AlphaSignal
    from src.factors.base import FactorData
    from src.factors.registry import FactorRegistry
    from src.indicators.regime import RegimeDetector as _RegimeDetector
    from src.risk.position_sizer import PositionSizer
    from src.risk.risk_manager import PositionInfo, RiskManager

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("main")

# Silence ib_insync internal noise at INFO level
logging.getLogger("ib_insync").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TradingAssistant:
    def __init__(self) -> None:
        self.feed = DataFeed()
        self.dashboard = Dashboard()
        self.auto_trader: AutoTrader | None = None
        self._running = False
        self._latest: Dict[str, AggregateResult] = {}

        # Choose pipeline mode
        if cfg.factor_mode:
            logger.info("FACTOR_MODE enabled — using factor-based pipeline")
            self.aggregator = None
            self._factor_combiner = AlphaCombiner()
            self._factor_sizer = PositionSizer(
                max_risk_per_trade=cfg.risk_per_trade,
                max_position_pct=cfg.max_position_pct,
                stop_atr_multiple=cfg.stop_loss_atr,
            )
            self._factor_risk = RiskManager(
                stop_loss_atr=cfg.stop_loss_atr,
                take_profit_atr=cfg.take_profit_atr,
                trailing_stop_atr=cfg.trailing_stop_atr,
                max_drawdown_pct=cfg.max_drawdown_pct,
            )
            self._factor_regime = _RegimeDetector()
            self._factors = FactorRegistry.create_all()
            self._latest_alpha: Dict[str, "AlphaSignal"] = {}
        else:
            self.aggregator = SignalAggregator()

    async def run(self) -> None:
        self._running = True
        mode_str = "FACTOR" if cfg.factor_mode else "INDICATOR"
        logger.info("Starting IBKR Signal Assistant… [mode=%s]", mode_str)
        logger.info("Symbols: %s", cfg.symbols)
        logger.info("Auto-trade: %s", cfg.auto_trade)
        if cfg.factor_mode:
            logger.info("Factor pipeline: %d factors registered", len(self._factors))
        else:
            logger.info("Thresholds: BUY>%.2f  SELL<%.2f", cfg.buy_threshold, cfg.sell_threshold)

        # Connect
        await self.feed.connect()

        # Set up auto-trader if enabled
        if cfg.auto_trade:
            self.auto_trader = AutoTrader(self.feed.ib)
            logger.warning(
                "AUTO-TRADE ENABLED — orders will be submitted to paper account."
            )

        # Start dashboard in the event loop (non-blocking)
        self.dashboard.start()

        # Seed history + subscribe to real-time bars.
        # on_ready: each symbol populates the dashboard as soon as its
        # historical data is loaded (no need to wait for all symbols).
        await self.feed.start(
            on_bar=self._on_bar,
            on_ready=self._on_initial,
        )

        # Keep running until interrupted
        try:
            while self._running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await self._shutdown()

    def _on_initial(self, symbol: str, df: pd.DataFrame) -> None:
        """Called per-symbol as soon as historical data is seeded."""
        try:
            if cfg.factor_mode:
                self._process_bar_factor(symbol, df)
            else:
                result = self.aggregator.process(symbol, df)
                self._latest[symbol] = result
                self.dashboard.update(symbol, result)
            logger.info("Initial analysis done for %s (%d bars).", symbol, len(df))
        except Exception as exc:
            logger.error("Initial analysis failed for %s: %s", symbol, exc)

    def _on_bar(self, symbol: str, df: pd.DataFrame) -> None:
        """Called by DataFeed on every new completed bar.
        Offloads computation to a thread pool to avoid blocking the ib_insync loop.
        """
        loop = asyncio.get_event_loop()
        if cfg.factor_mode:
            loop.run_in_executor(None, self._process_bar_factor, symbol, df)
        else:
            loop.run_in_executor(None, self._process_bar, symbol, df)

    def _process_bar(self, symbol: str, df: pd.DataFrame) -> None:
        """Indicator-mode: heavy work in thread pool."""
        result = self.aggregator.process(symbol, df)
        self._latest[symbol] = result
        self.dashboard.update(symbol, result)

        if result.confirmed and result.final_signal != "HOLD":
            logger.info(
                "[%s] %s  score=%.3f  bars=%d",
                symbol, result.final_signal, result.score, len(df),
            )

        if self.auto_trader and result.confirmed:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                self.auto_trader.on_signal(symbol, result.final_signal, df, result.score),
                loop,
            )

    def _process_bar_factor(self, symbol: str, df: pd.DataFrame) -> None:
        """Factor-mode: compute factors, combine, dispatch signal."""
        import pandas_ta as ta

        factor_data = FactorData(ohlcv=df, symbol=symbol)
        factor_results = [f.compute(factor_data) for f in self._factors]
        regime = self._factor_regime.detect(df)
        alpha = self._factor_combiner.combine(factor_results, regime=regime)
        self._latest_alpha[symbol] = alpha

        # Classify
        if alpha.score > cfg.buy_threshold:
            signal = "BUY"
        elif alpha.score < cfg.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Build a compatible AggregateResult for the existing dashboard
        from src.indicators.aggregator import AggregateResult, IndicatorSnapshot
        from datetime import datetime, timezone
        compat = AggregateResult(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            regime=regime,
            score=alpha.score,
            final_signal=signal,
            confirmed=signal != "HOLD",
            indicators=[],
            buy_count=0,
            sell_count=0,
            hold_count=0,
            price=float(df["close"].iloc[-1]),
        )
        self._latest[symbol] = compat
        self.dashboard.update(symbol, compat)

        if signal != "HOLD":
            logger.info(
                "[FACTOR] [%s] %s  score=%.3f  regime=%s  factors=%d",
                symbol, signal, alpha.score, regime, alpha.active_factor_count,
            )

        if self.auto_trader and signal != "HOLD":
            # Compute ATR-based size
            atr_val = 0.01
            try:
                atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
                if atr_series is not None:
                    clean = atr_series.dropna()
                    if len(clean):
                        atr_val = float(clean.iloc[-1])
            except Exception:
                pass

            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                self.auto_trader.on_signal(symbol, signal, df, alpha.score),
                loop,
            )

    async def _shutdown(self) -> None:
        logger.info("Shutting down…")
        self._running = False
        self.dashboard.stop()
        if self.aggregator is not None:
            self.aggregator.close()
        await self.feed.stop()
        logger.info("Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

async def _main() -> None:
    assistant = TradingAssistant()

    loop = asyncio.get_running_loop()

    def _handle_signal(signum, frame):  # noqa: ARG001
        logger.info("Signal %d received — stopping.", signum)
        assistant._running = False

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, lambda sig=s: _handle_signal(sig, None))
        except NotImplementedError:
            # Windows
            signal.signal(s, _handle_signal)

    await assistant.run()


if __name__ == "__main__":
    # Use ib_insync's event loop integration
    util.startLoop()
    asyncio.run(_main())
