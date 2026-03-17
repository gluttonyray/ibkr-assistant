"""
IBKR data feed: connects to TWS, pulls historical bars, subscribes to
real-time bars (keepUpToDate=True), and maintains a rolling OHLCV
DataFrame per symbol.

Supports both equities (Stock) and continuous futures (ContFuture).
Futures symbols use a "main" suffix convention (e.g. "ESmain", "YMmain").
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
from ib_insync import IB, BarData, BarDataList, Contract, ContFuture, Stock, util

from src.config import cfg

logger = logging.getLogger(__name__)

# Columns expected by all indicator code
OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# How many bars ≈ 1 trading day — used to decide history duration string.
# Approx for 15-min bars during a 6.5-hour NYSE session.
_BARS_PER_DAY = 26   # 6.5h × 4 bars/hour

# ─────────────────────────────────────────────────────────────────────────────
# Futures contract mapping  (root → (exchange, currency))
# Add more entries here as needed.
# ─────────────────────────────────────────────────────────────────────────────
_FUTURES_MAP: Dict[str, Tuple[str, str]] = {
    "ES":  ("CME",   "USD"),   # S&P 500 E-mini
    "NQ":  ("CME",   "USD"),   # Nasdaq 100 E-mini
    "RTY": ("CME",   "USD"),   # Russell 2000 E-mini
    "YM":  ("CBOT",  "USD"),   # Dow Jones E-mini
    "GC":  ("COMEX", "USD"),   # Gold
    "SI":  ("COMEX", "USD"),   # Silver
    "CL":  ("NYMEX", "USD"),   # Crude Oil WTI
    "ZB":  ("CBOT",  "USD"),   # 30-yr T-Bond
    "ZN":  ("CBOT",  "USD"),   # 10-yr T-Note
    "VX":  ("CFE",   "USD"),   # VIX futures
    "6E":  ("CME",   "USD"),   # Euro FX
    "6J":  ("CME",   "USD"),   # Japanese Yen
}


def _duration_str(n_bars: int, bar_size: str) -> str:
    """Convert a bar count to an IBKR duration string."""
    # For 1-min bars: 1 bar = 1 minute
    if "min" in bar_size or "secs" in bar_size:
        minutes = n_bars  # rough for 1-min; adjust for other sizes
        if "5" in bar_size:
            minutes = n_bars * 5
        elif "15" in bar_size:
            minutes = n_bars * 15
        days = max(1, (minutes // _BARS_PER_DAY) + 1)
        return f"{days} D"
    # daily bars
    return f"{n_bars} D"


def _bars_to_df(bars: BarDataList) -> pd.DataFrame:
    """Convert ib_insync BarDataList to a clean OHLCV DataFrame."""
    rows = [
        {
            "date": b.date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    if not rows:
        return pd.DataFrame(columns=["date"] + OHLCV_COLS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    for col in OHLCV_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class DataFeed:
    """
    Manages IBKR connection and per-symbol rolling DataFrames.

    Usage::

        feed = DataFeed()
        await feed.connect()
        await feed.start(on_bar_callback)
        # later:
        await feed.stop()
    """

    def __init__(self) -> None:
        self.ib = IB()
        # symbol → rolling DataFrame (max cfg.history_bars rows)
        self._bars: Dict[str, pd.DataFrame] = {}
        # symbol → live BarDataList subscription
        self._subscriptions: Dict[str, BarDataList] = {}
        self._on_bar: Optional[Callable[[str, pd.DataFrame], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to TWS / IB Gateway."""
        logger.info(
            "Connecting to IBKR at %s:%d (client %d)...",
            cfg.ibkr_host,
            cfg.ibkr_port,
            cfg.ibkr_client_id,
        )
        await self.ib.connectAsync(
            cfg.ibkr_host,
            cfg.ibkr_port,
            clientId=cfg.ibkr_client_id,
            timeout=20,
        )
        logger.info("Connected. Server version: %s", self.ib.client.serverVersion())

    async def start(
        self,
        on_bar: Callable[[str, pd.DataFrame], None],
        on_ready: Callable[[str, pd.DataFrame], None] | None = None,
    ) -> None:
        """
        Seed historical bars for every symbol, then subscribe to
        real-time bars.

        *on_bar* is called every time a new real-time bar arrives.
        *on_ready* (optional) is called per-symbol as soon as its
        historical data is seeded — useful for populating the dashboard
        before all symbols are loaded.
        """
        self._on_bar = on_bar
        failed = []
        for symbol in cfg.symbols:
            try:
                contract = self._make_contract(symbol)
                await self._qualify(contract)
                await self._seed_history(symbol, contract)
                # Notify caller so dashboard can show this symbol immediately
                if on_ready is not None:
                    df = self.get_bars(symbol)
                    if not df.empty:
                        on_ready(symbol, df)
                self._subscribe_realtime(symbol, contract)
            except Exception as exc:
                logger.error("Failed to start %s: %s — skipping.", symbol, exc)
                failed.append(symbol)
        active = [s for s in cfg.symbols if s not in failed]
        logger.info(
            "Subscribed %d/%d symbols.%s",
            len(active), len(cfg.symbols),
            f" Failed: {', '.join(failed)}" if failed else "",
        )

    def get_bars(self, symbol: str) -> pd.DataFrame:
        """Return current rolling DataFrame for *symbol* (copy)."""
        return self._bars.get(symbol, pd.DataFrame(columns=OHLCV_COLS)).copy()

    async def stop(self) -> None:
        """Cancel all subscriptions and disconnect."""
        for symbol, bars in self._subscriptions.items():
            try:
                self.ib.cancelHistoricalData(bars)
            except Exception as exc:
                logger.warning("Failed to cancel subscription for %s: %s", symbol, exc)
        self._subscriptions.clear()
        self.ib.disconnect()
        logger.info("Disconnected from IBKR.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_symbol(symbol: str) -> Tuple[str, bool]:
        """
        Return (root_symbol, is_futures).
        Symbols ending with 'main' (case-insensitive) are treated as
        continuous futures contracts.  Examples:
          'ESmain'  → ('ES',   True)
          'YMmain'  → ('YM',   True)
          'AAPL'    → ('AAPL', False)
        """
        if symbol.lower().endswith("main"):
            return symbol[:-4].upper(), True
        return symbol.upper(), False

    @staticmethod
    def _make_contract(symbol: str) -> Contract:
        """Return the appropriate ib_insync Contract for *symbol*."""
        root, is_futures = DataFeed._parse_symbol(symbol)
        if is_futures:
            exchange, currency = _FUTURES_MAP.get(root, ("CME", "USD"))
            if root not in _FUTURES_MAP:
                logger.warning(
                    "Unknown futures root '%s' for symbol '%s'; "
                    "defaulting to CME/USD.  Add it to _FUTURES_MAP if wrong.",
                    root, symbol,
                )
            return ContFuture(root, exchange, currency)
        return Stock(symbol, "SMART", "USD")

    @staticmethod
    def _use_rth(symbol: str) -> bool:
        """
        Return True if only Regular Trading Hours data should be requested.
        Futures trade nearly 24 h, so we fetch their full session (RTH=False).
        Stocks default to RTH=True to avoid pre/post-market noise.
        """
        _, is_futures = DataFeed._parse_symbol(symbol)
        return not is_futures  # stocks → RTH=True, futures → RTH=False

    async def _qualify(self, contract: Contract) -> None:
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(f"Could not qualify contract: {contract.symbol}")
        logger.debug("Qualified %s → conId=%s", contract.symbol, contract.conId)

    async def _seed_history(self, symbol: str, contract: Contract) -> None:
        """Fetch historical bars to seed the rolling buffer."""
        duration = _duration_str(cfg.history_bars, cfg.bar_size)
        logger.info("Fetching %s bars for %s (duration=%s)...", cfg.bar_size, symbol, duration)
        try:
            bars: BarDataList = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=cfg.bar_size,
                whatToShow="TRADES",
                useRTH=self._use_rth(symbol),
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as exc:
            logger.warning("History request failed for %s: %s — using empty DataFrame.", symbol, exc)
            self._bars[symbol] = pd.DataFrame(columns=OHLCV_COLS)
            return

        df = _bars_to_df(bars)
        # Keep only last N bars
        self._bars[symbol] = df.tail(cfg.history_bars).copy()
        logger.info(
            "Seeded %d bars for %s (requested %d).", len(self._bars[symbol]), symbol, cfg.history_bars
        )

    def _subscribe_realtime(self, symbol: str, contract: Contract) -> None:
        """Subscribe to real-time updates using keepUpToDate=True."""
        bars: BarDataList = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="2 D",
            barSizeSetting=cfg.bar_size,
            whatToShow="TRADES",
            useRTH=self._use_rth(symbol),
            formatDate=1,
            keepUpToDate=True,
        )
        bars.updateEvent += self._make_bar_handler(symbol)
        self._subscriptions[symbol] = bars
        logger.info("Subscribed to real-time bars for %s.", symbol)

    def _make_bar_handler(self, symbol: str) -> Callable:
        """Return a closure that handles bar updates for *symbol*."""

        def _handler(bars: BarDataList, has_new_bar: bool) -> None:
            if not has_new_bar:
                return  # tick update within the current bar — skip
            if not bars:
                return
            last: BarData = bars[-1]
            new_row = pd.DataFrame(
                [
                    {
                        "open": last.open,
                        "high": last.high,
                        "low": last.low,
                        "close": last.close,
                        "volume": last.volume,
                    }
                ],
                index=pd.DatetimeIndex([pd.to_datetime(last.date)]),
            )
            new_row.index.name = "date"
            current = self._bars.get(symbol, pd.DataFrame(columns=OHLCV_COLS))
            updated = pd.concat([current, new_row])
            # De-duplicate by index (in case of resubmit) and keep last N
            updated = updated[~updated.index.duplicated(keep="last")]
            updated.sort_index(inplace=True)
            self._bars[symbol] = updated.tail(cfg.history_bars)
            if self._on_bar:
                self._on_bar(symbol, self._bars[symbol].copy())

        return _handler
