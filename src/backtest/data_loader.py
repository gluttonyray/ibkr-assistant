"""
Data loader: CSV, Parquet, and IBKR historical data sources.

All loaders return a clean OHLCV DataFrame with DatetimeIndex.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and ensure OHLCV types."""
    df.columns = [c.strip().lower() for c in df.columns]

    # Common aliases
    renames = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("adj close", "adj_close", "adjclose"):
            continue  # skip adjusted close
        for target in OHLCV_COLS:
            if cl == target or cl.startswith(target):
                renames[col] = target
                break
    if renames:
        df = df.rename(columns=renames)

    # Handle date column → index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try parsing the existing index
        df.index = pd.to_datetime(df.index)

    df.index.name = "date"
    df.sort_index(inplace=True)

    # Ensure numeric
    for col in OHLCV_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only OHLCV
    available = [c for c in OHLCV_COLS if c in df.columns]
    df = df[available].dropna()

    if len(available) < 5:
        missing = set(OHLCV_COLS) - set(available)
        logger.warning("Missing columns after normalization: %s", missing)

    return df


def from_csv(path: str | Path) -> pd.DataFrame:
    """
    Load OHLCV from CSV.  Supports Yahoo Finance format and generic OHLCV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = _normalize(df)
    logger.info("Loaded %d bars from CSV: %s", len(df), path)
    return df


def from_parquet(path: str | Path) -> pd.DataFrame:
    """Load OHLCV from Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    df = pd.read_parquet(path)
    df = _normalize(df)
    logger.info("Loaded %d bars from Parquet: %s", len(df), path)
    return df


def _normalize_bar_size(bar_size: str) -> str:
    """
    Normalize bar size string to IBKR's exact format.

    IBKR legal values:
      1 secs, 5 secs, 10 secs, 15 secs, 30 secs,
      1 min, 2 mins, 3 mins, 5 mins, 10 mins, 15 mins, 20 mins, 30 mins,
      1 hour, 2 hours, 3 hours, 4 hours, 8 hours,
      1 day, 1W, 1M

    Note: "1 min" (singular), but "2 mins" and above (plural).
    Similarly "1 hour" vs "2 hours", "1 secs" is NOT valid — it's "1 secs".
    """
    s = bar_size.strip()

    # Already a known exact format
    known = {
        "1 secs", "5 secs", "10 secs", "15 secs", "30 secs",
        "1 min", "2 mins", "3 mins", "5 mins", "10 mins",
        "15 mins", "20 mins", "30 mins",
        "1 hour", "2 hours", "3 hours", "4 hours", "8 hours",
        "1 day", "1W", "1M",
    }
    if s in known:
        return s

    # Try to fix common variants: "15 min" → "15 mins", "5min" → "5 mins"
    import re
    m = re.match(r"(\d+)\s*(sec|min|hour|day|[WM])", s, re.IGNORECASE)
    if not m:
        return s  # return as-is, let IBKR report the error

    num, unit = int(m.group(1)), m.group(2).lower()

    if unit == "sec":
        return f"{num} secs"
    elif unit == "min":
        return "1 min" if num == 1 else f"{num} mins"
    elif unit == "hour":
        return "1 hour" if num == 1 else f"{num} hours"
    elif unit == "day":
        return f"{num} day"
    else:
        return s


def _max_step_duration(bar_size: str) -> str:
    """
    Return the maximum single-request duration for a given bar size,
    per IBKR historical data step-size limits.

    Reference: https://interactivebrokers.github.io/tws-api/historical_limitations.html

    Bar size → max duration:
      secs         → "1800 S" (30 min)      — we use "1800 S"
      1 min        → "1 D"
      2-5 mins     → "1 W"
      10-30 mins   → "1 M"
      1 hour+      → "1 Y"
      1 day        → "1 Y"
    """
    s = bar_size.lower().strip()
    if "sec" in s:
        return "1800 S"
    if s == "1 min":
        return "1 D"
    if any(s == x for x in ("2 mins", "3 mins", "5 mins")):
        return "1 W"
    if any(x in s for x in ("min",)):
        # 10 mins, 15 mins, 20 mins, 30 mins
        return "1 M"
    if "hour" in s:
        return "1 Y"
    if "day" in s or s in ("1w", "1m"):
        return "1 Y"
    return "1 M"  # conservative default


def _parse_duration_to_days(duration: str) -> int:
    """Parse an IBKR duration string like '2 Y', '6 M', '30 D' into approximate days."""
    parts = duration.strip().split()
    if len(parts) != 2:
        return 365
    num, unit = int(parts[0]), parts[1].upper()
    if unit == "Y":
        return num * 365
    elif unit == "M":
        return num * 30
    elif unit == "W":
        return num * 7
    elif unit == "D":
        return num
    elif unit == "S":
        return max(1, num // 86400)
    return 365


def _step_duration_to_days(step: str) -> int:
    """Convert a step duration to approximate calendar days for stepping."""
    return _parse_duration_to_days(step)


async def from_ibkr(
    symbol: str,
    duration: str = "2 Y",
    bar_size: str = "15 mins",
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Download historical data from IBKR and cache as CSV.

    Automatically splits the request into chunks that respect IBKR's
    per-bar-size step limits (e.g. 15-min bars max 1 month per request).
    Supports both stocks ("AAPL") and continuous futures ("GCmain").
    Uses client_id=10 to avoid conflicting with live trading (client_id=1).
    """
    import asyncio
    from datetime import datetime, timezone
    from ib_insync import IB, ContFuture
    from src.config import cfg
    from src.trading.data_feed import DataFeed, _bars_to_df

    bar_size = _normalize_bar_size(bar_size)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    safe_bar = bar_size.replace(" ", "_")
    csv_file = cache_path / f"{symbol}_{safe_bar}_{duration.replace(' ', '')}.csv"

    if csv_file.exists():
        logger.info("Using cached data: %s", csv_file)
        return from_csv(csv_file)

    ib = IB()
    try:
        # Low-level connect: TCP + API handshake only, no account sync.
        await ib.client.connectAsync(
            cfg.ibkr_host, cfg.ibkr_port, clientId=10, timeout=10,
        )

        # Build contract (Stock vs ContFuture)
        contract = DataFeed._make_contract(symbol)
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(
                f"Could not qualify contract for '{symbol}'. "
                f"For futures use 'GCmain' format, for stocks use 'AAPL'."
            )

        is_contfuture = isinstance(contract, ContFuture)
        use_rth = DataFeed._use_rth(symbol)
        step_dur = _max_step_duration(bar_size)
        total_days = _parse_duration_to_days(duration)
        step_days = _step_duration_to_days(step_dur)

        all_chunks: list[pd.DataFrame] = []
        chunk_idx = 0

        if is_contfuture:
            # IBKR Error 10339: ContFuture does not allow setting endDateTime.
            # Must use endDateTime="" (meaning "now"). Single request with step limit.
            # 15-min bars → max 1 M; 1 day bars → max 1 Y.
            logger.info(
                "Downloading %s bars for %s (ContFuture, endDateTime='' required, durationStr=%s)...",
                bar_size, symbol, step_dur,
            )

            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=step_dur,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False,
                timeout=120,
            )

            if bars:
                chunk_df = _bars_to_df(bars)
                all_chunks.append(chunk_df)
                logger.info(
                    "  ContFuture: %d bars (%s → %s)",
                    len(chunk_df), chunk_df.index[0], chunk_df.index[-1],
                )
            else:
                logger.warning("ContFuture '%s': 0 bars returned.", symbol)
        else:
            # Stocks: chunk backward with explicit endDateTime
            now = datetime.now(timezone.utc)
            cursor = now
            remaining_days = total_days

            logger.info(
                "Downloading %s of %s bars for %s (step=%s, ~%d chunks)...",
                duration, bar_size, symbol, step_dur,
                max(1, total_days // step_days),
            )

            while remaining_days > 0:
                end_str = cursor.strftime("%Y%m%d-%H:%M:%S")

                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_str,
                    durationStr=step_dur,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                    timeout=120,
                )

                if not bars:
                    logger.warning(
                        "Chunk %d: 0 bars returned (end=%s). "
                        "Possibly reached data boundary.",
                        chunk_idx, end_str,
                    )
                    break

                chunk_df = _bars_to_df(bars)
                all_chunks.append(chunk_df)
                chunk_idx += 1

                earliest = chunk_df.index[0]
                logger.info(
                    "  Chunk %d: %d bars (%s → %s)",
                    chunk_idx, len(chunk_df),
                    chunk_df.index[0], chunk_df.index[-1],
                )

                # Move cursor to just before the earliest bar we received
                cursor = earliest.to_pydatetime()
                remaining_days -= step_days

                # IBKR pacing: wait 2 seconds between requests to avoid violations
                if remaining_days > 0:
                    await asyncio.sleep(2)

        if not all_chunks:
            raise ValueError(
                f"IBKR returned 0 bars for {symbol} "
                f"(duration={duration}, bar_size={bar_size}). "
                f"Check your market data subscription."
            )

        # Concatenate, deduplicate, sort
        df = pd.concat(all_chunks)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        # Cache as CSV
        df.to_csv(csv_file)
        logger.info(
            "Downloaded %d bars total for %s, cached to %s",
            len(df), symbol, csv_file,
        )
        return df
    finally:
        ib.disconnect()
