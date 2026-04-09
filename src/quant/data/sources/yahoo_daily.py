"""Yahoo Finance 日频 OHLCV 数据源。

获取 12 支 ETF 跨资产 universe 和 VIX 的日频 K 线，
存储到三层 Parquet 系统并附带正确的 PIT 时间戳。

对于行情数据，``known_time == event_time``（收盘价在收盘时即公开）。

用法::

    source = YahooDailySource(store)
    await source.fetch_etf_universe(start="2020-01-01")
    await source.fetch_vix(start="2020-01-01")
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yaml
import yfinance as yf

from quant.data.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

_UTC = ZoneInfo("UTC")


class YahooDailySource:
    """从 Yahoo Finance 获取日频 OHLCV 并写入 ParquetStore。

    Parameters
    ----------
    store : ParquetStore
        目标存储。
    config_path : str or Path
        ``cross_asset.yaml`` 路径，用于定义 ETF universe。
    """

    def __init__(
        self,
        store: ParquetStore,
        config_path: str | Path = "configs/cross_asset.yaml",
    ) -> None:
        self._store = store
        self._config = self._load_config(config_path)

    @staticmethod
    def _load_config(path: str | Path) -> dict:
        path = Path(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return raw["cross_asset"]

    @property
    def etf_symbols(self) -> list[str]:
        """返回配置的 ETF universe 品种列表。"""
        return [e["symbol"] for e in self._config["etf_universe"]]

    async def fetch_etf_universe(
        self,
        start: str = "2015-01-01",
        end: str | None = None,
    ) -> list[str]:
        """获取跨资产 universe 中所有 ETF 的日频 OHLCV。

        Parameters
        ----------
        start : str
            起始日期（YYYY-MM-DD）。
        end : str or None
            结束日期，默认为今天。

        Returns
        -------
        list[str]
            成功获取的品种列表。
        """
        fetched = []
        for symbol in self.etf_symbols:
            try:
                self._fetch_single(symbol, start, end)
                fetched.append(symbol)
            except Exception:
                logger.exception("Failed to fetch %s", symbol)

        logger.info("Fetched %d/%d ETFs", len(fetched), len(self.etf_symbols))
        return fetched

    async def fetch_vix(
        self,
        start: str = "2015-01-01",
        end: str | None = None,
    ) -> None:
        """获取 VIX 日收盘价并存入 macro schema。

        VIX 存入 MACRO_SCHEMA（series_id="^VIX"）而非 OHLCV，
        因为它用作市场状态指标而非可交易标的。

        Parameters
        ----------
        start : str
            起始日期（YYYY-MM-DD）。
        end : str or None
            结束日期，默认为今天。
        """
        ticker = self._config["regime_indicators"]["vix"]["ticker"]
        logger.info("Fetching VIX (%s)", ticker)

        hist = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if hist.empty:
            logger.warning("No VIX data returned")
            return

        # 如果 yfinance 返回 MultiIndex 列则展平
        if hasattr(hist.columns, "levels") and hist.columns.nlevels > 1:
            hist.columns = hist.columns.get_level_values(0)

        records = []
        for idx, row in hist.iterrows():
            ts = datetime(idx.year, idx.month, idx.day, 21, 0, tzinfo=_UTC)
            records.append({
                "series_id": ticker,
                "event_time": ts,
                "known_time": ts,  # VIX 收盘时即公开
                "value": float(row["Close"]),
                "source": "yahoo",
            })

        df = pl.DataFrame(records)
        self._store.append_raw(df, "macro", "yahoo", "VIX")
        self._store.merge_to_clean("macro", "VIX", dedup_keys=["series_id", "event_time"])
        logger.info("Stored %d VIX observations", len(df))

    def _fetch_single(
        self,
        symbol: str,
        start: str,
        end: str | None,
    ) -> None:
        """获取并存储单个品种的日频 OHLCV。"""
        logger.info("Fetching %s daily OHLCV", symbol)

        hist = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        if hist.empty:
            logger.warning("No data returned for %s", symbol)
            return

        # 如果 yfinance 返回 MultiIndex 列则展平
        if hasattr(hist.columns, "levels") and hist.columns.nlevels > 1:
            hist.columns = hist.columns.get_level_values(0)

        records = []
        for idx, row in hist.iterrows():
            # NYSE 收盘时间 16:00 ET = 21:00 UTC (EST) 或 20:00 UTC (EDT)
            # 统一使用 21:00 UTC 作为稳定约定
            ts = datetime(idx.year, idx.month, idx.day, 21, 0, tzinfo=_UTC)
            records.append({
                "symbol": symbol,
                "event_time": ts,
                "known_time": ts,  # 行情数据收盘时即公开
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
                "adj_close": float(row.get("Adj Close", row["Close"])),
                "source": "yahoo",
            })

        df = pl.DataFrame(records)
        self._store.append_raw(df, "ohlcv", "yahoo", symbol)
        self._store.merge_to_clean("ohlcv", symbol, dedup_keys=["symbol", "event_time"])
        logger.info("Stored %d bars for %s", len(df), symbol)
