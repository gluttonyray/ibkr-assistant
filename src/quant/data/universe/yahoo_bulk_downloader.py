"""基于 Yahoo Finance 的批量日频 OHLCV 下载器。

为 10,000+ 品种下载日频 K 线，具备：
- 速率限制（批次 + sleep）
- 断点续传（跳过已下载的品种）
- 增量更新（仅获取新 K 线）
- 失败品种的错误日志
- 进度条（tqdm）

数据通过 :class:`ParquetStore` 写入，落入标准的
三层 Parquet 系统并附带正确的 PIT 时间戳。

用法::

    store = ParquetStore("data/parquet")
    universe = TickerUniverse()
    universe.load_defaults()

    downloader = YahooBulkDownloader(store)
    report = downloader.download_universe(universe.get_tickers())
    print(report)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import yfinance as yf
from tqdm import tqdm

from quant.data.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

_UTC = ZoneInfo("UTC")


@dataclass
class DownloadReport:
    """批量下载运行的汇总报告。"""

    succeeded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (品种, 错误信息)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.skipped) + len(self.failed)

    def __str__(self) -> str:
        mins = self.elapsed_seconds / 60
        lines = [
            f"Download report: {self.total} tickers in {mins:.1f} min",
            f"  Succeeded: {len(self.succeeded)}",
            f"  Skipped (already up-to-date): {len(self.skipped)}",
            f"  Failed: {len(self.failed)}",
        ]
        if self.failed:
            lines.append("  Failed tickers:")
            for ticker, err in self.failed[:20]:
                lines.append(f"    {ticker}: {err}")
            if len(self.failed) > 20:
                lines.append(f"    ... and {len(self.failed) - 20} more")
        return "\n".join(lines)

    def save_error_log(self, path: str | Path) -> None:
        """将失败品种写入 JSON 文件以便重试。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.now(_UTC).isoformat(),
            "failed_count": len(self.failed),
            "failed": [{"ticker": t, "error": e} for t, e in self.failed],
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Error log saved to %s", path)


class YahooBulkDownloader:
    """从 Yahoo Finance 批量下载日频 OHLCV 到 ParquetStore。

    Parameters
    ----------
    store : ParquetStore
        目标 Parquet 存储。
    batch_size : int
        每次 yfinance 批量调用的品种数。
        更大的批次更高效但部分失败风险更高。
    inter_batch_sleep : float
        批次间的 sleep 秒数，避免触发速率限制。
    max_retries : int
        单个失败品种的重试次数。
    """

    def __init__(
        self,
        store: ParquetStore,
        batch_size: int = 50,
        inter_batch_sleep: float = 2.0,
        max_retries: int = 2,
    ) -> None:
        self._store = store
        self._batch_size = batch_size
        self._inter_batch_sleep = inter_batch_sleep
        self._max_retries = max_retries

    def download_universe(
        self,
        tickers: list[str],
        start: str = "2010-01-01",
        end: str | None = None,
        skip_existing: bool = True,
        incremental: bool = True,
    ) -> DownloadReport:
        """下载品种列表的日频 OHLCV。

        Parameters
        ----------
        tickers : list[str]
            待下载的品种代码。
        start : str
            起始日期（YYYY-MM-DD）。
        end : str or None
            结束日期，默认为今天。
        skip_existing : bool
            若为 True，跳过已有数据覆盖到昨天（或 *end*）的品种。
            被 *incremental* 覆盖。
        incremental : bool
            若为 True，对已有数据的品种仅下载最后存储日期之后的 K 线。
            优先于 *skip_existing*。

        Returns
        -------
        DownloadReport
            包含成功、跳过和失败的汇总。
        """
        report = DownloadReport(start_time=time.monotonic())

        to_download: list[tuple[str, str]] = []  # (品种, 实际起始日期)

        existing_partitions = set(self._store.list_partitions("ohlcv", layer="clean"))

        for ticker in tickers:
            partition_key = _ticker_to_partition(ticker)
            if partition_key in existing_partitions:
                if incremental:
                    last_date = self._get_last_date(partition_key)
                    if last_date:
                        # 从最后存储的 K 线次日开始
                        effective_start = (
                            last_date.replace(tzinfo=None).strftime("%Y-%m-%d")
                        )
                        if end and effective_start >= end:
                            report.skipped.append(ticker)
                            continue
                        to_download.append((ticker, effective_start))
                    else:
                        to_download.append((ticker, start))
                elif skip_existing:
                    report.skipped.append(ticker)
                    continue
                else:
                    to_download.append((ticker, start))
            else:
                to_download.append((ticker, start))

        if not to_download:
            report.end_time = time.monotonic()
            logger.info("Nothing to download, all %d tickers up-to-date", len(tickers))
            return report

        logger.info(
            "Downloading %d tickers (%d skipped as up-to-date)",
            len(to_download), len(report.skipped),
        )

        # 按起始日期分组以便批量处理（大多数品种共享相同起始日期）
        by_start: dict[str, list[str]] = {}
        for ticker, eff_start in to_download:
            by_start.setdefault(eff_start, []).append(ticker)

        progress = tqdm(total=len(to_download), desc="Downloading", unit="ticker")

        for batch_start, batch_tickers in by_start.items():
            for i in range(0, len(batch_tickers), self._batch_size):
                batch = batch_tickers[i : i + self._batch_size]
                self._download_batch(batch, batch_start, end, report, progress)

                if i + self._batch_size < len(batch_tickers):
                    time.sleep(self._inter_batch_sleep)

        progress.close()
        report.end_time = time.monotonic()

        logger.info("%s", report)
        return report

    def download_tickers(
        self,
        tickers: list[str],
        start: str = "2010-01-01",
        end: str | None = None,
    ) -> DownloadReport:
        """下载指定品种，不使用跳过/增量逻辑。

        用于下载少量明确指定品种的便捷包装。
        """
        return self.download_universe(
            tickers, start=start, end=end, skip_existing=False, incremental=False,
        )

    def retry_failed(
        self,
        error_log_path: str | Path,
        start: str = "2010-01-01",
        end: str | None = None,
    ) -> DownloadReport:
        """从之前的错误日志中重试失败品种。

        Parameters
        ----------
        error_log_path : str or Path
            上次运行的 JSON 错误日志路径。
        start : str
            起始日期。
        end : str or None
            结束日期。

        Returns
        -------
        DownloadReport
        """
        with open(error_log_path) as fh:
            data = json.load(fh)

        tickers = [entry["ticker"] for entry in data["failed"]]
        logger.info("Retrying %d failed tickers from %s", len(tickers), error_log_path)
        return self.download_universe(
            tickers, start=start, end=end, skip_existing=False, incremental=False,
        )

    # -- 内部方法 ---------------------------------------------------------------

    def _download_batch(
        self,
        tickers: list[str],
        start: str,
        end: str | None,
        report: DownloadReport,
        progress: tqdm,
    ) -> None:
        """使用 yfinance 批量下载一组品种。"""
        ticker_str = " ".join(tickers)

        try:
            data = yf.download(
                ticker_str,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as e:
            logger.error("Batch download failed: %s", e)
            for ticker in tickers:
                report.failed.append((ticker, str(e)))
            progress.update(len(tickers))
            return

        if data.empty:
            for ticker in tickers:
                report.failed.append((ticker, "No data returned"))
            progress.update(len(tickers))
            return

        for ticker in tickers:
            try:
                self._process_single(ticker, data, tickers, report)
            except Exception as e:
                logger.debug("Failed to process %s: %s", ticker, e)
                # 逐个重试
                if not self._retry_single(ticker, start, end, report):
                    report.failed.append((ticker, str(e)))
            progress.update(1)

    def _process_single(
        self,
        ticker: str,
        data,
        batch_tickers: list[str],
        report: DownloadReport,
    ) -> None:
        """从批量结果中提取并存储单个品种的数据。"""
        if len(batch_tickers) == 1:
            hist = data
        else:
            # 多品种下载返回 MultiIndex 列
            try:
                hist = data[ticker]
            except KeyError:
                raise ValueError(f"Ticker {ticker} not in batch result")

        # 如果存在 MultiIndex 列则展平
        if hasattr(hist.columns, "levels") and hist.columns.nlevels > 1:
            hist.columns = hist.columns.get_level_values(0)

        # 丢弃所有 OHLCV 均为 NaN 的行（品种可能有数据空档）
        hist = hist.dropna(how="all")
        if hist.empty:
            raise ValueError(f"No valid data for {ticker}")

        records = []
        for idx, row in hist.iterrows():
            # 跳过收盘价为 NaN 的行（关键字段）
            close_val = row.get("Close")
            if close_val is None or (isinstance(close_val, float) and close_val != close_val):
                continue

            ts = datetime(idx.year, idx.month, idx.day, 21, 0, tzinfo=_UTC)
            records.append({
                "symbol": ticker,
                "event_time": ts,
                "known_time": ts,
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": float(close_val),
                "volume": _safe_float(row.get("Volume", 0.0)),
                "adj_close": _safe_float(row.get("Adj Close", close_val)),
                "source": "yahoo",
            })

        if not records:
            raise ValueError(f"No valid rows after filtering for {ticker}")

        df = pl.DataFrame(records)
        partition_key = _ticker_to_partition(ticker)
        self._store.append_raw(df, "ohlcv", "yahoo", partition_key)
        self._store.merge_to_clean("ohlcv", partition_key, dedup_keys=["symbol", "event_time"])

        report.succeeded.append(ticker)
        logger.debug("Stored %d bars for %s", len(df), ticker)

    def _retry_single(
        self,
        ticker: str,
        start: str,
        end: str | None,
        report: DownloadReport,
    ) -> bool:
        """逐个重试单个品种的下载。"""
        for attempt in range(1, self._max_retries + 1):
            time.sleep(self._inter_batch_sleep * attempt)
            try:
                hist = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                )
                if hist.empty:
                    continue

                if hasattr(hist.columns, "levels") and hist.columns.nlevels > 1:
                    hist.columns = hist.columns.get_level_values(0)

                self._process_single(ticker, hist, [ticker], report)
                return True
            except Exception as e:
                logger.debug("Retry %d/%d for %s failed: %s", attempt, self._max_retries, ticker, e)

        return False

    def _get_last_date(self, partition_key: str) -> datetime | None:
        """获取 clean 存储中某分区的最后 event_time。"""
        try:
            df = self._store.read_clean("ohlcv", partition_key)
            if df.is_empty():
                return None
            return df.select(pl.col("event_time").max()).item()
        except FileNotFoundError:
            return None


def _ticker_to_partition(ticker: str) -> str:
    """将品种代码转换为安全的文件系统分区键。

    替换文件路径中有问题的字符：
    ``=``（期货）、``^``（指数）、``.``（港股）。
    """
    return ticker.replace("=", "_").replace("^", "_").replace(".", "_")


def _safe_float(val) -> float:
    """转换为 float，NaN/None 返回 0.0。"""
    if val is None:
        return 0.0
    f = float(val)
    if f != f:  # NaN 检查
        return 0.0
    return f
