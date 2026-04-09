"""IBKR 历史 K 线数据源，基于 ``ib_insync``。

从 Interactive Brokers TWS/Gateway 获取日内和日频 K 线。
合约规格来自 ``InstrumentRegistry`` -- 无硬编码品种字典。

需要 ``ib_insync``（``pip install ib_insync``）。导入本模块无需该库；
实例化 ``IBKRHistoricalSource`` 时若缺少该库会抛出明确的 ``ImportError``。
"""
from __future__ import annotations

import asyncio
import logging
import re
import time as _time
from datetime import UTC, datetime, timedelta

import polars as pl

from quant.core.types import Instrument
from quant.data.sources._rate_limiter import IBKRRateLimiter
from quant.data.sources._trading_day import compute_bars_per_day
from quant.data.storage.schemas import OHLCV_SCHEMA
from quant.instrument.registry import InstrumentRegistry

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Contract
    from ib_insync import util as ib_util  # noqa: F401

    _HAS_IB_INSYNC = True
except ImportError:
    _HAS_IB_INSYNC = False


# ---------------------------------------------------------------------------
# K 线周期解析
# ---------------------------------------------------------------------------

_BAR_SIZE_RE = re.compile(r"(\d+)\s*(min|mins|sec|secs|hour|hours|day|days)")

_UNIT_TO_SECONDS = {
    "sec": 1,
    "secs": 1,
    "min": 60,
    "mins": 60,
    "hour": 3600,
    "hours": 3600,
    "day": 86400,
    "days": 86400,
}


def _parse_bar_seconds(bar_size: str) -> int:
    """将 IBKR K 线周期字符串解析为秒数。

    示例：``"15 mins"`` -> 900, ``"1 hour"`` -> 3600, ``"5 secs"`` -> 5。
    """
    m = _BAR_SIZE_RE.match(bar_size.strip())
    if not m:
        raise ValueError(f"Cannot parse bar size: {bar_size!r}")
    count = int(m.group(1))
    unit = m.group(2)
    return count * _UNIT_TO_SECONDS[unit]


def bars_per_day(instrument: Instrument, bar_size: str) -> int:
    """计算品种每个交易日预期的 K 线数量。

    Parameters
    ----------
    instrument : Instrument
        合约规格，包含交易时段窗口。
    bar_size : str
        IBKR K 线周期字符串（例如 ``"15 mins"``）。

    Returns
    -------
    int
        每个交易日的 K 线数量（最小为 1）。
    """
    bar_secs = _parse_bar_seconds(bar_size)
    return max(1, compute_bars_per_day(instrument, bar_secs))


# ---------------------------------------------------------------------------
# IBKR 合约构建器
# ---------------------------------------------------------------------------

def _make_contract(instrument: Instrument) -> Contract:
    """从 ``Instrument`` 规格构建 ``ib_insync.Contract``。"""
    contract = Contract()
    contract.symbol = instrument.symbol
    contract.secType = "FUT"
    contract.exchange = instrument.exchange
    contract.currency = instrument.currency.value
    return contract


# ---------------------------------------------------------------------------
# IBKRHistoricalSource
# ---------------------------------------------------------------------------

class IBKRHistoricalSource:
    """从 IBKR TWS/Gateway 获取历史 K 线数据。

    Parameters
    ----------
    registry : InstrumentRegistry
        用于将品种代码解析为合约规格。
    host : str
        TWS/Gateway 主机（默认 ``"127.0.0.1"``）。
    port : int
        TWS/Gateway 端口（默认 ``7497``）。
    client_id : int
        连接的客户端 ID（默认 ``10``）。
    rate_limiter : IBKRRateLimiter or None
        可选的自定义速率限制器。若为 None，创建默认实例。
    source_tag : str
        ``source`` 列的值（默认 ``"ibkr"``）。
    """

    def __init__(
        self,
        registry: InstrumentRegistry,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 10,
        rate_limiter: IBKRRateLimiter | None = None,
        source_tag: str = "ibkr",
    ) -> None:
        if not _HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync is required for IBKRHistoricalSource. "
                "Install it with: pip install ib_insync"
            )
        self._registry = registry
        self._host = host
        self._port = port
        self._client_id = client_id
        self._limiter = rate_limiter or IBKRRateLimiter()
        self._source = source_tag
        self._ib: IB | None = None

    def _ensure_connected(self) -> IB:
        """确保存在活跃的 IBKR 连接。"""
        if self._ib is None or not self._ib.isConnected():
            self._ib = IB()
            self._ib.connect(
                self._host, self._port, clientId=self._client_id
            )
            logger.info(
                "IBKR: connected to %s:%d (client_id=%d)",
                self._host,
                self._port,
                self._client_id,
            )
        return self._ib

    def disconnect(self) -> None:
        """断开 IBKR 连接。"""
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("IBKR: disconnected")
        self._ib = None

    def fetch_bars(
        self,
        symbol: str,
        bar_size: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """获取单个品种的历史 K 线。

        Parameters
        ----------
        symbol : str
            品种代码（必须在注册表中）。
        bar_size : str
            IBKR K 线周期字符串（例如 ``"15 mins"``、``"1 hour"``）。
        start, end : datetime
            时间范围（UTC）。

        Returns
        -------
        pl.DataFrame
            符合 ``OHLCV_SCHEMA``。
        """
        instrument = self._registry.get(symbol)
        ib = self._ensure_connected()
        contract = _make_contract(instrument)

        # 确认合约以获取正确的到期日
        ib.qualifyContracts(contract)

        bar_secs = _parse_bar_seconds(bar_size)

        # IBKR 根据 K 线周期限制每次请求的最大时长。
        # 例如 15 分钟 K 线最多约 20 天/次。按此分片请求。
        max_duration_days = self._max_duration_days(bar_secs)
        all_bars: list[dict] = []

        chunk_end = end
        while chunk_end > start:
            chunk_start = max(start, chunk_end - timedelta(days=max_duration_days))
            duration_str = self._duration_string(chunk_start, chunk_end)

            self._acquire_sync(symbol)

            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=chunk_end.strftime("%Y%m%d-%H:%M:%S"),
                    durationStr=duration_str,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,  # UTC
                )
                self._limiter.report_success()
            except Exception as exc:
                if "pacing" in str(exc).lower():
                    delay = self._limiter.report_pacing_violation()
                    _time.sleep(delay)
                    continue  # 重试当前分片
                raise

            if not bars:
                break

            for bar in bars:
                ts = bar.date
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)

                all_bars.append(
                    {
                        "symbol": symbol,
                        "event_time": ts,
                        "known_time": ts,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "adj_close": None,
                        "source": self._source,
                    }
                )

            # 向前移动窗口
            earliest = bars[0].date
            if isinstance(earliest, str):
                earliest = datetime.fromisoformat(earliest)
            if earliest.tzinfo is None:
                earliest = earliest.replace(tzinfo=UTC)
            chunk_end = earliest

        if not all_bars:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        df = pl.DataFrame(all_bars)

        # 转换为正确的 schema 类型
        df = df.with_columns(
            pl.col("event_time").cast(pl.Datetime("us", "UTC")),
            pl.col("known_time").cast(pl.Datetime("us", "UTC")),
        )

        # 过滤到请求范围并排序
        df = df.filter(
            (pl.col("event_time") >= start) & (pl.col("event_time") <= end)
        ).sort("event_time").unique(subset=["event_time"], keep="last")

        return df.select(list(OHLCV_SCHEMA.keys()))

    def fetch_batch(
        self,
        symbols: list[str],
        bar_size: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """批量获取多个品种的 K 线。

        Parameters
        ----------
        symbols : list[str]
            品种列表。
        bar_size, start, end
            同 ``fetch_bars``。

        Returns
        -------
        pl.DataFrame
            合并后的 OHLCV 数据。
        """
        frames: list[pl.DataFrame] = []
        for sym in symbols:
            try:
                df = self.fetch_bars(sym, bar_size, start, end)
                if not df.is_empty():
                    frames.append(df)
            except Exception:
                logger.exception("IBKR: failed to fetch %s", sym)

        if not frames:
            return pl.DataFrame(schema=OHLCV_SCHEMA)
        return pl.concat(frames)

    def update_incremental(
        self,
        symbol: str,
        bar_size: str,
        store: object,
    ) -> pl.DataFrame:
        """从上次存储的时间戳开始增量获取数据。

        Parameters
        ----------
        symbol : str
            品种代码。
        bar_size : str
            IBKR K 线周期字符串。
        store : ParquetStore
            存储实例，用于确定最新可用日期。

        Returns
        -------
        pl.DataFrame
            新获取的行。
        """
        from quant.data.storage.parquet_store import ParquetStore

        if not isinstance(store, ParquetStore):
            raise TypeError(f"Expected ParquetStore, got {type(store).__name__}")

        # 查找最后时间戳
        last_ts: datetime | None = None
        try:
            existing = store.read_clean("ohlcv", symbol)
            if not existing.is_empty():
                last_ts = existing["event_time"].max()
        except FileNotFoundError:
            pass

        if last_ts is not None:
            start = last_ts + timedelta(seconds=1)
        else:
            # 默认：30 天历史数据
            start = datetime.now(UTC) - timedelta(days=30)

        end = datetime.now(UTC)

        df = self.fetch_bars(symbol, bar_size, start, end)

        if not df.is_empty():
            store.append_raw(df, "ohlcv", self._source, symbol)
            logger.info(
                "IBKR: incremental update for %s: %d new bars", symbol, len(df)
            )

        return df

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _acquire_sync(self, symbol: str) -> None:
        """异步速率限制器的同步包装。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # 在事件循环内（例如 ib_insync 的循环）。
            # 使用 ib_insync 的 util.run 或创建任务。
            loop.run_until_complete(self._limiter.acquire(symbol))
        else:
            asyncio.run(self._limiter.acquire(symbol))

    @staticmethod
    def _max_duration_days(bar_secs: int) -> int:
        """IBKR 根据 K 线周期限制请求时长。"""
        if bar_secs <= 5:
            return 1
        elif bar_secs <= 60:
            return 5
        elif bar_secs <= 900:
            return 20
        elif bar_secs <= 3600:
            return 60
        else:
            return 365

    @staticmethod
    def _duration_string(start: datetime, end: datetime) -> str:
        """根据时间范围构建 IBKR duration 字符串。"""
        delta = end - start
        days = delta.days
        if days <= 0:
            days = 1
        if days <= 365:
            return f"{days} D"
        else:
            months = days // 30
            return f"{months} M"
