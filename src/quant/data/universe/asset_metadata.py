"""品种 Universe 的资产元数据管理。

存储每个品种的元数据（asset_type、exchange、currency、sector、
market_cap_tier），用于 vol-scaling、分类特征和筛选。

元数据通过 Yahoo Finance 的 ``info`` 接口获取，缓存到本地 JSON 文件。
重复获取通过节流控制避免触发速率限制。

用法::

    meta = AssetMetadata("data/universe/metadata.json")
    meta.fetch_batch(["AAPL", "MSFT", "GOOG"])
    info = meta.get("AAPL")
    print(info["sector"], info["market_cap_tier"])
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tqdm import tqdm

logger = logging.getLogger(__name__)

_UTC = ZoneInfo("UTC")


@dataclass
class TickerMetadata:
    """单个品种的元数据。"""

    symbol: str
    asset_type: str = "unknown"  # stock, etf, future, crypto, index
    exchange: str = ""
    currency: str = "USD"
    sector: str = ""
    industry: str = ""
    market_cap: float | None = None
    market_cap_tier: str = ""  # mega, large, mid, small, micro, nano
    name: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TickerMetadata:
        # 仅保留已知字段
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def _classify_market_cap(market_cap: float | None) -> str:
    """将市值分类为层级。"""
    if market_cap is None:
        return ""
    if market_cap >= 200e9:
        return "mega"
    if market_cap >= 10e9:
        return "large"
    if market_cap >= 2e9:
        return "mid"
    if market_cap >= 300e6:
        return "small"
    if market_cap >= 50e6:
        return "micro"
    return "nano"


def _infer_asset_type(info: dict, symbol: str) -> str:
    """从 Yahoo Finance info 字典推断资产类型。"""
    quote_type = info.get("quoteType", "").upper()
    if quote_type == "ETF":
        return "etf"
    if quote_type == "CRYPTOCURRENCY":
        return "crypto"
    if quote_type == "FUTURE":
        return "future"
    if quote_type == "INDEX":
        return "index"
    if quote_type == "EQUITY":
        return "stock"
    # 回退启发式规则
    if "=F" in symbol:
        return "future"
    if "-USD" in symbol:
        return "crypto"
    return "stock"


class AssetMetadata:
    """基于本地 JSON 文件缓存的资产元数据。

    Parameters
    ----------
    cache_path : str or Path
        JSON 缓存文件路径。
    inter_request_sleep : float
        Yahoo Finance info 请求之间的 sleep 秒数。
    """

    def __init__(
        self,
        cache_path: str | Path = "data/universe/metadata.json",
        inter_request_sleep: float = 0.5,
    ) -> None:
        self._cache_path = Path(cache_path)
        self._inter_request_sleep = inter_request_sleep
        self._data: dict[str, TickerMetadata] = {}

        if self._cache_path.exists():
            self._load_cache()
            logger.info("Loaded metadata cache: %d entries", len(self._data))

    # -- 查询 -------------------------------------------------------------------

    def get(self, symbol: str) -> TickerMetadata | None:
        """获取品种的元数据，未缓存返回 None。"""
        return self._data.get(symbol)

    def get_all(self) -> dict[str, TickerMetadata]:
        """返回所有缓存的元数据。"""
        return dict(self._data)

    def symbols(self) -> list[str]:
        """列出所有有缓存元数据的品种。"""
        return sorted(self._data.keys())

    def filter_by(
        self,
        asset_type: str | None = None,
        sector: str | None = None,
        market_cap_tier: str | None = None,
        exchange: str | None = None,
        currency: str | None = None,
    ) -> list[str]:
        """按元数据字段筛选缓存的品种。

        返回匹配品种的列表。
        """
        results = []
        for sym, meta in self._data.items():
            if asset_type and meta.asset_type != asset_type:
                continue
            if sector and meta.sector != sector:
                continue
            if market_cap_tier and meta.market_cap_tier != market_cap_tier:
                continue
            if exchange and meta.exchange != exchange:
                continue
            if currency and meta.currency != currency:
                continue
            results.append(sym)
        return sorted(results)

    @property
    def count(self) -> int:
        return len(self._data)

    # -- 获取 -------------------------------------------------------------------

    def fetch_batch(
        self,
        symbols: list[str],
        skip_cached: bool = True,
    ) -> tuple[int, int]:
        """从 Yahoo Finance 批量获取品种元数据。

        Parameters
        ----------
        symbols : list[str]
            待获取元数据的品种列表。
        skip_cached : bool
            若为 True，跳过已在缓存中的品种。

        Returns
        -------
        tuple[int, int]
            (成功数, 失败数) 计数。
        """
        import yfinance as yf

        to_fetch = symbols
        if skip_cached:
            to_fetch = [s for s in symbols if s not in self._data]

        if not to_fetch:
            logger.info("All %d symbols already cached", len(symbols))
            return 0, 0

        logger.info("Fetching metadata for %d symbols", len(to_fetch))
        succeeded = 0
        failed = 0

        for symbol in tqdm(to_fetch, desc="Fetching metadata", unit="ticker"):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info or {}

                if not info or info.get("regularMarketPrice") is None:
                    # 可获取的信息有限，仍存储已有数据
                    pass

                market_cap = info.get("marketCap")
                meta = TickerMetadata(
                    symbol=symbol,
                    asset_type=_infer_asset_type(info, symbol),
                    exchange=info.get("exchange", ""),
                    currency=info.get("currency", "USD"),
                    sector=info.get("sector", ""),
                    industry=info.get("industry", ""),
                    market_cap=market_cap,
                    market_cap_tier=_classify_market_cap(market_cap),
                    name=info.get("shortName", info.get("longName", "")),
                    fetched_at=datetime.now(_UTC).isoformat(),
                )
                self._data[symbol] = meta
                succeeded += 1

            except Exception as e:
                logger.debug("Failed to fetch metadata for %s: %s", symbol, e)
                failed += 1

            time.sleep(self._inter_request_sleep)

        self._save_cache()
        logger.info("Metadata fetch complete: %d succeeded, %d failed", succeeded, failed)
        return succeeded, failed

    # -- 持久化 ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """保存元数据缓存到 JSON。"""
        out = Path(path) if path else self._cache_path
        self._save_cache(out)
        return out

    def _save_cache(self, path: Path | None = None) -> None:
        out = path or self._cache_path
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {sym: meta.to_dict() for sym, meta in self._data.items()}
        with open(out, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.debug("Saved metadata cache: %d entries to %s", len(data), out)

    def _load_cache(self) -> None:
        with open(self._cache_path) as fh:
            raw = json.load(fh)
        for sym, data in raw.items():
            self._data[sym] = TickerMetadata.from_dict(data)

    # -- 手动设置 ----------------------------------------------------------------

    def set_metadata(self, symbol: str, **kwargs) -> None:
        """手动设置或更新品种元数据。

        适用于期货等 Yahoo info 信息不全的品种。
        """
        if symbol in self._data:
            for k, v in kwargs.items():
                if hasattr(self._data[symbol], k):
                    setattr(self._data[symbol], k, v)
        else:
            self._data[symbol] = TickerMetadata(symbol=symbol, **kwargs)
