"""从交易所官方来源自动构建 10,000+ 品种 Universe。

数据来源：
- NASDAQ 上市品种（~3,500）：nasdaqtrader.com 官方品种目录
- NYSE/AMEX 上市品种（~2,300）：nasdaqtrader.com 其他交易所品种
- 美国期货：静态列表（CME E-mini、国债、大宗商品、外汇）
- 港股：HSI 成分 + 默认列表
- 加密货币：Top-50 主流币

品种列表经过过滤（排除外国私募、测试品种、权证等），
结果填充到 :class:`TickerUniverse` 并可选缓存到本地。

用法::

    builder = UniverseBuilder()
    universe = builder.build()
    print(universe.summary())
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from quant.data.universe.ticker_universe import (
    _DEFAULT_TICKERS,
    AssetClass,
    TickerUniverse,
)

logger = logging.getLogger(__name__)

# NASDAQ 官方品种目录 URL
_NASDAQ_LISTED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)
_OTHER_LISTED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
)

# 过滤规则：排除含这些关键词的品种（case insensitive）
_EXCLUDE_KEYWORDS = re.compile(r"warr|unit|right", re.IGNORECASE)

# Top-50 加密货币（扩展默认列表）
_CRYPTO_TOP50 = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "MATIC-USD",
    "LINK-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "ETC-USD",
    "XLM-USD", "NEAR-USD", "BCH-USD", "FIL-USD", "APT-USD",
    "ICP-USD", "HBAR-USD", "VET-USD", "ALGO-USD", "QNT-USD",
    "ARB-USD", "OP-USD", "GRT-USD", "AAVE-USD", "MKR-USD",
    "STX-USD", "SAND-USD", "MANA-USD", "AXS-USD", "THETA-USD",
    "FTM-USD", "RUNE-USD", "EGLD-USD", "XTZ-USD", "FLOW-USD",
    "IMX-USD", "KAVA-USD", "MINA-USD", "SNX-USD", "CRV-USD",
    "LDO-USD", "RPL-USD", "INJ-USD", "SUI-USD", "SEI-USD",
]


def _is_valid_us_ticker(symbol: str) -> bool:
    """检查美股品种代码是否合法。

    过滤规则：
    1. 排除含 ^ 或空格的品种代码
    2. 排除代码长度 > 5 的美股（外国私募）
    3. 排除含 $ 的代码（测试品种）
    4. 保留 =F 结尾的期货连续合约
    5. 过滤含 warr/unit/right 的品种
    """
    if not symbol or not symbol.strip():
        return False
    # 期货连续合约始终保留
    if symbol.endswith("=F"):
        return True
    if " " in symbol or "^" in symbol or "$" in symbol:
        return False
    if len(symbol) > 5:
        return False
    if _EXCLUDE_KEYWORDS.search(symbol):
        return False
    return True


class UniverseBuilder:
    """从交易所官方来源自动构建品种 Universe。

    Parameters
    ----------
    cache_dir : Path or str
        HTTP 响应缓存目录。
    cache_ttl_days : int
        缓存有效期（天）。过期后重新下载。
    """

    def __init__(
        self,
        cache_dir: Path | str = "data/universe_cache",
        cache_ttl_days: int = 7,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_ttl_days = cache_ttl_days

    def build(
        self,
        include_stocks: bool = True,
        include_etfs: bool = True,
        include_futures: bool = True,
        include_crypto: bool = True,
        include_hk: bool = True,
        max_stocks: int | None = None,
    ) -> TickerUniverse:
        """构建完整品种 Universe。

        Parameters
        ----------
        include_stocks : bool
            是否包含美股个股。
        include_etfs : bool
            是否包含美股 ETF。
        include_futures : bool
            是否包含美国期货。
        include_crypto : bool
            是否包含加密货币。
        include_hk : bool
            是否包含港股。
        max_stocks : int or None
            美股个股数量上限（None 表示不限制）。

        Returns
        -------
        TickerUniverse
            填充好的品种 Universe 实例。
        """
        universe = TickerUniverse()

        if include_stocks or include_etfs:
            # 从 NASDAQ 官方目录获取品种列表
            nasdaq_stocks, nasdaq_etfs = self._fetch_nasdaq_listed()
            other_stocks, other_etfs = self._fetch_other_listed()

            all_stocks = nasdaq_stocks + other_stocks
            all_etfs = nasdaq_etfs + other_etfs

            if include_stocks:
                stocks = all_stocks
                if max_stocks is not None:
                    stocks = stocks[:max_stocks]
                universe.add_tickers(stocks, AssetClass.US_STOCKS)
                logger.info("添加 %d 只美股个股", len(stocks))

            if include_etfs:
                # 合并默认 ETF 列表
                default_etfs = _DEFAULT_TICKERS.get(AssetClass.US_ETFS, [])
                combined_etfs = list(set(all_etfs + default_etfs))
                universe.add_tickers(combined_etfs, AssetClass.US_ETFS)
                logger.info("添加 %d 只美股 ETF", len(combined_etfs))

        if include_futures:
            # 复用默认期货列表
            futures = _DEFAULT_TICKERS.get(AssetClass.US_FUTURES, [])
            universe.add_tickers(futures, AssetClass.US_FUTURES)
            logger.info("添加 %d 只美国期货", len(futures))

        if include_crypto:
            # 扩展到 Top-50 加密货币
            universe.add_tickers(_CRYPTO_TOP50, AssetClass.CRYPTO)
            logger.info("添加 %d 只加密货币", len(_CRYPTO_TOP50))

        if include_hk:
            # 复用默认港股列表
            hk_tickers = _DEFAULT_TICKERS.get(AssetClass.HK_STOCKS, [])
            universe.add_tickers(hk_tickers, AssetClass.HK_STOCKS)
            logger.info("添加 %d 只港股", len(hk_tickers))

        logger.info(
            "Universe 构建完成: %d 品种 (%s)",
            universe.total_count,
            universe.summary(),
        )
        return universe

    def _fetch_nasdaq_listed(self) -> tuple[list[str], list[str]]:
        """从 NASDAQ 官方目录获取品种列表。

        Returns
        -------
        tuple[list[str], list[str]]
            (stocks, etfs) 两个列表。
        """
        raw = self._load_or_fetch(_NASDAQ_LISTED_URL, "nasdaqlisted")
        stocks: list[str] = []
        etfs: list[str] = []

        for line in raw.strip().splitlines():
            # 跳过表头和文件创建时间行
            if line.startswith("Symbol|") or line.startswith("File Creation Time"):
                continue

            parts = line.split("|")
            if len(parts) < 2:
                continue

            symbol = parts[0].strip()
            if not _is_valid_us_ticker(symbol):
                continue

            # ETF 标记在第 7 列（索引 6），值为 "Y" 或 "N"
            is_etf = len(parts) > 6 and parts[6].strip().upper() == "Y"

            if is_etf:
                etfs.append(symbol)
            else:
                stocks.append(symbol)

        logger.info(
            "NASDAQ 列表: %d stocks, %d ETFs", len(stocks), len(etfs)
        )
        return stocks, etfs

    def _fetch_other_listed(self) -> tuple[list[str], list[str]]:
        """从 NYSE/AMEX 其他交易所获取品种列表。

        Returns
        -------
        tuple[list[str], list[str]]
            (stocks, etfs) 两个列表。
        """
        raw = self._load_or_fetch(_OTHER_LISTED_URL, "otherlisted")
        stocks: list[str] = []
        etfs: list[str] = []

        for line in raw.strip().splitlines():
            # 跳过表头和文件创建时间行
            if line.startswith("ACT Symbol|") or line.startswith(
                "File Creation Time"
            ):
                continue

            parts = line.split("|")
            if len(parts) < 2:
                continue

            # ACT Symbol 在第 1 列（索引 0）
            symbol = parts[0].strip()
            if not _is_valid_us_ticker(symbol):
                continue

            # ETF 标记在第 5 列（索引 4），值为 "Y" 或 "N"
            is_etf = len(parts) > 4 and parts[4].strip().upper() == "Y"

            if is_etf:
                etfs.append(symbol)
            else:
                stocks.append(symbol)

        logger.info(
            "其他交易所列表: %d stocks, %d ETFs", len(stocks), len(etfs)
        )
        return stocks, etfs

    def _load_or_fetch(self, url: str, cache_key: str) -> str:
        """从缓存或 HTTP 获取数据，带 7 天 TTL。

        Parameters
        ----------
        url : str
            远程 URL。
        cache_key : str
            本地缓存文件名前缀。

        Returns
        -------
        str
            响应文本内容。
        """
        cache_file = self._cache_dir / f"{cache_key}.txt"

        # 检查缓存是否有效
        if cache_file.exists():
            age_seconds = time.time() - cache_file.stat().st_mtime
            age_days = age_seconds / 86400
            if age_days < self._cache_ttl_days:
                logger.debug("缓存命中: %s (%.1f 天)", cache_key, age_days)
                return cache_file.read_text(encoding="utf-8")
            logger.debug("缓存过期: %s (%.1f 天)", cache_key, age_days)

        # 尝试 HTTP 下载
        try:
            logger.info("正在下载: %s", url)
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")

            # 写入缓存
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(data, encoding="utf-8")
            logger.info("已缓存: %s (%d bytes)", cache_key, len(data))
            return data

        except (URLError, OSError) as e:
            logger.warning("下载失败: %s — %s", url, e)
            # 失败时回退到过期缓存
            if cache_file.exists():
                logger.info("使用过期缓存: %s", cache_key)
                return cache_file.read_text(encoding="utf-8")
            raise RuntimeError(
                f"无法获取 {url}，且无本地缓存可用"
            ) from e
