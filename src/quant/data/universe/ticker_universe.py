"""大规模数据管道的品种 Universe 管理。

管理 10,000+ 品种的分类列表。
支持从 CSV 文件加载（例如 NASDAQ/NYSE 官方品种列表）
及按资产类别筛选。

用法::

    universe = TickerUniverse()
    universe.load_csv("data/universe/nasdaq_listed.csv", AssetClass.US_STOCKS)
    universe.add_tickers(["SPY", "QQQ", "IWM"], AssetClass.US_ETFS)
    all_us = universe.get_tickers(AssetClass.US_STOCKS, AssetClass.US_ETFS)
"""
from __future__ import annotations

import csv
import enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AssetClass(str, enum.Enum):
    """支持的资产类别分类。"""

    US_STOCKS = "us_stocks"
    HK_STOCKS = "hk_stocks"
    CN_STOCKS = "cn_stocks"
    US_ETFS = "us_etfs"
    US_FUTURES = "us_futures"
    CRYPTO = "crypto"


# 用于快速测试/引导的默认品种列表。
# 均为高流动性、数据覆盖好的品种。
_DEFAULT_TICKERS: dict[AssetClass, list[str]] = {
    AssetClass.US_ETFS: [
        # 宽基股票
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
        # 国际市场
        "EFA", "EEM", "VWO", "VEA",
        # 固定收益
        "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "BND",
        # 大宗商品
        "GLD", "SLV", "USO", "UNG", "DBC", "CPER",
        # 外汇/美元
        "UUP", "FXE", "FXY", "FXB",
        # 行业板块
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE",
        # 波动率
        "VIXY",
    ],
    AssetClass.US_FUTURES: [
        # CME E-mini / Micro（Yahoo 使用连续合约代码）
        "ES=F", "NQ=F", "YM=F", "RTY=F",
        # 国债期货
        "ZB=F", "ZN=F", "ZF=F",
        # 大宗商品
        "GC=F", "SI=F", "CL=F", "NG=F", "HG=F",
        # 外汇
        "6E=F", "6J=F", "6B=F",
    ],
    AssetClass.CRYPTO: [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "MATIC-USD",
    ],
    AssetClass.HK_STOCKS: [
        # 主要港股（Yahoo 格式：ticker.HK）
        "0700.HK", "9988.HK", "9618.HK", "3690.HK", "1810.HK",
        "0005.HK", "0941.HK", "0388.HK", "2318.HK", "1299.HK",
        "0883.HK", "0027.HK", "0011.HK", "0016.HK", "0066.HK",
        "1038.HK", "0002.HK", "0003.HK", "0006.HK", "0012.HK",
        # 港股指数 ETF
        "2800.HK", "2828.HK", "3067.HK",
    ],
}


class TickerUniverse:
    """管理分类品种集合。

    Parameters
    ----------
    snapshot_path : str or Path or None
        JSON 快照文件路径，用于持久化。若文件存在，
        初始化时自动加载。调用 :meth:`save` 保存。
    """

    def __init__(self, snapshot_path: str | Path | None = None) -> None:
        self._tickers: dict[AssetClass, set[str]] = {ac: set() for ac in AssetClass}
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None

        if self._snapshot_path and self._snapshot_path.exists():
            self._load_snapshot()
            logger.info(
                "Loaded universe snapshot: %d total tickers",
                self.total_count,
            )

    # -- 修改操作 ---------------------------------------------------------------

    def add_tickers(self, tickers: list[str], asset_class: AssetClass) -> int:
        """添加品种到分类。返回新增数量。"""
        before = len(self._tickers[asset_class])
        self._tickers[asset_class].update(tickers)
        added = len(self._tickers[asset_class]) - before
        if added:
            logger.debug("Added %d tickers to %s", added, asset_class.value)
        return added

    def remove_tickers(self, tickers: list[str], asset_class: AssetClass) -> int:
        """从分类中移除品种。返回移除数量。"""
        before = len(self._tickers[asset_class])
        self._tickers[asset_class] -= set(tickers)
        removed = before - len(self._tickers[asset_class])
        return removed

    def load_defaults(self) -> int:
        """加载内置默认品种。返回总新增数量。"""
        total = 0
        for ac, tickers in _DEFAULT_TICKERS.items():
            total += self.add_tickers(tickers, ac)
        logger.info("Loaded %d default tickers", total)
        return total

    def load_csv(
        self,
        path: str | Path,
        asset_class: AssetClass,
        ticker_column: str = "Symbol",
        encoding: str = "utf-8",
    ) -> int:
        """从 CSV 文件加载品种（例如 NASDAQ/NYSE 官方列表）。

        Parameters
        ----------
        path : str or Path
            CSV 文件路径。
        asset_class : AssetClass
            加载品种的归属分类。
        ticker_column : str
            包含品种代码的列名。
        encoding : str
            文件编码。

        Returns
        -------
        int
            新增品种数量。
        """
        path = Path(path)
        tickers = []
        with open(path, newline="", encoding=encoding) as fh:
            reader = csv.DictReader(fh)
            if ticker_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Column '{ticker_column}' not found in {path}. "
                    f"Available: {reader.fieldnames}"
                )
            for row in reader:
                symbol = row[ticker_column].strip()
                if symbol and not symbol.startswith("#"):
                    tickers.append(symbol)

        added = self.add_tickers(tickers, asset_class)
        logger.info(
            "Loaded %d tickers from %s (%d new) into %s",
            len(tickers), path.name, added, asset_class.value,
        )
        return added

    # -- 查询操作 ----------------------------------------------------------------

    def get_tickers(self, *asset_classes: AssetClass) -> list[str]:
        """返回指定分类的排序品种列表。

        若未指定分类，返回所有品种。
        """
        if not asset_classes:
            asset_classes = tuple(AssetClass)

        combined: set[str] = set()
        for ac in asset_classes:
            combined |= self._tickers[ac]
        return sorted(combined)

    def get_class(self, ticker: str) -> AssetClass | None:
        """返回品种的资产类别，未找到返回 None。"""
        for ac, tickers in self._tickers.items():
            if ticker in tickers:
                return ac
        return None

    def count(self, asset_class: AssetClass) -> int:
        """某个分类中的品种数量。"""
        return len(self._tickers[asset_class])

    @property
    def total_count(self) -> int:
        """所有分类中的唯一品种总数。"""
        return len(set().union(*self._tickers.values()))

    def summary(self) -> dict[str, int]:
        """返回 {资产类别名: 数量} 映射。"""
        return {ac.value: len(tickers) for ac, tickers in self._tickers.items()}

    # -- 持久化 ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """保存 universe 到 JSON 快照。

        Parameters
        ----------
        path : str or Path or None
            输出路径。默认使用初始化时的 snapshot_path。

        Returns
        -------
        Path
            写入文件的路径。
        """
        out = Path(path) if path else self._snapshot_path
        if out is None:
            raise ValueError("No snapshot path provided")

        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            ac.value: sorted(tickers)
            for ac, tickers in self._tickers.items()
            if tickers
        }
        with open(out, "w") as fh:
            json.dump(data, fh, indent=2)

        logger.info("Saved universe snapshot: %d tickers to %s", self.total_count, out)
        return out

    def _load_snapshot(self) -> None:
        """从 JSON 快照加载。"""
        with open(self._snapshot_path) as fh:  # type: ignore[arg-type]
            data = json.load(fh)

        for ac_name, tickers in data.items():
            try:
                ac = AssetClass(ac_name)
            except ValueError:
                logger.warning("Unknown asset class in snapshot: %s", ac_name)
                continue
            self._tickers[ac].update(tickers)
