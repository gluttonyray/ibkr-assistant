#!/usr/bin/env python3
"""下载完整训练 Universe 数据（10,000+ 品种，10年日频 OHLCV）。

用法::

    PYTHONPATH=src .venv/bin/python scripts/download_universe.py

支持断点续传：再次运行会跳过已下载的品种，仅下载新增/缺失部分。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保 src 在 path 中
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from quant.data.storage.parquet_store import ParquetStore
from quant.data.universe.universe_builder import UniverseBuilder
from quant.data.universe.yahoo_bulk_downloader import YahooBulkDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 数据存储目录
DATA_DIR = _ROOT / "data" / "training_universe"
PARQUET_DIR = DATA_DIR / "parquet"
CACHE_DIR = DATA_DIR / "universe_cache"
ERROR_LOG = DATA_DIR / "errors.json"


def main() -> None:
    # 1. 构建完整品种列表
    logger.info("=== 步骤 1: 构建品种 Universe ===")
    builder = UniverseBuilder(cache_dir=CACHE_DIR)
    universe = builder.build()

    all_tickers = universe.get_tickers()
    logger.info(
        "Universe 总计 %d 品种: %s",
        len(all_tickers),
        universe.summary(),
    )

    # 保存 universe 快照
    snapshot_path = DATA_DIR / "universe_snapshot.json"
    universe.save(snapshot_path)

    # 2. 批量下载日频 OHLCV
    logger.info("=== 步骤 2: 批量下载 OHLCV ===")
    store = ParquetStore(PARQUET_DIR)
    downloader = YahooBulkDownloader(
        store,
        batch_size=100,
        inter_batch_sleep=2.0,
        max_retries=2,
    )

    report = downloader.download_universe(
        all_tickers,
        start="2010-01-01",
        incremental=True,
    )

    # 3. 保存错误日志
    if report.failed:
        report.save_error_log(ERROR_LOG)
        logger.info("错误日志已保存: %s", ERROR_LOG)

    # 4. 打印最终报告
    logger.info("=== 下载完成 ===")
    print(report)
    print(f"\nParquet 存储目录: {PARQUET_DIR}")
    print(f"成功率: {len(report.succeeded)}/{report.total} "
          f"({100 * len(report.succeeded) / max(report.total, 1):.1f}%)")


if __name__ == "__main__":
    main()
