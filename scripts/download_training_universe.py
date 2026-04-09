#!/usr/bin/env python3
"""下载完整训练 Universe 数据（目标 10,000+ 品种）。

使用 UniverseBuilder 获取品种列表，然后用 YahooBulkDownloader
下载 2010-01-01 至今的日频 OHLCV 数据。

用法::

    cd new/
    .venv/bin/python scripts/download_training_universe.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from quant.data.storage.parquet_store import ParquetStore
from quant.data.universe.universe_builder import UniverseBuilder
from quant.data.universe.yahoo_bulk_downloader import YahooBulkDownloader

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────────────

DATA_DIR = _PROJECT_ROOT / "data" / "training_universe"
PARQUET_DIR = DATA_DIR / "parquet"
ERROR_LOG = DATA_DIR / "errors.json"
CACHE_DIR = DATA_DIR / "universe_cache"

START_DATE = "2010-01-01"
BATCH_SIZE = 100
INTER_BATCH_SLEEP = 2.0
MAX_RETRIES = 2


def main() -> None:
    """主函数：构建品种列表 -> 下载 -> 报告。"""
    # 1. 构建品种列表
    logger.info("=== 构建品种 Universe ===")
    builder = UniverseBuilder(cache_dir=CACHE_DIR)
    universe = builder.build()
    tickers = universe.get_tickers()

    logger.info("品种总数: %d", len(tickers))
    logger.info("分类统计: %s", universe.summary())

    # 2. 初始化下载器
    logger.info("=== 开始下载 ===")
    store = ParquetStore(PARQUET_DIR)
    downloader = YahooBulkDownloader(
        store,
        batch_size=BATCH_SIZE,
        inter_batch_sleep=INTER_BATCH_SLEEP,
        max_retries=MAX_RETRIES,
    )

    # 3. 执行下载（增量模式，支持断点续传）
    report = downloader.download_universe(
        tickers,
        start=START_DATE,
        incremental=True,
    )

    # 4. 保存错误日志
    if report.failed:
        report.save_error_log(ERROR_LOG)
        logger.info("错误日志已保存: %s", ERROR_LOG)

    # 5. 输出最终报告
    logger.info("=== 下载完成 ===")
    print(report)
    print(f"\n数据目录: {PARQUET_DIR}")
    print(f"成功: {len(report.succeeded)}")
    print(f"跳过: {len(report.skipped)}")
    print(f"失败: {len(report.failed)}")
    print(f"耗时: {report.elapsed_seconds / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
