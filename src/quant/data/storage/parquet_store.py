"""统一的 Parquet 读写接口，支持 Point-in-Time 查询。

存储布局::

    {base_dir}/
        raw/{source}/{schema_name}/{symbol_or_series}.parquet
        clean/{schema_name}/{symbol_or_series}.parquet
        feature/{feature_name}.parquet

写入路径：
    1. 验证 schema（``validate_schema``）
    2. 追加到 raw 层（按数据源分区）
    3. 去重 + 合并到 clean 层

读取路径：
    1. 加载 clean 层 Parquet
    2. 应用 PIT 过滤：``known_time <= as_of_date``
    3. 返回 Polars DataFrame
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from quant.data.storage.schemas import SCHEMAS, validate_schema

logger = logging.getLogger(__name__)


class ParquetStore:
    """三层 Parquet 存储，支持 PIT 安全查询。

    Parameters
    ----------
    base_dir : str or Path
        所有 Parquet 文件的根目录。
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)

    # -- 写入 ------------------------------------------------------------------

    def append_raw(
        self,
        df: pl.DataFrame,
        schema_name: str,
        source: str,
        partition_key: str,
    ) -> Path:
        """追加数据到 raw 层（仅追加，不去重）。

        Parameters
        ----------
        df : pl.DataFrame
            待写入的数据，会按 *schema_name* 进行 schema 验证。
        schema_name : str
            ``"ohlcv"``、``"macro"`` 或 ``"fundamental"`` 之一。
        source : str
            数据源标识（``"yahoo"``、``"fred"``、``"ibkr"``）。
        partition_key : str
            文件名主干（symbol 或 series_id），例如 ``"SPY"``、``"DGS10"``。

        Returns
        -------
        Path
            写入的 Parquet 文件路径。
        """
        validated = validate_schema(df, schema_name)
        out_dir = self._base / "raw" / source / schema_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{partition_key}.parquet"

        if out_path.exists():
            existing = pl.read_parquet(out_path)
            validated = pl.concat([existing, validated])

        validated.write_parquet(
            out_path,
            compression="zstd",
            statistics=True,
        )
        logger.debug("Wrote raw %s/%s/%s (%d rows)", source, schema_name, partition_key, len(validated))
        return out_path

    def merge_to_clean(
        self,
        schema_name: str,
        partition_key: str,
        dedup_keys: list[str] | None = None,
    ) -> Path:
        """对 raw 数据去重后写入 clean 层。

        扫描所有 raw 源中 *schema_name/partition_key* 的数据，合并后
        按 *dedup_keys* 去重（保留最后一条），写入单一 clean Parquet 文件。

        Parameters
        ----------
        schema_name : str
            Schema 名称（``"ohlcv"``、``"macro"``、``"fundamental"``）。
        partition_key : str
            分区文件名主干。
        dedup_keys : list[str] or None
            用于去重的列。默认使用 schema 的前两列
            （通常是 symbol/series_id + event_time）。

        Returns
        -------
        Path
            clean 层 Parquet 文件路径。
        """
        raw_root = self._base / "raw"
        frames: list[pl.DataFrame] = []

        if raw_root.exists():
            for source_dir in sorted(raw_root.iterdir()):
                parquet_path = source_dir / schema_name / f"{partition_key}.parquet"
                if parquet_path.exists():
                    frames.append(pl.read_parquet(parquet_path))

        if not frames:
            raise FileNotFoundError(
                f"No raw data found for {schema_name}/{partition_key}"
            )

        combined = pl.concat(frames)

        # 去重
        schema = SCHEMAS[schema_name]
        if dedup_keys is None:
            schema_cols = list(schema.keys())
            dedup_keys = schema_cols[:2]  # 例如 ["symbol", "event_time"]

        combined = (
            combined
            .sort(dedup_keys + ["known_time"])
            .unique(subset=dedup_keys, keep="last")
            .sort(dedup_keys)
        )

        out_dir = self._base / "clean" / schema_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{partition_key}.parquet"

        combined.write_parquet(
            out_path,
            compression="zstd",
            statistics=True,
        )
        logger.debug("Wrote clean %s/%s (%d rows)", schema_name, partition_key, len(combined))
        return out_path

    # -- 读取 ------------------------------------------------------------------

    def read_clean(
        self,
        schema_name: str,
        partition_key: str,
        as_of: datetime | None = None,
    ) -> pl.DataFrame:
        """读取 clean 层数据，可选 PIT 过滤。

        Parameters
        ----------
        schema_name : str
            Schema 名称。
        partition_key : str
            分区文件名主干。
        as_of : datetime or None
            若提供，过滤 ``known_time <= as_of``（Point-in-Time 安全）。
            若为 None，返回全部数据（实时模式）。

        Returns
        -------
        pl.DataFrame
            clean 层数据，按 event_time 排序。

        Raises
        ------
        FileNotFoundError
            clean 层 Parquet 文件不存在。
        """
        path = self._base / "clean" / schema_name / f"{partition_key}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Clean data not found: {path}")

        df = pl.read_parquet(path)

        if as_of is not None:
            as_of_utc = as_of if as_of.tzinfo else as_of.replace(
                tzinfo=__import__("zoneinfo").ZoneInfo("UTC")
            )
            df = df.filter(pl.col("known_time") <= as_of_utc)

        return df.sort("event_time")

    def read_clean_multi(
        self,
        schema_name: str,
        partition_keys: list[str],
        as_of: datetime | None = None,
    ) -> pl.DataFrame:
        """读取并合并多个 clean 层分区。

        Parameters
        ----------
        schema_name : str
            Schema 名称。
        partition_keys : list[str]
            分区键列表。
        as_of : datetime or None
            PIT 过滤时间戳。

        Returns
        -------
        pl.DataFrame
            合并后的 clean 数据。
        """
        frames = []
        for key in partition_keys:
            try:
                frames.append(self.read_clean(schema_name, key, as_of=as_of))
            except FileNotFoundError:
                logger.warning("Clean data missing for %s/%s, skipping", schema_name, key)

        if not frames:
            raise FileNotFoundError(
                f"No clean data found for {schema_name} with keys {partition_keys}"
            )

        return pl.concat(frames).sort("event_time")

    # -- Feature 层 ------------------------------------------------------------

    def write_feature(self, df: pl.DataFrame, feature_name: str) -> Path:
        """将因子矩阵写入 feature 层。

        Parameters
        ----------
        df : pl.DataFrame
            因子数据（不强制 schema -- 因子 schema 各异）。
        feature_name : str
            因子文件名主干。

        Returns
        -------
        Path
            写入的 Parquet 文件路径。
        """
        out_dir = self._base / "feature"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{feature_name}.parquet"

        df.write_parquet(
            out_path,
            compression="zstd",
            statistics=True,
        )
        logger.debug("Wrote feature %s (%d rows)", feature_name, len(df))
        return out_path

    def read_feature(self, feature_name: str) -> pl.DataFrame:
        """读取因子矩阵。

        Parameters
        ----------
        feature_name : str
            因子文件名主干。

        Returns
        -------
        pl.DataFrame

        Raises
        ------
        FileNotFoundError
            因子文件不存在。
        """
        path = self._base / "feature" / f"{feature_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature data not found: {path}")
        return pl.read_parquet(path)

    # -- 工具方法 ---------------------------------------------------------------

    def list_partitions(self, schema_name: str, layer: str = "clean") -> list[str]:
        """列出某个 schema 在指定层中可用的分区键。

        Parameters
        ----------
        schema_name : str
            Schema 名称。
        layer : str
            ``"raw"``、``"clean"`` 或 ``"feature"``。

        Returns
        -------
        list[str]
            排序后的分区键主干列表。
        """
        if layer == "feature":
            scan_dir = self._base / "feature"
        else:
            scan_dir = self._base / layer / schema_name

        if not scan_dir.exists():
            return []

        return sorted(
            p.stem for p in scan_dir.glob("*.parquet")
        )
