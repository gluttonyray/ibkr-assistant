"""三层 Parquet 存储系统的 Polars schema 定义。

存储层级结构::

    Raw     -- 仅追加写入，按数据源分区
    Clean   -- 去重、UTC 标准化、双 PIT 时间戳
    Feature -- 按交易日对齐的因子矩阵

每条记录携带双时间戳：
    event_time  -- 事件发生时间（业务时间）
    known_time  -- 数据公开可用时间

回测查询强制 ``known_time <= as_of_date`` 以防止前视偏差。
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# OHLCV -- 跨资产 ETF 和期货的日频 K 线
# ---------------------------------------------------------------------------

OHLCV_SCHEMA = pl.Schema(
    {
        "symbol": pl.Utf8,
        "event_time": pl.Datetime("us", "UTC"),  # K 线收盘时间
        "known_time": pl.Datetime("us", "UTC"),  # 行情数据 = event_time
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "adj_close": pl.Float64,  # 拆股/分红调整后价格（仅 ETF）
        "source": pl.Utf8,  # "yahoo", "ibkr", "databento"
    }
)

# ---------------------------------------------------------------------------
# 宏观指标 -- FRED 序列、VIX 等
# ---------------------------------------------------------------------------

MACRO_SCHEMA = pl.Schema(
    {
        "series_id": pl.Utf8,  # 例如 "DGS10", "BAMLC0A0CM", "^VIX"
        "event_time": pl.Datetime("us", "UTC"),  # 观测日期
        "known_time": pl.Datetime("us", "UTC"),  # 发布日期（FRED 修订值）
        "value": pl.Float64,
        "source": pl.Utf8,  # "fred", "yahoo"
    }
)

# ---------------------------------------------------------------------------
# 基本面数据 -- 季度财务、财报日期
# ---------------------------------------------------------------------------

FUNDAMENTAL_SCHEMA = pl.Schema(
    {
        "symbol": pl.Utf8,
        "event_time": pl.Datetime("us", "UTC"),  # 财务期末日
        "known_time": pl.Datetime("us", "UTC"),  # 披露/发布日期
        "metric": pl.Utf8,  # "revenue", "eps", "book_value" 等
        "value": pl.Float64,
        "period": pl.Utf8,  # "2024Q3", "2024FY"
        "source": pl.Utf8,  # "sec_edgar", "yahoo"
    }
)


# ---------------------------------------------------------------------------
# Schema 注册表 -- 用于验证和分发
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, pl.Schema] = {
    "ohlcv": OHLCV_SCHEMA,
    "macro": MACRO_SCHEMA,
    "fundamental": FUNDAMENTAL_SCHEMA,
}


def validate_schema(df: pl.DataFrame, schema_name: str) -> pl.DataFrame:
    """将 *df* 强制转换为目标 schema，不兼容列会抛出异常。

    缺失的可空列用 null 填充。多余列会被丢弃。
    此行为设计为严格模式：schema 漂移应尽早发现。

    Parameters
    ----------
    df : pl.DataFrame
        待验证的数据。
    schema_name : str
        ``SCHEMAS`` 中的键（``"ohlcv"``, ``"macro"``, ``"fundamental"``）。

    Returns
    -------
    pl.DataFrame
        转换为目标 schema 的 DataFrame，列按规范顺序排列。

    Raises
    ------
    KeyError
        *schema_name* 不在 ``SCHEMAS`` 中。
    pl.exceptions.SchemaError
        必需列缺失或类型转换不可能。
    """
    schema = SCHEMAS[schema_name]

    # 缺失列用 null 补齐，类型与 schema 一致
    for col_name, dtype in schema.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col_name))

    # 仅选取 schema 中的列，按规范顺序排列并转换类型
    exprs = []
    for col_name, dtype in schema.items():
        exprs.append(pl.col(col_name).cast(dtype))

    return df.select(exprs)
