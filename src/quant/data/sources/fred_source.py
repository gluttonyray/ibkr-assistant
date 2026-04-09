"""FRED（美联储经济数据）数据源。

通过 ``requests`` 调用公开的 FRED API 获取宏观指标序列。
输出符合 ``quant.data.storage.schemas`` 中的 ``MACRO_SCHEMA``。

需要 FRED API key（免费申请：https://fred.stlouisfed.org/docs/api/api_key.html）。
通过构造函数传入或设置 ``FRED_API_KEY`` 环境变量。
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone

import polars as pl
import requests

from quant.data.storage.schemas import MACRO_SCHEMA

logger = logging.getLogger(__name__)

_FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class FREDSource:
    """从 FRED API 获取宏观指标数据。

    Parameters
    ----------
    api_key : str or None
        FRED API key。若为 None，从 ``FRED_API_KEY`` 环境变量读取。
    source_tag : str
        ``source`` 列的值（默认 ``"fred"``）。
    timeout : float
        HTTP 请求超时秒数（默认 30）。
    """

    def __init__(
        self,
        api_key: str | None = None,
        source_tag: str = "fred",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "FRED API key is required. Pass api_key= or set FRED_API_KEY env var. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self._source = source_tag
        self._timeout = timeout

    def fetch_series(
        self,
        series_id: str,
        start_date: date | str,
        end_date: date | str,
    ) -> pl.DataFrame:
        """获取单个 FRED 序列。

        Parameters
        ----------
        series_id : str
            FRED 序列标识（例如 ``"DGS10"``、``"BAMLC0A0CM"``）。
        start_date, end_date : date or str
            观测日期范围。

        Returns
        -------
        pl.DataFrame
            符合 ``MACRO_SCHEMA``。
        """
        logger.debug("FRED: fetching %s [%s, %s]", series_id, start_date, end_date)

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": str(start_date),
            "observation_end": str(end_date),
            "sort_order": "asc",
        }

        resp = requests.get(_FRED_BASE_URL, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            logger.warning("FRED: no observations for %s", series_id)
            return pl.DataFrame(schema=MACRO_SCHEMA)

        # 解析观测值
        dates: list[datetime] = []
        values: list[float | None] = []

        for obs in observations:
            obs_date = obs["date"]  # "YYYY-MM-DD"
            obs_val = obs["value"]

            dt = datetime.strptime(obs_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            dates.append(dt)

            # FRED 用 "." 表示缺失值
            if obs_val == ".":
                values.append(None)
            else:
                values.append(float(obs_val))

        df = pl.DataFrame(
            {
                "series_id": [series_id] * len(dates),
                "event_time": dates,
                "known_time": dates,  # PIT：公开数据，观测时即可知
                "value": values,
                "source": [self._source] * len(dates),
            }
        )

        # 将时间戳转换为正确的 dtype
        df = df.with_columns(
            pl.col("event_time").cast(pl.Datetime("us", "UTC")),
            pl.col("known_time").cast(pl.Datetime("us", "UTC")),
        )

        # 丢弃空值
        df = df.drop_nulls(subset=["value"])

        return df.select(list(MACRO_SCHEMA.keys()))

    def fetch_yield_spread(
        self,
        start_date: date | str,
        end_date: date | str,
    ) -> pl.DataFrame:
        """获取 10Y-2Y 美债收益率利差（DGS10 - DGS2）。

        返回 ``series_id="yield_spread"``、``value`` 为差值的 DataFrame。

        Parameters
        ----------
        start_date, end_date : date or str
            观测日期范围。

        Returns
        -------
        pl.DataFrame
            符合 ``MACRO_SCHEMA``。
        """
        dgs10 = self.fetch_series("DGS10", start_date, end_date)
        dgs2 = self.fetch_series("DGS2", start_date, end_date)

        if dgs10.is_empty() or dgs2.is_empty():
            logger.warning("FRED: cannot compute yield spread, missing data")
            return pl.DataFrame(schema=MACRO_SCHEMA)

        # 按 event_time 连接，计算利差
        spread = dgs10.join(
            dgs2.select(["event_time", pl.col("value").alias("value_2y")]),
            on="event_time",
            how="inner",
        ).with_columns(
            (pl.col("value") - pl.col("value_2y")).alias("value"),
            pl.lit("yield_spread").alias("series_id"),
        ).drop("value_2y")

        return spread.select(list(MACRO_SCHEMA.keys()))

    def fetch_credit_spread(
        self,
        start_date: date | str,
        end_date: date | str,
    ) -> pl.DataFrame:
        """获取企业信用利差（BofA OAS）。

        主要序列：``BAMLC0A0CM``（美国企业 Master OAS）。
        备用序列：``BAMLC0A4CBBB``（BBB OAS），当主序列不可用时使用。

        Parameters
        ----------
        start_date, end_date : date or str
            观测日期范围。

        Returns
        -------
        pl.DataFrame
            符合 ``MACRO_SCHEMA``。
        """
        primary_id = "BAMLC0A0CM"
        fallback_id = "BAMLC0A4CBBB"

        df = self.fetch_series(primary_id, start_date, end_date)

        if df.is_empty():
            logger.info(
                "FRED: primary credit spread %s empty, trying fallback %s",
                primary_id,
                fallback_id,
            )
            df = self.fetch_series(fallback_id, start_date, end_date)

        # 无论源序列如何，统一 series_id 为 "credit_spread"
        if not df.is_empty():
            df = df.with_columns(pl.lit("credit_spread").alias("series_id"))

        return df.select(list(MACRO_SCHEMA.keys())) if not df.is_empty() else df
