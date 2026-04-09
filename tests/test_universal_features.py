"""FeatureBuilderV2 通用特征向量的测试。

覆盖：固定维度、多资产类型输入、数据量不足时的安全降级、
波动率缩放特征范围，以及类别特征编码。
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone

from quant.core.types import Bar, Currency, Instrument, InstrumentType, TradingWindow
from quant.strategy.layer1.features import (
    ASSET_CLASS_MAP,
    CURRENCY_MAP,
    EXCHANGE_MAP,
    REGION_MAP,
    FeatureBuilderV2,
    GlobalState,
    GlobalStateBuilder,
)

UTC = timezone.utc


def _make_instrument(
    symbol: str,
    itype: InstrumentType = InstrumentType.FUTURE,
    exchange: str = "CME",
    currency: Currency = Currency.USD,
) -> Instrument:
    return Instrument(
        symbol=symbol, instrument_type=itype,
        exchange=exchange, currency=currency,
        multiplier=50.0, tick_size=0.25,
        margin_initial=15200.0, margin_maintenance=13800.0,
        sessions=(TradingWindow(start="17:00", end="16:00", timezone="America/Chicago", label="Globex"),),
    )


def _make_bars(instrument: Instrument, n: int, start_price: float = 4000.0,
               step: float = 0.5) -> list[Bar]:
    """生成 n 根 K 线，收盘价以 step 逐步递增。"""
    bars: list[Bar] = []
    price = start_price
    for i in range(n):
        ts = datetime(2024, 1, 2 + i // 96, (i % 96) // 4, ((i % 96) % 4) * 15, 0, tzinfo=UTC)
        bars.append(Bar(
            instrument=instrument, timestamp=ts,
            open=price, high=price + 2, low=price - 2, close=price,
            volume=float(1000 + i % 100),
        ))
        price += step
    return bars


def _empty_global_state() -> GlobalState:
    """返回空的全局状态对象。"""
    return GlobalState()


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_feature_dim_fixed():
    """无论标的池大小，特征维度应固定为 37。"""
    fb = FeatureBuilderV2()
    assert fb.FEATURE_DIM == 37

    es = _make_instrument("ES")
    bars = _make_bars(es, 100)
    gs = _empty_global_state()
    as_of = bars[-1].timestamp

    vec = fb.build(es, bars, gs, as_of)
    assert vec.shape == (37,), f"Expected shape (37,), got {vec.shape}"
    assert vec.dtype == np.float32


def test_multi_asset_type_input():
    """FUTURE 和 EQUITY 类型的标的均应产生有效的 37 维特征向量。"""
    fb = FeatureBuilderV2()
    gs = _empty_global_state()

    # 期货
    es = _make_instrument("ES", InstrumentType.FUTURE, "CME", Currency.USD)
    bars_es = _make_bars(es, 100)
    vec_es = fb.build(es, bars_es, gs, bars_es[-1].timestamp)
    assert vec_es.shape == (37,)
    assert np.all(np.isfinite(vec_es))

    # 股票（模拟）
    aapl = _make_instrument("AAPL", InstrumentType.EQUITY, "NASDAQ", Currency.USD)
    bars_aapl = _make_bars(aapl, 100, start_price=150.0, step=0.1)
    vec_aapl = fb.build(aapl, bars_aapl, gs, bars_aapl[-1].timestamp)
    assert vec_aapl.shape == (37,)
    assert np.all(np.isfinite(vec_aapl))

    # 类别特征 asset_class 应不同
    # FUTURE -> "futures" -> 2，EQUITY -> "stock" -> 0
    assert vec_es[33] == pytest.approx(ASSET_CLASS_MAP["futures"])
    assert vec_aapl[33] == pytest.approx(ASSET_CLASS_MAP["stock"])


def test_short_data_fills_zero():
    """K 线数量极少（< 21）时，需要历史数据的特征应为 0。"""
    fb = FeatureBuilderV2()
    gs = _empty_global_state()

    es = _make_instrument("ES")
    bars = _make_bars(es, 5)
    vec = fb.build(es, bars, gs, bars[-1].timestamp)

    assert vec.shape == (37,)
    # vol_20（索引 5）需要 20+ 根 K 线的对数收益 -> 应为 0
    assert vec[5] == pytest.approx(0.0), f"vol_20 should be 0 with 5 bars, got {vec[5]}"
    # ret_60（索引 3）需要 61 根 K 线 -> 应为 0
    assert vec[3] == pytest.approx(0.0), f"ret_60 should be 0 with 5 bars, got {vec[3]}"
    # ret_120（索引 4）需要 121 根 K 线 -> 应为 0
    assert vec[4] == pytest.approx(0.0), f"ret_120 should be 0 with 5 bars, got {vec[4]}"
    # 全向量不应出现 NaN
    assert not np.any(np.isnan(vec)), f"No NaN allowed in feature vector: {vec}"


def test_vol_scaled_features_in_range():
    """波动率缩放收益特征应为有限值（不含 NaN/Inf）。"""
    fb = FeatureBuilderV2()
    gs = _empty_global_state()

    # 使用随机游走价格，确保波动率非零
    np.random.seed(42)
    es = _make_instrument("ES")
    n = 200
    prices = 4000.0 + np.cumsum(np.random.randn(n) * 2.0)
    prices = np.abs(prices) + 100  # 确保价格为正
    bars: list[Bar] = []
    for i in range(n):
        ts = datetime(2024, 1, 2 + i // 96, (i % 96) // 4, ((i % 96) % 4) * 15, 0, tzinfo=UTC)
        p = float(prices[i])
        bars.append(Bar(
            instrument=es, timestamp=ts,
            open=p, high=p + 2, low=p - 2, close=p,
            volume=float(1000 + i % 100),
        ))

    vec = fb.build(es, bars, gs, bars[-1].timestamp)

    # ret_vol_scaled_20 在索引 9，ret_vol_scaled_60 在索引 10
    rvs_20 = vec[9]
    rvs_60 = vec[10]

    assert np.isfinite(rvs_20), f"ret_vol_scaled_20 should be finite, got {rvs_20}"
    assert np.isfinite(rvs_60), f"ret_vol_scaled_60 should be finite, got {rvs_60}"
    # 随机游走数据经过正确的波动率归一化后，数值应有界
    assert abs(rvs_20) < 500, f"ret_vol_scaled_20 out of range: {rvs_20}"
    assert abs(rvs_60) < 500, f"ret_vol_scaled_60 out of range: {rvs_60}"


def test_categorical_features_correct():
    """类别特征应正确编码 asset_class、exchange、currency 和 region。"""
    fb = FeatureBuilderV2()
    gs = _empty_global_state()

    # CME 期货，USD 计价
    es = _make_instrument("ES", InstrumentType.FUTURE, "CME", Currency.USD)
    bars = _make_bars(es, 50)
    vec = fb.build(es, bars, gs, bars[-1].timestamp)

    assert vec[33] == pytest.approx(ASSET_CLASS_MAP["futures"]), "asset_class should be futures=2"
    assert vec[34] == pytest.approx(EXCHANGE_MAP["CME"]), "exchange should be CME=0"
    assert vec[35] == pytest.approx(CURRENCY_MAP["USD"]), "currency should be USD=0"
    assert vec[36] == pytest.approx(REGION_MAP["US"]), "region should be US=0"

    # HKEX 期货，HKD 计价
    hsi = _make_instrument("HSI", InstrumentType.FUTURE, "HKEX", Currency.HKD)
    bars_hsi = _make_bars(hsi, 50, start_price=20000.0)
    vec_hsi = fb.build(hsi, bars_hsi, gs, bars_hsi[-1].timestamp)

    assert vec_hsi[33] == pytest.approx(ASSET_CLASS_MAP["futures"])
    assert vec_hsi[34] == pytest.approx(EXCHANGE_MAP["HKEX"])
    assert vec_hsi[35] == pytest.approx(CURRENCY_MAP["HKD"])
    assert vec_hsi[36] == pytest.approx(REGION_MAP["HK"])
