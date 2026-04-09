"""FeatureBuilder 新特征（每品种 10 维）的测试。

覆盖：feature_dim、ret60/ret120 安全降级、trend_consistency_20、cross_corr_20。
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone

from quant.strategy.layer1.features import FeatureBuilder
from quant.core.types import Bar, Currency, Instrument, InstrumentType, TradingWindow

UTC = timezone.utc

# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

_ES = Instrument(
    symbol="ES", instrument_type=InstrumentType.FUTURE,
    exchange="CME", currency=Currency.USD,
    multiplier=50.0, tick_size=0.25,
    margin_initial=15200.0, margin_maintenance=13800.0,
    sessions=(TradingWindow(start="17:00", end="16:00", timezone="America/Chicago", label="Globex"),),
)

_HSI = Instrument(
    symbol="HSI", instrument_type=InstrumentType.FUTURE,
    exchange="HKEX", currency=Currency.HKD,
    multiplier=50.0, tick_size=1.0,
    margin_initial=132180.0, margin_maintenance=105744.0,
    sessions=(
        TradingWindow(start="09:15", end="12:00", timezone="Asia/Hong_Kong", label="Morning"),
        TradingWindow(start="17:15", end="03:00", timezone="Asia/Hong_Kong", label="T+1"),
    ),
)


def _make_bars(instrument: Instrument, n: int, start_price: float = 4000.0,
               direction: float = 0.0) -> list[Bar]:
    """生成 n 根 K 线。direction > 0 表示上涨趋势，< 0 表示下跌趋势。"""
    bars: list[Bar] = []
    price = start_price
    base_time = datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
    for i in range(n):
        price = price + direction
        high = price + 1.0
        low = price - 1.0
        bars.append(Bar(
            instrument=instrument,
            timestamp=base_time.replace(
                minute=(i * 15) % 60,
                hour=((i * 15) // 60) % 24,
            ),
            open=price, high=high, low=low, close=price,
            volume=1000.0,
        ))
    return bars


@pytest.fixture
def fb() -> FeatureBuilder:
    return FeatureBuilder(loo_groups={"ES": [], "HSI": []})


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_feature_dim_is_10_per_symbol(fb: FeatureBuilder):
    """feature_dim(N) 应返回 10 * N。"""
    assert fb.feature_dim(1) == 10
    assert fb.feature_dim(3) == 30
    assert fb.feature_dim(7) == 70


def test_ret60_with_short_bars(fb: FeatureBuilder):
    """K 线数量不足 61 时，ret60 应安全返回 0。"""
    bars = _make_bars(_HSI, 50, start_price=20000.0)
    as_of = bars[-1].timestamp
    # HSI 为目标品种，ES 提供特征（不在 LOO 排除列表中）
    windows = {"HSI": bars, "ES": bars}
    features = fb.build("HSI", windows, as_of)
    # ES 的 10 维特征：[ret1, ret5, ret20, ret60, ret120, ...]
    # ret60 在 10 维块中的索引为 3
    ret60 = features[3]
    assert ret60 == pytest.approx(0.0), f"ret60 should be 0 with only 50 bars, got {ret60}"


def test_ret120_with_short_bars(fb: FeatureBuilder):
    """K 线数量不足 121 时，ret120 应安全返回 0。"""
    bars = _make_bars(_HSI, 100, start_price=20000.0, direction=1.0)
    as_of = bars[-1].timestamp
    windows = {"HSI": bars, "ES": bars}
    features = fb.build("HSI", windows, as_of)
    # ret120 在 10 维块中的索引为 4
    ret120 = features[4]
    assert ret120 == pytest.approx(0.0), f"ret120 should be 0 with only 100 bars, got {ret120}"


def test_trend_consistency_all_up(fb: FeatureBuilder):
    """价格单调上涨时，trend_consistency_20 应接近 1.0。"""
    bars = _make_bars(_HSI, 200, start_price=20000.0, direction=2.0)
    as_of = bars[-1].timestamp
    windows = {"HSI": bars, "ES": bars}
    features = fb.build("HSI", windows, as_of)
    # trend_consistency_20 在 10 维块中的索引为 8
    trend_consistency = features[8]
    assert trend_consistency >= 0.9, (
        f"trend_consistency_20 should be ~1.0 for monotonic up, got {trend_consistency}"
    )


def test_trend_consistency_all_down(fb: FeatureBuilder):
    """价格单调下跌时，trend_consistency_20 应接近 1.0（一致性下跌也是一种趋势一致性）。"""
    bars = _make_bars(_HSI, 200, start_price=25000.0, direction=-2.0)
    as_of = bars[-1].timestamp
    windows = {"HSI": bars, "ES": bars}
    features = fb.build("HSI", windows, as_of)
    # trend_consistency_20 在 10 维块中的索引为 8
    #
    # 实现说明：
    #   overall_direction = sign(segment[-1] - segment[0])  -> 下跌为负
    #   bar_directions = sign(returns)  -> 每根 K 线均为负
    #   mean(bar_directions == overall_direction) -> 全为 True -> 1.0
    #
    # 全部下跌是一种一致性趋势（每根 K 线均与整体方向一致），
    # 因此 trend_consistency_20 ≈ 1.0。
    trend_consistency = features[8]
    # 全部下跌是一致性趋势（每根 K 线均与整体方向匹配）
    assert trend_consistency >= 0.9, (
        f"trend_consistency_20 should be ~1.0 for consistent downtrend, got {trend_consistency}"
    )


def test_cross_corr_shape(fb: FeatureBuilder):
    """存在多个品种时，cross_corr_20 应在 [-1, 1] 范围内。"""
    bars_hsi = _make_bars(_HSI, 200, start_price=20000.0, direction=1.0)
    bars_es = _make_bars(_ES, 200, start_price=4000.0, direction=0.5)
    as_of = bars_hsi[-1].timestamp
    windows = {"HSI": bars_hsi, "ES": bars_es}
    features = fb.build("HSI", windows, as_of)
    # cross_corr_20 在 10 维块中的索引为 9
    cross_corr = features[9]
    assert -1.0 <= cross_corr <= 1.0, (
        f"cross_corr_20 should be in [-1, 1], got {cross_corr}"
    )
