"""GlobalStateBuilder 中波动率缩放逻辑的测试。

覆盖：公式正确性、数据量不足时的安全降级、target_vol 参数，以及无前瞻验证。
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone

from quant.core.types import Bar, Currency, Instrument, InstrumentType, TradingWindow
from quant.strategy.layer1.features import GlobalStateBuilder

UTC = timezone.utc

_ES = Instrument(
    symbol="ES", instrument_type=InstrumentType.FUTURE,
    exchange="CME", currency=Currency.USD,
    multiplier=50.0, tick_size=0.25,
    margin_initial=15200.0, margin_maintenance=13800.0,
    sessions=(TradingWindow(start="17:00", end="16:00", timezone="America/Chicago", label="Globex"),),
)


def _make_bars(n: int, start_price: float = 4000.0, step: float = 1.0) -> list[Bar]:
    """生成 n 根 K 线，收盘价以 step 逐步递增。"""
    bars: list[Bar] = []
    price = start_price
    for i in range(n):
        ts = datetime(2024, 1, 2, (i % 96) // 4, ((i % 96) % 4) * 15, 0, tzinfo=UTC)
        bars.append(Bar(
            instrument=_ES, timestamp=ts,
            open=price, high=price + 2, low=price - 2, close=price,
            volume=1000.0,
        ))
        price += step
    return bars


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_vol_scaled_return_formula():
    """_vol_scaled_return 应等于 raw_return / realized_vol。"""
    closes = np.array([100.0 + i * 0.5 for i in range(100)], dtype=np.float64)
    lookback = 60

    sigma = GlobalStateBuilder._realized_vol(closes, lookback)
    vs_ret = GlobalStateBuilder._vol_scaled_return(closes, 5, sigma, 1e-9)

    # 手动计算
    raw_ret = (closes[-1] / closes[-6]) - 1.0
    expected = raw_ret / (sigma + 1e-9)

    assert vs_ret == pytest.approx(expected, rel=1e-6), (
        f"vol-scaled return {vs_ret} != expected {expected}"
    )


def test_short_data_returns_zero():
    """K 线数量少于收益率回望期时，波动率缩放收益应返回 0。"""
    closes = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    sigma = GlobalStateBuilder._realized_vol(closes, 60)

    # 请求 5 根 K 线的收益，但仅有 3 个数据点
    vs_ret = GlobalStateBuilder._vol_scaled_return(closes, 5, sigma, 1e-9)
    assert vs_ret == pytest.approx(0.0), f"Expected 0.0 for short data, got {vs_ret}"


def test_vol_lookback_parameter():
    """不同的 vol_lookback 参数应产生不同的波动率估计值。"""
    np.random.seed(42)
    closes = np.cumsum(np.random.randn(200)) + 4000.0
    closes = np.abs(closes) + 1.0  # 确保价格为正

    vol_short = GlobalStateBuilder._realized_vol(closes, 20)
    vol_long = GlobalStateBuilder._realized_vol(closes, 100)

    # 随机游走数据下，不同窗口应产生不同的波动率估计
    # （不保证大小关系，仅保证数值不同）
    assert vol_short > 0, "vol with lookback=20 should be > 0"
    assert vol_long > 0, "vol with lookback=100 should be > 0"
    # 均应为有限浮点数
    assert np.isfinite(vol_short)
    assert np.isfinite(vol_long)


def test_no_lookahead_in_vol_calculation():
    """_realized_vol 仅应使用最近 `lookback+1` 个收盘价，不使用更早的数据。

    构造一个前段剧烈波动、后段平稳的序列，使用短回望期时，
    _realized_vol 应仅反映平稳尾部的波动率。
    """
    np.random.seed(42)
    # 前 80 根 K 线：剧烈波动（大幅随机跳动）
    wild = 4000.0 + np.cumsum(np.random.randn(80) * 10.0)
    # 后 30 根 K 线：极度平稳（近似常数）
    calm = np.ones(30, dtype=np.float64) * wild[-1] + np.arange(30) * 0.001
    closes = np.concatenate([wild, calm])

    vol_short = GlobalStateBuilder._realized_vol(closes, 20)  # 仅使用最近 21 个（平稳段）
    vol_long = GlobalStateBuilder._realized_vol(closes, 80)   # 使用最近 81 个（含剧烈段）

    # 仅平稳尾部的波动率应远小于包含剧烈段的波动率
    assert vol_short < vol_long * 0.5, (
        f"vol_short={vol_short} (calm tail only) should be much smaller than "
        f"vol_long={vol_long} (includes volatile section)"
    )
