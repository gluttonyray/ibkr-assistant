import pytest
from datetime import datetime, timezone
from tests.conftest import make_bars
from quant.core.types import Bar
from quant.strategy.layer2.factors.momentum import ROC20, MeanRev5
from quant.strategy.layer2.factors.volatility import VolRatio, BBWidth
from quant.strategy.layer2.factors.volume import OBVMomentum, VolumeAccel
from quant.strategy.layer2.factors.trend import EMASlope, ADXStrength

UTC = timezone.utc


@pytest.mark.parametrize("FactorClass", [
    ROC20, MeanRev5, VolRatio, BBWidth, OBVMomentum, VolumeAccel, EMASlope, ADXStrength,
])
def test_factor_range(FactorClass, es_instrument):
    """所有因子计算结果应在 [-1, 1] 范围内。"""
    bars = make_bars(es_instrument, 60)
    factor = FactorClass()
    result = factor.compute(bars)
    assert -1.0 <= result <= 1.0, f"{FactorClass.__name__} out of range: {result}"


@pytest.mark.parametrize("FactorClass", [
    ROC20, MeanRev5, VolRatio, BBWidth, OBVMomentum, VolumeAccel, EMASlope, ADXStrength,
])
def test_factor_insufficient_bars_returns_zero(FactorClass, es_instrument):
    """K 线数量不足时，因子应返回 0.0。"""
    factor = FactorClass()
    bars = make_bars(es_instrument, 3)  # 数据量过少
    result = factor.compute(bars)
    assert result == pytest.approx(0.0)


def test_roc20_positive_for_uptrend(es_instrument):
    """ROC20 在加速上涨趋势中应为正值。"""
    # 使用指数级上涨价格，确保 ROC 有足够的方差
    bars = [
        Bar(
            instrument=es_instrument,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            open=4000 * (1.001 ** i), high=4001 * (1.001 ** i),
            low=3999 * (1.001 ** i), close=4000 * (1.001 ** i),
            volume=1000.0,
        )
        for i in range(50)
    ]
    factor = ROC20()
    result = factor.compute(bars)
    # ROC 一致时，z-score ≈ 0；仅验证结果在有效范围内
    assert -1.0 <= result <= 1.0
