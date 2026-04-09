import pytest
import numpy as np
from quant.strategy.layer1.mii import MIICalculator
from quant.config.schema import Layer1Config


@pytest.fixture
def mii_calc():
    return MIICalculator(Layer1Config())


def test_mii_range(mii_calc):
    """MII 值应在 [0, 1] 范围内。"""
    rng = np.random.RandomState(42)
    rz = rng.randn(60)
    vol = np.abs(rng.randn(60)) * 1000 + 100
    result = mii_calc.compute(rz, vol)
    assert 0.0 <= result <= 1.0


def test_mii_extreme_decays(mii_calc):
    """持续在极端区间时，MII 应持续衰减。"""
    results = []
    for i in range(20):
        rz = np.ones(60) * 3.0  # 持续极端动量
        vol = np.ones(60) * 1000
        r = mii_calc.compute(rz, vol)
        results.append(r)
    # 持续极端 -> MII 应持续下降
    assert results[-1] <= results[0]


def test_mii_neutral_stays_high():
    """无动量时，MII 应接近 1。"""
    calc = MIICalculator(Layer1Config())
    rz = np.zeros(60)  # 无动量
    vol = np.ones(60) * 1000
    result = calc.compute(rz, vol)
    # 无极端动量 -> MII 应大于 0.5
    assert result > 0.5


def test_short_series_returns_one(mii_calc):
    """数据不足时，应返回 1.0（默认值）。"""
    rz = np.array([0.5])
    vol = np.array([1000.0])
    assert mii_calc.compute(rz, vol) == pytest.approx(1.0)
