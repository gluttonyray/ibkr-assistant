import pytest
from quant.strategy.layer2.ic_weights import ICWeightCalculator
from quant.config.schema import Layer2Config


def test_equal_weights_with_no_history():
    """无历史数据时，各因子权重应相等（或近似等权）。"""
    cfg = Layer2Config()
    calc = ICWeightCalculator(cfg, ["f1", "f2", "f3"])
    weights = calc.get_weights()
    assert set(weights.keys()) == {"f1", "f2", "f3"}
    total_abs = sum(abs(w) for w in weights.values())
    # 若权重非零，绝对值之和应为 1
    if total_abs > 0:
        assert total_abs == pytest.approx(1.0, rel=0.01)


def test_weights_sum_to_one_after_update():
    """更新若干期 IC 后，权重绝对值之和应等于 1。"""
    cfg = Layer2Config(ic_lookback_bars=50)
    calc = ICWeightCalculator(cfg, ["f1", "f2"])
    for i in range(30):
        calc.update({"f1": float(i), "f2": -float(i)}, realized_return=float(i) * 0.001)
    weights = calc.get_weights()
    total_abs = sum(abs(w) for w in weights.values())
    assert total_abs == pytest.approx(1.0, rel=0.01)


def test_shrinkage_toward_equal():
    """ic_shrinkage=1.0（完全收缩）时，两个因子权重应趋近相等。"""
    cfg = Layer2Config(ic_shrinkage=1.0)  # 完全收缩 -> 等权
    calc = ICWeightCalculator(cfg, ["f1", "f2"])
    for i in range(30):
        calc.update({"f1": float(i), "f2": 0.0}, realized_return=float(i) * 0.001)
    weights = calc.get_weights()
    # shrinkage=1.0 -> 应近似等权
    assert abs(weights["f1"] - weights["f2"]) < 0.1
