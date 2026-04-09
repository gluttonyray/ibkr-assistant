import pytest
import dataclasses
from quant.portfolio.cost_model import FuturesCostModel
from quant.config.schema import CostConfig
from quant.core.types import InstrumentType


def test_cme_cost(es_instrument):
    """测试 CME 期货的佣金、交易所费用和滑点计算。"""
    model = FuturesCostModel(CostConfig())
    comm, fees, slip = model.calculate(es_instrument, qty=2, fill_price=4000.0)
    assert comm == pytest.approx(0.85 * 2)
    assert fees == pytest.approx((1.28 + 0.02) * 2)
    assert slip == pytest.approx(0.25 * 50 * 1.0 * 2)  # tick_size * multiplier * slippage_ticks * qty


def test_hkex_cost(hsi_instrument):
    """测试 HKEX 期货的佣金和滑点（港币计价）。"""
    model = FuturesCostModel(CostConfig())
    comm, fees, slip = model.calculate(hsi_instrument, qty=1, fill_price=20000.0)
    assert comm == pytest.approx(20.0)
    assert slip == pytest.approx(1.0 * 50 * 1.0 * 1)  # 港币


def test_non_future_raises(es_instrument):
    """非期货合约应抛出 ValueError。"""
    equity = dataclasses.replace(es_instrument, instrument_type=InstrumentType.EQUITY)
    model = FuturesCostModel(CostConfig())
    with pytest.raises(ValueError):
        model.calculate(equity, qty=1, fill_price=100.0)
