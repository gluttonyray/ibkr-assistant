import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from quant.engine.backtest import BacktestEngine
from quant.core.types import Bar, Signal, SignalType
from quant.core.strategy import Layer1Result, Layer2Result
from quant.config.schema import AppConfig
from quant.instrument.registry import InstrumentRegistry

UTC = timezone.utc


def make_dummy_layer1():
    """返回一个 compute 返回空字典的 Mock 对象。"""
    m = MagicMock()
    m.compute.return_value = {}
    return m


def make_dummy_layer2():
    m = MagicMock()
    m.compute.return_value = Layer2Result(alpha_score=0.0)
    return m


def make_dummy_composer(es_inst):
    m = MagicMock()
    m.compose.return_value = Signal(
        instrument=es_inst, signal_type=SignalType.HOLD,
        score=0.0, regime="RANGING",
        layer1_score=0.0, layer2_score=0.0, mii=1.0,
        timestamp=datetime.now(UTC),
    )
    return m


def make_dummy_pre_risk():
    m = MagicMock()
    m.check.return_value = (False, 0, "no trade in test")
    return m


def make_dummy_post_risk():
    m = MagicMock()
    m.check.return_value = []
    return m


def make_dummy_sizer():
    m = MagicMock()
    m.compute_size.return_value = 0
    return m


def test_engine_runs_without_error(es_instrument):
    """回测引擎在基础配置下应能正常运行并返回结果。"""
    registry = InstrumentRegistry({"ES": es_instrument})
    cfg = AppConfig()
    cfg.backtest.warmup_bars = 10

    bars = [
        Bar(
            instrument=es_instrument,
            timestamp=datetime(2024, 1, 2, i // 4, (i % 4) * 15, 0, tzinfo=UTC),
            open=4000.0 + i, high=4001.0 + i, low=3999.0 + i,
            close=4000.0 + i, volume=1000.0,
        )
        for i in range(50)
    ]

    engine = BacktestEngine(
        config=cfg,
        registry=registry,
        layer1=make_dummy_layer1(),
        layer2=make_dummy_layer2(),
        composer=make_dummy_composer(es_instrument),
        pre_risk=make_dummy_pre_risk(),
        post_risk=make_dummy_post_risk(),
        sizer=make_dummy_sizer(),
    )
    result = engine.run({"ES": bars})
    assert result is not None
    assert len(result.equity_curve) > 0
    assert result.metrics.total_return == pytest.approx(0.0, abs=0.01)


def test_warmup_bars_skipped(es_instrument):
    """预热期内，Layer1 不应被调用。"""
    registry = InstrumentRegistry({"ES": es_instrument})
    cfg = AppConfig()
    cfg.backtest.warmup_bars = 30

    bars = [
        Bar(
            instrument=es_instrument,
            timestamp=datetime(2024, 1, 2, i // 4, (i % 4) * 15, 0, tzinfo=UTC),
            open=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=1000.0,
        )
        for i in range(40)
    ]
    l1 = make_dummy_layer1()
    engine = BacktestEngine(
        config=cfg, registry=registry, layer1=l1,
        layer2=make_dummy_layer2(),
        composer=make_dummy_composer(es_instrument),
        pre_risk=make_dummy_pre_risk(),
        post_risk=make_dummy_post_risk(),
        sizer=make_dummy_sizer(),
    )
    engine.run({"ES": bars})
    # warmup_bars=30，共 40 根 K 线 -> layer1 最多被调用 10 次
    assert l1.compute.call_count <= 10
