"""BacktestEngine 预训练模型集成测试。"""
from __future__ import annotations

import inspect
import json
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from quant.config.schema import AppConfig
from quant.core.strategy import Layer1Result, Layer2Result
from quant.core.types import (
    Bar,
    Currency,
    Instrument,
    InstrumentType,
    Signal,
    SignalType,
    TradingWindow,
)
from quant.engine.backtest import BacktestEngine
from quant.instrument.registry import InstrumentRegistry

UTC = timezone.utc

_ES = Instrument(
    symbol="ES",
    instrument_type=InstrumentType.FUTURE,
    exchange="CME",
    currency=Currency.USD,
    multiplier=50.0,
    tick_size=0.25,
    margin_initial=15200.0,
    margin_maintenance=13800.0,
    sessions=(
        TradingWindow(
            start="17:00", end="16:00", timezone="America/Chicago", label="Globex"
        ),
    ),
)


def _make_bars(n: int, start_price: float = 4000.0) -> list[Bar]:
    bars = []
    price = start_price
    for i in range(n):
        ts = datetime(
            2024, 1, 2 + i // 96,
            (i % 96) // 4, ((i % 96) % 4) * 15, 0,
            tzinfo=UTC,
        )
        bars.append(
            Bar(
                instrument=_ES,
                timestamp=ts,
                open=price, high=price + 1, low=price - 1, close=price,
                volume=1000.0,
            )
        )
        price += 0.25
    return bars


def _make_mock_layer1():
    m = MagicMock()
    m.compute.return_value = {
        _ES: Layer1Result(
            regime="RANGING", beta_score=0.0, mii=1.0, confidence=0.0,
        )
    }
    return m


def _make_mock_layer2():
    m = MagicMock()
    m.compute.return_value = Layer2Result(alpha_score=0.0)
    return m


def _make_mock_composer():
    m = MagicMock()
    m.compose.return_value = Signal(
        instrument=_ES, signal_type=SignalType.HOLD,
        score=0.0, regime="RANGING",
        layer1_score=0.0, layer2_score=0.0, mii=1.0,
        timestamp=datetime.now(UTC),
    )
    return m


def _make_mock_pre_risk():
    m = MagicMock()
    m.check.return_value = (False, 0, "no trade in test")
    return m


def _make_mock_post_risk():
    m = MagicMock()
    m.check.return_value = []
    return m


def _make_mock_sizer():
    m = MagicMock()
    m.compute_size.return_value = 0
    return m


def test_run_has_no_training_universe_param():
    """BacktestEngine.run() 不再有 training_universe 参数。"""
    sig = inspect.signature(BacktestEngine.run)
    params = list(sig.parameters.keys())
    assert "training_universe" not in params
    # 只有 self 和 data
    assert params == ["self", "data"]


def test_backtest_with_pretrained_lgbm_model():
    """BacktestEngine 用预训练 lgbm 模型运行回测返回正确 result。"""
    cfg = AppConfig()
    cfg.backtest.warmup_bars = 10

    bars = _make_bars(50)
    registry = InstrumentRegistry({"ES": _ES})

    # 用一个有 predict 方法的 mock layer1
    l1 = _make_mock_layer1()

    engine = BacktestEngine(
        config=cfg,
        registry=registry,
        layer1=l1,
        layer2=_make_mock_layer2(),
        composer=_make_mock_composer(),
        pre_risk=_make_mock_pre_risk(),
        post_risk=_make_mock_post_risk(),
        sizer=_make_mock_sizer(),
    )

    result = engine.run({"ES": bars})

    assert result is not None
    assert len(result.equity_curve) > 0
    assert result.metrics is not None
    # 确认 layer1.compute 被调用（非预热期间）
    assert l1.compute.call_count > 0


def test_backtest_engine_does_not_call_fit():
    """BacktestEngine 不再调用 layer1.fit_window() 或 layer1.fit()。"""
    cfg = AppConfig()
    cfg.backtest.warmup_bars = 5

    bars = _make_bars(80)
    registry = InstrumentRegistry({"ES": _ES})

    l1 = _make_mock_layer1()
    l1.fit_window = MagicMock()
    l1.fit = MagicMock()

    engine = BacktestEngine(
        config=cfg,
        registry=registry,
        layer1=l1,
        layer2=_make_mock_layer2(),
        composer=_make_mock_composer(),
        pre_risk=_make_mock_pre_risk(),
        post_risk=_make_mock_post_risk(),
        sizer=_make_mock_sizer(),
    )
    engine.run({"ES": bars})

    assert l1.fit_window.call_count == 0, "fit_window should not be called by BacktestEngine"
    assert l1.fit.call_count == 0, "fit should not be called by BacktestEngine"
