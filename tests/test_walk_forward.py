"""滚动训练触发逻辑的测试。

覆盖：WalkForwardScheduler 调度逻辑、BacktestEngine 集成 fit_window，
以及 forward_horizon_bars 配置传播。
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

from quant.config.schema import AppConfig, Layer1Config
from quant.core.types import Bar, Signal, SignalType, Currency, Instrument, InstrumentType, TradingWindow
from quant.core.strategy import Layer1Result, Layer2Result
from quant.engine.backtest import BacktestEngine
from quant.instrument.registry import InstrumentRegistry
from quant.strategy.layer1.walk_forward import WalkForwardScheduler

UTC = timezone.utc

_ES = Instrument(
    symbol="ES", instrument_type=InstrumentType.FUTURE,
    exchange="CME", currency=Currency.USD,
    multiplier=50.0, tick_size=0.25,
    margin_initial=15200.0, margin_maintenance=13800.0,
    sessions=(TradingWindow(start="17:00", end="16:00", timezone="America/Chicago", label="Globex"),),
)


def _make_bars(n: int, start_price: float = 4000.0) -> list[Bar]:
    bars: list[Bar] = []
    price = start_price
    base_time = datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
    for i in range(n):
        ts = datetime(
            2024, 1, 2 + i // 96,
            (i % 96) // 4, ((i % 96) % 4) * 15, 0,
            tzinfo=UTC,
        )
        bars.append(Bar(
            instrument=_ES,
            timestamp=ts,
            open=price, high=price + 1, low=price - 1, close=price,
            volume=1000.0,
        ))
        price += 0.25
    return bars


def _make_mock_layer1():
    m = MagicMock()
    m.compute.return_value = {_ES: Layer1Result(
        regime="RANGING", beta_score=0.0, mii=1.0, confidence=0.0,
    )}
    m.fit_window = MagicMock()
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


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


def test_layer1_fit_not_called_by_engine():
    """训练已从 BacktestEngine 中移除；fit_window() 不再被调用。"""
    warmup = 10
    window = 20
    horizon = 5
    interval = 50

    cfg = AppConfig()
    cfg.backtest.warmup_bars = warmup
    cfg.strategy.layer1.retrain_window_bars = window
    cfg.strategy.layer1.forward_horizon_bars = horizon
    cfg.strategy.layer1.retrain_interval_bars = interval

    total_bars = 40
    bars = _make_bars(total_bars)
    registry = InstrumentRegistry({"ES": _ES})
    l1 = _make_mock_layer1()

    engine = BacktestEngine(
        config=cfg, registry=registry, layer1=l1,
        layer2=_make_mock_layer2(),
        composer=_make_mock_composer(),
        pre_risk=_make_mock_pre_risk(),
        post_risk=_make_mock_post_risk(),
        sizer=_make_mock_sizer(),
    )
    engine.run({"ES": bars})

    # 训练已移出引擎，fit_window 不应被调用
    assert l1.fit_window.call_count == 0, (
        f"fit_window should NOT be called by BacktestEngine, but was called {l1.fit_window.call_count} times"
    )


def test_layer1_fit_never_called_regardless_of_bar_count():
    """无论有多少 bar，BacktestEngine 都不再调用 fit_window()。"""
    warmup = 5
    window = 10
    horizon = 3
    interval = 20

    cfg = AppConfig()
    cfg.backtest.warmup_bars = warmup
    cfg.strategy.layer1.retrain_window_bars = window
    cfg.strategy.layer1.forward_horizon_bars = horizon
    cfg.strategy.layer1.retrain_interval_bars = interval

    total_bars = 80
    bars = _make_bars(total_bars)
    registry = InstrumentRegistry({"ES": _ES})
    l1 = _make_mock_layer1()

    engine = BacktestEngine(
        config=cfg, registry=registry, layer1=l1,
        layer2=_make_mock_layer2(),
        composer=_make_mock_composer(),
        pre_risk=_make_mock_pre_risk(),
        post_risk=_make_mock_post_risk(),
        sizer=_make_mock_sizer(),
    )
    engine.run({"ES": bars})

    # 训练已移出引擎
    assert l1.fit_window.call_count == 0, (
        f"fit_window should NOT be called by BacktestEngine, "
        f"but was called {l1.fit_window.call_count} times"
    )


def test_forward_horizon_from_config():
    """Layer1.fit() 应使用 cfg.forward_horizon_bars（而非硬编码的 5）。

    使用 feature_version=1（v1 逐品种路径）以确保样本数可预测。
    """
    from quant.strategy.layer1.layer1 import Layer1

    cfg = Layer1Config(
        feature_version=1,
        forward_horizon_bars=10,
        retrain_window_bars=200,
        retrain_interval_bars=100,
        loo_correlation_groups={"ES": [], "HSI": []},
    )
    layer1 = Layer1(cfg)

    bars = _make_bars(200)

    with patch("quant.strategy.layer1.layer1.Layer1Model") as MockModel:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = (0.0, 0.0)
        MockModel.return_value = mock_instance

        windows = {_ES: bars}
        layer1.fit(windows)

        assert mock_instance.fit.called, "Layer1Model.fit should have been called"
        X_arg = mock_instance.fit.call_args[0][0]
        n_samples = len(X_arg)
        # v1 路径：range(50, len(bars) - forward) = range(50, 190) = 140 个样本
        assert n_samples == 200 - 50 - 10, (
            f"Expected {200 - 50 - 10} training samples (forward=10), got {n_samples}"
        )
