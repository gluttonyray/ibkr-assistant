"""端到端集成测试：Layer1 V2 + 大规模 Universe。

验证完整 Layer1 V2 流水线（FeatureBuilderV2、BacktestEngine 预训练模式）
使用合成数据，不依赖真实市场数据下载。
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from quant.config.schema import AppConfig, Layer1Config
from quant.core.strategy import Layer1Result, Layer2Result
from quant.core.types import (
    Bar, Currency, Instrument, InstrumentType, Signal, SignalType,
    TradingWindow,
)
from quant.engine.backtest import BacktestEngine
from quant.instrument.registry import InstrumentRegistry
from quant.strategy.layer1.features import FeatureBuilderV2, GlobalStateBuilder
from quant.strategy.layer1.layer1 import Layer1
from quant.strategy.layer1.walk_forward import WalkForwardScheduler

UTC = timezone.utc


# ---------------------------------------------------------------------------
# 合成数据构造 Helper
# ---------------------------------------------------------------------------

def _make_instrument(symbol: str, exchange: str = "CME") -> Instrument:
    """构造一个测试用 Instrument。"""
    currency = Currency.HKD if exchange == "HKEX" else Currency.USD
    return Instrument(
        symbol=symbol,
        instrument_type=InstrumentType.FUTURE,
        exchange=exchange,
        currency=currency,
        multiplier=50.0,
        tick_size=0.25,
        margin_initial=15000.0,
        margin_maintenance=12000.0,
        sessions=(
            TradingWindow(
                start="17:00", end="16:00",
                timezone="America/Chicago", label="Globex",
            ),
        ),
    )


def make_synthetic_bars(
    symbol: str,
    n: int = 200,
    seed: int = 42,
    exchange: str = "CME",
    start_price: float = 100.0,
) -> list[Bar]:
    """生成 n 根合成 K 线，价格做几何随机游走。"""
    rng = np.random.default_rng(seed)
    prices = start_price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    inst = _make_instrument(symbol, exchange)
    bars: list[Bar] = []
    t = datetime(2023, 1, 2, tzinfo=UTC)
    for i, p in enumerate(prices):
        high = p * (1 + abs(rng.normal(0, 0.005)))
        low = p * (1 - abs(rng.normal(0, 0.005)))
        bars.append(Bar(
            instrument=inst,
            timestamp=t + timedelta(minutes=15 * i),
            open=p,
            high=high,
            low=low,
            close=p,
            volume=float(rng.integers(1000, 50000)),
        ))
    return bars


# ---------------------------------------------------------------------------
# Mock 辅助函数（用于 BacktestEngine）
# ---------------------------------------------------------------------------

def _make_dummy_layer2():
    m = MagicMock()
    m.compute.return_value = Layer2Result(alpha_score=0.0)
    return m


def _make_dummy_composer(inst: Instrument):
    m = MagicMock()
    m.compose.return_value = Signal(
        instrument=inst, signal_type=SignalType.HOLD,
        score=0.0, regime="RANGING",
        layer1_score=0.0, layer2_score=0.0, mii=1.0,
        timestamp=datetime.now(UTC),
    )
    return m


def _make_dummy_pre_risk():
    m = MagicMock()
    m.check.return_value = (False, 0, "no trade in test")
    return m


def _make_dummy_post_risk():
    m = MagicMock()
    m.check.return_value = []
    return m


def _make_dummy_sizer():
    m = MagicMock()
    m.compute_size.return_value = 0
    return m


# ===========================================================================
# 测试用例
# ===========================================================================


class TestV2FeatureBuilder37Dims:
    """验证 FeatureBuilderV2.build() 输出维度为 37。"""

    def test_v2_feature_builder_37dims(self):
        bars = make_synthetic_bars("ES", n=200, seed=42)
        inst = bars[0].instrument

        # 构建 GlobalState
        gsb = GlobalStateBuilder(
            n_components=5, pca_update_freq=20,
            pca_window=100, vol_lookback=60,
        )
        sym_windows = {"ES": bars[:100]}
        gs = gsb.update(bar_index=99, windows=sym_windows)

        # 构建特征向量
        fb = FeatureBuilderV2(vol_lookback=60)
        feat = fb.build(
            instrument=inst,
            asset_bars=bars[:100],
            global_state=gs,
            as_of=bars[99].timestamp,
        )

        assert feat is not None
        assert feat.shape == (37,), f"期望维度 (37,)，实际为 {feat.shape}"
        assert feat.dtype == np.float32
        # 类别型特征应有合理值
        assert feat[33] >= 0  # asset_class
        assert feat[34] >= 0  # exchange
        assert feat[35] >= 0  # currency
        assert feat[36] >= 0  # region


class TestWalkForwardSchedulerNoLookahead:
    """验证 WalkForwardScheduler 的 PIT（时点准确性）约束。"""

    def test_walk_forward_scheduler_no_lookahead(self):
        cfg = Layer1Config(
            retrain_interval_bars=50,
            retrain_window_bars=500,
            forward_horizon_bars=5,
        )
        s = WalkForwardScheduler(cfg=cfg, warmup_bars=100)

        # bar_index=604 时，training_slice 的 end 应 == bar_index - horizon + 1 = 600
        start, end = s.training_slice(604)
        assert end == 600, f"期望 end=600（无前视），实际为 {end}"

        # start 应为 max(0, 600 - 500) = 100
        assert start == 100, f"期望 start=100，实际为 {start}"

        # 验证 should_retrain 在第一次合格 bar 触发
        # first_train_bar = warmup + window + horizon = 100 + 500 + 5 = 605
        assert not s.should_retrain(604)
        assert s.should_retrain(605)

        # 标记完成后，下一次在 605 + 50 = 655
        s.mark_retrained(605)
        assert not s.should_retrain(654)
        assert s.should_retrain(655)


def _lightgbm_available() -> bool:
    """检测 LightGBM 的原生库是否可用。"""
    try:
        import lightgbm  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


class TestPooledTrainingPipeline:
    """用 3 个合成品种验证完整 fit -> predict 流程（feature_version=2）。

    若 LightGBM 原生库不可用（例如缺少 libomp），测试仍然验证
    特征构建 + 标签生成 + 样本权重 + compute 推理的完整流水线，
    只是模型返回零值（未训练状态的预期行为）。
    """

    def test_pooled_training_pipeline(self, monkeypatch):
        cfg = Layer1Config(
            feature_version=2,
            retrain_window_bars=200,
            forward_horizon_bars=3,
            retrain_interval_bars=30,
            vol_lookback=20,
            pca_window=100,
            pca_update_freq=20,
            pca_n_components=3,
        )
        layer1 = Layer1(cfg)

        # 构造 3 个品种各 200 根合成 K 线
        symbols = ["ES", "NQ", "YM"]
        windows: dict[Instrument, list[Bar]] = {}
        for i, sym in enumerate(symbols):
            bars = make_synthetic_bars(sym, n=200, seed=42 + i)
            windows[bars[0].instrument] = bars

        lgb_ok = _lightgbm_available()

        if not lgb_ok:
            # LightGBM 不可用时，patch model.fit 为 no-op，
            # 避免 OSError 中断池化特征构建流程的验证。
            from quant.strategy.layer1 import model as layer1_model_mod
            monkeypatch.setattr(
                layer1_model_mod.Layer1Model, "fit",
                lambda self, X, y, sample_weights=None: None,
            )

        # 训练（内部执行完整的特征构建 + 标签生成 + 样本权重计算）
        layer1.fit(windows)

        # 推理
        results = layer1.compute(windows)

        assert len(results) == 3, f"期望 3 个品种结果，实际为 {len(results)}"
        for inst, res in results.items():
            assert isinstance(res, Layer1Result)
            assert isinstance(res.regime, str)
            assert isinstance(res.beta_score, float)
            assert isinstance(res.mii, float)
            assert 0.0 <= res.mii <= 1.0, f"{inst.symbol} MII 超出 [0,1] 范围: {res.mii}"
            assert isinstance(res.confidence, float)
            assert 0.0 <= res.confidence <= 1.0


class TestBacktestEnginePretrainedMode:
    """验证 BacktestEngine 在预训练模式下正常运行（训练已分离）。"""

    def test_backtest_engine_inference_only(self):
        # 构造 2 个交易品种
        trading_syms = ["ES", "NQ"]
        trading_bars: dict[str, list[Bar]] = {}
        instruments: dict[str, Instrument] = {}
        for i, sym in enumerate(trading_syms):
            bars = make_synthetic_bars(sym, n=150, seed=100 + i)
            trading_bars[sym] = bars
            instruments[sym] = bars[0].instrument

        cfg = AppConfig()
        cfg.backtest.warmup_bars = 10
        registry = InstrumentRegistry(instruments)

        # 使用 mock layer1（避免 LightGBM 依赖问题，聚焦集成测试）
        layer1_mock = MagicMock()
        layer1_mock.compute.return_value = {
            instruments[sym]: Layer1Result(
                regime="RANGING", beta_score=0.0, mii=1.0, confidence=0.0,
            )
            for sym in trading_syms
        }

        first_inst = instruments[trading_syms[0]]

        engine = BacktestEngine(
            config=cfg,
            registry=registry,
            layer1=layer1_mock,
            layer2=_make_dummy_layer2(),
            composer=_make_dummy_composer(first_inst),
            pre_risk=_make_dummy_pre_risk(),
            post_risk=_make_dummy_post_risk(),
            sizer=_make_dummy_sizer(),
        )

        result = engine.run(data=trading_bars)

        # 验证结果基本结构
        assert isinstance(result.trades, list), "trade_records 应为 list"
        assert len(result.equity_curve) > 0, "equity_curve 不应为空"
        assert result.metrics is not None
        # 确认训练不再由引擎触发
        assert not hasattr(layer1_mock, 'fit_window') or layer1_mock.fit_window.call_count == 0


class TestFullTestSuiteCount:
    """验证总测试数 >= 107（含本文件 5 个）。"""

    def test_full_test_suite_count(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/", "--collect-only", "-q",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/ray/Desktop/Claude Code/ibkr-signal-assistant",
            timeout=60,
        )
        # 解析输出的最后几行，形如 "112 tests collected"
        output = result.stdout + result.stderr
        count = 0
        for line in output.splitlines():
            if "test" in line and "collected" in line:
                # 提取数字，如 "112 tests collected in 0.82s"
                parts = line.strip().split()
                for p in parts:
                    if p.isdigit():
                        count = int(p)
                        break
                break

        assert count >= 107, (
            f"总测试数为 {count}，期望 >= 107（含本文件 5 个集成测试）"
        )
