"""SequenceFeatureBuilder 测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from quant.core.types import Bar, Instrument
from quant.strategy.layer1.sequence_features import (
    SEQUENCE_FEATURE_DIM,
    SequenceFeatureBuilder,
)


def _make_instrument() -> Instrument:
    return Instrument(
        symbol="TEST",
        instrument_type="FUTURE",
        exchange="CME",
        currency="USD",
        multiplier=50.0,
        tick_size=0.25,
        margin_initial=12000.0,
        margin_maintenance=10000.0,
    )


def _make_bars(n: int, base_price: float = 100.0, seed: int = 42) -> list[Bar]:
    """生成 n 根模拟 Bar。"""
    rng = np.random.RandomState(seed)
    inst = _make_instrument()
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = base_price
    for i in range(n):
        ret = rng.normal(0, 0.002)
        price = price * (1 + ret)
        volume = float(rng.uniform(100, 1000))
        bars.append(
            Bar(
                instrument=inst,
                timestamp=t0 + timedelta(minutes=15 * i),
                open=price * 0.999,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                volume=volume,
            )
        )
    return bars


class TestBuildSequence:
    def test_shape_and_dtype(self):
        """build_sequence 应返回 (T, 14) float32。"""
        builder = SequenceFeatureBuilder()
        bars = _make_bars(300)
        T = 32
        result = builder.build_sequence(bars, lookback_T=T)
        assert result is not None
        assert result.shape == (T, SEQUENCE_FEATURE_DIM)
        assert result.dtype == np.float32

    def test_returns_none_when_insufficient_bars(self):
        """bars 不足时应返回 None。"""
        builder = SequenceFeatureBuilder()
        # 需要 vol_lookback(60) + T(32) + 1 = 93 根 bar
        bars = _make_bars(50)
        result = builder.build_sequence(bars, lookback_T=32)
        assert result is None

    def test_no_lookahead(self):
        """增加未来 bars 不应改变过去时步的特征值。"""
        builder = SequenceFeatureBuilder()
        T = 20

        bars_200 = _make_bars(200)
        bars_250 = _make_bars(250)  # 同 seed，前 200 根相同

        # 使用 as_of 限制到第 200 根 bar 的时间戳
        as_of = bars_200[-1].timestamp

        result_200 = builder.build_sequence(bars_200, lookback_T=T)
        result_250 = builder.build_sequence(bars_250, lookback_T=T, as_of=as_of)

        assert result_200 is not None
        assert result_250 is not None
        np.testing.assert_array_almost_equal(result_200, result_250, decimal=6)

    def test_no_nan_or_inf(self):
        """输出不应包含 NaN 或 Inf。"""
        builder = SequenceFeatureBuilder()
        bars = _make_bars(300)
        result = builder.build_sequence(bars, lookback_T=32)
        assert result is not None
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestBuildTrainingSequences:
    def test_shape(self):
        """build_training_sequences 应返回正确形状。"""
        builder = SequenceFeatureBuilder()
        bars = _make_bars(300)
        T = 32
        fwd = 10
        result = builder.build_training_sequences(bars, lookback_T=T, forward_horizon=fwd)
        assert result is not None
        X, y = result
        assert X.ndim == 3
        assert X.shape[1] == T
        assert X.shape[2] == SEQUENCE_FEATURE_DIM
        assert X.dtype == np.float32
        assert y.ndim == 1
        assert len(y) == len(X)
        assert y.dtype == np.float64

    def test_returns_none_when_insufficient(self):
        """bars 不足时应返回 None。"""
        builder = SequenceFeatureBuilder()
        bars = _make_bars(50)
        result = builder.build_training_sequences(bars, lookback_T=32, forward_horizon=10)
        assert result is None

    def test_multiple_windows(self):
        """应生成多个训练窗口。"""
        builder = SequenceFeatureBuilder()
        bars = _make_bars(300)
        result = builder.build_training_sequences(bars, lookback_T=32, forward_horizon=10)
        assert result is not None
        X, y = result
        # 应该有相当数量的窗口
        assert len(X) > 50
