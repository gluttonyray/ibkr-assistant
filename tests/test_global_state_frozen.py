"""GlobalStateBuilder frozen 模式测试。"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from quant.strategy.layer1.features import GlobalStateBuilder


def _make_simple_windows(n_bars: int = 100) -> dict[str, list]:
    """构建简单的测试 windows。"""
    from quant.core.types import Bar, Currency, Instrument, InstrumentType

    inst = Instrument(
        symbol="ES",
        instrument_type=InstrumentType.FUTURE,
        exchange="CME",
        currency=Currency.USD,
        multiplier=50.0,
        tick_size=0.25,
        margin_initial=15200.0,
        margin_maintenance=13800.0,
    )

    from datetime import datetime, timezone

    bars = []
    price = 4000.0
    rng = np.random.RandomState(42)
    for i in range(n_bars):
        change = rng.normal(0, price * 0.001)
        close = max(price + change, 1.0)
        bars.append(
            Bar(
                instrument=inst,
                timestamp=datetime(2024, 1, 2, i // 4, (i % 4) * 15, 0, tzinfo=timezone.utc),
                open=price,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1000.0,
            )
        )
        price = close

    return {"ES": bars}


def test_frozen_mode_skips_pca_refit():
    """frozen=True 时 PCA 不重拟合。"""
    builder = GlobalStateBuilder(
        n_components=2, pca_update_freq=5, pca_window=50, vol_lookback=10,
        frozen=True,
    )
    # 设置一个假的 PCA 状态
    builder._pca_components = np.eye(2, 3, dtype=np.float64)
    builder._pca_syms_fitted = ["ES", "NQ", "YM"]

    windows = _make_simple_windows(80)

    with patch.object(builder, "_recompute_pca") as mock_pca:
        builder.update(bar_index=0, windows=windows)
        builder.update(bar_index=10, windows=windows)
        builder.update(bar_index=20, windows=windows)

        # frozen 模式下 _recompute_pca 不应被调用
        assert mock_pca.call_count == 0


def test_non_frozen_mode_refits_pca():
    """frozen=False 时 PCA 按频率重拟合。"""
    builder = GlobalStateBuilder(
        n_components=2, pca_update_freq=5, pca_window=50, vol_lookback=10,
        frozen=False,
    )

    windows = _make_simple_windows(80)

    with patch.object(builder, "_recompute_pca") as mock_pca:
        builder.update(bar_index=0, windows=windows)
        # 首次应该调用（_pca_components is None）
        assert mock_pca.call_count >= 1


def test_save_load_pca_state_roundtrip():
    """save_pca_state / load_pca_state 往返一致性。"""
    builder = GlobalStateBuilder(n_components=3, pca_update_freq=5)

    # 设置 PCA 状态
    original_components = np.random.randn(3, 5).astype(np.float64)
    original_syms = ["ES", "NQ", "YM", "RTY", "HSI"]
    builder._pca_components = original_components.copy()
    builder._pca_syms_fitted = original_syms.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_state.pkl"
        builder.save_pca_state(path)

        # 加载到新的 builder
        new_builder = GlobalStateBuilder(n_components=3, pca_update_freq=5)
        assert not new_builder._frozen
        new_builder.load_pca_state(path)

        np.testing.assert_array_almost_equal(
            new_builder._pca_components, original_components
        )
        assert new_builder._pca_syms_fitted == original_syms


def test_load_pca_state_enables_frozen():
    """load_pca_state 后 frozen=True。"""
    builder = GlobalStateBuilder(n_components=2, pca_update_freq=5)
    assert not builder._frozen

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.pkl"
        # 手工写一个状态文件
        payload = {
            "pca_components": np.eye(2, 3),
            "pca_syms_fitted": ["A", "B", "C"],
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        builder.load_pca_state(path)
        assert builder._frozen is True
