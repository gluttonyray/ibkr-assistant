"""Layer1 后端路由测试。"""
from __future__ import annotations

import json
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quant.config.schema import Layer1Config
from quant.core.types import Bar, Currency, Instrument, InstrumentType, TradingWindow
from quant.strategy.layer1.layer1 import Layer1, _POOLED_KEY

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


def _make_bars(n: int = 100, start_price: float = 4000.0) -> list[Bar]:
    rng = np.random.RandomState(42)
    bars = []
    price = start_price
    for i in range(n):
        change = rng.normal(0, price * 0.001)
        close = max(price + change, 1.0)
        bars.append(
            Bar(
                instrument=_ES,
                timestamp=datetime(
                    2024, 1, 2 + i // 96, (i % 96) // 4, ((i % 96) % 4) * 15, 0,
                    tzinfo=UTC,
                ),
                open=price,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1000.0,
            )
        )
        price = close
    return bars


def test_lgbm_uses_tabular_path():
    """model_type='lgbm' 时使用 tabular 路径。"""
    cfg = Layer1Config(model_type="lgbm", feature_version=2)
    layer1 = Layer1(cfg)

    # 模拟一个 tabular backend
    mock_backend = MagicMock()
    mock_backend.input_type = "tabular"
    mock_backend.predict.return_value = (0.5, 0.8)
    layer1._models[_POOLED_KEY] = mock_backend

    bars = _make_bars(100)
    windows = {_ES: bars}

    results = layer1.compute(windows)

    assert _ES in results
    # tabular 路径下，使用 FeatureBuilderV2 构建特征
    result = results[_ES]
    assert result.beta_score == pytest.approx(0.5)
    assert result.confidence == pytest.approx(0.8)


def test_load_model_with_lgbm(tmp_path):
    """load_model() 加载 lgbm 后 backend 正确注册到 _models。"""
    cfg = Layer1Config(model_type="lgbm", feature_version=2)

    # 训练一个真正的 LGBMBackend 并保存
    from quant.strategy.layer1.backends.lgbm_backend import LGBMBackend

    backend = LGBMBackend(cfg)
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 10)
    y_train = X_train[:, 0] * 0.5 + rng.randn(200) * 0.1
    backend.fit(X_train, y_train)

    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    backend.save(model_dir)

    layer1 = Layer1(cfg)
    layer1.load_model(model_dir)

    # 模型应已加载到 _models
    assert _POOLED_KEY in layer1._models
    loaded_backend = layer1._models[_POOLED_KEY]
    assert loaded_backend.is_fitted
    assert loaded_backend.input_type == "tabular"

    # 直接用 10 维特征验证 predict 正常工作
    test_x = rng.randn(10)
    beta_score, confidence = loaded_backend.predict(test_x)
    assert isinstance(beta_score, float)
    assert isinstance(confidence, float)


def test_sequential_backend_routes_correctly():
    """sequential backend 应路由到 _compute_v2_sequential。"""
    cfg = Layer1Config(model_type="lgbm", feature_version=2, seq_lookback_T=60)
    layer1 = Layer1(cfg)

    # 模拟一个 sequential backend
    mock_backend = MagicMock()
    mock_backend.input_type = "sequential"
    mock_backend.predict.return_value = (0.7, 0.9)
    layer1._models[_POOLED_KEY] = mock_backend

    bars = _make_bars(200)
    windows = {_ES: bars}

    results = layer1.compute(windows)
    assert _ES in results
    result = results[_ES]
    assert result.beta_score == pytest.approx(0.7)
    assert result.metadata.get("input_type") == "sequential"


def test_feature_version_mismatch_raises(tmp_path):
    """meta.json 的 feature_version 与 config 不匹配时应抛错。"""
    cfg = Layer1Config(model_type="lgbm", feature_version=2)

    # 先训练一个真正的模型
    from quant.strategy.layer1.backends.lgbm_backend import LGBMBackend

    backend = LGBMBackend(cfg)
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    y = X[:, 0] * 0.5 + rng.randn(200) * 0.1
    backend.fit(X, y)

    model_dir = tmp_path / "mismatched_model"
    model_dir.mkdir()
    backend.save(model_dir)

    # 篡改 meta.json 中的 feature_version
    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["feature_version"] = 1
    meta_path.write_text(json.dumps(meta))

    layer1 = Layer1(cfg)
    with pytest.raises(ValueError, match="feature_version"):
        layer1.load_model(model_dir)
