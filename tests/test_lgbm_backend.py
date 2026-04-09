"""LGBMBackend 测试：fit/predict/save/load。"""
from __future__ import annotations

import numpy as np
import pytest

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backend_protocol import ModelBackend
from quant.strategy.layer1.backends.lgbm_backend import LGBMBackend


@pytest.fixture
def cfg() -> Layer1Config:
    return Layer1Config()


@pytest.fixture
def training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    y = X[:, 0] * 0.5 + rng.randn(200) * 0.1
    return X, y


def test_implements_protocol(cfg: Layer1Config) -> None:
    backend = LGBMBackend(cfg)
    assert isinstance(backend, ModelBackend)


def test_predict_unfitted_returns_zeros(cfg: Layer1Config) -> None:
    backend = LGBMBackend(cfg)
    beta, conf = backend.predict(np.zeros(10))
    assert beta == 0.0
    assert conf == 0.0


def test_fit_and_predict(cfg: Layer1Config, training_data) -> None:
    X, y = training_data
    backend = LGBMBackend(cfg)
    result = backend.fit(X, y)

    assert backend.is_fitted
    assert "best_iteration" in result
    assert "val_loss" in result

    beta, conf = backend.predict(X[0])
    assert isinstance(beta, float)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_meta_after_fit(cfg: Layer1Config, training_data) -> None:
    X, y = training_data
    backend = LGBMBackend(cfg)
    backend.fit(X, y)

    meta = backend.meta
    assert meta.model_type == "lgbm"
    assert meta.input_type == "tabular"
    assert meta.feature_dim == 10


def test_save_load_roundtrip(cfg: Layer1Config, training_data, tmp_path) -> None:
    X, y = training_data
    backend = LGBMBackend(cfg)
    backend.fit(X, y)

    # 保存前的预测
    beta_before, conf_before = backend.predict(X[0])

    save_dir = tmp_path / "lgbm_model"
    backend.save(save_dir)

    # 验证文件存在
    assert (save_dir / "model.pkl").exists()
    assert (save_dir / "meta.json").exists()

    # 加载到新实例
    backend2 = LGBMBackend(cfg)
    backend2.load(save_dir)

    assert backend2.is_fitted
    beta_after, conf_after = backend2.predict(X[0])

    assert beta_before == pytest.approx(beta_after)
    assert conf_before == pytest.approx(conf_after)


def test_fit_insufficient_samples(cfg: Layer1Config) -> None:
    X = np.random.randn(10, 5)
    y = np.random.randn(10)
    backend = LGBMBackend(cfg)
    result = backend.fit(X, y)

    assert not backend.is_fitted
    assert "error" in result
