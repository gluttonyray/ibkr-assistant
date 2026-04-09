"""sklearn 系列后端测试：Ridge / ElasticNet / RandomForest。"""
from __future__ import annotations

import numpy as np
import pytest

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backend_protocol import ModelBackend
from quant.strategy.layer1.backends import create_backend


@pytest.fixture
def cfg() -> Layer1Config:
    return Layer1Config()


@pytest.fixture
def training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.randn(200) * 0.1
    return X, y


@pytest.mark.parametrize("backend_name", ["ridge", "elasticnet", "rf"])
def test_fit_and_predict(
    cfg: Layer1Config,
    training_data: tuple[np.ndarray, np.ndarray],
    backend_name: str,
) -> None:
    X, y = training_data
    backend = create_backend(backend_name, cfg)

    assert isinstance(backend, ModelBackend)
    assert not backend.is_fitted

    result = backend.fit(X, y)

    assert backend.is_fitted
    assert "val_mse" in result

    beta, conf = backend.predict(X[0])
    assert isinstance(beta, float)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


@pytest.mark.parametrize("backend_name", ["ridge", "elasticnet", "rf"])
def test_predict_unfitted_returns_zeros(
    cfg: Layer1Config,
    backend_name: str,
) -> None:
    backend = create_backend(backend_name, cfg)
    beta, conf = backend.predict(np.zeros(10))
    assert beta == 0.0
    assert conf == 0.0


@pytest.mark.parametrize("backend_name", ["ridge", "elasticnet", "rf"])
def test_save_load_roundtrip(
    cfg: Layer1Config,
    training_data: tuple[np.ndarray, np.ndarray],
    backend_name: str,
    tmp_path,
) -> None:
    X, y = training_data
    backend = create_backend(backend_name, cfg)
    backend.fit(X, y)

    beta_before, conf_before = backend.predict(X[0])

    save_dir = tmp_path / f"{backend_name}_model"
    backend.save(save_dir)

    assert (save_dir / "model.joblib").exists()
    assert (save_dir / "meta.json").exists()

    backend2 = create_backend(backend_name, cfg)
    backend2.load(save_dir)

    assert backend2.is_fitted
    beta_after, conf_after = backend2.predict(X[0])

    assert beta_before == pytest.approx(beta_after)
    assert conf_before == pytest.approx(conf_after)


@pytest.mark.parametrize("backend_name", ["ridge", "elasticnet", "rf"])
def test_meta_after_fit(
    cfg: Layer1Config,
    training_data: tuple[np.ndarray, np.ndarray],
    backend_name: str,
) -> None:
    X, y = training_data
    backend = create_backend(backend_name, cfg)
    backend.fit(X, y)

    meta = backend.meta
    assert meta.input_type == "tabular"
    assert meta.feature_dim == 10
