"""Backend registry 工厂函数测试。"""
from __future__ import annotations

import pytest

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backend_protocol import ModelBackend
from quant.strategy.layer1.backends import create_backend


@pytest.fixture
def cfg() -> Layer1Config:
    return Layer1Config()


def test_create_lgbm_backend(cfg: Layer1Config) -> None:
    backend = create_backend("lgbm", cfg)
    assert isinstance(backend, ModelBackend)
    assert backend.input_type == "tabular"
    assert not backend.is_fitted


def test_unknown_backend_raises(cfg: Layer1Config) -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("nonexistent", cfg)


def test_create_ridge_backend(cfg: Layer1Config) -> None:
    backend = create_backend("ridge", cfg)
    assert isinstance(backend, ModelBackend)
    assert backend.input_type == "tabular"


def test_create_elasticnet_backend(cfg: Layer1Config) -> None:
    backend = create_backend("elasticnet", cfg)
    assert isinstance(backend, ModelBackend)


def test_create_rf_backend(cfg: Layer1Config) -> None:
    backend = create_backend("rf", cfg)
    assert isinstance(backend, ModelBackend)
