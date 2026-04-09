"""Layer1 后端注册表与工厂函数。"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backend_protocol import ModelBackend

_REGISTRY: dict[str, Callable[[Layer1Config], Any]] = {}


def register_backend(name: str) -> Callable:
    """装饰器：注册模型后端工厂函数。"""

    def decorator(factory: Callable) -> Callable:
        _REGISTRY[name] = factory
        return factory

    return decorator


def create_backend(name: str, cfg: Layer1Config) -> ModelBackend:
    """根据名称创建模型后端实例。未安装依赖时抛出 ImportError。"""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](cfg)


# -- 注册 tabular 后端 --------------------------------------------------


@register_backend("lgbm")
def _create_lgbm(cfg: Layer1Config) -> Any:
    from quant.strategy.layer1.backends.lgbm_backend import LGBMBackend

    return LGBMBackend(cfg)


@register_backend("xgb")
def _create_xgb(cfg: Layer1Config) -> Any:
    try:
        from quant.strategy.layer1.backends.xgb_backend import XGBBackend
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. Install with: pip install xgboost"
        )
    return XGBBackend(cfg)


@register_backend("ridge")
def _create_ridge(cfg: Layer1Config) -> Any:
    from quant.strategy.layer1.backends.ridge_backend import RidgeBackend

    return RidgeBackend(cfg)


@register_backend("elasticnet")
def _create_elasticnet(cfg: Layer1Config) -> Any:
    from quant.strategy.layer1.backends.elasticnet_backend import ElasticNetBackend

    return ElasticNetBackend(cfg)


@register_backend("rf")
def _create_rf(cfg: Layer1Config) -> Any:
    from quant.strategy.layer1.backends.rf_backend import RandomForestBackend

    return RandomForestBackend(cfg)


# -- 注册 sequential 后端（需要 PyTorch）------------------------------------

try:
    @register_backend("dlinear")
    def _create_dlinear(cfg: Layer1Config) -> Any:
        from quant.strategy.layer1.backends.dlinear_backend import DLinearBackend

        return DLinearBackend(cfg)

    @register_backend("patchtst")
    def _create_patchtst(cfg: Layer1Config) -> Any:
        from quant.strategy.layer1.backends.patchtst_backend import PatchTSTBackend

        return PatchTSTBackend(cfg)

    @register_backend("lstm")
    def _create_lstm(cfg: Layer1Config) -> Any:
        from quant.strategy.layer1.backends.lstm_backend import LSTMBackend

        return LSTMBackend(cfg)

    @register_backend("gru")
    def _create_gru(cfg: Layer1Config) -> Any:
        from quant.strategy.layer1.backends.lstm_backend import GRUBackend

        return GRUBackend(cfg)

except ImportError:
    pass
