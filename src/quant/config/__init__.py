"""Configuration package -- the single entry point for all settings.

Usage::

    from quant.config import load_config

    cfg = load_config()                         # defaults + env vars
    cfg = load_config("configs/prod.yaml")      # with override file
    cfg.strategy.layer1.weight                  # structured access
"""
from quant.config.loader import load_config
from quant.config.schema import (
    AppConfig,
    BacktestConfig,
    CostConfig,
    IBKRConfig,
    Layer1Config,
    Layer2Config,
    RiskConfig,
    StrategyConfig,
)

__all__ = [
    "load_config",
    "AppConfig",
    "IBKRConfig",
    "StrategyConfig",
    "Layer1Config",
    "Layer2Config",
    "CostConfig",
    "RiskConfig",
    "BacktestConfig",
]
