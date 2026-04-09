"""向后兼容导入。Layer1Model 已迁移至 backends/lgbm_backend.py。"""
from quant.strategy.layer1.backends.lgbm_backend import LGBMBackend as Layer1Model

__all__ = ["Layer1Model"]
