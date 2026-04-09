"""XGBoost 回归后端（可选依赖）。"""
from __future__ import annotations

from typing import Any

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backends._sklearn_base import SklearnTabularBase


class XGBBackend(SklearnTabularBase):
    """XGBRegressor 后端。类别特征使用 ordinal 编码（不声明 enable_categorical）。"""

    @property
    def _model_type(self) -> str:
        return "xgb"

    def _create_estimator(self) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
