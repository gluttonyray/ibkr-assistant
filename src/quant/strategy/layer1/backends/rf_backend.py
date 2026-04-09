"""Random Forest 回归后端。"""
from __future__ import annotations

from typing import Any

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backends._sklearn_base import SklearnTabularBase


class RandomForestBackend(SklearnTabularBase):
    """RandomForestRegressor 后端。"""

    @property
    def _model_type(self) -> str:
        return "rf"

    def _create_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
        )
