"""Ridge 回归后端。"""
from __future__ import annotations

from typing import Any

from quant.strategy.layer1.backends._sklearn_base import SklearnTabularBase


class RidgeBackend(SklearnTabularBase):
    """Pipeline(StandardScaler, Ridge) 后端。"""

    @property
    def _model_type(self) -> str:
        return "ridge"

    def _create_estimator(self) -> Any:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
