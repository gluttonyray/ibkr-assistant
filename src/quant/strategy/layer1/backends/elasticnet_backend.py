"""ElasticNet 回归后端。"""
from __future__ import annotations

from typing import Any

from quant.strategy.layer1.backends._sklearn_base import SklearnTabularBase


class ElasticNetBackend(SklearnTabularBase):
    """Pipeline(StandardScaler, ElasticNet) 后端。"""

    @property
    def _model_type(self) -> str:
        return "elasticnet"

    def _create_estimator(self) -> Any:
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ])
