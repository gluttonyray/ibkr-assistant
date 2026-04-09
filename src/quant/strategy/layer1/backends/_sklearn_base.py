"""sklearn 系列 tabular 后端的公共基类。"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backend_protocol import BackendMeta

logger = logging.getLogger(__name__)

_EPS = 1e-9


class SklearnTabularBase(ABC):
    """sklearn 系列 tabular 后端的公共基类。

    子类只需实现 ``_model_type`` 和 ``_create_estimator()``。
    基类处理：fit / predict / save / load / confidence 校准。
    """

    def __init__(self, cfg: Layer1Config) -> None:
        self._cfg = cfg
        self._estimator: Any | None = None
        self._is_fitted: bool = False
        self._confidence_scale: float = 1.0
        self._feature_dim: int = 0

    @property
    def input_type(self) -> Literal["tabular"]:
        return "tabular"

    @property
    @abstractmethod
    def _model_type(self) -> str: ...

    @abstractmethod
    def _create_estimator(self) -> Any:
        """返回一个 sklearn 兼容的 estimator（带 fit/predict 方法）。"""
        ...

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def meta(self) -> BackendMeta:
        return BackendMeta(
            model_type=self._model_type,
            input_type="tabular",
            feature_dim=self._feature_dim,
            extra={"confidence_scale": self._confidence_scale},
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        categorical_indices: list[int] | None = None,
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        self._feature_dim = X.shape[1]
        self._estimator = self._create_estimator()

        # 时序 80/20 切分
        n = len(X)
        split = int(n * (1.0 - val_fraction))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        if sample_weights is not None:
            self._estimator.fit(X_train, y_train, sample_weight=sample_weights[:split])
        else:
            self._estimator.fit(X_train, y_train)

        self._is_fitted = True

        # confidence_scale = P90(|y_pred_train|)
        y_pred_train = self._estimator.predict(X_train)
        self._confidence_scale = float(
            np.percentile(np.abs(y_pred_train), 90) + _EPS
        )

        # 验证集 MSE
        y_pred_val = self._estimator.predict(X_val)
        val_mse = float(np.mean((y_pred_val - y_val) ** 2))

        logger.debug(
            "%s trained: N=%d, confidence_scale=%.4f, val_mse=%.6f",
            self._model_type,
            n,
            self._confidence_scale,
            val_mse,
        )
        return {"val_mse": val_mse, "model_type": self._model_type}

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        if not self._is_fitted or self._estimator is None:
            return 0.0, 0.0

        pred = self._estimator.predict(X.reshape(1, -1))
        beta_score = float(pred[0])
        confidence = float(
            min(abs(beta_score) / (self._confidence_scale + _EPS), 1.0)
        )
        return beta_score, confidence

    def save(self, directory: Path) -> None:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_file = "model.joblib"
        joblib.dump(self._estimator, directory / model_file)

        meta = {
            "model_type": self._model_type,
            "model_file": model_file,
            "feature_dim": self._feature_dim,
            "confidence_scale": self._confidence_scale,
        }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.debug("%s saved to %s", self._model_type, directory)

    def load(self, directory: Path) -> None:
        import joblib

        directory = Path(directory)

        with open(directory / "meta.json") as f:
            meta = json.load(f)

        model_file = meta.get("model_file", "model.joblib")
        self._feature_dim = meta.get("feature_dim", 0)
        self._confidence_scale = meta.get("confidence_scale", 1.0)

        self._estimator = joblib.load(directory / model_file)
        self._is_fitted = True

        logger.debug("%s loaded from %s", self._model_type, directory)
