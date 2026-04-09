"""LightGBM Backend — 重构自 model.py，实现 ModelBackend Protocol。"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backend_protocol import BackendMeta

logger = logging.getLogger(__name__)

_EPS = 1e-9


class LGBMBackend:
    """LightGBM Huber 回归后端。

    保留原生 lgb.train() API、categorical_feature 声明和 early_stopping。
    """

    def __init__(self, cfg: Layer1Config) -> None:
        self._cfg = cfg
        self._model: object | None = None  # lgb.Booster
        self._is_fitted: bool = False
        self._confidence_scale: float = 1.0
        self._feature_dim: int = 0

    @property
    def input_type(self) -> Literal["tabular"]:
        return "tabular"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def meta(self) -> BackendMeta:
        return BackendMeta(
            model_type="lgbm",
            input_type="tabular",
            feature_dim=self._feature_dim,
            extra={"confidence_scale": self._confidence_scale},
        )

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        categorical_indices: list[int] | None = None,
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not available -- LGBMBackend will return zeros")
            self._is_fitted = False
            return {"error": "lightgbm not installed"}

        if len(X) < 50:
            logger.warning(
                "LGBMBackend.fit: only %d samples (need >= 50), skipping", len(X)
            )
            self._is_fitted = False
            return {"error": "insufficient_samples", "n_samples": len(X)}

        self._feature_dim = X.shape[1]
        cat_cols = list(categorical_indices) if categorical_indices else []

        n = len(X)
        split = int(n * (1.0 - val_fraction))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        w_train = sample_weights[:split] if sample_weights is not None else None
        w_val = sample_weights[split:] if sample_weights is not None else None

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=w_train,
            categorical_feature=cat_cols if cat_cols else "auto",
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            weight=w_val,
            categorical_feature=cat_cols if cat_cols else "auto",
            reference=train_data,
            free_raw_data=False,
        )

        params: dict = {
            "objective": self._cfg.lgb_objective,
            "alpha": self._cfg.lgb_huber_delta,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 5,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "random_state": 42,
        }

        self._model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        self._is_fitted = True

        # confidence_scale = |训练预测值| 的第 90 百分位数
        y_pred_train = self._model.predict(X_train)
        self._confidence_scale = float(
            np.percentile(np.abs(y_pred_train), 90) + _EPS
        )

        best_iteration = getattr(self._model, "best_iteration", -1)
        logger.debug(
            "LGBMBackend trained: N=%d, confidence_scale=%.4f, best_iteration=%s",
            n,
            self._confidence_scale,
            best_iteration,
        )

        # 计算验证集 loss
        y_pred_val = self._model.predict(X_val)
        val_loss = float(np.mean((y_pred_val - y_val) ** 2))

        return {"best_iteration": best_iteration, "val_loss": val_loss}

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        if not self._is_fitted or self._model is None:
            return 0.0, 0.0

        pred = self._model.predict(X.reshape(1, -1))
        beta_score = float(pred[0])
        confidence = float(
            min(abs(beta_score) / (self._confidence_scale + _EPS), 1.0)
        )
        return beta_score, confidence

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_file = "model.pkl"
        with open(directory / model_file, "wb") as f:
            pickle.dump(
                {"model": self._model, "confidence_scale": self._confidence_scale},
                f,
            )

        meta = {
            "model_type": "lgbm",
            "model_file": model_file,
            "feature_dim": self._feature_dim,
            "confidence_scale": self._confidence_scale,
        }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.debug("LGBMBackend saved to %s", directory)

    def load(self, directory: Path) -> None:
        directory = Path(directory)

        meta_path = directory / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_file = meta.get("model_file", "model.pkl")
            self._feature_dim = meta.get("feature_dim", 0)
        else:
            model_file = "model.pkl"

        with open(directory / model_file, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            self._model = payload["model"]
            self._confidence_scale = float(
                payload.get("confidence_scale", 1.0)
            )
        else:
            # 向后兼容：旧格式直接存储裸 Booster
            self._model = payload
            self._confidence_scale = 1.0

        self._is_fitted = True
        logger.debug("LGBMBackend loaded from %s", directory)
