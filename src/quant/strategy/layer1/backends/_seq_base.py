"""PyTorch Sequential 后端基类。

子类需实现：
    _model_type: str
    _build_model(T, D, cfg) -> nn.Module
    _default_hparams(cfg) -> dict
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backend_protocol import BackendMeta
from quant.strategy.layer1.sequence_features import SEQUENCE_FEATURE_DIM, SEQUENCE_FEATURE_NAMES

logger = logging.getLogger(__name__)

_EPS = 1e-9


class SequentialBackendBase(ABC):
    """PyTorch sequential 后端共用基类。"""

    input_type: Literal["sequential"] = "sequential"

    @property
    @abstractmethod
    def _model_type(self) -> str: ...

    @abstractmethod
    def _build_model(self, T: int, D: int, cfg: Layer1Config) -> Any:
        """构建 PyTorch nn.Module。返回 nn.Module 实例。"""
        ...

    @abstractmethod
    def _default_hparams(self, cfg: Layer1Config) -> dict: ...

    def __init__(self, cfg: Layer1Config) -> None:
        self._cfg = cfg
        self._model: Any = None  # nn.Module, lazy
        self._is_fitted: bool = False
        self._confidence_scale: float = 1.0
        self._temperature: float = cfg.confidence_temperature if hasattr(cfg, "confidence_temperature") else 1.0
        self._model_T: int = 0
        self._model_D: int = SEQUENCE_FEATURE_DIM
        self._hparams: dict = {}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def meta(self) -> BackendMeta:
        return BackendMeta(
            model_type=self._model_type,
            input_type="sequential",
            feature_dim=self._model_D,
            lookback_T=self._model_T,
            feature_names=list(SEQUENCE_FEATURE_NAMES),
            extra={
                "confidence_scale": self._confidence_scale,
                "hparams": self._hparams,
            },
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
        """训练模型。X: (N, T, D), y: (N,)。"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        N, T, D = X.shape
        self._model_T = T
        self._model_D = D
        self._hparams = self._default_hparams(self._cfg)

        # 构建模型
        model = self._build_model(T, D, self._cfg)

        # 时序切分（不打乱顺序）
        split = int(N * (1 - val_fraction))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        w_train = sample_weights[:split] if sample_weights is not None else None

        # 转换为 tensor
        X_t = torch.from_numpy(X_train).float()
        y_t = torch.from_numpy(y_train).float()
        X_v = torch.from_numpy(X_val).float()
        y_v = torch.from_numpy(y_val).float()

        if w_train is not None:
            w_t = torch.from_numpy(w_train).float()
            train_ds = TensorDataset(X_t, y_t, w_t)
        else:
            train_ds = TensorDataset(X_t, y_t)

        batch_size = getattr(self._cfg, "seq_batch_size", 256)
        epochs = getattr(self._cfg, "seq_epochs", 50)
        lr = getattr(self._cfg, "seq_learning_rate", 1e-3)
        patience = getattr(self._cfg, "seq_early_stopping_patience", 10)
        huber_delta = self._cfg.lgb_huber_delta

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

        # 设备选择：训练用 GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.HuberLoss(delta=huber_delta)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                if len(batch) == 3:
                    xb, yb, wb = batch
                    xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    wb = None

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                if wb is not None:
                    loss = (loss * wb).mean() if loss.dim() > 0 else loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # 验证
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v.to(device))
                val_loss = float(criterion(val_pred, y_v.to(device)).item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # 恢复最佳模型并移到 CPU
        if best_state is not None:
            model.load_state_dict(best_state)
        model = model.cpu()
        model.eval()
        self._model = model
        self._is_fitted = True

        # confidence_scale = P90(|y_pred_train|)
        with torch.no_grad():
            y_pred_train = model(X_t).numpy()
        self._confidence_scale = float(
            np.percentile(np.abs(y_pred_train), 90) + _EPS
        )

        return {
            "val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "model_type": self._model_type,
            "confidence_scale": self._confidence_scale,
        }

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        """单样本推理，返回 (beta_score, confidence)。"""
        if not self._is_fitted or self._model is None:
            return 0.0, 0.0

        import torch

        # X shape: (T, D) 或 (1, T, D)
        if X.ndim == 2:
            X = X[np.newaxis, ...]  # (1, T, D)

        self._model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).float()
            pred = self._model(x_tensor)
            beta_score = float(pred.item())

        temperature = self._temperature
        confidence = min(
            abs(beta_score) / (self._confidence_scale * temperature + _EPS), 1.0
        )
        return beta_score, confidence

    def save(self, directory: Path) -> None:
        """保存模型到目录。"""
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            torch.save(self._model.state_dict(), directory / "checkpoint.pt")

        # 保存模型配置（用于重建结构）
        model_config = {
            "T": self._model_T,
            "D": self._model_D,
            "hparams": self._hparams,
        }
        (directory / "model_config.json").write_text(
            json.dumps(model_config, indent=2)
        )

        # meta.json
        meta = {
            "model_type": self._model_type,
            "input_type": "sequential",
            "model_file": "checkpoint.pt",
            "feature_dim": self._model_D,
            "lookback_T": self._model_T,
            "confidence_scale": self._confidence_scale,
            "is_fitted": self._is_fitted,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, directory: Path) -> None:
        """从目录加载模型。"""
        import torch

        directory = Path(directory)

        # 读取模型配置
        config_path = directory / "model_config.json"
        config = json.loads(config_path.read_text())
        T = config["T"]
        D = config["D"]
        self._hparams = config.get("hparams", {})
        self._model_T = T
        self._model_D = D

        # 读取 meta
        meta_path = directory / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._confidence_scale = meta.get("confidence_scale", 1.0)

        # 重建模型并加载权重
        model = self._build_model(T, D, self._cfg)
        state_dict = torch.load(
            directory / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        self._is_fitted = True
