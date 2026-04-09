"""Layer1 ModelBackend Protocol + BackendMeta dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BackendMeta:
    """模型元数据，随模型一同保存至 meta.json。"""

    model_type: str
    input_type: Literal["tabular", "sequential"]
    feature_dim: int
    lookback_T: int = 1
    feature_names: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelBackend(Protocol):
    """Layer1 模型后端统一接口。

    Tabular 后端:  fit(X, y, ...) X: (N, D)   -> predict(X) X: (1, D)
    Sequential 后端: fit(X, y, ...) X: (N, T, D) -> predict(X) X: (1, T, D)
    """

    @property
    def input_type(self) -> Literal["tabular", "sequential"]: ...

    @property
    def is_fitted(self) -> bool: ...

    @property
    def meta(self) -> BackendMeta: ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        categorical_indices: list[int] | None = None,
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        """训练模型。返回训练日志。"""
        ...

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        """单样本推理，返回 (beta_score, confidence)。"""
        ...

    def save(self, directory: Path) -> None:
        """将模型权重 + meta.json 保存到目录。"""
        ...

    def load(self, directory: Path) -> None:
        """从目录加载模型。"""
        ...
