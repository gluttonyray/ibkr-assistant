from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from quant.core.types import Bar


class AlphaFactor(ABC):
    """Alpha 因子基类。所有因子返回值 ∈ [-1, +1]，+1 强多头信号，-1 强空头信号。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """计算该因子所需最少 bar 数量。"""

    @abstractmethod
    def compute(self, bars: list[Bar]) -> float:
        """返回因子值 ∈ [-1, +1]。若 bars 不足 min_bars，返回 0.0。"""

    def _safe_zscore(self, series: list[float] | np.ndarray, clip: float = 3.0) -> float:
        """对序列末尾元素做 z-score 标准化并截断到 [-clip, clip]，映射到 [-1, +1]。"""
        arr = np.array(series, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        mu, sigma = arr[:-1].mean(), arr[:-1].std() + 1e-9
        z = (arr[-1] - mu) / sigma
        return float(np.clip(z / clip, -1.0, 1.0))
