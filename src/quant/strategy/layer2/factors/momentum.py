from __future__ import annotations

import numpy as np

from quant.core.types import Bar
from quant.strategy.layer2.factors.base import AlphaFactor


class ROC20(AlphaFactor):
    """20-bar 收益率 z-score（时序动量）。"""

    name = "ROC20"
    min_bars = 40

    def compute(self, bars: list[Bar]) -> float:
        if len(bars) < self.min_bars:
            return 0.0
        closes = np.array([b.close for b in bars], dtype=np.float64)
        return self._safe_zscore([
            (closes[i] / closes[i - 20] - 1)
            for i in range(20, len(closes))
            if closes[i - 20] != 0
        ])


class MeanRev5(AlphaFactor):
    """5-bar 反转（短周期均值回归）。方向与 ROC20 相反，捕捉过冲回调。"""

    name = "MeanRev5"
    min_bars = 20

    def compute(self, bars: list[Bar]) -> float:
        if len(bars) < self.min_bars:
            return 0.0
        closes = np.array([b.close for b in bars], dtype=np.float64)
        rocs = [
            (closes[i] / closes[i - 5] - 1)
            for i in range(5, len(closes))
            if closes[i - 5] != 0
        ]
        z = self._safe_zscore(rocs)
        return -z  # 反转：高 ROC5 → 空头偏向
