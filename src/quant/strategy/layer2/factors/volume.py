from __future__ import annotations

import numpy as np

from quant.core.types import Bar
from quant.strategy.layer2.factors.base import AlphaFactor


class OBVMomentum(AlphaFactor):
    """OBV 10-bar 变化率 z-score。OBV 上升 → 多头资金流入 → 偏多。"""

    name = "OBVMomentum"
    min_bars = 30

    def compute(self, bars: list[Bar]) -> float:
        if len(bars) < self.min_bars:
            return 0.0
        obv = 0.0
        obvs = []
        for i in range(1, len(bars)):
            if bars[i].close > bars[i - 1].close:
                obv += bars[i].volume
            elif bars[i].close < bars[i - 1].close:
                obv -= bars[i].volume
            obvs.append(obv)
        if len(obvs) < 11:
            return 0.0
        roc10 = [
            (obvs[i] - obvs[i - 10]) / (abs(obvs[i - 10]) + 1e-9)
            for i in range(10, len(obvs))
        ]
        return self._safe_zscore(roc10)


class VolumeAccel(AlphaFactor):
    """成交量加速度（当前成交量 vs 历史均值之比的变化率）。放量突破 → 偏多。"""

    name = "VolumeAccel"
    min_bars = 25

    def compute(self, bars: list[Bar]) -> float:
        if len(bars) < self.min_bars:
            return 0.0
        vols = np.array([b.volume for b in bars], dtype=np.float64)
        vol_ma = vols[-20:].mean() + 1e-9
        vol_ratio = vols[-1] / vol_ma
        # 成交量放大 + 价格上涨 → 偏多
        closes = np.array([b.close for b in bars], dtype=np.float64)
        price_dir = 1.0 if closes[-1] > closes[-2] else -1.0
        score = (vol_ratio - 1.0) * price_dir
        return float(np.clip(score, -1.0, 1.0))
