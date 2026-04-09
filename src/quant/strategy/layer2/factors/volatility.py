from __future__ import annotations

import numpy as np

from quant.core.types import Bar
from quant.strategy.layer2.factors.base import AlphaFactor


class VolRatio(AlphaFactor):
    """短期/长期波动率比值。短期波动率骤升 → 空头偏向（不确定性上升）。"""

    name = "VolRatio"
    min_bars = 30

    def compute(self, bars: list[Bar]) -> float:
        if len(bars) < self.min_bars:
            return 0.0
        closes = np.array([b.close for b in bars], dtype=np.float64)
        rets = np.diff(np.log(closes + 1e-9))
        vol5 = rets[-5:].std() + 1e-9
        vol20 = rets[-20:].std() + 1e-9
        ratio = vol5 / vol20
        # ratio > 1 → 短期波动率高于长期 → 不确定 → 偏空
        return float(np.clip(-(ratio - 1.0) * 2, -1.0, 1.0))


class BBWidth(AlphaFactor):
    """布林带宽度变化率。宽度收窄后突破往往带来强趋势。"""

    name = "BBWidth"
    min_bars = 25

    def compute(self, bars: list[Bar]) -> float:
        if len(bars) < self.min_bars:
            return 0.0
        closes = np.array([b.close for b in bars[-25:]], dtype=np.float64)
        ma = closes[-20:].mean()
        std = closes[-20:].std() + 1e-9
        # 当前价相对布林带位置 → 正值偏多，负值偏空
        position = (closes[-1] - ma) / (2 * std + 1e-9)
        return float(np.clip(position, -1.0, 1.0))
