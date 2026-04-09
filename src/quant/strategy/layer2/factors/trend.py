from __future__ import annotations

import numpy as np

from quant.core.types import Bar
from quant.strategy.layer2.factors.base import AlphaFactor


class EMASlope(AlphaFactor):
    """EMA(20) 斜率 z-score。斜率向上 → 趋势多头。"""

    name = "EMASlope"
    min_bars = 30

    def compute(self, bars: list[Bar]) -> float:
        if len(bars) < self.min_bars:
            return 0.0
        closes = np.array([b.close for b in bars], dtype=np.float64)
        alpha = 2.0 / 21.0
        ema = closes[0]
        emas = [ema]
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema
            emas.append(ema)
        emas_arr = np.array(emas)
        slopes = np.diff(emas_arr)
        return self._safe_zscore(slopes.tolist())


class ADXStrength(AlphaFactor):
    """ADX 方向x强度（纯 numpy 实现，不依赖 TA-Lib）。"""

    name = "ADXStrength"
    min_bars = 30
    _period = 14

    def compute(self, bars: list[Bar]) -> float:
        if len(bars) < self.min_bars:
            return 0.0
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        closes = np.array([b.close for b in bars], dtype=np.float64)
        p = self._period

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                abs(highs[1:] - closes[:-1]),
                abs(lows[1:] - closes[:-1]),
            ),
        )

        # Directional Movement
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]
        pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth with Wilder (EMA-like)
        def wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
            result = np.zeros_like(arr)
            result[period - 1] = arr[:period].sum()
            for i in range(period, len(arr)):
                result[i] = result[i - 1] - result[i - 1] / period + arr[i]
            return result

        if len(tr) < p:
            return 0.0

        atr_s = wilder_smooth(tr, p)
        pdm_s = wilder_smooth(pdm, p)
        ndm_s = wilder_smooth(ndm, p)

        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = np.where(atr_s > 0, 100 * pdm_s / atr_s, 0.0)
            ndi = np.where(atr_s > 0, 100 * ndm_s / atr_s, 0.0)
            dx = np.where(
                (pdi + ndi) > 0, 100 * abs(pdi - ndi) / (pdi + ndi), 0.0
            )

        adx = wilder_smooth(dx[p - 1 :], p)
        if len(adx) == 0 or adx[-1] == 0:
            return 0.0

        direction = 1.0 if pdi[-1] > ndi[-1] else -1.0
        strength = float(np.clip(adx[-1] / 50.0, 0.0, 1.0))  # ADX > 50 → 最强
        return float(np.clip(direction * strength, -1.0, 1.0))
