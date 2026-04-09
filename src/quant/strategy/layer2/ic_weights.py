from __future__ import annotations

from collections import deque

import numpy as np

from quant.config.schema import Layer2Config


class ICWeightCalculator:
    """滚动 IC（信息系数）计算，用于因子权重。

    IC = Spearman corr(因子值, 前瞻收益率)，用历史 ic_lookback_bars 的 IC 均值加权。
    Shrinkage：向等权缩减，避免单因子过度主导。
    """

    def __init__(self, cfg: Layer2Config, factor_names: list[str]) -> None:
        self._cfg = cfg
        self._names = factor_names
        self._history: dict[str, deque[tuple[float, float]]] = {
            name: deque(maxlen=cfg.ic_lookback_bars) for name in factor_names
        }  # 存 (factor_value, realized_return)

    def update(self, factor_values: dict[str, float], realized_return: float) -> None:
        """每个 bar 调用一次，记录 (factor_value, 下一期实现收益率)。"""
        for name, val in factor_values.items():
            if name in self._history:
                self._history[name].append((val, realized_return))

    def get_weights(self) -> dict[str, float]:
        """返回 IC-based 权重（含 shrinkage），各权重绝对值之和 = 1。"""
        n = len(self._names)
        if n == 0:
            return {}
        equal_w = 1.0 / n

        ic_scores: dict[str, float] = {}
        for name in self._names:
            hist = self._history[name]
            if len(hist) < 10:
                ic_scores[name] = 0.0
                continue
            factors = np.array([h[0] for h in hist])
            returns = np.array([h[1] for h in hist])
            # Spearman IC（rank correlation）
            rank_f = np.argsort(np.argsort(factors)).astype(float)
            rank_r = np.argsort(np.argsort(returns)).astype(float)
            ic = float(np.corrcoef(rank_f, rank_r)[0, 1])
            ic_scores[name] = ic if not np.isnan(ic) else 0.0

        total_abs_ic = sum(abs(v) for v in ic_scores.values()) + 1e-9
        shrinkage = self._cfg.ic_shrinkage
        weights: dict[str, float] = {}
        for name in self._names:
            ic_w = ic_scores[name] / total_abs_ic
            weights[name] = (1 - shrinkage) * ic_w + shrinkage * equal_w

        # 归一化（绝对值之和 = 1）
        total = sum(abs(w) for w in weights.values()) + 1e-9
        return {k: v / total for k, v in weights.items()}
