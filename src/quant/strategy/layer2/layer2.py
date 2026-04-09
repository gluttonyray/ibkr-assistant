from __future__ import annotations

from collections import deque

import numpy as np

from quant.config.schema import Layer2Config
from quant.core.types import Bar, Instrument
from quant.core.strategy import Layer1Result, Layer2Result
from quant.strategy.layer2.factors.momentum import ROC20, MeanRev5
from quant.strategy.layer2.factors.volatility import VolRatio, BBWidth
from quant.strategy.layer2.factors.volume import OBVMomentum, VolumeAccel
from quant.strategy.layer2.factors.trend import EMASlope, ADXStrength
from quant.strategy.layer2.ic_weights import ICWeightCalculator

_ALL_FACTORS = [
    ROC20(), MeanRev5(), VolRatio(), BBWidth(),
    OBVMomentum(), VolumeAccel(), EMASlope(), ADXStrength(),
]


class Layer2:
    """Alpha 因子层，实现 Layer2Strategy Protocol。"""

    def __init__(self, cfg: Layer2Config) -> None:
        self._cfg = cfg
        self._factors = _ALL_FACTORS
        self._ic_calc: dict[str, ICWeightCalculator] = {}
        self._signal_history: dict[str, deque[float]] = {}  # anti-flicker

    def compute(
        self,
        instrument: Instrument,
        window: list[Bar],
        layer1: Layer1Result,
    ) -> Layer2Result:
        sym = instrument.symbol

        # 因子计算
        factor_values: dict[str, float] = {}
        for f in self._factors:
            if len(window) >= f.min_bars:
                factor_values[f.name] = f.compute(window)
            else:
                factor_values[f.name] = 0.0

        # IC 加权
        if sym not in self._ic_calc:
            self._ic_calc[sym] = ICWeightCalculator(
                self._cfg, [f.name for f in self._factors]
            )
        weights = self._ic_calc[sym].get_weights()

        if weights:
            alpha_raw = sum(
                factor_values[k] * weights.get(k, 1.0 / len(self._factors))
                for k in factor_values
            )
        else:
            # 初期无 IC 历史：等权
            n = len(factor_values)
            alpha_raw = sum(factor_values.values()) / (n + 1e-9)

        alpha_raw = float(np.clip(alpha_raw, -1.0, 1.0))

        # Anti-flicker：取最近 confirm_bars 的平均
        if sym not in self._signal_history:
            self._signal_history[sym] = deque(maxlen=self._cfg.confirm_bars)
        self._signal_history[sym].append(alpha_raw)
        alpha_score = float(np.mean(self._signal_history[sym]))

        return Layer2Result(
            alpha_score=alpha_score,
            factor_details=factor_values,
        )

    def update_ic(
        self,
        instrument: Instrument,
        factor_values: dict[str, float],
        realized_return: float,
    ) -> None:
        """在每个 ic_forward_horizon 之后调用，更新 IC 历史。"""
        sym = instrument.symbol
        if sym in self._ic_calc:
            self._ic_calc[sym].update(factor_values, realized_return)
