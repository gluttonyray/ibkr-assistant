from __future__ import annotations

from datetime import datetime

from quant.config.schema import StrategyConfig
from quant.core.strategy import Layer1Result, Layer2Result
from quant.core.types import Instrument, Signal, SignalType


class DefaultComposer:
    """合成 Layer1 + Layer2 信号。实现 SignalComposer Protocol。

    公式：S_combined = w_L1 * S_L1 * MII + w_L2 * S_L2
    MII -> 0：Layer1 贡献趋近 0，Layer2 独立运作。
    """

    def __init__(self, cfg: StrategyConfig) -> None:
        self._cfg = cfg

    def compose(
        self,
        instrument: Instrument,
        layer1: Layer1Result,
        layer2: Layer2Result,
        timestamp: datetime,
    ) -> Signal:
        w1 = self._cfg.layer1.weight
        w2 = self._cfg.layer2.weight

        s_combined = w1 * layer1.beta_score * layer1.mii + w2 * layer2.alpha_score

        # 阈值判断
        entry_thr = self._cfg.layer2.entry_threshold
        exit_thr = self._cfg.layer2.exit_threshold

        if s_combined > entry_thr:
            signal_type = SignalType.LONG_ENTRY
        elif s_combined < -entry_thr:
            signal_type = SignalType.SHORT_ENTRY
        elif abs(s_combined) < exit_thr:
            signal_type = SignalType.LONG_EXIT  # 实际平仓方向由引擎根据持仓决定
        else:
            signal_type = SignalType.HOLD

        return Signal(
            instrument=instrument,
            signal_type=signal_type,
            score=s_combined,
            regime=layer1.regime,
            layer1_score=layer1.beta_score,
            layer2_score=layer2.alpha_score,
            mii=layer1.mii,
            timestamp=timestamp,
            metadata={
                "confidence": layer1.confidence,
                **{f"factor_{k}": v for k, v in layer2.factor_details.items()},
            },
        )
