from __future__ import annotations
import numpy as np
from quant.config.schema import Layer1Config

class MIICalculator:
    """动量信息含量指数（Momentum Information Index），∈ [0, 1]。

    MII → 1：动量持续有效，Layer1 满权重输入合并信号。
    MII → 0：动量耗尽，Layer1 贡献趋近中性，Layer2 独立运作。
    """

    def __init__(self, cfg: Layer1Config) -> None:
        self._cfg = cfg
        self._duration_in_extreme: int = 0  # 持续在极端区间的 bar 数

    def compute(self, returns_z: np.ndarray, volume: np.ndarray) -> float:
        """计算 MII。

        Args:
            returns_z: z-score 标准化的滚动收益率序列，形状 (N,)，最后一个是当前 bar。
            volume: 对应的成交量序列，形状 (N,)。
        Returns:
            MII ∈ [0, 1]
        """
        if len(returns_z) < 2:
            return 1.0

        current_z = returns_z[-1]
        prev_z = returns_z[-2]

        # ── 1. 幅度维度 ──────────────────────────────────────────────────
        # |z| 超过历史 extreme_percentile → 衰减
        extreme_threshold = np.percentile(np.abs(returns_z[:-1]), self._cfg.mii_extreme_percentile)
        amplitude_in_extreme = abs(current_z) > extreme_threshold

        # ── 2. 加速度维度 ──────────────────────────────────────────────
        # 动量减速（|current_z| < |prev_z|）→ 加速衰减
        decelerating = abs(current_z) < abs(prev_z)

        # ── 3. 持续时间维度 ────────────────────────────────────────────
        if amplitude_in_extreme:
            self._duration_in_extreme += 1
        else:
            self._duration_in_extreme = 0

        halflife = self._cfg.mii_duration_halflife
        duration_decay = 0.5 ** (self._duration_in_extreme / halflife)

        # ── 4. 成交量修正 ──────────────────────────────────────────────
        # 若成交量仍在放大，MII 衰减放缓（区分技术性耗尽 vs 信息驱动）
        vol_ma = volume[-20:].mean() if len(volume) >= 20 else volume.mean()
        volume_expanding = volume[-1] > vol_ma * 1.2

        # ── 合成 MII ───────────────────────────────────────────────────
        mii = 1.0
        if amplitude_in_extreme:
            mii *= 0.6  # 在极端区间基础衰减
        if decelerating and amplitude_in_extreme:
            mii *= 0.7  # 减速叠加衰减
        mii *= duration_decay
        if volume_expanding and amplitude_in_extreme:
            mii = mii + (1.0 - mii) * 0.5  # 放量时拉回一半衰减

        return float(np.clip(mii, 0.0, 1.0))

    def reset(self) -> None:
        """重置持续时间计数（每个 instrument 独立实例，不需要共享状态）。"""
        self._duration_in_extreme = 0
