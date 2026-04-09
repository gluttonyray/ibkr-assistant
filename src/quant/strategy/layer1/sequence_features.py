"""Layer1 序列特征构建器。

为 sequential 后端（DLinear, PatchTST, LSTM/GRU）构建 (T, 14) 时序特征张量。
与 FeatureBuilderV2 共享相同的 14 个逐资产特征定义，
但输出是完整的时序窗口而非单时步快照。

No-lookahead 保证：第 t 步只使用 bars[:end_idx_t+1]。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from quant.core.types import Bar

SEQUENCE_FEATURE_NAMES = [
    "ret_1", "ret_5", "ret_20", "ret_60", "ret_120",
    "vol_20", "vol_60", "vol_ratio", "volume_ratio",
    "ret_vol_scaled_20", "ret_vol_scaled_60",
    "trend_consistency_20", "staleness_weight", "market_beta_20",
]
SEQUENCE_FEATURE_DIM = 14

# 年化因子（26 bars/天 @ 15 分钟）
_ANNUALIZE = np.sqrt(26.0)
_EPS = 1e-9


class SequenceFeatureBuilder:
    """从 Bar 序列构建 (T, D) 特征张量。"""

    def build_sequence(
        self,
        asset_bars: list[Bar],
        lookback_T: int,
        as_of: datetime | None = None,
    ) -> np.ndarray | None:
        """返回 (lookback_T, 14) float32 or None（bars 不足时）。

        只使用 as_of 之前（含）的 bar，严格 no-lookahead。
        bars 不足 vol_lookback + lookback_T 时返回 None。
        """
        vol_lookback = 60  # 需要至少 60 bars 的历史来计算 vol_60

        # 确定有效 bar 范围
        if as_of is not None:
            # 找到 as_of 之前（含）的最后一根 bar
            end_idx = None
            for i in range(len(asset_bars) - 1, -1, -1):
                if asset_bars[i].timestamp <= as_of:
                    end_idx = i
                    break
            if end_idx is None:
                return None
            bars = asset_bars[: end_idx + 1]
        else:
            bars = asset_bars

        # 需要至少 vol_lookback + lookback_T + 1 根 bar
        # (+1 是因为计算 ret_1 需要前一根 bar)
        min_required = vol_lookback + lookback_T + 1
        if len(bars) < min_required:
            return None

        closes = np.array([b.close for b in bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars], dtype=np.float64)

        # 构建 (T, 14) 特征矩阵
        result = np.zeros((lookback_T, SEQUENCE_FEATURE_DIM), dtype=np.float32)

        for t in range(lookback_T):
            # 当前时步对应 bars 中的绝对索引
            idx = len(bars) - lookback_T + t
            result[t] = self._compute_features_at(closes, volumes, idx)

        # 替换 NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def build_training_sequences(
        self,
        asset_bars: list[Bar],
        lookback_T: int,
        forward_horizon: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """sliding window，返回 (X: (N,T,14) float32, y: (N,) float64) or None。

        y[i] 是第 i 个窗口结束后 forward_horizon bars 的波动率缩放收益率。
        严格 no-lookahead：y[i] 使用 end_idx + forward_horizon 处的 close。
        """
        vol_lookback = 60

        # 最小 bar 数：vol_lookback + lookback_T + forward_horizon + 1
        min_required = vol_lookback + lookback_T + forward_horizon + 1
        if len(asset_bars) < min_required:
            return None

        closes = np.array([b.close for b in asset_bars], dtype=np.float64)
        volumes = np.array([b.volume for b in asset_bars], dtype=np.float64)

        # 第一个窗口的结束位置
        first_end = vol_lookback + lookback_T
        # 最后一个窗口的结束位置（需要留出 forward_horizon）
        last_end = len(asset_bars) - forward_horizon

        if first_end >= last_end:
            return None

        X_list = []
        y_list = []

        for end_idx in range(first_end, last_end):
            # 窗口 [end_idx - lookback_T, end_idx) 的特征
            seq = np.zeros((lookback_T, SEQUENCE_FEATURE_DIM), dtype=np.float32)
            for t in range(lookback_T):
                idx = end_idx - lookback_T + t
                seq[t] = self._compute_features_at(closes, volumes, idx)

            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            X_list.append(seq)

            # 标签：波动率缩放的前向收益率
            fwd_close = closes[end_idx + forward_horizon]
            cur_close = closes[end_idx]
            if cur_close > _EPS:
                raw_ret = (fwd_close / cur_close) - 1.0
            else:
                raw_ret = 0.0

            # 使用 end_idx 处的 vol_20 做缩放
            log_rets = np.diff(np.log(closes[: end_idx + 1] + _EPS))
            if len(log_rets) >= 20:
                vol = float(log_rets[-20:].std()) * _ANNUALIZE
            else:
                vol = float(log_rets.std()) * _ANNUALIZE if len(log_rets) > 0 else 1.0

            y_val = raw_ret / (vol + _EPS)
            y_list.append(y_val)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float64)
        return X, y

    @staticmethod
    def _compute_features_at(
        closes: np.ndarray,
        volumes: np.ndarray,
        idx: int,
    ) -> np.ndarray:
        """计算单个时步的 14 维特征向量。

        closes/volumes 是完整的价格/成交量数组。
        idx 是当前时步在数组中的绝对索引。
        """
        out = np.zeros(SEQUENCE_FEATURE_DIM, dtype=np.float32)

        # ret_1
        if idx >= 1 and closes[idx - 1] > _EPS:
            out[0] = (closes[idx] / closes[idx - 1]) - 1.0
        # ret_5
        if idx >= 5 and closes[idx - 5] > _EPS:
            out[1] = (closes[idx] / closes[idx - 5]) - 1.0
        # ret_20
        if idx >= 20 and closes[idx - 20] > _EPS:
            out[2] = (closes[idx] / closes[idx - 20]) - 1.0
        # ret_60
        if idx >= 60 and closes[idx - 60] > _EPS:
            out[3] = (closes[idx] / closes[idx - 60]) - 1.0
        # ret_120
        if idx >= 120 and closes[idx - 120] > _EPS:
            out[4] = (closes[idx] / closes[idx - 120]) - 1.0

        # 计算 log returns 截至 idx
        log_rets = np.diff(np.log(closes[: idx + 1] + _EPS))

        # vol_20 = std(ret_1[-20:]) * sqrt(26)
        if len(log_rets) >= 20:
            vol_20 = float(log_rets[-20:].std()) * _ANNUALIZE
        else:
            vol_20 = 0.0
        out[5] = vol_20

        # vol_60 = std(ret_1[-60:]) * sqrt(26)
        if len(log_rets) >= 60:
            vol_60 = float(log_rets[-60:].std()) * _ANNUALIZE
        elif vol_20 > 0:
            vol_60 = vol_20  # fallback
        else:
            vol_60 = 0.0
        out[6] = vol_60

        # vol_ratio = vol_20 / vol_60
        if vol_60 > _EPS:
            out[7] = vol_20 / vol_60
        else:
            out[7] = 1.0

        # volume_ratio = volume[idx] / mean(volume[-20:])
        start_v = max(0, idx - 19)
        vol_slice = volumes[start_v: idx + 1]
        vol_mean = float(vol_slice.mean()) if len(vol_slice) > 0 else 0.0
        if vol_mean > _EPS:
            out[8] = volumes[idx] / vol_mean
        else:
            out[8] = 1.0

        # ret_vol_scaled_20 = ret_20 / (vol_20 + eps)
        out[9] = out[2] / (vol_20 + _EPS)

        # ret_vol_scaled_60 = ret_60 / (vol_60 + eps)
        out[10] = out[3] / (vol_60 + _EPS)

        # trend_consistency_20：过去 20 bars 中 ret_1 > 0 的比例 → [-1, 1]
        if len(log_rets) >= 20:
            recent = log_rets[-20:]
            proportion = float(np.mean(recent > 0))
            out[11] = 2.0 * proportion - 1.0

        # staleness_weight = 1.0（简化）
        out[12] = 1.0

        # market_beta_20 = 0（单资产时填 0）
        out[13] = 0.0

        return out
