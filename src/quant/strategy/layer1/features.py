"""Layer1 特征构建器。

v1 与 v2 两种实现在过渡期内共存：

FeatureBuilder   (v1)
    按目标合约构建跨资产特征矩阵。
    每个合约 10 个特征；维度 = 10 × (N_universe - N_excluded)。
    保持不变，以确保 ``Layer1Config.feature_version == 1`` 时现有代码仍可运行。

FeatureBuilderV2  (v2)
    每个 (资产, 时间) 样本对应固定 37 维特征向量的池化架构。
    架构详见 docs/universal_feature_spec.md。

GlobalState
    每根 K 线更新一次的全局市场状态（PCA 每 pca_update_freq 根 K 线更新一次）。
    传入 FeatureBuilderV2.build() 以避免为每个资产重复计算全局信号。
"""
from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from quant.core.types import Bar, Instrument

# ---------------------------------------------------------------------------
# GlobalState — 全局市场状态（规格 §12.2）
# ---------------------------------------------------------------------------

# 类别型整数编码映射 —— 在模块加载时固定。
# 训练与推理时必须保持稳定。
ASSET_CLASS_MAP: dict[str, int] = {
    "stock": 0, "etf": 1, "futures": 2,
    "crypto": 3, "bond": 4, "fx": 5,
}
EXCHANGE_MAP: dict[str, int] = {
    "CME": 0, "CBOT": 1, "HKEX": 2, "NYSE": 3,
    "NASDAQ": 4, "CRYPTO": 5, "OTHER": 6,
}
CURRENCY_MAP: dict[str, int] = {
    "USD": 0, "HKD": 1, "EUR": 2, "JPY": 3,
    "CNY": 4, "GBP": 5, "OTHER": 6,
}
REGION_MAP: dict[str, int] = {
    "US": 0, "HK": 1, "EU": 2, "JP": 3,
    "CN": 4, "GLOBAL": 5, "OTHER": 6,
}

# 交易所 → 区域推断（当 Instrument.region 不可用时使用）
_EXCHANGE_TO_REGION: dict[str, str] = {
    "CME": "US", "CBOT": "US", "NASDAQ": "US", "NYSE": "US",
    "HKEX": "HK",
}

# InstrumentType → 资产类别推断
_ITYPE_TO_ASSET_CLASS: dict[str, str] = {
    "FUTURE": "futures",
    "EQUITY": "stock",
}


@dataclass
class GlobalState:
    """给定 K 线时刻，所有资产共享的全局市场状态。

    复合特征每根 K 线更新一次；PCA 得分每 ``pca_update_freq`` 根
    K 线更新一次，两次更新之间缓存复用。

    字段说明
    ------
    pca_scores : dict[str, np.ndarray]
        symbol -> 形状 (n_components,) 的数组，当前 PCA 投影。
    pca_momentum_5 : dict[str, np.ndarray]
        symbol -> (n_components,)，PCA 得分的 5 bar 变化量。
    pca_momentum_20 : dict[str, np.ndarray]
        symbol -> (n_components,)，PCA 得分的 20 bar 变化量。
    composite_ew_ret_5 : float
        全品种等权波动率缩放 5 bar 收益率。
    composite_ew_ret_20 : float
        全品种等权波动率缩放 20 bar 收益率。
    composite_iv_ret_5 : float
        逆波动率加权波动率缩放 5 bar 收益率。
    composite_iv_ret_20 : float
        逆波动率加权波动率缩放 20 bar 收益率。
    composite_ew_ret_history : np.ndarray
        形状 (≤20,)：最近若干根 K 线的等权复合 1-bar 收益率
        （1-bar 波动率缩放 EW 均值），最旧值在前。
        用于 FeatureBuilderV2 中的 market_beta_20 计算（规格 §13 Q2）。
    """
    pca_scores: dict[str, np.ndarray] = field(default_factory=dict)
    pca_momentum_5: dict[str, np.ndarray] = field(default_factory=dict)
    pca_momentum_20: dict[str, np.ndarray] = field(default_factory=dict)
    composite_ew_ret_5: float = 0.0
    composite_ew_ret_20: float = 0.0
    composite_iv_ret_5: float = 0.0
    composite_iv_ret_20: float = 0.0
    composite_ew_ret_history: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# GlobalStateBuilder — 每根 K 线计算 GlobalState（规格 §3、§4）
# ---------------------------------------------------------------------------

class GlobalStateBuilder:
    """每根 K 线计算并缓存 GlobalState。

    Parameters
    ----------
    n_components : int
        PCA 主成分数量（规格 §3.3，默认 5）。
    pca_update_freq : int
        每 N 根 K 线重新计算 PCA 以降低计算成本。
    vol_lookback : int
        波动率缩放的回看窗口（规格 §2，默认 60）。
    eps : float
        波动率分母的数值下限。
    """

    def __init__(
        self,
        n_components: int = 5,
        pca_update_freq: int = 20,
        pca_window: int = 500,
        vol_lookback: int = 60,
        eps: float = 1e-6,
        frozen: bool = False,
    ) -> None:
        self._n_components = n_components
        self._pca_update_freq = pca_update_freq
        self._pca_window = pca_window
        self._vol_lookback = vol_lookback
        self._eps = eps
        self._frozen = frozen

        # PCA 状态
        self._pca_components: np.ndarray | None = None  # (n_components, n_symbols)
        self._pca_syms_fitted: list[str] = []
        self._last_pca_bar: int = -1

        # 用于计算动量的得分历史（最旧值在前）
        # symbol -> list[np.ndarray(n_components)]
        self._score_history: dict[str, list[np.ndarray]] = {}

        # 1-bar 等权复合收益率历史（用于 market_beta_20，规格 §13 Q2）
        # 每根 K 线存储各品种波动率缩放 1-bar 收益率的等权均值。
        self._ew_bar_ret_history: list[float] = []
        _HISTORY_MAX = 25  # 多保留一点，略超过 20 以保证安全

    def update(
        self,
        bar_index: int,
        windows: dict[str, list[Bar]],
        pca_universe: Sequence[str] | None = None,
    ) -> GlobalState:
        """计算当前 K 线的 GlobalState。

        Parameters
        ----------
        bar_index : int
            当前 K 线索引（用于控制 PCA 重计算频率）。
        windows : dict[str, list[Bar]]
            symbol -> 最近 K 线列表（已由引擎截取好窗口）。
        pca_universe : sequence of str, optional
            用于 PCA 的品种列表。None = 使用 windows 中的所有品种。
        """
        symbols = list(windows.keys())
        pca_syms = list(pca_universe) if pca_universe else symbols

        # -- 计算各品种波动率缩放收益率 ---------------------------
        vs_ret_1: dict[str, float] = {}   # 1-bar 波动率缩放收益率，用于 EW 历史
        vs_ret_5: dict[str, float] = {}
        vs_ret_20: dict[str, float] = {}
        vols: dict[str, float] = {}

        for sym, bars in windows.items():
            closes = np.array([b.close for b in bars], dtype=np.float64)
            sigma = self._realized_vol(closes, self._vol_lookback)
            vols[sym] = sigma
            vs_ret_1[sym] = self._vol_scaled_return(closes, 1, sigma, self._eps)
            vs_ret_5[sym] = self._vol_scaled_return(closes, 5, sigma, self._eps)
            vs_ret_20[sym] = self._vol_scaled_return(closes, 20, sigma, self._eps)

        # 累积 1-bar 等权复合收益率历史（规格 §13 Q2）
        _HISTORY_MAX = 25
        if vs_ret_1:
            ew_bar = float(np.mean(list(vs_ret_1.values())))
        else:
            ew_bar = 0.0
        self._ew_bar_ret_history.append(ew_bar)
        if len(self._ew_bar_ret_history) > _HISTORY_MAX:
            self._ew_bar_ret_history = self._ew_bar_ret_history[-_HISTORY_MAX:]

        # -- 滚动 PCA（规格 §3.3）-----------------------------------------
        if not self._frozen:
            needs_pca_update = (
                self._pca_components is None
                or (bar_index - self._last_pca_bar) >= self._pca_update_freq
            )
            if needs_pca_update:
                self._recompute_pca(windows, pca_syms)
                self._last_pca_bar = bar_index
        # else: 冻结模式下使用已加载的 _pca_components

        # -- 将各品种投影到 PCA 空间 ------------------------------
        pca_scores: dict[str, np.ndarray] = {}
        for sym in symbols:
            pca_scores[sym] = self._project(sym, windows, pca_syms)

        # -- 更新得分历史并计算 PC 动量 --------------------------
        for sym, scores in pca_scores.items():
            hist = self._score_history.setdefault(sym, [])
            hist.append(scores.copy())
            # 只保留所需的最大回看量（mom20 需要 21 个）
            if len(hist) > 25:
                self._score_history[sym] = hist[-25:]

        pca_mom_5: dict[str, np.ndarray] = {}
        pca_mom_20: dict[str, np.ndarray] = {}
        zero_pc = np.zeros(self._n_components, dtype=np.float32)
        for sym in symbols:
            hist = self._score_history.get(sym, [])
            cur = pca_scores.get(sym, zero_pc)
            pca_mom_5[sym] = (cur - hist[-6]).astype(np.float32) if len(hist) >= 6 else zero_pc.copy()
            pca_mom_20[sym] = (cur - hist[-21]).astype(np.float32) if len(hist) >= 21 else zero_pc.copy()

        # -- 复合收益率（规格 §4）-------------------------------------
        active_vs5 = [vs_ret_5[s] for s in symbols if s in vs_ret_5]
        active_vs20 = [vs_ret_20[s] for s in symbols if s in vs_ret_20]
        _active_vols = [vols[s] for s in symbols if vols.get(s, 0) > self._eps]  # reserved

        composite_ew_ret_5 = float(np.mean(active_vs5)) if active_vs5 else 0.0
        composite_ew_ret_20 = float(np.mean(active_vs20)) if active_vs20 else 0.0

        # 逆波动率加权复合收益率
        composite_iv_ret_5 = self._iv_composite(symbols, vs_ret_5, vols, self._eps)
        composite_iv_ret_20 = self._iv_composite(symbols, vs_ret_20, vols, self._eps)

        ew_hist = np.array(self._ew_bar_ret_history[-20:], dtype=np.float32)

        return GlobalState(
            pca_scores=pca_scores,
            pca_momentum_5=pca_mom_5,
            pca_momentum_20=pca_mom_20,
            composite_ew_ret_5=composite_ew_ret_5,
            composite_ew_ret_20=composite_ew_ret_20,
            composite_iv_ret_5=composite_iv_ret_5,
            composite_iv_ret_20=composite_iv_ret_20,
            composite_ew_ret_history=ew_hist,
        )

    # -- PCA 辅助方法 ----------------------------------------------------------

    def _recompute_pca(
        self,
        windows: dict[str, list[Bar]],
        pca_syms: list[str],
    ) -> None:
        """在 pca_syms 的波动率缩放收益率上拟合滚动 PCA。

        使用最近 ``pca_window`` 根 K 线。
        符号修正：ES PC1 载荷固定为正（规格 §3.4）。
        """
        # 构建矩阵：行=K 线，列=品种（取交集）
        available = [s for s in pca_syms if s in windows and len(windows[s]) >= self._vol_lookback + 2]
        if len(available) < self._n_components:
            self._pca_components = None
            return

        # 对齐最后 pca_window 根 K 线构建收益率矩阵
        cols: list[np.ndarray] = []
        for sym in available:
            bars = windows[sym]
            closes = np.array([b.close for b in bars], dtype=np.float64)
            sigma = self._realized_vol(closes, self._vol_lookback)
            if sigma < self._eps:
                continue
            # 波动率缩放 1-bar 对数收益率
            log_rets = np.diff(np.log(closes + 1e-9))
            vs = log_rets / sigma
            cols.append(vs[-self._pca_window:] if len(vs) >= self._pca_window else vs)

        if len(cols) < self._n_components:
            self._pca_components = None
            return

        # 对齐长度
        min_len = min(len(c) for c in cols)
        mat = np.column_stack([c[-min_len:] for c in cols])  # (T, N)

        # 标准化列
        col_std = mat.std(axis=0) + self._eps
        mat_std = mat / col_std

        # 通过 SVD 实现 PCA（无需 sklearn 依赖）
        try:
            _, _, Vt = np.linalg.svd(mat_std, full_matrices=False)
        except np.linalg.LinAlgError:
            self._pca_components = None
            return

        components = Vt[: self._n_components]  # (n_components, N)

        # 符号修正：固定 ES（或第一个可用的美股期货）PC1 > 0
        anchor_sym = "ES" if "ES" in available else available[0]
        anchor_idx = available.index(anchor_sym)
        if components[0, anchor_idx] < 0:
            components[0] *= -1

        self._pca_components = components  # (n_components, N)
        self._pca_syms_fitted = available

    def save_pca_state(self, path: str | Path) -> None:
        """序列化 PCA 冻结状态。"""
        payload = {
            "pca_components": self._pca_components,
            "pca_syms_fitted": getattr(self, "_pca_syms_fitted", []),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_pca_state(self, path: str | Path) -> None:
        """加载 PCA 冻结状态，并启用冻结模式。"""
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        self._pca_components = payload["pca_components"]
        self._pca_syms_fitted = payload["pca_syms_fitted"]
        self._frozen = True

    def _project(
        self,
        sym: str,
        windows: dict[str, list[Bar]],
        pca_syms: list[str],
    ) -> np.ndarray:
        """将目标品种的波动率缩放收益率投影到 PCA 载荷上。"""
        if self._pca_components is None or not hasattr(self, "_pca_syms_fitted"):
            return np.zeros(self._n_components, dtype=np.float32)

        available = self._pca_syms_fitted
        if sym not in available:
            # 品种不在 PCA 域内 —— 返回全零
            return np.zeros(self._n_components, dtype=np.float32)

        sym_idx = available.index(sym)
        # 该品种在所有 PC 上的载荷向量：形状 (n_components,)
        loadings = self._pca_components[:, sym_idx]  # (n_components,)

        bars = windows.get(sym, [])
        if not bars:
            return np.zeros(self._n_components, dtype=np.float32)
        closes = np.array([b.close for b in bars], dtype=np.float64)
        sigma = self._realized_vol(closes, self._vol_lookback)
        if sigma < self._eps or len(closes) < 2:
            return np.zeros(self._n_components, dtype=np.float32)

        # 当前波动率缩放 1-bar 收益率作为"坐标"
        cur_vs_ret = (closes[-1] - closes[-2]) / (closes[-2] + self._eps) / sigma
        scores = loadings * cur_vs_ret  # 逐元素：score_k = loading_k * vs_ret
        return scores.astype(np.float32)

    # -- 静态辅助方法 -------------------------------------------------------

    @staticmethod
    def _realized_vol(closes: np.ndarray, lookback: int, eps: float = 1e-9) -> float:
        if len(closes) < lookback + 1:
            log_rets = np.diff(np.log(closes + eps))
            return float(log_rets.std()) if len(log_rets) > 0 else 0.0
        log_rets = np.diff(np.log(closes[-lookback - 1:] + eps))
        return float(log_rets.std())

    @staticmethod
    def _vol_scaled_return(
        closes: np.ndarray, n: int, sigma: float, eps: float
    ) -> float:
        if len(closes) <= n or closes[-n - 1] == 0:
            return 0.0
        raw_ret = (closes[-1] / closes[-n - 1]) - 1.0
        return float(raw_ret / (sigma + eps))

    @staticmethod
    def _iv_composite(
        symbols: list[str],
        vs_rets: dict[str, float],
        vols: dict[str, float],
        eps: float,
    ) -> float:
        inv_vols = np.array([1.0 / (vols.get(s, eps) + eps) for s in symbols], dtype=np.float64)
        total = inv_vols.sum()
        if total < eps:
            return 0.0
        weights = inv_vols / total
        rets = np.array([vs_rets.get(s, 0.0) for s in symbols], dtype=np.float64)
        return float(np.dot(weights, rets))


# ---------------------------------------------------------------------------
# FeatureBuilderV2 — 37 维池化特征向量（规格 §8）
# ---------------------------------------------------------------------------

class FeatureBuilderV2:
    """为单个 (资产, 时间) 样本构建固定 37 维特征向量。

    特征布局（规格 §8.4）：
        [0:14]   14 个逐资产 OHLCV 特征
        [14:29]  15 个全局 PCA 特征（5 PC × 3：得分、mom5、mom20）
        [29:33]  4 个复合收益率特征
        [33:37]  4 个类别型整数特征（asset_class、exchange、currency、region）

    类别型列位于固定索引 [33, 34, 35, 36]，
    创建 lgb.Dataset 时必须声明为 ``categorical_feature``。

    Parameters
    ----------
    vol_lookback : int
        用于特征 #10、#11 波动率缩放的回看窗口（默认 60）。
    staleness_halflife_hours : float
        陈旧权重特征的半衰期（默认 8.0）。
    eps : float
        除法的数值下限（默认 1e-9）。
    """

    FEATURE_DIM: int = 37
    # 37 维向量中 LightGBM 类别型声明的索引
    CATEGORICAL_FEATURE_INDICES: tuple[int, ...] = (33, 34, 35, 36)

    def __init__(
        self,
        vol_lookback: int = 60,
        staleness_halflife_hours: float = 8.0,
        eps: float = 1e-9,
    ) -> None:
        self._vol_lookback = vol_lookback
        self._halflife = staleness_halflife_hours
        self._eps = eps

    def build(
        self,
        instrument: Instrument,
        asset_bars: list[Bar],
        global_state: GlobalState,
        as_of: datetime,
    ) -> np.ndarray:
        """返回 37 维 float32 特征向量。

        Parameters
        ----------
        instrument : Instrument
            目标合约规格（提供交易所、货币、类型）。
        asset_bars : list[Bar]
            该资产的最近 K 线（必须按时间排序，无前视）。
        global_state : GlobalState
            本 K 线预先计算的全局状态（PCA、复合特征）。
        as_of : datetime
            当前 K 线时间戳（用于陈旧度计算）。
        """
        sym = instrument.symbol
        vec = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # -- 第 1 块：逐资产特征 [0:14] -------------------------------
        if asset_bars:
            closes = np.array([b.close for b in asset_bars], dtype=np.float64)
            volumes = np.array([b.volume for b in asset_bars], dtype=np.float64)
            vec[0:14] = self._per_asset_features(
                closes, volumes, as_of, asset_bars[-1].timestamp,
                global_state.composite_ew_ret_history,
            )
        # 否则保持全零（规格 §1.4 安全填充）

        # -- 第 2 块：PCA 特征 [14:29]（5 PC × 3）-----------------------
        pc_scores = global_state.pca_scores.get(sym)
        pc_mom5 = global_state.pca_momentum_5.get(sym)
        pc_mom20 = global_state.pca_momentum_20.get(sym)
        n_pc = 5
        zero_pc = np.zeros(n_pc, dtype=np.float32)
        scores = pc_scores if pc_scores is not None else zero_pc
        mom5 = pc_mom5 if pc_mom5 is not None else zero_pc
        mom20 = pc_mom20 if pc_mom20 is not None else zero_pc
        # 交错排列：pc1_score, pc1_mom5, pc1_mom20, pc2_score, ...
        for k in range(n_pc):
            base = 14 + k * 3
            vec[base] = float(scores[k]) if k < len(scores) else 0.0
            vec[base + 1] = float(mom5[k]) if k < len(mom5) else 0.0
            vec[base + 2] = float(mom20[k]) if k < len(mom20) else 0.0

        # -- 第 3 块：复合特征 [29:33] ------------------------------
        vec[29] = np.float32(global_state.composite_ew_ret_5)
        vec[30] = np.float32(global_state.composite_ew_ret_20)
        vec[31] = np.float32(global_state.composite_iv_ret_5)
        vec[32] = np.float32(global_state.composite_iv_ret_20)

        # -- 第 4 块：类别型特征 [33:37] ----------------------------
        vec[33] = np.float32(self._asset_class_id(instrument))
        vec[34] = np.float32(EXCHANGE_MAP.get(instrument.exchange, EXCHANGE_MAP["OTHER"]))
        vec[35] = np.float32(CURRENCY_MAP.get(str(instrument.currency), CURRENCY_MAP["OTHER"]))
        vec[36] = np.float32(REGION_MAP.get(self._infer_region(instrument), REGION_MAP["OTHER"]))

        return vec

    # -- 逐资产 14 个特征（规格 §1.1）------------------------------------

    def _per_asset_features(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        as_of: datetime,
        last_bar_time: datetime,
        composite_ew_ret_history: np.ndarray,
    ) -> np.ndarray:
        """计算 14 个逐资产特征（规格 §1.1、§13 Q2）。

        特征 #14 为 ``market_beta_20``：该资产 1-bar 对数收益率
        对等权复合 1-bar 收益率的滚动 OLS Beta。
        替代在池化模式下无意义的 ``cross_corr_20``。
        """
        eps = self._eps
        out = np.zeros(14, dtype=np.float64)

        def safe_ret(n: int) -> float:
            if len(closes) <= n or closes[-n - 1] == 0:
                return 0.0
            return float((closes[-1] / closes[-n - 1]) - 1.0)

        # #1-5：动量收益率
        out[0] = safe_ret(1)    # ret_1
        out[1] = safe_ret(5)    # ret_5
        out[2] = safe_ret(20)   # ret_20
        out[3] = safe_ret(60)   # ret_60
        out[4] = safe_ret(120)  # ret_120

        # #6：vol_20
        log_rets = np.diff(np.log(closes + eps))
        vol_20 = float(log_rets[-20:].std()) if len(log_rets) >= 20 else 0.0
        out[5] = vol_20

        # #7：vol_60
        vol_60 = float(log_rets[-60:].std()) if len(log_rets) >= 60 else 0.0
        out[6] = vol_60

        # #8：vol_ratio = vol_20 / vol_60
        out[7] = float(vol_20 / (vol_60 + eps)) if vol_60 > eps else 0.0

        # #9：volume_ratio
        vol_ma = float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean() + eps)
        out[8] = float(volumes[-1] / (vol_ma + eps)) if vol_ma > 0 else 1.0

        # #10：ret_vol_scaled_20
        out[9] = float(out[2] / (vol_20 + eps)) if vol_20 > eps else 0.0

        # #11：ret_vol_scaled_60
        out[10] = float(out[3] / (vol_60 + eps)) if vol_60 > eps else 0.0

        # #12：trend_consistency_20
        out[11] = _trend_consistency(closes, 20)

        # #13：staleness_weight（陈旧度权重）
        hours_diff = (as_of - last_bar_time).total_seconds() / 3600.0
        out[12] = float(np.exp(-hours_diff / self._halflife))

        # #14：market_beta_20（规格 §13 Q2）
        # cov(asset_1bar_ret, composite_1bar_ret, 20) / var(composite_1bar_ret, 20)
        # 使用 GlobalState.composite_ew_ret_history（1-bar 等权复合收益率）。
        out[13] = _market_beta(log_rets, composite_ew_ret_history, window=20, eps=eps)

        return out.astype(np.float32)

    # -- 静态辅助方法 -------------------------------------------------------

    @staticmethod
    def _asset_class_id(instrument: Instrument) -> int:
        cls = _ITYPE_TO_ASSET_CLASS.get(str(instrument.instrument_type), "stock")
        return ASSET_CLASS_MAP.get(cls, ASSET_CLASS_MAP["stock"])

    @staticmethod
    def _infer_region(instrument: Instrument) -> str:
        return _EXCHANGE_TO_REGION.get(instrument.exchange, "OTHER")


# ---------------------------------------------------------------------------
# FeatureBuilder (v1) — 保留供 feature_version == 1 使用
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """为 Layer1 模型构建跨资产特征矩阵（LOO + staleness）。

    每个品种贡献 10 个特征：
        ret1, ret5, ret20, ret60, ret120,
        vol_20, volume_ratio, staleness_weight,
        trend_consistency_20, cross_corr_20
    """

    # 每个品种的特征数量
    FEATURES_PER_SYMBOL: int = 10

    def __init__(
        self,
        loo_groups: dict[str, list[str]],
        staleness_halflife_hours: float = 8.0,
    ) -> None:
        self._loo = loo_groups
        self._halflife = staleness_halflife_hours

    def build(
        self,
        target_symbol: str,
        windows: dict[str, list[Bar]],
        as_of: datetime,
    ) -> np.ndarray:
        """构建单行特征向量（1D array），用于 target_symbol 的 Layer1 预测。"""
        excluded = set(self._loo.get(target_symbol, []))
        excluded.add(target_symbol)

        target_bars = windows.get(target_symbol, [])
        target_closes = np.array([b.close for b in target_bars], dtype=np.float64) if target_bars else np.array([], dtype=np.float64)

        feature_parts: list[np.ndarray] = []
        for sym, bars in sorted(windows.items()):
            if sym in excluded:
                continue
            feature_parts.append(self._symbol_features(bars, as_of, target_closes))

        if not feature_parts:
            return np.zeros(self.FEATURES_PER_SYMBOL, dtype=np.float32)
        return np.concatenate(feature_parts).astype(np.float32)

    def _symbol_features(
        self,
        bars: list[Bar],
        as_of: datetime,
        target_closes: np.ndarray,
    ) -> np.ndarray:
        closes = np.array([b.close for b in bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars], dtype=np.float64)

        def safe_return(n: int) -> float:
            if len(closes) <= n or closes[-n - 1] == 0:
                return 0.0
            return float((closes[-1] / closes[-n - 1]) - 1)

        ret1 = safe_return(1)
        ret5 = safe_return(5)
        ret20 = safe_return(20)
        ret60 = safe_return(60)
        ret120 = safe_return(120)

        vol20 = float(np.std(np.diff(np.log(closes[-21:] + 1e-9)))) if len(closes) >= 21 else 0.0

        vol_ma = float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean() + 1e-9)
        volume_ratio = float(volumes[-1] / (vol_ma + 1e-9)) if vol_ma > 0 else 1.0

        last_bar_time = bars[-1].timestamp if bars else as_of
        hours_diff = (as_of - last_bar_time).total_seconds() / 3600.0
        staleness_weight = float(np.exp(-hours_diff / self._halflife))

        trend_consistency_20 = _trend_consistency(closes, 20)
        cross_corr_20 = _cross_correlation(closes, target_closes, 20)

        return np.array(
            [ret1, ret5, ret20, ret60, ret120, vol20, volume_ratio,
             staleness_weight, trend_consistency_20, cross_corr_20],
            dtype=np.float32,
        )

    def feature_dim(self, n_symbols_universe: int) -> int:
        """预测特征维度（用于模型初始化检查）。"""
        return self.FEATURES_PER_SYMBOL * n_symbols_universe


# ---------------------------------------------------------------------------
# 共享纯 NumPy 辅助函数
# ---------------------------------------------------------------------------

def _market_beta(
    asset_log_rets: np.ndarray,
    composite_bar_rets: np.ndarray,
    window: int = 20,
    eps: float = 1e-9,
) -> float:
    """资产对等权复合指数的滚动 OLS Beta（规格 §13 Q2）。

    beta = cov(asset_ret[-window:], composite_ret[-window:]) / var(composite_ret[-window:])

    数据不足或方差近似为零时返回 0.0。
    """
    if len(asset_log_rets) < window or len(composite_bar_rets) < window:
        return 0.0
    a_ret = asset_log_rets[-window:]
    m_ret = composite_bar_rets[-window:].astype(np.float64)
    var_m = float(np.var(m_ret))
    if var_m < eps:
        return 0.0
    cov_am = float(np.cov(a_ret, m_ret)[0, 1])
    return float(cov_am / (var_m + eps))


def _trend_consistency(closes: np.ndarray, window: int) -> float:
    """与整体趋势方向一致的 K 线收益率占比。"""
    if len(closes) < window + 1:
        return 0.0
    segment = closes[-(window + 1):]
    returns = np.diff(segment)
    overall_direction = np.sign(segment[-1] - segment[0])
    if overall_direction == 0:
        return 0.0
    bar_directions = np.sign(returns)
    return float(np.mean(bar_directions == overall_direction))


def _cross_correlation(
    closes_a: np.ndarray,
    closes_b: np.ndarray,
    window: int,
) -> float:
    """最近窗口收益率的皮尔逊相关系数（纯 NumPy 实现）。"""
    min_len = window + 1
    if len(closes_a) < min_len or len(closes_b) < min_len:
        return 0.0
    ret_a = np.diff(closes_a[-min_len:])
    ret_b = np.diff(closes_b[-min_len:])
    std_a = ret_a.std()
    std_b = ret_b.std()
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    corr = np.corrcoef(ret_a, ret_b)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)
