"""Layer1 策略 —— 宏观/状态层。

v2（feature_version == 2）：池化单模型架构
    - 所有合约共享一个全局 LightGBM 模型。
    - 使用 FeatureBuilderV2 生成固定 37 维特征向量。
    - 波动率缩放 Huber 回归标签（规格 §6）。
    - 逆频率样本权重（规格 §13 Q5）。
    - GlobalState 由 GlobalStateBuilder 每根 K 线计算一次。

v1（feature_version == 1）：逐品种跨资产模型（已保留）
    - N 个独立 LightGBM 模型，每个 10*M 维特征。
    - Phase 2 的原始架构。
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from quant.config.schema import Layer1Config
from quant.core.strategy import Layer1Result
from quant.core.types import Bar, Instrument
from quant.strategy.layer1.backends import create_backend
from quant.strategy.layer1.features import (
    FeatureBuilder,
    FeatureBuilderV2,
    GlobalState,
    GlobalStateBuilder,
)
from quant.strategy.layer1.mii import MIICalculator
from quant.strategy.layer1.model import Layer1Model

logger = logging.getLogger(__name__)

_EPS = 1e-9
# 在 self._models 中用于单一池化模型的键
_POOLED_KEY = "_pooled_v2"


# ---------------------------------------------------------------------------
# 状态分类辅助函数
# ---------------------------------------------------------------------------

def _classify_regime(beta_score: float, mii: float) -> str:
    """将 beta_score + mii 映射为状态标签字符串。"""
    if mii < 0.3:
        return "EXHAUSTED"
    if abs(beta_score) > 0.5:
        return "TRENDING_STRONG" if mii > 0.6 else "TRENDING_WEAK"
    return "RANGING"


# ---------------------------------------------------------------------------
# Layer1 — 实现 Layer1Strategy 协议
# ---------------------------------------------------------------------------

class Layer1:
    """宏观动量层。

    实现 ``Layer1Strategy`` 协议。通过 ``cfg.feature_version`` 控制，
    支持 v1（逐品种跨资产）和 v2（池化 37 维）两种架构。

    滚动训练接口
    ----------------------
    ``fit_window(windows, bar_index, start_idx, end_idx)``
        由 BacktestEngine 在每次重训练触发时调用。
        派发至 ``_fit_pooled()``（v2）或逐品种循环（v1）。

    ``fit(windows)``
        初始/离线训练的便捷封装。
    """

    def __init__(self, cfg: Layer1Config) -> None:
        self._cfg = cfg
        self._use_v2 = (cfg.feature_version >= 2)

        # 模型字典：v2 -> {"_pooled_v2": backend}
        #           v1 -> {symbol: Layer1Model}
        self._models: dict[str, Any] = {}

        # 每个品种的 MII 计算器（v1 和 v2 均使用）
        self._miis: dict[str, MIICalculator] = {}

        # v2：共享 GlobalStateBuilder + 缓存当前 GlobalState
        if self._use_v2:
            self._global_builder = GlobalStateBuilder(
                n_components=cfg.pca_n_components,
                pca_update_freq=cfg.pca_update_freq,
                pca_window=cfg.pca_window,
                vol_lookback=cfg.vol_lookback,
            )
            self._feature_builder_v2 = FeatureBuilderV2(
                vol_lookback=cfg.vol_lookback,
                staleness_halflife_hours=cfg.staleness_halflife_hours,
            )
            self._last_global_state: GlobalState | None = None
        else:
            self._feature_builder = FeatureBuilder(
                loo_groups=cfg.loo_correlation_groups,
                staleness_halflife_hours=cfg.staleness_halflife_hours,
            )

    # ------------------------------------------------------------------
    # load_model() — 加载预训练模型
    # ------------------------------------------------------------------

    def load_model(self, model_dir: str | Path) -> None:
        """加载预训练的 Layer1 模型目录。

        Parameters
        ----------
        model_dir : str | Path
            包含 meta.json 和模型文件的目录路径。
        """
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        model_type = meta.get("model_type", "lgbm")

        backend = create_backend(model_type, self._cfg)
        backend.load(model_dir)
        self._models[_POOLED_KEY] = backend

        if self._use_v2 and (model_dir / "global_state.pkl").exists():
            self._global_builder.load_pca_state(model_dir / "global_state.pkl")

        if meta.get("feature_version") and meta["feature_version"] != self._cfg.feature_version:
            raise ValueError(
                f"Model feature_version={meta['feature_version']} "
                f"!= config={self._cfg.feature_version}"
            )

        logger.info(
            "Loaded pretrained model from %s (type=%s)", model_dir, model_type
        )

    # ------------------------------------------------------------------
    # compute() — Layer1Strategy 协议
    # ------------------------------------------------------------------

    def compute(
        self,
        windows: dict[Instrument, list[Bar]],
        external_data: dict[str, Any] | None = None,
    ) -> dict[Instrument, Layer1Result]:
        """计算所有合约的 Layer1Result。"""
        if self._use_v2:
            backend = self._models.get(_POOLED_KEY)
            if backend is not None and hasattr(backend, "input_type") and backend.input_type == "sequential":
                return self._compute_v2_sequential(windows)
            return self._compute_v2(windows)
        return self._compute_v1(windows)

    # ------------------------------------------------------------------
    # fit() — 便捷封装（初始/离线训练）
    # ------------------------------------------------------------------

    def fit(self, windows: dict[Instrument, list[Bar]]) -> None:
        """在完整提供的窗口上训练。

        使用 ``cfg.forward_horizon_bars`` 构造标签。
        滚动增量重训练建议使用 ``fit_window()``。
        """
        forward = self._cfg.forward_horizon_bars
        sym_windows = {inst.symbol: bars for inst, bars in windows.items()}

        if self._use_v2:
            temp_builder = GlobalStateBuilder(
                n_components=self._cfg.pca_n_components,
                pca_update_freq=self._cfg.pca_update_freq,
                pca_window=self._cfg.pca_window,
                vol_lookback=self._cfg.vol_lookback,
            )
            self._fit_pooled(windows, sym_windows, forward, temp_builder)
        else:
            min_bars = forward + 50
            for inst, bars in windows.items():
                sym = inst.symbol
                if len(bars) < min_bars:
                    logger.warning(
                        "Not enough bars to train %s: %d (need %d)",
                        sym, len(bars), min_bars,
                    )
                    continue
                self._fit_one(sym, bars, sym_windows, forward)

    # ------------------------------------------------------------------
    # fit_window() — 滚动训练接口（由 BacktestEngine 调用）
    # ------------------------------------------------------------------

    def fit_window(
        self,
        windows: dict[Instrument, list[Bar]],
        bar_index: int,
        start_idx: int,
        end_idx: int,
    ) -> None:
        """由 BacktestEngine 触发的滚动增量训练。

        Parameters
        ----------
        windows : dict[Instrument, list[Bar]]
            所有合约的完整 K 线列表（调用方不预先切片）。
        bar_index : int
            当前 K 线索引（仅用于日志记录）。
        start_idx : int
            训练窗口起始索引（含）。
        end_idx : int
            训练窗口结束索引（不含）。调用方保证
            ``end_idx = bar_index - forward_horizon_bars + 1``。
        """
        forward = self._cfg.forward_horizon_bars
        slice_len = end_idx - start_idx

        logger.info(
            "WalkForward retrain at bar %d: slice [%d, %d) = %d bars",
            bar_index, start_idx, end_idx, slice_len,
        )

        # 将所有窗口切片至训练范围
        sliced: dict[Instrument, list[Bar]] = {
            inst: bars[start_idx:end_idx]
            for inst, bars in windows.items()
        }
        sliced_sym = {inst.symbol: bars for inst, bars in sliced.items()}

        if self._use_v2:
            temp_builder = GlobalStateBuilder(
                n_components=self._cfg.pca_n_components,
                pca_update_freq=self._cfg.pca_update_freq,
                pca_window=self._cfg.pca_window,
                vol_lookback=self._cfg.vol_lookback,
            )
            self._fit_pooled(sliced, sliced_sym, forward, temp_builder)
        else:
            min_bars = forward + 50
            for inst, bars in sliced.items():
                sym = inst.symbol
                if len(bars) < min_bars:
                    logger.warning(
                        "Skipping WalkForward retrain for %s: %d bars (need %d)",
                        sym, len(bars), min_bars,
                    )
                    continue
                self._fit_one(sym, bars, sliced_sym, forward)

    # ------------------------------------------------------------------
    # v2 池化训练
    # ------------------------------------------------------------------

    def _fit_pooled(
        self,
        windows: dict[Instrument, list[Bar]],
        sym_windows: dict[str, list[Bar]],
        forward: int,
        global_builder: GlobalStateBuilder,
    ) -> None:
        """构建池化 (资产, 时间) 训练集并拟合一个全局模型。

        步骤（规格 §9.3）：
        1. 对于 [50, min_len - forward) 中的每个时间步 i：
           对于每个资产：
             a. 构建截至 K 线 i 的子窗口（无前视）
             b. 在 K 线 i 处更新 GlobalState
             c. 构建 37 维特征向量
             d. 计算波动率缩放前向收益率标签（已 Winsorize）
        2. 应用逆频率样本权重（规格 §13 Q5）
        3. 用 Huber 目标函数拟合 Layer1Model
        """
        forward_h = forward
        vol_lb = self._cfg.vol_lookback
        winsorize_sigma = self._cfg.label_winsorize_sigma
        min_bars = forward_h + 50

        insts = list(windows.keys())
        if not insts:
            logger.warning("_fit_pooled: no instruments, skipping")
            return

        valid_insts = [
            inst for inst in insts
            if len(windows[inst]) >= min_bars
        ]
        if not valid_insts:
            logger.warning("_fit_pooled: no instruments with enough bars, skipping")
            return

        # 遍历所有有效合约共同的时间步
        max_end = min(len(windows[inst]) for inst in valid_insts)
        start_i = max(50, vol_lb + 2)  # 需要足够的历史数据用于波动率特征

        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        sym_labels: list[str] = []  # 用于计算逆频率权重

        pca_universe = list(self._cfg.pca_universe_symbols) or None

        for i in range(start_i, max_end - forward_h):
            # 截至 K 线 i 的子窗口，用于 GlobalState（严格无前视）
            sub_sym: dict[str, list[Bar]] = {
                sym: bars[:i] for sym, bars in sym_windows.items()
            }

            try:
                gs: GlobalState = global_builder.update(
                    bar_index=i,
                    windows=sub_sym,
                    pca_universe=pca_universe,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("GlobalStateBuilder error at i=%d: %s", i, exc)
                continue

            for inst in valid_insts:
                sym = inst.symbol
                bars = windows[inst]
                if i >= len(bars) - forward_h:
                    continue

                asset_bars_i = bars[:i]
                if len(asset_bars_i) < max(vol_lb + 1, 21):
                    continue

                # 37 维特征向量
                feat = self._feature_builder_v2.build(
                    instrument=inst,
                    asset_bars=asset_bars_i,
                    global_state=gs,
                    as_of=bars[i].timestamp,
                )
                if feat is None or len(feat) != FeatureBuilderV2.FEATURE_DIM:
                    continue

                # 波动率缩放前向收益率标签（规格 §6.2）
                c_slice = bars[max(0, i - vol_lb - 1): i + 1]
                closes = np.array([b.close for b in c_slice], dtype=np.float64)
                sigma_i = _realized_vol(closes, vol_lb)
                if sigma_i < _EPS:
                    continue

                fwd_ret = (bars[i + forward_h].close - bars[i].close) / (
                    bars[i].close + _EPS
                )
                y_label = float(np.clip(fwd_ret / sigma_i,
                                        -winsorize_sigma, winsorize_sigma))

                X_rows.append(feat)
                y_rows.append(y_label)
                sym_labels.append(sym)

        if len(X_rows) < 50:
            logger.warning(
                "_fit_pooled: only %d valid samples, skipping (need ≥50)",
                len(X_rows),
            )
            return

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.float64)
        sample_weights = _inverse_frequency_weights(sym_labels)

        model = Layer1Model(self._cfg)
        model.fit(X, y, sample_weights=sample_weights)
        self._models[_POOLED_KEY] = model
        logger.info(
            "_fit_pooled: trained on %d samples, %d instruments, forward=%d",
            len(X_rows), len(valid_insts), forward_h,
        )

    # ------------------------------------------------------------------
    # v1 逐品种训练（保持不变）
    # ------------------------------------------------------------------

    def _fit_one(
        self,
        sym: str,
        bars: list[Bar],
        sym_windows: dict[str, list[Bar]],
        forward: int,
    ) -> None:
        """训练单个合约的 Layer1Model（v1 路径）。"""
        X_list, y_list = [], []

        for i in range(50, len(bars) - forward):
            sub_windows = {s: bs[:i] for s, bs in sym_windows.items()}
            X_row = self._feature_builder.build(sym, sub_windows, bars[i].timestamp)
            if len(X_row) == 0:
                continue
            fwd_return = (bars[i + forward].close - bars[i].close) / (
                bars[i].close + _EPS
            )
            X_list.append(X_row)
            y_list.append(fwd_return)

        if len(X_list) < 50:
            logger.warning(
                "Not enough valid samples to train %s: %d (need 50)",
                sym, len(X_list),
            )
            return

        model = Layer1Model(self._cfg)
        model.fit(np.array(X_list), np.array(y_list))
        self._models[sym] = model
        logger.info(
            "Trained Layer1Model for %s on %d samples (forward=%d)",
            sym, len(X_list), forward,
        )

    # ------------------------------------------------------------------
    # v2 推理路径
    # ------------------------------------------------------------------

    def _compute_v2(
        self,
        windows: dict[Instrument, list[Bar]],
    ) -> dict[Instrument, Layer1Result]:
        """使用单一池化模型 + GlobalState 进行推理。"""
        as_of = datetime.now(timezone.utc)
        sym_windows = {inst.symbol: bars for inst, bars in windows.items()}

        pca_universe = list(self._cfg.pca_universe_symbols) or None
        try:
            gs = self._global_builder.update(
                bar_index=0,
                windows=sym_windows,
                pca_universe=pca_universe,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GlobalStateBuilder error in compute: %s", exc)
            gs = GlobalState()
        self._last_global_state = gs

        pooled_model = self._models.get(_POOLED_KEY)

        results: dict[Instrument, Layer1Result] = {}
        for inst, bars in windows.items():
            sym = inst.symbol

            if len(bars) < 20:
                results[inst] = Layer1Result(
                    regime="RANGING", beta_score=0.0, mii=1.0, confidence=0.0
                )
                continue

            feat = self._feature_builder_v2.build(
                instrument=inst,
                asset_bars=bars,
                global_state=gs,
                as_of=as_of,
            )

            if pooled_model is None:
                beta_score, confidence = 0.0, 0.0
            else:
                beta_score, confidence = pooled_model.predict(feat)

            mii = self._compute_mii(sym, bars)
            regime = _classify_regime(beta_score, mii)

            results[inst] = Layer1Result(
                regime=regime,
                beta_score=beta_score,
                mii=mii,
                confidence=confidence,
                metadata={
                    "feature_version": 2,
                    "feature_dim": len(feat),
                },
            )

        return results

    # ------------------------------------------------------------------
    # v2 sequential 推理路径
    # ------------------------------------------------------------------

    def _compute_v2_sequential(
        self,
        windows: dict[Instrument, list[Bar]],
    ) -> dict[Instrument, Layer1Result]:
        """使用 sequential 后端进行推理。"""
        from quant.strategy.layer1.sequence_features import SequenceFeatureBuilder

        seq_builder = getattr(self, "_seq_builder", None)
        if seq_builder is None:
            self._seq_builder = SequenceFeatureBuilder()
            seq_builder = self._seq_builder

        backend = self._models.get(_POOLED_KEY)

        results: dict[Instrument, Layer1Result] = {}
        for inst, bars in windows.items():
            sym = inst.symbol

            if len(bars) < 20:
                results[inst] = Layer1Result(
                    regime="RANGING", beta_score=0.0, mii=1.0, confidence=0.0
                )
                continue

            seq = seq_builder.build_sequence(
                list(bars), self._cfg.seq_lookback_T,
            )

            if seq is None or backend is None:
                beta_score, confidence = 0.0, 0.0
            else:
                beta_score, confidence = backend.predict(seq)

            mii = self._compute_mii(sym, bars)
            regime = _classify_regime(beta_score, mii)

            results[inst] = Layer1Result(
                regime=regime,
                beta_score=beta_score,
                mii=mii,
                confidence=confidence,
                metadata={
                    "feature_version": 2,
                    "input_type": "sequential",
                },
            )

        return results

    # ------------------------------------------------------------------
    # v1 推理路径（保持不变）
    # ------------------------------------------------------------------

    def _compute_v1(
        self,
        windows: dict[Instrument, list[Bar]],
    ) -> dict[Instrument, Layer1Result]:
        """使用逐品种模型进行推理（v1 路径）。"""
        results: dict[Instrument, Layer1Result] = {}
        as_of = datetime.now(timezone.utc)
        sym_windows = {inst.symbol: bars for inst, bars in windows.items()}

        for inst, bars in windows.items():
            sym = inst.symbol
            if len(bars) < 20:
                results[inst] = Layer1Result(
                    regime="RANGING", beta_score=0.0, mii=1.0, confidence=0.0
                )
                continue

            X = self._feature_builder.build(sym, sym_windows, as_of)

            model = self._models.get(sym)
            if model is None:
                model = Layer1Model(self._cfg)
                self._models[sym] = model
            beta_score, confidence = model.predict(X)

            mii = self._compute_mii(sym, bars)
            regime = _classify_regime(beta_score, mii)

            results[inst] = Layer1Result(
                regime=regime,
                beta_score=beta_score,
                mii=mii,
                confidence=confidence,
                metadata={"feature_dim": len(X)},
            )

        return results

    # ------------------------------------------------------------------
    # 共享 MII 辅助方法
    # ------------------------------------------------------------------

    def _compute_mii(self, sym: str, bars: list[Bar]) -> float:
        """根据品种的历史 K 线计算 MII。"""
        mii_calc = self._miis.get(sym)
        if mii_calc is None:
            mii_calc = MIICalculator(self._cfg)
            self._miis[sym] = mii_calc

        closes = np.array([b.close for b in bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars], dtype=np.float64)
        log_returns = np.diff(np.log(closes + _EPS))

        lookback = self._cfg.mii_amplitude_lookback
        if len(log_returns) >= lookback:
            window_ret = log_returns[-lookback:]
            std = window_ret.std() + _EPS
            returns_z = window_ret / std
        else:
            std = log_returns.std() + _EPS
            returns_z = log_returns / std

        return mii_calc.compute(returns_z, volumes[-len(returns_z):])


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------

def _realized_vol(closes: np.ndarray, lookback: int, eps: float = 1e-9) -> float:
    """从收盘价数组计算事前已实现波动率。"""
    if len(closes) < lookback + 1:
        rets = np.diff(np.log(closes + eps))
        return float(rets.std()) if len(rets) > 0 else 0.0
    rets = np.diff(np.log(closes[-(lookback + 1):] + eps))
    return float(rets.std())


def _inverse_frequency_weights(sym_labels: list[str]) -> np.ndarray:
    """逐样本逆频率权重（规格 §13 Q5）。

    无论每个品种提供多少训练样本，
    每个品种对总损失的贡献均等。

    weight_i = total / (n_symbols * count_for_symbol_i)

    随后归一化使得 mean(weight) == 1，
    以保持 LightGBM 损失量纲稳定。
    """
    counts = Counter(sym_labels)
    total = len(sym_labels)
    n_sym = len(counts)
    weights = np.array(
        [total / (n_sym * counts[sym] + _EPS) for sym in sym_labels],
        dtype=np.float64,
    )
    mean_w = weights.mean()
    if mean_w > _EPS:
        weights /= mean_w
    return weights
