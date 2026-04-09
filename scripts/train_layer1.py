#!/usr/bin/env python3
"""离线训练 Layer1 模型。

训练从 BacktestEngine 中分离后的独立脚本。
支持 tabular（LightGBM, XGBoost, Ridge 等）和 sequential（DLinear, PatchTST, LSTM）后端。

用法：
  cd new/
  .venv/bin/python scripts/train_layer1.py --data-dir data/training_universe/ --config configs/app.yaml
  .venv/bin/python scripts/train_layer1.py --data-dir data/training_universe/ --config configs/app.yaml --train-end 2026-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 确保 src 在 import 路径中
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant.config.loader import load_config
from quant.config.schema import AppConfig
from quant.core.types import Bar, Currency, Instrument, InstrumentType
from quant.instrument.registry import InstrumentRegistry
from quant.strategy.layer1.backends import create_backend
from quant.strategy.layer1.features import (
    FeatureBuilderV2,
    GlobalState,
    GlobalStateBuilder,
)
from quant.strategy.layer1.sequence_features import SequenceFeatureBuilder

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _realized_vol(closes: np.ndarray, lookback: int, eps: float = 1e-9) -> float:
    if len(closes) < lookback + 1:
        rets = np.diff(np.log(closes + eps))
        return float(rets.std()) if len(rets) > 0 else 0.0
    rets = np.diff(np.log(closes[-(lookback + 1):] + eps))
    return float(rets.std())


def _inverse_frequency_weights(sym_labels: list[str]) -> np.ndarray:
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


def load_bars_from_dir(
    data_dir: Path, registry: InstrumentRegistry
) -> dict[str, list[Bar]]:
    """加载目录中的 Parquet 文件为 dict[symbol, list[Bar]]。"""
    import polars as pl

    ohlcv_subdir = data_dir / "clean" / "ohlcv"
    search_dir = ohlcv_subdir if ohlcv_subdir.exists() else data_dir

    result: dict[str, list[Bar]] = {}
    for parquet_file in sorted(search_dir.glob("*.parquet")):
        symbol = parquet_file.stem.split("_")[0].upper()
        df = pl.read_parquet(parquet_file)

        try:
            inst = registry.get(symbol)
        except KeyError:
            inst = Instrument(
                symbol=symbol,
                instrument_type=InstrumentType.EQUITY,
                exchange="OTHER",
                currency=Currency.USD,
                multiplier=1.0,
                tick_size=0.01,
                margin_initial=0.0,
                margin_maintenance=0.0,
            )

        bars: list[Bar] = []
        for row in df.iter_rows(named=True):
            ts = (
                row.get("event_time")
                or row.get("timestamp")
                or row.get("date")
                or row.get("datetime")
            )
            if ts is None:
                continue
            if hasattr(ts, "replace"):
                ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
            bars.append(
                Bar(
                    instrument=inst,
                    timestamp=ts,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=float(row.get("volume", 0)),
                )
            )
        if bars:
            result[symbol] = bars

    return result


def build_tabular_dataset(
    bars_dict: dict[str, list[Bar]],
    cfg: AppConfig,
    train_end_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GlobalStateBuilder]:
    """构建 tabular 训练集（复用 _fit_pooled 逻辑）。

    Returns (X, y, sample_weights, global_builder)
    """
    l1cfg = cfg.strategy.layer1
    forward_h = l1cfg.forward_horizon_bars
    vol_lb = l1cfg.vol_lookback
    winsorize_sigma = l1cfg.label_winsorize_sigma
    min_bars = forward_h + 50

    global_builder = GlobalStateBuilder(
        n_components=l1cfg.pca_n_components,
        pca_update_freq=l1cfg.pca_update_freq,
        pca_window=l1cfg.pca_window,
        vol_lookback=l1cfg.vol_lookback,
    )
    feature_builder = FeatureBuilderV2(
        vol_lookback=l1cfg.vol_lookback,
        staleness_halflife_hours=l1cfg.staleness_halflife_hours,
    )

    # 构建 Instrument 映射
    inst_map: dict[str, Instrument] = {}
    for sym, bars in bars_dict.items():
        if bars:
            inst_map[sym] = bars[0].instrument

    valid_syms = [sym for sym, bars in bars_dict.items() if len(bars) >= min_bars]
    if not valid_syms:
        raise ValueError("No instruments with enough bars for training")

    max_end = min(len(bars_dict[sym]) for sym in valid_syms)
    if train_end_idx is not None:
        max_end = min(max_end, train_end_idx)
    start_i = max(50, vol_lb + 2)

    pca_universe = list(l1cfg.pca_universe_symbols) or None

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    sym_labels: list[str] = []

    for i in range(start_i, max_end - forward_h):
        sub_sym: dict[str, list[Bar]] = {
            sym: bars_dict[sym][:i] for sym in bars_dict
        }

        try:
            gs: GlobalState = global_builder.update(
                bar_index=i, windows=sub_sym, pca_universe=pca_universe
            )
        except Exception:
            continue

        for sym in valid_syms:
            bars = bars_dict[sym]
            inst = inst_map[sym]
            if i >= len(bars) - forward_h:
                continue

            asset_bars_i = bars[:i]
            if len(asset_bars_i) < max(vol_lb + 1, 21):
                continue

            feat = feature_builder.build(
                instrument=inst,
                asset_bars=asset_bars_i,
                global_state=gs,
                as_of=bars[i].timestamp,
            )
            if feat is None or len(feat) != FeatureBuilderV2.FEATURE_DIM:
                continue

            c_slice = bars[max(0, i - vol_lb - 1) : i + 1]
            closes = np.array([b.close for b in c_slice], dtype=np.float64)
            sigma_i = _realized_vol(closes, vol_lb)
            if sigma_i < _EPS:
                continue

            fwd_ret = (bars[i + forward_h].close - bars[i].close) / (
                bars[i].close + _EPS
            )
            y_label = float(
                np.clip(fwd_ret / sigma_i, -winsorize_sigma, winsorize_sigma)
            )

            X_rows.append(feat)
            y_rows.append(y_label)
            sym_labels.append(sym)

    if len(X_rows) < 50:
        raise ValueError(f"Only {len(X_rows)} valid samples (need >= 50)")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float64)
    sample_weights = _inverse_frequency_weights(sym_labels)

    return X, y, sample_weights, global_builder


def build_sequential_dataset(
    bars_dict: dict[str, list[Bar]],
    cfg: AppConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 sequential 训练集。

    Returns (X: (N,T,D), y: (N,), sample_weights: (N,))
    """
    l1cfg = cfg.strategy.layer1
    seq_builder = SequenceFeatureBuilder()
    winsorize_sigma = l1cfg.label_winsorize_sigma

    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    sym_labels: list[str] = []

    for sym, bars in bars_dict.items():
        result = seq_builder.build_training_sequences(
            bars, l1cfg.seq_lookback_T, l1cfg.forward_horizon_bars
        )
        if result is None:
            continue
        X_sym, y_sym = result
        X_all.append(X_sym)
        y_all.append(y_sym)
        sym_labels.extend([sym] * len(y_sym))

    if not X_all:
        raise ValueError("No valid training sequences")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # Winsorize
    y = np.clip(y, -winsorize_sigma, winsorize_sigma)
    sample_weights = _inverse_frequency_weights(sym_labels)

    return X, y, sample_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Layer1 model offline")
    parser.add_argument(
        "--data-dir", required=True, help="训练数据 Parquet 文件目录"
    )
    parser.add_argument("--config", default="configs/app.yaml", help="配置文件路径")
    parser.add_argument(
        "--train-end", default=None, help="训练数据截止日期（YYYY-MM-DD）"
    )
    parser.add_argument(
        "--output-dir", default=None, help="模型输出目录（默认使用 config 中的 model_dir）"
    )
    parser.add_argument(
        "--validate", action="store_true", default=True, help="OOS 验证"
    )
    parser.add_argument(
        "--no-validate", action="store_false", dest="validate", help="跳过 OOS 验证"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    cfg = load_config(args.config)
    l1cfg = cfg.strategy.layer1
    registry = InstrumentRegistry.from_yaml(cfg.instruments_path)

    # 校验 pca_universe_symbols
    if l1cfg.model_type in ("lgbm", "xgb", "ridge", "elasticnet", "rf"):
        if not l1cfg.pca_universe_symbols:
            logger.warning(
                "pca_universe_symbols is empty; PCA will use all available symbols"
            )

    # 加载数据
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录不存在：{data_dir}")
        sys.exit(1)

    logger.info(f"加载训练数据：{data_dir}")
    bars_dict = load_bars_from_dir(data_dir, registry)
    logger.info(f"加载完成：{len(bars_dict)} 个品种")

    if not bars_dict:
        logger.error("未找到有效的训练数据")
        sys.exit(1)

    # 创建 backend
    backend = create_backend(l1cfg.model_type, l1cfg)
    logger.info(f"Backend: {l1cfg.model_type} (input_type={backend.input_type})")

    # 构建训练集
    global_builder = None
    if backend.input_type == "tabular":
        logger.info("构建 tabular 训练集...")
        X, y, sample_weights, global_builder = build_tabular_dataset(
            bars_dict, cfg
        )
    else:
        logger.info("构建 sequential 训练集...")
        X, y, sample_weights = build_sequential_dataset(bars_dict, cfg)

    logger.info(f"训练集：X={X.shape}, y={y.shape}")

    # 训练
    logger.info("开始训练...")
    train_log = backend.fit(X, y, sample_weights=sample_weights)
    logger.info(f"训练完成：{train_log}")

    # OOS 验证
    if args.validate and backend.is_fitted:
        n = len(X)
        split = int(n * 0.8)
        X_oos = X[split:]
        y_oos = y[split:]
        if len(X_oos) > 10:
            preds = []
            for i in range(len(X_oos)):
                bs, _ = backend.predict(X_oos[i])
                preds.append(bs)
            preds = np.array(preds)

            # IC (rank correlation)
            from scipy.stats import spearmanr

            ic, _ = spearmanr(preds, y_oos)
            ic_mean = ic if not np.isnan(ic) else 0.0

            # 简化 IC IR：使用单折的 IC 值
            ic_ir = abs(ic_mean) / (0.1 + _EPS)  # 简化估计

            logger.info(f"OOS IC mean: {ic_mean:.4f}")
            if abs(ic_mean) < 0.02:
                logger.warning("OOS IC mean < 0.02 — 模型预测能力较弱")
            if abs(ic_mean) / 0.1 < 0.3:
                logger.warning("OOS IC IR < 0.3 — 信噪比较低")
        else:
            logger.warning("OOS 样本不足，跳过验证")
            ic_mean = 0.0
            ic_ir = 0.0
    else:
        ic_mean = 0.0
        ic_ir = 0.0

    # 保存模型
    version_tag = f"v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else Path(l1cfg.model_dir)
    model_dir = output_dir / version_tag
    model_dir.mkdir(parents=True, exist_ok=True)

    backend.save(model_dir)
    logger.info(f"模型已保存至：{model_dir}")

    # 保存 PCA 状态（tabular 路径）
    if global_builder is not None:
        global_builder.save_pca_state(model_dir / "global_state.pkl")
        logger.info("PCA 状态已保存")

    # 更新 meta.json（补充额外字段）
    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta.update(
        {
            "model_type": l1cfg.model_type,
            "input_type": backend.input_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_train_samples": len(X),
            "oos_ic_mean": float(ic_mean),
            "oos_ic_ir": float(ic_ir),
            "feature_version": l1cfg.feature_version,
        }
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"训练完成。模型目录：{model_dir}")


if __name__ == "__main__":
    main()
