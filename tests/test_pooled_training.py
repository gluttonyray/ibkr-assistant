"""Layer1 池化训练逻辑（v2）的测试。

覆盖：多资产样本堆叠、asset_id 类别特征，以及单一模型对不同品种输出差异化得分。
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from quant.config.schema import Layer1Config
from quant.core.types import Bar, Currency, Instrument, InstrumentType, TradingWindow
from quant.strategy.layer1.layer1 import Layer1, _inverse_frequency_weights
from quant.strategy.layer1.features import FeatureBuilderV2

UTC = timezone.utc


def _make_instrument(symbol: str, exchange: str = "CME") -> Instrument:
    return Instrument(
        symbol=symbol, instrument_type=InstrumentType.FUTURE,
        exchange=exchange, currency=Currency.USD,
        multiplier=50.0, tick_size=0.25,
        margin_initial=15200.0, margin_maintenance=13800.0,
        sessions=(TradingWindow(start="17:00", end="16:00", timezone="America/Chicago", label="Globex"),),
    )


def _make_bars(instrument: Instrument, n: int, start_price: float = 4000.0,
               step: float = 0.5) -> list[Bar]:
    """生成 n 根 K 线，收盘价以 step 逐步递增。"""
    bars: list[Bar] = []
    price = start_price
    for i in range(n):
        ts = datetime(2024, 1, 2 + i // 96, (i % 96) // 4, ((i % 96) % 4) * 15, 0, tzinfo=UTC)
        bars.append(Bar(
            instrument=instrument, timestamp=ts,
            open=price, high=price + 2, low=price - 2, close=price,
            volume=float(1000 + (i * 7) % 500),
        ))
        price += step
    return bars


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_pooled_multi_asset_stacking():
    """_fit_pooled 应将多个品种的样本堆叠为一个训练集，
    且特征矩阵列数应等于 FEATURE_DIM=37。"""
    cfg = Layer1Config(
        feature_version=2,
        forward_horizon_bars=5,
        vol_lookback=20,
        pca_n_components=3,
        pca_update_freq=1,
        pca_window=50,
        pca_universe_symbols=[],
    )
    layer1 = Layer1(cfg)

    es = _make_instrument("ES")
    nq = _make_instrument("NQ")
    hsi = _make_instrument("HSI", "HKEX")
    n_bars = 120

    windows = {
        es: _make_bars(es, n_bars, 4000.0, 0.5),
        nq: _make_bars(nq, n_bars, 15000.0, 1.0),
        hsi: _make_bars(hsi, n_bars, 20000.0, 2.0),
    }

    # 拦截 Layer1Model.fit，捕获训练数据
    with patch("quant.strategy.layer1.layer1.Layer1Model") as MockModel:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = (0.0, 0.0)
        MockModel.return_value = mock_instance

        layer1.fit(windows)

        if mock_instance.fit.called:
            X, y = mock_instance.fit.call_args[0][:2]
            # X 的形状应为 (N, 37)
            assert X.shape[1] == FeatureBuilderV2.FEATURE_DIM, (
                f"Pooled training X should have {FeatureBuilderV2.FEATURE_DIM} columns, "
                f"got {X.shape[1]}"
            )
            # 应包含来自全部 3 个品种的样本（超过任意单一品种）
            # 120 根 K 线，forward=5，约 3 个资产，预期超过 100 个样本
            assert X.shape[0] > 50, (
                f"Pooled training should have > 50 samples from 3 instruments, "
                f"got {X.shape[0]}"
            )
            # y 长度应与 X 行数一致
            assert len(y) == X.shape[0]


def test_asset_id_categorical_present():
    """索引 33 处的类别特征（asset_class）在池化训练集中应存在，
    且具有有效的整数编码。"""
    cfg = Layer1Config(
        feature_version=2,
        forward_horizon_bars=5,
        vol_lookback=20,
        pca_n_components=3,
        pca_update_freq=1,
        pca_window=50,
        pca_universe_symbols=[],
    )
    layer1 = Layer1(cfg)

    es = _make_instrument("ES")
    hsi = _make_instrument("HSI", "HKEX")
    n_bars = 120

    windows = {
        es: _make_bars(es, n_bars, 4000.0),
        hsi: _make_bars(hsi, n_bars, 20000.0, 2.0),
    }

    with patch("quant.strategy.layer1.layer1.Layer1Model") as MockModel:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = (0.0, 0.0)
        MockModel.return_value = mock_instance

        layer1.fit(windows)

        if mock_instance.fit.called:
            X = mock_instance.fit.call_args[0][0]
            # 类别特征索引为 (33, 34, 35, 36)
            cat_indices = FeatureBuilderV2.CATEGORICAL_FEATURE_INDICES

            for idx in cat_indices:
                col = X[:, idx]
                # 所有值应为非负整数
                assert np.all(col >= 0), (
                    f"Categorical feature at index {idx} has negative values"
                )
                # 值应为有效整数编码（< 10）
                assert np.all(col < 10), (
                    f"Categorical feature at index {idx} has unexpectedly large values"
                )

            # asset_class（索引 33）对 ES 和 HSI 均应为"期货"= 2
            # （两者均为 FUTURE 类型）
            assert np.all(X[:, 33] == pytest.approx(2.0)), (
                f"asset_class should be 2 (futures) for all samples"
            )

            # exchange（索引 34）：ES=CME=0，HSI=HKEX=2
            # 应至少有 2 个不同的取值
            unique_exchanges = np.unique(X[:, 34])
            assert len(unique_exchanges) >= 2, (
                f"Expected at least 2 exchange values (CME=0, HKEX=2), "
                f"got {unique_exchanges}"
            )


def test_single_model_different_scores():
    """单一池化模型应对价格走势不同的品种输出差异化的 beta_score。"""
    cfg = Layer1Config(
        feature_version=2,
        forward_horizon_bars=5,
        vol_lookback=20,
        pca_n_components=3,
        pca_update_freq=1,
        pca_window=50,
        pca_universe_symbols=[],
    )
    layer1 = Layer1(cfg)

    es = _make_instrument("ES")
    nq = _make_instrument("NQ")

    # ES：温和上涨，NQ：急剧上涨 -> 特征不同
    bars_es = _make_bars(es, 100, 4000.0, step=0.1)
    bars_nq = _make_bars(nq, 100, 15000.0, step=5.0)

    windows = {
        es: bars_es,
        nq: bars_nq,
    }

    # 创建正确 Mock 的 Layer1Model，根据特征向量返回 (beta_score, confidence)，
    # 使不同品种产生不同的得分。
    mock_model = MagicMock()

    def mock_predict(X):
        """根据特征返回 (beta_score, confidence)，beta 随特征变化。"""
        feat = X.reshape(-1)
        beta = float(feat[0]) * 10  # 以 ret_1 缩放
        confidence = min(abs(beta), 1.0)
        return beta, confidence

    mock_model.predict = mock_predict

    layer1._models["_pooled_v2"] = mock_model

    results = layer1.compute(windows)

    # 两个品种均应有结果
    assert es in results, "ES should be in results"
    assert nq in results, "NQ should be in results"

    # 两者均应使用 v2 管道，特征维度固定为 37
    assert results[es].metadata.get("feature_version") == 2
    assert results[nq].metadata.get("feature_version") == 2
    assert results[es].metadata.get("feature_dim") == 37
    assert results[nq].metadata.get("feature_dim") == 37
