"""Sequential 后端 (DLinear/PatchTST/LSTM/GRU) 测试。"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quant.config.schema import Layer1Config


def _make_cfg(**overrides) -> Layer1Config:
    """创建测试用 Layer1Config（快速训练参数）。"""
    defaults = {
        "seq_epochs": 3,
        "seq_early_stopping_patience": 2,
        "seq_batch_size": 64,
        "seq_learning_rate": 1e-3,
        "lgb_huber_delta": 1.0,
    }
    defaults.update(overrides)
    return Layer1Config(**defaults)


@pytest.fixture
def training_data():
    """生成 (200, 32, 14) 训练数据。"""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 32, 14).astype(np.float32)
    y = rng.randn(200).astype(np.float64)
    return X, y


class TestDLinearBackend:
    def test_fit_predict(self, training_data):
        from quant.strategy.layer1.backends.dlinear_backend import DLinearBackend

        X, y = training_data
        backend = DLinearBackend(_make_cfg())
        result = backend.fit(X, y)
        assert "val_loss" in result
        assert result["model_type"] == "dlinear"
        assert backend.is_fitted

        # predict 单样本
        score, conf = backend.predict(X[0])
        assert isinstance(score, float)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_not_fitted_returns_zero(self):
        from quant.strategy.layer1.backends.dlinear_backend import DLinearBackend

        backend = DLinearBackend(_make_cfg())
        score, conf = backend.predict(np.zeros((32, 14), dtype=np.float32))
        assert score == 0.0
        assert conf == 0.0

    def test_save_load_roundtrip(self, training_data, tmp_path):
        from quant.strategy.layer1.backends.dlinear_backend import DLinearBackend

        X, y = training_data
        backend = DLinearBackend(_make_cfg())
        backend.fit(X, y)

        score_before, conf_before = backend.predict(X[0])

        save_dir = tmp_path / "dlinear_model"
        backend.save(save_dir)

        backend2 = DLinearBackend(_make_cfg())
        backend2.load(save_dir)
        score_after, conf_after = backend2.predict(X[0])

        assert abs(score_before - score_after) < 1e-5
        assert abs(conf_before - conf_after) < 1e-5


class TestPatchTSTBackend:
    def test_fit_predict(self, training_data):
        from quant.strategy.layer1.backends.patchtst_backend import PatchTSTBackend

        X, y = training_data
        backend = PatchTSTBackend(_make_cfg())
        result = backend.fit(X, y)
        assert "val_loss" in result
        assert result["model_type"] == "patchtst"
        assert backend.is_fitted

        score, conf = backend.predict(X[0])
        assert isinstance(score, float)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_not_fitted_returns_zero(self):
        from quant.strategy.layer1.backends.patchtst_backend import PatchTSTBackend

        backend = PatchTSTBackend(_make_cfg())
        score, conf = backend.predict(np.zeros((32, 14), dtype=np.float32))
        assert score == 0.0
        assert conf == 0.0

    def test_save_load_roundtrip(self, training_data, tmp_path):
        from quant.strategy.layer1.backends.patchtst_backend import PatchTSTBackend

        X, y = training_data
        backend = PatchTSTBackend(_make_cfg())
        backend.fit(X, y)

        score_before, conf_before = backend.predict(X[0])

        save_dir = tmp_path / "patchtst_model"
        backend.save(save_dir)

        backend2 = PatchTSTBackend(_make_cfg())
        backend2.load(save_dir)
        score_after, conf_after = backend2.predict(X[0])

        assert abs(score_before - score_after) < 1e-5
        assert abs(conf_before - conf_after) < 1e-5


class TestLSTMBackend:
    def test_fit_predict(self, training_data):
        from quant.strategy.layer1.backends.lstm_backend import LSTMBackend

        X, y = training_data
        backend = LSTMBackend(_make_cfg())
        result = backend.fit(X, y)
        assert "val_loss" in result
        assert result["model_type"] == "lstm"
        assert backend.is_fitted

        score, conf = backend.predict(X[0])
        assert isinstance(score, float)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_not_fitted_returns_zero(self):
        from quant.strategy.layer1.backends.lstm_backend import LSTMBackend

        backend = LSTMBackend(_make_cfg())
        score, conf = backend.predict(np.zeros((32, 14), dtype=np.float32))
        assert score == 0.0
        assert conf == 0.0

    def test_save_load_roundtrip(self, training_data, tmp_path):
        from quant.strategy.layer1.backends.lstm_backend import LSTMBackend

        X, y = training_data
        backend = LSTMBackend(_make_cfg())
        backend.fit(X, y)

        score_before, conf_before = backend.predict(X[0])

        save_dir = tmp_path / "lstm_model"
        backend.save(save_dir)

        backend2 = LSTMBackend(_make_cfg())
        backend2.load(save_dir)
        score_after, conf_after = backend2.predict(X[0])

        assert abs(score_before - score_after) < 1e-5
        assert abs(conf_before - conf_after) < 1e-5


class TestGRUBackend:
    def test_fit_predict(self, training_data):
        from quant.strategy.layer1.backends.lstm_backend import GRUBackend

        X, y = training_data
        backend = GRUBackend(_make_cfg())
        result = backend.fit(X, y)
        assert "val_loss" in result
        assert result["model_type"] == "gru"
        assert backend.is_fitted

        score, conf = backend.predict(X[0])
        assert isinstance(score, float)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_save_load_roundtrip(self, training_data, tmp_path):
        from quant.strategy.layer1.backends.lstm_backend import GRUBackend

        X, y = training_data
        backend = GRUBackend(_make_cfg())
        backend.fit(X, y)

        score_before, conf_before = backend.predict(X[0])

        save_dir = tmp_path / "gru_model"
        backend.save(save_dir)

        backend2 = GRUBackend(_make_cfg())
        backend2.load(save_dir)
        score_after, conf_after = backend2.predict(X[0])

        assert abs(score_before - score_after) < 1e-5
        assert abs(conf_before - conf_after) < 1e-5
