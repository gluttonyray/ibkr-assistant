"""模型持久化 save/load 往返测试。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quant.config.schema import Layer1Config
from quant.strategy.layer1.backends.lgbm_backend import LGBMBackend


def _make_training_data(n: int = 200, d: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """生成简单的训练数据（有信号，避免 LightGBM 退化）。"""
    rng = np.random.RandomState(42)
    X = rng.randn(n, d)
    y = X[:, 0] * 0.5 + rng.randn(n) * 0.1
    return X, y


@pytest.fixture
def lgbm_cfg() -> Layer1Config:
    return Layer1Config(model_type="lgbm")


def test_meta_json_contains_model_file(lgbm_cfg):
    """meta.json 包含 model_file 字段。"""
    backend = LGBMBackend(lgbm_cfg)
    X, y = _make_training_data()
    backend.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        backend.save(model_dir)

        meta_path = model_dir / "meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert "model_file" in meta
        assert meta["model_file"] == "model.pkl"
        assert (model_dir / meta["model_file"]).exists()


def test_lgbm_save_load_roundtrip_predict(lgbm_cfg):
    """save/load 往返后 predict 输出一致。"""
    backend = LGBMBackend(lgbm_cfg)
    X, y = _make_training_data()
    backend.fit(X, y)

    # 在保存前做一些预测
    test_x = X[0]
    score_before, conf_before = backend.predict(test_x)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        backend.save(model_dir)

        # 创建新 backend 并加载
        new_backend = LGBMBackend(lgbm_cfg)
        assert not new_backend.is_fitted
        new_backend.load(model_dir)
        assert new_backend.is_fitted

        score_after, conf_after = new_backend.predict(test_x)

    assert score_after == pytest.approx(score_before, abs=1e-6)
    assert conf_after == pytest.approx(conf_before, abs=1e-6)


def test_meta_json_contains_confidence_scale(lgbm_cfg):
    """meta.json 包含 confidence_scale。"""
    backend = LGBMBackend(lgbm_cfg)
    X, y = _make_training_data()
    backend.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        backend.save(model_dir)

        with open(model_dir / "meta.json") as f:
            meta = json.load(f)

        assert "confidence_scale" in meta
        assert meta["confidence_scale"] > 0
