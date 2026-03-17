"""Tests for FactorRegistry: registration, discovery, and lifecycle."""
from __future__ import annotations

import pytest

from src.factors.base import BaseFactor, FactorData, FactorResult
from src.factors.registry import FactorRegistry


# ──────────────────────────────────────────────
# Helpers: isolated registry test using save/restore
# ──────────────────────────────────────────────

@pytest.fixture
def isolated_registry():
    """Save and restore registry state around each test."""
    saved = dict(FactorRegistry._factors)
    yield FactorRegistry
    FactorRegistry._factors.clear()
    FactorRegistry._factors.update(saved)


class DummyFactor(BaseFactor):
    name = "dummy_factor_for_registry_test"
    category = "technical"
    frequency = "intraday"
    lookback_bars = 5
    weight = 1.0

    def _compute(self, data: FactorData) -> float:
        return 42.0


class SentimentFactor(BaseFactor):
    name = "dummy_sentiment_for_registry_test"
    category = "sentiment"
    frequency = "daily"
    lookback_bars = 5
    weight = 0.5

    def _compute(self, data: FactorData) -> float:
        return -1.0


# ──────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────

class TestRegistration:
    def test_register_decorator_adds_to_registry(self, isolated_registry):
        FactorRegistry.register(DummyFactor)
        assert DummyFactor.name in FactorRegistry.names()

    def test_register_returns_class_unchanged(self, isolated_registry):
        result = FactorRegistry.register(DummyFactor)
        assert result is DummyFactor

    def test_register_multiple_factors(self, isolated_registry):
        FactorRegistry.register(DummyFactor)
        FactorRegistry.register(SentimentFactor)
        names = FactorRegistry.names()
        assert DummyFactor.name in names
        assert SentimentFactor.name in names


# ──────────────────────────────────────────────
# create_all
# ──────────────────────────────────────────────

class TestCreateAll:
    def test_create_all_returns_instances(self, isolated_registry):
        FactorRegistry.register(DummyFactor)
        FactorRegistry.register(SentimentFactor)
        instances = FactorRegistry.create_all()
        assert len(instances) >= 2
        assert any(isinstance(i, DummyFactor) for i in instances)
        assert any(isinstance(i, SentimentFactor) for i in instances)

    def test_create_all_returns_fresh_instances(self, isolated_registry):
        FactorRegistry.register(DummyFactor)
        i1 = FactorRegistry.create_all()
        i2 = FactorRegistry.create_all()
        dummy1 = next(x for x in i1 if isinstance(x, DummyFactor))
        dummy2 = next(x for x in i2 if isinstance(x, DummyFactor))
        assert dummy1 is not dummy2

    def test_create_all_empty_registry(self, isolated_registry):
        FactorRegistry.clear()
        assert FactorRegistry.create_all() == []


# ──────────────────────────────────────────────
# by_category
# ──────────────────────────────────────────────

class TestByCategory:
    def test_by_category_filters_correctly(self, isolated_registry):
        FactorRegistry.register(DummyFactor)
        FactorRegistry.register(SentimentFactor)
        technical = FactorRegistry.by_category("technical")
        sentiment = FactorRegistry.by_category("sentiment")
        assert any(isinstance(i, DummyFactor) for i in technical)
        assert not any(isinstance(i, SentimentFactor) for i in technical)
        assert any(isinstance(i, SentimentFactor) for i in sentiment)

    def test_by_category_unknown_returns_empty(self, isolated_registry):
        FactorRegistry.register(DummyFactor)
        result = FactorRegistry.by_category("nonexistent_category")
        assert not any(isinstance(i, DummyFactor) for i in result)


# ──────────────────────────────────────────────
# get
# ──────────────────────────────────────────────

class TestGet:
    def test_get_returns_class(self, isolated_registry):
        FactorRegistry.register(DummyFactor)
        cls = FactorRegistry.get(DummyFactor.name)
        assert cls is DummyFactor

    def test_get_missing_raises(self, isolated_registry):
        with pytest.raises(KeyError):
            FactorRegistry.get("nonexistent_factor")


# ──────────────────────────────────────────────
# clear
# ──────────────────────────────────────────────

class TestClear:
    def test_clear_removes_all_factors(self, isolated_registry):
        FactorRegistry.register(DummyFactor)
        FactorRegistry.clear()
        assert DummyFactor.name not in FactorRegistry.names()

    def test_names_empty_after_clear(self, isolated_registry):
        FactorRegistry.clear()
        assert FactorRegistry.names() == []


# ──────────────────────────────────────────────
# Integration: technical factors are auto-registered on import
# ──────────────────────────────────────────────

class TestTechnicalFactorsRegistered:
    def test_all_11_technical_factors_registered(self):
        import src.factors.technical  # noqa: F401
        expected = [
            "momentum_rsi",
            "momentum_roc",
            "trend_ema_slope",
            "trend_macd_hist",
            "trend_adx_signed",
            "volatility_bb_pctb",
            "volatility_atr_ratio",
            "volume_obv_slope",
            "volume_vwap_dev",
            "mean_reversion_bb_dev",
            "mean_reversion_rsi_extreme",
        ]
        registered = FactorRegistry.names()
        for name in expected:
            assert name in registered, f"{name!r} not registered"

    def test_create_all_includes_technical_factors(self):
        import src.factors.technical  # noqa: F401
        instances = FactorRegistry.create_all()
        names = [f.name for f in instances]
        assert "momentum_rsi" in names
        assert "trend_ema_slope" in names
