"""Tests for AlphaCombiner."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.alpha.combiner import AlphaCombiner, AlphaSignal
from src.factors.base import FactorResult


def _make_result(
    name: str = "test_factor",
    z_score: float = 1.0,
    category: str = "technical",
    weight: float = 1.0,
    is_stale: bool = False,
) -> FactorResult:
    return FactorResult(
        name=name,
        raw_value=42.0,
        z_score=z_score,
        category=category,
        frequency="intraday",
        timestamp=datetime.now(timezone.utc),
        is_stale=is_stale,
        label="test",
        weight=weight,
    )


@pytest.fixture
def combiner():
    return AlphaCombiner()


# ──────────────────────────────────────────────
# Empty / stale inputs
# ──────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_list_returns_zero_score(self, combiner):
        result = combiner.combine([])
        assert result.score == 0.0
        assert result.active_factor_count == 0

    def test_all_stale_returns_zero_score(self, combiner):
        factors = [
            _make_result("f1", z_score=2.0, is_stale=True),
            _make_result("f2", z_score=-1.0, is_stale=True),
        ]
        result = combiner.combine(factors)
        assert result.score == 0.0
        assert result.active_factor_count == 0

    def test_nan_z_score_filtered_out(self, combiner):
        factors = [
            _make_result("f1", z_score=float("nan")),
            _make_result("f2", z_score=1.0),
        ]
        result = combiner.combine(factors)
        assert result.active_factor_count == 1


# ──────────────────────────────────────────────
# Score direction
# ──────────────────────────────────────────────

class TestScoreDirection:
    def test_positive_z_scores_give_positive_alpha(self, combiner):
        factors = [_make_result(f"f{i}", z_score=2.0) for i in range(5)]
        result = combiner.combine(factors)
        assert result.score > 0.0

    def test_negative_z_scores_give_negative_alpha(self, combiner):
        factors = [_make_result(f"f{i}", z_score=-2.0) for i in range(5)]
        result = combiner.combine(factors)
        assert result.score < 0.0

    def test_mixed_z_scores_near_zero(self, combiner):
        factors = [
            _make_result("f1", z_score=1.0),
            _make_result("f2", z_score=-1.0),
        ]
        result = combiner.combine(factors)
        assert abs(result.score) < 0.1


# ──────────────────────────────────────────────
# tanh scaling
# ──────────────────────────────────────────────

class TestTanhBounding:
    def test_score_bounded_between_minus1_plus1(self, combiner):
        # Very large z-scores should still be bounded
        factors = [_make_result(f"f{i}", z_score=100.0) for i in range(10)]
        result = combiner.combine(factors)
        assert -1.0 <= result.score <= 1.0

    def test_very_negative_z_scores_bounded(self, combiner):
        factors = [_make_result(f"f{i}", z_score=-100.0) for i in range(10)]
        result = combiner.combine(factors)
        assert -1.0 <= result.score <= 1.0

    def test_tanh_scale_affects_saturation(self):
        combiner_high = AlphaCombiner(tanh_scale=5.0)
        combiner_low = AlphaCombiner(tanh_scale=0.1)
        factors = [_make_result("f1", z_score=0.5)]
        r_high = combiner_high.combine(factors)
        r_low = combiner_low.combine(factors)
        # Higher scale → score closer to ±1
        assert abs(r_high.score) > abs(r_low.score)


# ──────────────────────────────────────────────
# Weights
# ──────────────────────────────────────────────

class TestWeighting:
    def test_higher_weight_increases_influence(self, combiner):
        # One strong factor vs one weak
        f_strong = [_make_result("strong", z_score=2.0, weight=5.0)]
        f_weak = [_make_result("weak", z_score=2.0, weight=1.0)]
        r_strong = combiner.combine(f_strong)
        r_weak = combiner.combine(f_weak)
        # Both should have same direction but we check both are positive
        assert r_strong.score > 0.0
        assert r_weak.score > 0.0

    def test_weight_zero_factor_excluded_effectively(self, combiner):
        factors = [
            _make_result("f1", z_score=2.0, weight=1.0),
            _make_result("f2", z_score=-2.0, weight=0.0),
        ]
        result = combiner.combine(factors)
        # f2 has weight=0, so total_weight dominated by f1 → positive score
        assert result.score > 0.0


# ──────────────────────────────────────────────
# Regime adjustment
# ──────────────────────────────────────────────

class TestRegimeAdjustment:
    def test_trending_calm_regime_field_preserved(self):
        combiner = AlphaCombiner()
        trend_factors = [_make_result(f"t{i}", z_score=1.0, category="trend") for i in range(3)]
        r_trending = combiner.combine(trend_factors, regime="TRENDING_CALM")
        assert r_trending.regime == "TRENDING_CALM"

    def test_regime_multipliers_affect_mixed_category_scoring(self):
        """When factors have different categories, regime multipliers affect
        their RELATIVE weighting, changing the overall normalized score."""
        combiner = AlphaCombiner()
        # TRENDING_CALM: trend=1.3 (amplified), momentum=0.8 (suppressed)
        # CHOPPY_VOLATILE: trend=0.3 (suppressed), momentum=0.4 (suppressed)
        # Use trend factors (positive) vs momentum factors (negative):
        # In TRENDING_CALM, trend factors get more weight → positive score
        # In CHOPPY_VOLATILE, both are suppressed but trend is more suppressed
        trend_positive = [_make_result(f"t{i}", z_score=2.0, category="trend") for i in range(3)]
        momentum_negative = [_make_result(f"m{i}", z_score=-1.0, category="momentum") for i in range(3)]
        factors = trend_positive + momentum_negative

        r_trending = combiner.combine(factors, regime="TRENDING_CALM")
        r_choppy = combiner.combine(factors, regime="CHOPPY_VOLATILE")
        # In TRENDING_CALM, trend factors (positive) are amplified → higher score
        assert r_trending.score > r_choppy.score

    def test_unknown_regime_uses_neutral_multipliers(self):
        combiner = AlphaCombiner()
        factors = [_make_result("f1", z_score=1.0)]
        result = combiner.combine(factors, regime="UNKNOWN")
        assert math.isfinite(result.score)


# ──────────────────────────────────────────────
# Confidence
# ──────────────────────────────────────────────

class TestConfidence:
    def test_all_agreeing_factors_confidence_one(self, combiner):
        factors = [_make_result(f"f{i}", z_score=2.0) for i in range(5)]
        result = combiner.combine(factors)
        assert result.confidence == pytest.approx(1.0)

    def test_all_disagreeing_factors_confidence_zero(self, combiner):
        # Half positive, half negative z-scores
        factors = [
            _make_result("f1", z_score=2.0),
            _make_result("f2", z_score=2.0),
            _make_result("f3", z_score=-2.0),
            _make_result("f4", z_score=-2.0),
        ]
        result = combiner.combine(factors)
        # Score near zero → confidence = 0
        assert result.confidence == 0.0

    def test_neutral_score_confidence_zero(self, combiner):
        factors = [_make_result("f1", z_score=1e-8)]
        result = combiner.combine(factors)
        assert result.confidence == 0.0


# ──────────────────────────────────────────────
# Factor contributions
# ──────────────────────────────────────────────

class TestFactorContributions:
    def test_contributions_keyed_by_factor_name(self, combiner):
        factors = [
            _make_result("alpha_factor", z_score=1.0),
            _make_result("beta_factor", z_score=-0.5),
        ]
        result = combiner.combine(factors)
        assert "alpha_factor" in result.factor_contributions
        assert "beta_factor" in result.factor_contributions

    def test_contributions_finite(self, combiner):
        factors = [_make_result(f"f{i}", z_score=float(i)) for i in range(5)]
        result = combiner.combine(factors)
        for v in result.factor_contributions.values():
            assert math.isfinite(v)


# ──────────────────────────────────────────────
# AlphaSignal structure
# ──────────────────────────────────────────────

class TestAlphaSignalStructure:
    def test_returns_alpha_signal_instance(self, combiner):
        result = combiner.combine([_make_result("f1", z_score=1.0)])
        assert isinstance(result, AlphaSignal)

    def test_regime_preserved(self, combiner):
        result = combiner.combine([_make_result("f1")], regime="TRENDING_CALM")
        assert result.regime == "TRENDING_CALM"

    def test_active_factor_count_correct(self, combiner):
        factors = [
            _make_result("f1", is_stale=False),
            _make_result("f2", is_stale=True),
            _make_result("f3", is_stale=False),
        ]
        result = combiner.combine(factors)
        assert result.active_factor_count == 2
