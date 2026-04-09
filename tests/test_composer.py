import pytest
from datetime import datetime, timezone
from quant.strategy.layer2.composer import DefaultComposer
from quant.core.strategy import Layer1Result, Layer2Result
from quant.core.types import SignalType
from quant.config.schema import StrategyConfig

UTC = timezone.utc


@pytest.fixture
def composer():
    return DefaultComposer(StrategyConfig())


def test_strong_long_signal(composer, es_instrument):
    """强势多头：w1*1*1 + w2*1 = 0.35+0.65 = 1.0，应触发 LONG_ENTRY。"""
    l1 = Layer1Result(regime="TRENDING_STRONG", beta_score=1.0, mii=1.0, confidence=0.9)
    l2 = Layer2Result(alpha_score=1.0)
    sig = composer.compose(es_instrument, l1, l2, datetime.now(UTC))
    # w1*1*1 + w2*1 = 0.35+0.65 = 1.0 > entry_threshold(0.3)
    assert sig.signal_type == SignalType.LONG_ENTRY


def test_strong_short_signal(composer, es_instrument):
    """强势空头：beta_score=-1, alpha_score=-1，应触发 SHORT_ENTRY。"""
    l1 = Layer1Result(regime="TRENDING_STRONG", beta_score=-1.0, mii=1.0, confidence=0.9)
    l2 = Layer2Result(alpha_score=-1.0)
    sig = composer.compose(es_instrument, l1, l2, datetime.now(UTC))
    assert sig.signal_type == SignalType.SHORT_ENTRY


def test_mii_zero_neutralizes_l1(composer, es_instrument):
    """MII=0 时，Layer1 贡献为零，由 Layer2 单独决定方向。"""
    l1 = Layer1Result(regime="EXHAUSTED", beta_score=1.0, mii=0.0, confidence=0.5)
    l2 = Layer2Result(alpha_score=0.5)  # 0.65*0.5=0.325 > 0.3
    sig = composer.compose(es_instrument, l1, l2, datetime.now(UTC))
    assert sig.signal_type == SignalType.LONG_ENTRY
    assert sig.mii == pytest.approx(0.0)


def test_score_formula(composer, es_instrument):
    """验证综合得分公式：score = w1 * beta_score * mii + w2 * alpha_score。"""
    l1 = Layer1Result(regime="RANGING", beta_score=0.5, mii=0.8, confidence=0.6)
    l2 = Layer2Result(alpha_score=0.4)
    sig = composer.compose(es_instrument, l1, l2, datetime.now(UTC))
    expected = 0.35 * 0.5 * 0.8 + 0.65 * 0.4  # = 0.14 + 0.26 = 0.40
    assert sig.score == pytest.approx(expected, rel=0.01)
