"""Tests for sentiment factors (VIX, VIX term structure, put/call ratio)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.factors.base import FactorData
import src.factors.sentiment  # noqa: F401
from src.factors.sentiment.vix import SentimentVIX, SentimentVIXTerm
from src.factors.sentiment.put_call import SentimentPutCall


def _make_ohlcv(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min")
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [1e6] * n,
    }, index=idx)


def _data(sentiment: dict | None = None) -> FactorData:
    return FactorData(ohlcv=_make_ohlcv(), symbol="TEST", sentiment=sentiment)


# ──────────────────────────────────────────────
# SentimentVIX
# ──────────────────────────────────────────────

class TestSentimentVIX:
    def test_stale_when_no_sentiment_data(self):
        f = SentimentVIX()
        result = f.compute(_data(sentiment=None))
        assert result.is_stale is True

    def test_stale_when_vix_key_missing(self):
        f = SentimentVIX()
        result = f.compute(_data(sentiment={"other_key": 1.0}))
        assert result.is_stale is True

    def test_high_vix_gives_negative_raw(self):
        f = SentimentVIX()
        result = f.compute(_data(sentiment={"vix": 30.0}))
        assert result.raw_value == pytest.approx(-30.0)

    def test_low_vix_gives_less_negative_raw(self):
        f = SentimentVIX()
        result_high = f.compute(_data(sentiment={"vix": 30.0}))
        result_low = f.compute(_data(sentiment={"vix": 10.0}))
        assert result_high.raw_value < result_low.raw_value

    def test_label_contains_vix(self):
        f = SentimentVIX()
        result = f.compute(_data(sentiment={"vix": 20.0}))
        assert "VIX" in result.label

    def test_category_is_sentiment(self):
        f = SentimentVIX()
        result = f.compute(_data(sentiment={"vix": 20.0}))
        assert result.category == "sentiment"


# ──────────────────────────────────────────────
# SentimentVIXTerm
# ──────────────────────────────────────────────

class TestSentimentVIXTerm:
    def test_stale_when_no_sentiment(self):
        f = SentimentVIXTerm()
        result = f.compute(_data(sentiment=None))
        assert result.is_stale is True

    def test_stale_when_vix3m_missing(self):
        f = SentimentVIXTerm()
        result = f.compute(_data(sentiment={"vix": 20.0}))
        assert result.is_stale is True

    def test_contango_gives_positive_raw(self):
        """VIX < VIX3M = contango = bullish."""
        f = SentimentVIXTerm()
        result = f.compute(_data(sentiment={"vix": 15.0, "vix3m": 20.0}))
        assert result.raw_value > 0.0

    def test_backwardation_gives_negative_raw(self):
        """VIX > VIX3M = backwardation = bearish fear spike."""
        f = SentimentVIXTerm()
        result = f.compute(_data(sentiment={"vix": 25.0, "vix3m": 20.0}))
        assert result.raw_value < 0.0

    def test_equal_vix_vix3m_gives_zero(self):
        f = SentimentVIXTerm()
        result = f.compute(_data(sentiment={"vix": 20.0, "vix3m": 20.0}))
        assert result.raw_value == pytest.approx(0.0)


# ──────────────────────────────────────────────
# SentimentPutCall
# ──────────────────────────────────────────────

class TestSentimentPutCall:
    def test_stale_when_no_sentiment(self):
        f = SentimentPutCall()
        result = f.compute(_data(sentiment=None))
        assert result.is_stale is True

    def test_returns_raw_pcr_value(self):
        f = SentimentPutCall()
        result = f.compute(_data(sentiment={"put_call_ratio": 1.2}))
        assert result.raw_value == pytest.approx(1.2)

    def test_category_is_sentiment(self):
        f = SentimentPutCall()
        result = f.compute(_data(sentiment={"put_call_ratio": 1.0}))
        assert result.category == "sentiment"

    def test_label_contains_ratio(self):
        f = SentimentPutCall()
        result = f.compute(_data(sentiment={"put_call_ratio": 0.8}))
        assert "P/C" in result.label or "/" in result.label
