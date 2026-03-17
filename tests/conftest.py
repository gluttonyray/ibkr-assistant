"""
Shared test fixtures: synthetic OHLCV DataFrames for various market scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_df(
    n: int = 500,
    start_price: float = 100.0,
    trend: float = 0.001,   # fractional daily drift per bar
    vol: float = 0.008,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with configurable trend."""
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(n - 1):
        ret = trend + vol * rng.standard_normal()
        prices.append(prices[-1] * (1 + ret))

    closes = np.array(prices)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = closes * (1 + abs(rng.normal(0, 0.003, n)))
    lows = closes * (1 - abs(rng.normal(0, 0.003, n)))
    volumes = rng.integers(100_000, 5_000_000, size=n).astype(float)

    idx = pd.date_range("2024-01-02 09:31", periods=n, freq="1min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


@pytest.fixture
def bullish_df():
    """Strong uptrend data."""
    return _make_df(n=500, trend=0.003, vol=0.004, seed=1)


@pytest.fixture
def bearish_df():
    """Strong downtrend data."""
    return _make_df(n=500, trend=-0.003, vol=0.004, seed=2)


@pytest.fixture
def flat_df():
    """Sideways/flat market."""
    return _make_df(n=500, trend=0.0, vol=0.001, seed=3)


@pytest.fixture
def short_df():
    """Too-short DataFrame (10 bars) — tests insufficient data handling."""
    return _make_df(n=10, seed=99)


@pytest.fixture
def truly_flat_df():
    """
    Completely static price data: constant OHLC with no directional drift.
    Used for tests that require ADX < 20 (no trend signal).
    With constant consecutive highs/lows, DM+ = DM- = 0, so ADX ≈ 0.
    """
    n = 100
    idx = pd.date_range("2024-01-02 09:31", periods=n, freq="1min")
    price = 100.0
    return pd.DataFrame(
        {
            "open":   [price] * n,
            "high":   [price * 1.001] * n,  # static high — DM+ = 0
            "low":    [price * 0.999] * n,  # static low  — DM- = 0
            "close":  [price] * n,
            "volume": [1_000_000.0] * n,
        },
        index=idx,
    )


@pytest.fixture
def standard_df():
    """Standard 500-bar neutral DataFrame."""
    return _make_df(n=500, trend=0.0005, vol=0.006, seed=42)
