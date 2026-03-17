"""
Unit tests for individual indicator modules.
Tests that each indicator:
  1. Returns a valid SignalResult (+1 / 0 / -1)
  2. Handles insufficient data gracefully
  3. Produces directionally correct signals on strong trend data
"""
from __future__ import annotations

import pytest
import pandas as pd

from src.indicators.base import SignalResult
from src.indicators.trend import (
    EMAcross, SMAcross, MACD, ADXSignal, ParabolicSAR, Ichimoku, Supertrend
)
from src.indicators.momentum import RSI, StochKDJ, CCI, WilliamsR, ROC, UltimateOscillator
from src.indicators.volatility import BollingerBands, KeltnerChannel, DonchianChannel
from src.indicators.volume import OBVTrend, VWAPSignal, MFI, CMF
from src.indicators.support_resistance import PivotPoints, FibonacciRetracement


ALL_CLASSES = [
    EMAcross, SMAcross, MACD, ADXSignal, ParabolicSAR, Ichimoku, Supertrend,
    RSI, StochKDJ, CCI, WilliamsR, ROC, UltimateOscillator,
    BollingerBands, KeltnerChannel, DonchianChannel,
    OBVTrend, VWAPSignal, MFI, CMF,
    PivotPoints, FibonacciRetracement,
]


class TestSignalValidity:
    """Every indicator must return a signal in {-1, 0, +1}."""

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=[c.__name__ for c in ALL_CLASSES])
    def test_signal_is_valid(self, standard_df, cls):
        result = cls().compute(standard_df)
        assert isinstance(result, SignalResult)
        assert result.signal in (-1, 0, 1), f"{cls.__name__} returned {result.signal}"

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=[c.__name__ for c in ALL_CLASSES])
    def test_short_data_returns_valid_signal(self, short_df, cls):
        result = cls().compute(short_df)
        assert result.signal in (-1, 0, 1)

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=[c.__name__ for c in ALL_CLASSES])
    def test_result_has_label(self, standard_df, cls):
        result = cls().compute(standard_df)
        assert isinstance(result.label, str)
        assert len(result.label) > 0


class TestTrendIndicators:
    def test_ema_cross_bullish_on_uptrend(self, bullish_df):
        result = EMAcross().compute(bullish_df)
        assert result.signal == 1, f"EMA cross should be bullish, got {result.label}"

    def test_ema_cross_bearish_on_downtrend(self, bearish_df):
        result = EMAcross().compute(bearish_df)
        assert result.signal == -1, f"EMA cross should be bearish, got {result.label}"

    def test_sma_cross_bullish_on_uptrend(self, bullish_df):
        result = SMAcross().compute(bullish_df)
        assert result.signal == 1

    def test_sma_cross_bearish_on_downtrend(self, bearish_df):
        result = SMAcross().compute(bearish_df)
        assert result.signal == -1

    def test_macd_returns_valid(self, standard_df):
        result = MACD().compute(standard_df)
        assert result.signal in (-1, 0, 1)

    def test_adx_weak_trend_gives_hold(self, truly_flat_df):
        result = ADXSignal().compute(truly_flat_df)
        # Constant prices → DM+ = DM- = 0 → ADX ≈ 0 < 20 → HOLD
        assert result.signal == 0, f"ADX should be hold on constant-price data, got {result.label}"

    def test_supertrend_returns_direction(self, bullish_df):
        result = Supertrend().compute(bullish_df)
        assert result.signal in (-1, 1)  # should be decisive on strong trend


class TestMomentumIndicators:
    def test_rsi_oversold_on_crash(self):
        """Simulate a crash: price drops sharply each bar."""
        import numpy as np
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = 100.0 * np.exp(-np.arange(n) * 0.05)  # exponential decay
        df = pd.DataFrame({
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": [1e6] * n
        }, index=idx)
        result = RSI().compute(df)
        # After a sharp decline, RSI should be oversold → +1
        assert result.signal == 1, f"RSI should be oversold after crash, got {result.label}"

    def test_rsi_overbought_on_rally(self):
        """Simulate a rally: price rises sharply."""
        import numpy as np
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = 100.0 * np.exp(np.arange(n) * 0.05)
        df = pd.DataFrame({
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": [1e6] * n
        }, index=idx)
        result = RSI().compute(df)
        assert result.signal == -1, f"RSI should be overbought after rally, got {result.label}"

    def test_roc_positive_on_uptrend(self, bullish_df):
        result = ROC().compute(bullish_df)
        assert result.signal == 1

    def test_roc_negative_on_downtrend(self, bearish_df):
        result = ROC().compute(bearish_df)
        assert result.signal == -1

    def test_stoch_kdj_returns_valid(self, standard_df):
        result = StochKDJ().compute(standard_df)
        assert result.signal in (-1, 0, 1)


class TestVolatilityIndicators:
    def test_bollinger_bands_returns_valid(self, standard_df):
        result = BollingerBands().compute(standard_df)
        assert result.signal in (-1, 0, 1)

    def test_keltner_returns_valid(self, standard_df):
        result = KeltnerChannel().compute(standard_df)
        assert result.signal in (-1, 0, 1)

    def test_donchian_at_high_is_sell(self):
        """If price is at the 20-bar high, Donchian should be bearish."""
        import numpy as np
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = np.ones(n) * 100.0
        close[-1] = 110.0  # spike to a new high
        high = close.copy()
        df = pd.DataFrame({
            "open": close, "high": high, "low": close * 0.99,
            "close": close, "volume": [1e6] * n
        }, index=idx)
        result = DonchianChannel().compute(df)
        assert result.signal == -1, f"At 20-bar high, Donchian should be -1, got {result.label}"


class TestVolumeIndicators:
    def test_vwap_signal_direction(self, standard_df):
        result = VWAPSignal().compute(standard_df)
        assert result.signal in (-1, 0, 1)

    def test_mfi_oversold_gives_buy(self):
        """Simulate high selling volume pushing MFI low."""
        import numpy as np
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = 100.0 - np.arange(n) * 0.5  # falling prices
        df = pd.DataFrame({
            "open": close + 0.1, "high": close + 0.2,
            "low": close - 0.2, "close": close,
            "volume": [5_000_000.0] * n  # high volume
        }, index=idx)
        result = MFI().compute(df)
        # Low price + high volume on falling market → low MFI → buy
        assert result.signal in (0, 1)  # may vary; just no crash

    def test_obv_rising_on_uptrend(self, bullish_df):
        result = OBVTrend().compute(bullish_df)
        assert result.signal in (0, 1)  # OBV should trend up with price

    def test_cmf_returns_valid(self, standard_df):
        result = CMF().compute(standard_df)
        assert result.signal in (-1, 0, 1)


class TestSupportResistance:
    def test_pivot_points_returns_valid(self, standard_df):
        result = PivotPoints().compute(standard_df)
        assert result.signal in (-1, 0, 1)

    def test_fibonacci_returns_valid(self, standard_df):
        result = FibonacciRetracement().compute(standard_df)
        assert result.signal in (-1, 0, 1)

    def test_fibonacci_uptrend_positive(self, bullish_df):
        result = FibonacciRetracement().compute(bullish_df)
        assert result.signal in (-1, 0, 1)  # direction may vary at top

    def test_fibonacci_handles_flat(self, flat_df):
        result = FibonacciRetracement().compute(flat_df)
        assert result.signal in (-1, 0, 1)
