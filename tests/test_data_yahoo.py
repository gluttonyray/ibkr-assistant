"""Tests for Yahoo Finance data source (mocked yfinance calls)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_yahoo_df(n: int = 30, cols=None) -> pd.DataFrame:
    cols = cols or ["Open", "High", "Low", "Close", "Volume"]
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")
    data = {c: np.random.default_rng(0).random(n) * 100 + 50 for c in cols}
    return pd.DataFrame(data, index=idx)


class TestFetchOHLCV:
    @patch("src.data.yahoo._yf")
    def test_returns_dataframe_with_ohlcv(self, mock_yf_fn):
        mock_yf = MagicMock()
        mock_yf_fn.return_value = mock_yf
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.history.return_value = _make_yahoo_df()

        from src.data.yahoo import fetch_ohlcv
        df = fetch_ohlcv("AAPL", "2024-01-01", "2024-02-01")
        assert "open" in df.columns
        assert "close" in df.columns
        assert len(df) == 30

    @patch("src.data.yahoo._yf")
    def test_empty_response_returns_empty_df(self, mock_yf_fn):
        mock_yf = MagicMock()
        mock_yf_fn.return_value = mock_yf
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        from src.data.yahoo import fetch_ohlcv
        df = fetch_ohlcv("UNKNOWN", "2024-01-01", "2024-02-01")
        assert df.empty


class TestFetchVIX:
    @patch("src.data.yahoo._yf")
    def test_returns_series_named_vix(self, mock_yf_fn):
        mock_yf = MagicMock()
        mock_yf_fn.return_value = mock_yf
        df = pd.DataFrame({"Close": [18.5, 19.2, 20.1]},
                          index=pd.date_range("2024-01-02", periods=3, freq="D"))
        mock_yf.download.return_value = df

        from src.data.yahoo import fetch_vix
        result = fetch_vix("2024-01-01", "2024-01-05")
        assert result.name == "vix"
        assert len(result) == 3

    @patch("src.data.yahoo._yf")
    def test_empty_vix_returns_empty_series(self, mock_yf_fn):
        mock_yf = MagicMock()
        mock_yf_fn.return_value = mock_yf
        mock_yf.download.return_value = pd.DataFrame()

        from src.data.yahoo import fetch_vix
        result = fetch_vix("2024-01-01", "2024-01-05")
        assert result.empty
        assert result.name == "vix"


class TestFetchFundamentals:
    @patch("src.data.yahoo._yf")
    def test_returns_dict_with_expected_keys(self, mock_yf_fn):
        mock_yf = MagicMock()
        mock_yf_fn.return_value = mock_yf
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = {
            "trailingPE": 25.0,
            "priceToBook": 6.5,
            "revenueGrowth": 0.08,
            "earningsGrowth": 0.12,
        }

        from src.data.yahoo import fetch_fundamentals
        result = fetch_fundamentals("AAPL")
        assert "trailing_pe" in result
        assert result["trailing_pe"] == 25.0

    @patch("src.data.yahoo._yf")
    def test_handles_missing_fields(self, mock_yf_fn):
        mock_yf = MagicMock()
        mock_yf_fn.return_value = mock_yf
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = {}

        from src.data.yahoo import fetch_fundamentals
        result = fetch_fundamentals("AAPL")
        assert result["trailing_pe"] is None

    @patch("src.data.yahoo._yf")
    def test_handles_ticker_exception(self, mock_yf_fn):
        mock_yf = MagicMock()
        mock_yf_fn.return_value = mock_yf
        mock_yf.Ticker.side_effect = Exception("network error")

        from src.data.yahoo import fetch_fundamentals
        result = fetch_fundamentals("AAPL")
        assert result == {}
