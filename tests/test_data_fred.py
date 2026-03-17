"""Tests for FRED data source (mocked fredapi calls)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestFetchSeries:
    @patch("src.data.fred._fred_client")
    def test_returns_series(self, mock_client_fn):
        mock_fred = MagicMock()
        mock_client_fn.return_value = mock_fred
        idx = pd.date_range("2024-01-02", periods=5, freq="D")
        mock_fred.get_series.return_value = pd.Series([4.5, 4.6, 4.5, 4.4, 4.4], index=idx)

        from src.data.fred import fetch_series
        result = fetch_series("DFF", "2024-01-01", "2024-01-10")
        assert len(result) == 5
        assert result.name == "dff"

    @patch("src.data.fred._fred_client")
    def test_exception_returns_empty_series(self, mock_client_fn):
        mock_fred = MagicMock()
        mock_client_fn.return_value = mock_fred
        mock_fred.get_series.side_effect = Exception("API error")

        from src.data.fred import fetch_series
        result = fetch_series("DGS10", "2024-01-01", "2024-01-10")
        assert result.empty


class TestFetchYieldCurve:
    @patch("src.data.fred.fetch_series")
    def test_computes_spread(self, mock_fetch):
        def side_effect(series_id, *args, **kwargs):
            idx = pd.date_range("2024-01-02", periods=3, freq="D")
            if series_id == "DGS2":
                return pd.Series([4.5, 4.6, 4.5], index=idx, name="dgs2")
            elif series_id == "DGS10":
                return pd.Series([4.2, 4.3, 4.1], index=idx, name="dgs10")
            return pd.Series(dtype=float)

        mock_fetch.side_effect = side_effect

        from src.data.fred import fetch_yield_curve
        df = fetch_yield_curve("2024-01-01", "2024-01-05")
        assert "spread" in df.columns
        # spread = (10Y - 2Y) * 100 bps
        expected_spread = (4.2 - 4.5) * 100  # -30 bps
        assert abs(df["spread"].iloc[0] - expected_spread) < 0.01

    @patch("src.data.fred.fetch_series")
    def test_empty_when_no_data(self, mock_fetch):
        mock_fetch.return_value = pd.Series(dtype=float)

        from src.data.fred import fetch_yield_curve
        df = fetch_yield_curve("2024-01-01", "2024-01-05")
        assert df.empty or len(df) == 0


class TestFredClientValidation:
    def test_raises_import_error_when_fredapi_missing(self):
        import sys
        # Save state
        fredapi_backup = sys.modules.get("fredapi")
        sys.modules["fredapi"] = None  # type: ignore

        try:
            # Re-import to trigger fresh _fred_client call
            from importlib import reload
            import src.data.fred as fred_module
            reload(fred_module)
            with pytest.raises((ImportError, TypeError, AttributeError)):
                fred_module._fred_client("testkey")
        finally:
            # Restore
            if fredapi_backup is not None:
                sys.modules["fredapi"] = fredapi_backup
            elif "fredapi" in sys.modules:
                del sys.modules["fredapi"]

    def test_raises_value_error_when_no_api_key(self):
        import os
        # Temporarily remove key
        original = os.environ.pop("FRED_API_KEY", None)
        try:
            with pytest.raises((ValueError, ImportError)):
                from src.data.fred import _fred_client
                _fred_client(api_key=None)
        finally:
            if original is not None:
                os.environ["FRED_API_KEY"] = original
