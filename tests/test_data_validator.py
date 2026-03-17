"""Tests for DataValidator: gap detection, outliers, cross-source comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validator import DataValidator


def _make_df(n: int = 100, seed: int = 0, freq: str = "1min") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq=freq)
    return pd.DataFrame({
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture
def validator():
    return DataValidator(outlier_sigma=5.0, max_gap_pct=0.05, divergence_pct=0.5)


# ──────────────────────────────────────────────
# Basic validation
# ──────────────────────────────────────────────

class TestBasicValidation:
    def test_clean_data_passes_unchanged(self, validator):
        df = _make_df(100)
        result, report = validator.validate(df, source="test")
        assert len(result) == len(df)
        assert report.rows_out == 100

    def test_empty_df_returns_empty(self, validator):
        df = pd.DataFrame()
        result, report = validator.validate(df)
        assert result.empty

    def test_report_has_source_name(self, validator):
        df = _make_df(50)
        _, report = validator.validate(df, source="MySource")
        assert report.source == "MySource"


# ──────────────────────────────────────────────
# Duplicate timestamps
# ──────────────────────────────────────────────

class TestDuplicates:
    def test_duplicate_timestamps_removed(self, validator):
        df = _make_df(10)
        df = pd.concat([df, df.iloc[:3]])  # add 3 duplicates
        result, report = validator.validate(df)
        assert report.duplicate_count == 3
        assert len(result) == 10

    def test_no_duplicates_no_removal(self, validator):
        df = _make_df(10)
        result, report = validator.validate(df)
        assert report.duplicate_count == 0
        assert len(result) == 10


# ──────────────────────────────────────────────
# Zero/negative prices
# ──────────────────────────────────────────────

class TestZeroPrices:
    def test_zero_price_rows_removed(self, validator):
        df = _make_df(10)
        df.iloc[5, df.columns.get_loc("close")] = 0.0
        result, report = validator.validate(df)
        assert len(result) == 9
        assert len(report.warnings) > 0

    def test_negative_price_rows_removed(self, validator):
        df = _make_df(10)
        df.iloc[3, df.columns.get_loc("close")] = -10.0
        result, report = validator.validate(df)
        assert len(result) == 9


# ──────────────────────────────────────────────
# Outlier detection
# ──────────────────────────────────────────────

class TestOutliers:
    def test_no_outliers_in_clean_data(self, validator):
        df = _make_df(100)
        _, report = validator.validate(df)
        assert report.outlier_count == 0

    def test_spike_detected_as_outlier(self, validator):
        df = _make_df(100)
        # Inject a 20σ spike
        df.iloc[50, df.columns.get_loc("close")] *= 3.0
        _, report = validator.validate(df)
        assert report.outlier_count > 0

    def test_outlier_data_retained(self, validator):
        """Policy: flag but don't remove outliers."""
        df = _make_df(100)
        df.iloc[50, df.columns.get_loc("close")] *= 3.0
        result, report = validator.validate(df)
        assert report.outlier_count > 0
        assert len(result) == 100  # data kept


# ──────────────────────────────────────────────
# Gap detection
# ──────────────────────────────────────────────

class TestGapDetection:
    def test_no_gaps_in_complete_data(self, validator):
        df = _make_df(100, freq="15min")
        _, report = validator.validate(df, expected_freq="15min")
        assert report.gap_count == 0

    def test_gaps_detected(self, validator):
        df = _make_df(100, freq="15min")
        # Remove middle 10 rows (creating a gap)
        df_gapped = pd.concat([df.iloc[:40], df.iloc[50:]])
        _, report = validator.validate(df_gapped, expected_freq="15min")
        assert report.gap_count > 0


# ──────────────────────────────────────────────
# Cross-source comparison
# ──────────────────────────────────────────────

class TestCrossSourceComparison:
    def test_identical_sources_no_warnings(self, validator):
        df = _make_df(50)
        warnings = validator.compare_sources(df, df.copy())
        assert len(warnings) == 0

    def test_divergent_sources_emit_warning(self, validator):
        df1 = _make_df(50)
        df2 = df1.copy()
        # Inflate prices by 2%
        df2["close"] = df2["close"] * 1.02
        warnings = validator.compare_sources(df1, df2)
        assert len(warnings) > 0

    def test_no_overlap_no_warnings(self, validator):
        df1 = _make_df(20, freq="D")
        idx2 = pd.date_range("2024-06-01", periods=20, freq="D")
        df2 = _make_df(20, freq="D")
        df2.index = idx2
        warnings = validator.compare_sources(df1, df2)
        assert len(warnings) == 0
