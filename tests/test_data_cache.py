"""Tests for DataCache: put/get, TTL, invalidation."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache import DataCache


@pytest.fixture
def tmp_cache(tmp_path):
    return DataCache(cache_dir=tmp_path / "cache")


def _make_df(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    return pd.DataFrame({"value": np.random.default_rng(0).random(n)}, index=idx)


class TestPutGet:
    def test_get_returns_none_when_empty(self, tmp_cache):
        assert tmp_cache.get("missing_key") is None

    def test_put_then_get_returns_data(self, tmp_cache):
        df = _make_df()
        tmp_cache.put("test_key", df)
        result = tmp_cache.get("test_key")
        assert result is not None
        assert len(result) == len(df)

    def test_get_returns_correct_values(self, tmp_cache):
        df = _make_df()
        tmp_cache.put("test_key", df)
        result = tmp_cache.get("test_key")
        pd.testing.assert_frame_equal(df, result)

    def test_put_overwrites_existing(self, tmp_cache):
        df1 = _make_df(5)
        df2 = _make_df(10)
        tmp_cache.put("key", df1)
        tmp_cache.put("key", df2)
        result = tmp_cache.get("key")
        assert len(result) == 10


class TestTTL:
    def test_fresh_data_returned(self, tmp_cache):
        df = _make_df()
        tmp_cache.put("fresh", df)
        result = tmp_cache.get("fresh", max_age_hours=24)
        assert result is not None

    def test_expired_data_returns_none(self, tmp_path):
        """Test with very short TTL (1 second = ~0.0003 hours)."""
        cache = DataCache(cache_dir=tmp_path / "ttl_cache")
        df = _make_df()
        cache.put("stale", df)

        # Touch the file to make it look old
        path = cache._path("stale")
        # Set mtime to 2 hours ago
        old_time = time.time() - 7200
        import os
        os.utime(path, (old_time, old_time))

        result = cache.get("stale", max_age_hours=1)
        assert result is None


class TestInvalidate:
    def test_invalidate_removes_entry(self, tmp_cache):
        df = _make_df()
        tmp_cache.put("to_remove", df)
        tmp_cache.invalidate("to_remove")
        assert tmp_cache.get("to_remove") is None

    def test_invalidate_missing_key_no_error(self, tmp_cache):
        tmp_cache.invalidate("nonexistent")  # should not raise

    def test_clear_removes_all(self, tmp_cache):
        for i in range(3):
            tmp_cache.put(f"key_{i}", _make_df())
        tmp_cache.clear()
        for i in range(3):
            assert tmp_cache.get(f"key_{i}") is None


class TestKeyHandling:
    def test_key_with_slashes_sanitized(self, tmp_cache):
        df = _make_df()
        tmp_cache.put("path/to/data", df)
        result = tmp_cache.get("path/to/data")
        assert result is not None

    def test_key_with_colons_sanitized(self, tmp_cache):
        df = _make_df()
        tmp_cache.put("symbol:interval:date", df)
        result = tmp_cache.get("symbol:interval:date")
        assert result is not None
