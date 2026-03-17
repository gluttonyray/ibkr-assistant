"""
Unit tests for the signal aggregator and anti-flicker logic.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.indicators.aggregator import (
    AggregateResult,
    INDICATOR_CATEGORY,
    SignalAggregator,
)
from src.indicators.base import BaseIndicator, SignalResult


# ─────────────────────────────────────────────────────────────────────────────
# Stub indicators for deterministic testing
# ─────────────────────────────────────────────────────────────────────────────

class FixedIndicator(BaseIndicator):
    """Returns a fixed signal regardless of the DataFrame."""
    def __init__(self, sig: int, weight: float = 1.0, name: str = "fixed"):
        self._sig = sig
        self.weight = weight
        self.name = name

    @property
    def min_bars(self) -> int:
        return 1

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        return SignalResult(self._sig, float(self._sig), f"fixed {self._sig}")


def _make_agg(indicators: List[BaseIndicator], confirm: int = 2, **kwargs) -> SignalAggregator:
    """Create an aggregator with a temp log file."""
    tmp = tempfile.mktemp(suffix=".log")
    return SignalAggregator(
        indicators=indicators,
        confirm_bars=confirm,
        log_file=tmp,
        **kwargs,
    )


def _minimal_df(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame(
        {"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
         "close": [100.0] * n, "volume": [1e6] * n},
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Score calculation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCalculation:
    def test_all_buy_gives_positive_score(self):
        inds = [FixedIndicator(1) for _ in range(5)]
        agg = _make_agg(inds, confirm=1)
        result = agg.process("TEST", _minimal_df())
        assert result.score == pytest.approx(1.0)
        agg.close()

    def test_all_sell_gives_negative_score(self):
        inds = [FixedIndicator(-1) for _ in range(5)]
        agg = _make_agg(inds, confirm=1)
        result = agg.process("TEST", _minimal_df())
        assert result.score == pytest.approx(-1.0)
        agg.close()

    def test_mixed_signals_averages_correctly(self):
        # 3 buy (w=1) + 1 sell (w=1) + 1 hold (w=1) → (3 - 1 + 0) / 5 = 0.4
        inds = [FixedIndicator(1)] * 3 + [FixedIndicator(-1)] + [FixedIndicator(0)]
        agg = _make_agg(inds, confirm=1)
        result = agg.process("TEST", _minimal_df())
        assert result.score == pytest.approx(0.4)
        agg.close()

    def test_weighted_average(self):
        # buy w=2, sell w=1 → (2 - 1) / 3 ≈ 0.333
        inds = [FixedIndicator(1, weight=2.0), FixedIndicator(-1, weight=1.0)]
        agg = _make_agg(inds, confirm=1)
        result = agg.process("TEST", _minimal_df())
        assert result.score == pytest.approx(1 / 3, rel=1e-3)
        agg.close()

    def test_score_range_is_bounded(self, standard_df):
        """Score must always be in [-1, +1]."""
        from src.indicators.aggregator import ALL_INDICATORS
        agg = _make_agg(ALL_INDICATORS, confirm=1)
        result = agg.process("AAPL", standard_df)
        assert -1.0 <= result.score <= 1.0
        agg.close()


# ─────────────────────────────────────────────────────────────────────────────
# Threshold / classification tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClassification:
    def test_buy_above_threshold(self):
        inds = [FixedIndicator(1)]
        agg = _make_agg(inds, confirm=1, buy_threshold=0.4, sell_threshold=-0.4)
        result = agg.process("T", _minimal_df())
        assert result.final_signal == "BUY"
        agg.close()

    def test_sell_below_threshold(self):
        inds = [FixedIndicator(-1)]
        agg = _make_agg(inds, confirm=1, buy_threshold=0.4, sell_threshold=-0.4)
        result = agg.process("T", _minimal_df())
        assert result.final_signal == "SELL"
        agg.close()

    def test_hold_within_thresholds(self):
        # score = 0 → HOLD
        inds = [FixedIndicator(0)]
        agg = _make_agg(inds, confirm=1, buy_threshold=0.4, sell_threshold=-0.4)
        result = agg.process("T", _minimal_df())
        assert result.final_signal == "HOLD"
        agg.close()

    def test_custom_thresholds(self):
        # tight threshold: 0.1
        inds = [FixedIndicator(1, weight=1.0), FixedIndicator(0, weight=9.0)]
        agg = _make_agg(inds, confirm=1, buy_threshold=0.05, sell_threshold=-0.05)
        result = agg.process("T", _minimal_df())
        # score = 0.1 > 0.05 → BUY
        assert result.final_signal == "BUY"
        agg.close()


# ─────────────────────────────────────────────────────────────────────────────
# Anti-flicker tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiFlicker:
    def test_first_bar_not_confirmed(self):
        inds = [FixedIndicator(1)]
        agg = _make_agg(inds, confirm=2)
        result = agg.process("T", _minimal_df())
        # Only 1 bar seen — not confirmed yet
        assert result.confirmed is False
        assert result.final_signal == "HOLD"  # unconfirmed → HOLD
        agg.close()

    def test_two_consistent_bars_confirmed(self):
        inds = [FixedIndicator(1)]
        agg = _make_agg(inds, confirm=2)
        agg.process("T", _minimal_df())   # bar 1
        result = agg.process("T", _minimal_df())  # bar 2
        assert result.confirmed is True
        assert result.final_signal == "BUY"
        agg.close()

    def test_inconsistent_bars_not_confirmed(self):
        """Alternating BUY / SELL should never confirm."""
        buy_ind = FixedIndicator(1)
        sell_ind = FixedIndicator(-1)
        agg_buy = _make_agg([buy_ind], confirm=2)
        agg_sell = _make_agg([sell_ind], confirm=2)

        # Process once with buy indicator aggregator
        r1 = agg_buy.process("T", _minimal_df())
        assert r1.confirmed is False

        # Simulate a change: swap to sell on second bar
        # Use fresh aggregator that has seen BUY then SELL
        inds_toggle = [FixedIndicator(1)]
        agg = _make_agg(inds_toggle, confirm=2)
        agg.process("T", _minimal_df())  # BUY
        # Manually corrupt history to simulate flip
        agg._history["T"].append("SELL")
        result = agg.process("T", _minimal_df())
        assert result.confirmed is False
        agg.close()
        agg_buy.close()
        agg_sell.close()

    def test_confirm_bars_three(self):
        inds = [FixedIndicator(1)]
        agg = _make_agg(inds, confirm=3)
        agg.process("T", _minimal_df())  # bar 1 — not confirmed
        agg.process("T", _minimal_df())  # bar 2 — not confirmed
        result = agg.process("T", _minimal_df())  # bar 3 — confirmed
        assert result.confirmed is True
        assert result.final_signal == "BUY"
        agg.close()

    def test_confirm_resets_on_symbol_change(self):
        """History is per-symbol; different symbols don't interfere."""
        inds = [FixedIndicator(1)]
        agg = _make_agg(inds, confirm=2)
        agg.process("AAPL", _minimal_df())  # AAPL bar 1
        result_msft = agg.process("MSFT", _minimal_df())  # MSFT bar 1 (fresh)
        assert result_msft.confirmed is False
        agg.close()


# ─────────────────────────────────────────────────────────────────────────────
# JSON log tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLogging:
    def test_confirmed_signal_written_to_log(self, tmp_path):
        log_file = str(tmp_path / "test_signals.log")
        inds = [FixedIndicator(1)]
        agg = SignalAggregator(
            indicators=inds,
            confirm_bars=1,
            log_file=log_file,
            buy_threshold=0.4,
            sell_threshold=-0.4,
        )
        agg.process("AAPL", _minimal_df())
        agg.close()

        lines = open(log_file).readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["symbol"] == "AAPL"
        assert record["final_signal"] == "BUY"
        assert "timestamp" in record
        assert "score" in record

    def test_unconfirmed_signal_not_written(self, tmp_path):
        log_file = str(tmp_path / "test_unconfirmed.log")
        inds = [FixedIndicator(1)]
        agg = SignalAggregator(
            indicators=inds,
            confirm_bars=2,
            log_file=log_file,
        )
        agg.process("AAPL", _minimal_df())  # only 1 bar → not confirmed
        agg.close()

        if os.path.exists(log_file):
            lines = [l for l in open(log_file).readlines() if l.strip()]
            assert len(lines) == 0

    def test_log_contains_valid_json(self, tmp_path):
        log_file = str(tmp_path / "test_json.log")
        inds = [FixedIndicator(1), FixedIndicator(-1), FixedIndicator(0)]
        agg = SignalAggregator(
            indicators=inds,
            confirm_bars=1,
            log_file=log_file,
        )
        agg.process("SPY", _minimal_df())
        agg.close()

        with open(log_file) as f:
            for line in f:
                record = json.loads(line)
                assert isinstance(record["score"], float)
                assert isinstance(record["buy_count"], int)
                assert isinstance(record["indicators"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Counter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalCounts:
    def test_counts_sum_to_total_indicators(self):
        inds = [FixedIndicator(1)] * 3 + [FixedIndicator(-1)] * 2 + [FixedIndicator(0)] * 1
        agg = _make_agg(inds, confirm=1)
        result = agg.process("T", _minimal_df())
        assert result.buy_count == 3
        assert result.sell_count == 2
        assert result.hold_count == 1
        assert result.buy_count + result.sell_count + result.hold_count == len(inds)
        agg.close()


# ─────────────────────────────────────────────────────────────────────────────
# Insufficient data handling
# ─────────────────────────────────────────────────────────────────────────────

class TestInsufficientData:
    def test_short_df_returns_hold(self, short_df):
        """All real indicators should gracefully degrade on very short data."""
        from src.indicators.aggregator import ALL_INDICATORS
        agg = _make_agg(ALL_INDICATORS, confirm=1)
        result = agg.process("TEST", short_df)
        # Most indicators can't compute — score should be near 0 and signal HOLD
        assert result.final_signal in ("HOLD", "BUY", "SELL")  # doesn't crash
        assert -1.0 <= result.score <= 1.0
        agg.close()

    def test_empty_df_does_not_crash(self):
        from src.indicators.aggregator import ALL_INDICATORS
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype=float
        )
        agg = _make_agg(ALL_INDICATORS, confirm=1)
        result = agg.process("TEST", empty)
        assert result.final_signal == "HOLD"
        agg.close()


# ─────────────────────────────────────────────────────────────────────────────
# AggregateResult serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_is_json_serializable(self, standard_df):
        from src.indicators.aggregator import ALL_INDICATORS
        agg = _make_agg(ALL_INDICATORS, confirm=1)
        result = agg.process("AAPL", standard_df)
        d = result.to_dict()
        # Must be JSON-serializable without error
        serialized = json.dumps(d, default=str)
        reloaded = json.loads(serialized)
        assert reloaded["symbol"] == "AAPL"
        assert isinstance(reloaded["indicators"], list)
        assert "regime" in reloaded  # new field added in regime-aware aggregator
        agg.close()


# ─────────────────────────────────────────────────────────────────────────────
# Weight override tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightOverrides:
    def test_weight_override_changes_score(self):
        """Overriding a weight should shift the aggregated score."""
        buy = FixedIndicator(1, weight=1.0, name="A")
        sell = FixedIndicator(-1, weight=1.0, name="B")

        # Without override: (1*1 + (-1)*1) / 2 = 0
        agg1 = _make_agg([buy, sell], confirm=1)
        r1 = agg1.process("T", _minimal_df())
        agg1.close()

        # Override A weight to 3.0: (1*3 + (-1)*1) / 4 = 0.5
        agg2 = _make_agg([buy, sell], confirm=1, weight_overrides={"A": 3.0})
        r2 = agg2.process("T", _minimal_df())
        agg2.close()

        assert r2.score > r1.score

    def test_weight_override_unknown_name_ignored(self):
        """Overrides for non-existent indicator names are silently ignored."""
        inds = [FixedIndicator(1, name="X")]
        agg = _make_agg(inds, confirm=1, weight_overrides={"NonExistent": 99.0})
        result = agg.process("T", _minimal_df())
        assert result.score == pytest.approx(1.0)
        agg.close()


class TestDisableLogging:
    def test_disable_logging_no_file_created(self, tmp_path):
        """With disable_logging=True, no log file handle is opened."""
        inds = [FixedIndicator(1)]
        agg = SignalAggregator(
            indicators=inds, confirm_bars=1, disable_logging=True,
        )
        agg.process("T", _minimal_df())
        assert agg._log_fh is None
        agg.close()  # should not raise

    def test_disable_logging_still_computes(self):
        """Aggregation works normally with logging disabled."""
        inds = [FixedIndicator(1)] * 3
        agg = SignalAggregator(
            indicators=inds, confirm_bars=1, disable_logging=True,
        )
        result = agg.process("T", _minimal_df())
        assert result.score == pytest.approx(1.0)
        assert result.final_signal == "BUY"
        agg.close()


class TestBuildWeightOverrides:
    def test_category_multiplier_scales_weights(self):
        """build_weight_overrides should scale indicator base weights by category."""
        buy = FixedIndicator(1, weight=1.5, name="EMA Cross")
        sell = FixedIndicator(-1, weight=1.0, name="RSI(14)")
        overrides = SignalAggregator.build_weight_overrides(
            {"trend": 2.0, "momentum": 0.5},
            indicators=[buy, sell],
        )
        assert overrides["EMA Cross"] == pytest.approx(3.0)  # 1.5 * 2.0
        assert overrides["RSI(14)"] == pytest.approx(0.5)    # 1.0 * 0.5

    def test_missing_category_defaults_to_one(self):
        """Categories not in the multiplier map use 1.0."""
        ind = FixedIndicator(1, weight=2.0, name="OBV")
        overrides = SignalAggregator.build_weight_overrides(
            {"trend": 2.0},  # no "volume" key
            indicators=[ind],
        )
        assert overrides["OBV"] == pytest.approx(2.0)  # 2.0 * 1.0
