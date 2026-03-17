"""
Signal aggregator: runs all active indicators, computes a regime-adjusted
weighted score, applies anti-flicker confirmation, and emits BUY/SELL/HOLD.

Key design changes vs. original:
  - Reduced from 22 → 11 indicators (removed high-correlation duplicates).
  - Added RegimeDetector: dynamically re-weights indicator categories and
    adjusts signal thresholds based on market state (trending / ranging /
    choppy / volatile).
  - Fixed RSI and CCI to use only extreme-zone signals (no midline).
  - Fixed PivotPoints to use real prior-day OHLC from the DatetimeIndex.
  - CONFIRM_BARS default raised to 5 (75-minute window on 15-min bars).
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import pandas as pd

from src.config import cfg
from src.indicators.base import BaseIndicator, SignalResult
from src.indicators.trend import EMAcross, MACD, ADXSignal
from src.indicators.momentum import RSI, StochKDJ, CCI, ROC
from src.indicators.volatility import BollingerBands
from src.indicators.volume import OBVTrend, VWAPSignal
from src.indicators.support_resistance import PivotPoints
from src.indicators.regime import RegimeDetector

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Active indicator set (11 — de-correlated across five factor categories)
# ─────────────────────────────────────────────────────────────────────────────

ALL_INDICATORS: List[BaseIndicator] = [
    # Trend (3) — kept as the most complementary trio; SMAcross/SAR/Supertrend
    # removed due to >0.85 correlation with EMAcross + MACD
    EMAcross(),       # multi-EMA stack alignment          weight=1.5
    MACD(),           # EMA-difference oscillator          weight=1.5
    ADXSignal(),      # trend-strength filter + DI cross   weight=1.2

    # Momentum (4) — overbought/oversold only; midline logic removed from RSI/CCI
    RSI(),            # extreme-zone mean-reversion        weight=1.2
    StochKDJ(),       # %K/%D zone + cross                 weight=1.0
    CCI(),            # ±100 extreme zones                 weight=0.8
    ROC(),            # raw price rate-of-change           weight=0.8

    # Volatility (1) — Keltner/Donchian removed (redundant with BB)
    BollingerBands(), # price vs. band position            weight=1.2

    # Volume (2) — MFI/CMF removed (redundant with OBV)
    OBVTrend(),       # on-balance volume slope            weight=1.0
    VWAPSignal(),     # intraday VWAP anchor               weight=1.2

    # Support / Resistance (1) — uses real prior-day OHLC via DatetimeIndex
    PivotPoints(),    # classic session pivot levels       weight=1.0
]


# ─────────────────────────────────────────────────────────────────────────────
# Regime-based dynamic weight configuration
# ─────────────────────────────────────────────────────────────────────────────

# Maps each indicator's display name to its factor category
INDICATOR_CATEGORY: Dict[str, str] = {
    "EMA Cross":    "trend",
    "MACD":         "trend",
    "ADX+DI":       "trend",
    "RSI(14)":      "momentum",
    "KDJ/Stoch":    "momentum",
    "CCI(20)":      "momentum",
    "ROC(12)":      "momentum",
    "Bollinger":    "volatility",
    "OBV":          "volume",
    "VWAP":         "volume",
    "Pivot Points": "sr",
}

# Category weight multipliers per regime.
# Values < 1 down-weight a category; > 1 amplify it.
REGIME_WEIGHT_MULT: Dict[str, Dict[str, float]] = {
    # Clean directional move — amplify trend signals, reduce mean-reversion noise
    "TRENDING_CALM":     {"trend": 1.3, "momentum": 0.8, "volatility": 1.0, "volume": 1.0, "sr": 1.0},
    # Directional but noisy — keep trend lead, tighten via threshold multiplier
    "TRENDING_VOLATILE": {"trend": 1.5, "momentum": 0.5, "volatility": 1.0, "volume": 1.0, "sr": 0.7},
    # Sideways / range-bound — mean-reversion indicators take the lead
    "RANGING":           {"trend": 0.5, "momentum": 1.5, "volatility": 1.0, "volume": 1.2, "sr": 1.2},
    # Choppy + extreme volatility — raise threshold dramatically; avoid most signals
    "CHOPPY_VOLATILE":   {"trend": 0.3, "momentum": 0.4, "volatility": 1.3, "volume": 0.8, "sr": 0.5},
    # Insufficient data to classify
    "UNKNOWN":           {"trend": 1.0, "momentum": 1.0, "volatility": 1.0, "volume": 1.0, "sr": 1.0},
}

# Score threshold multiplier per regime.
# Applied to both buy_threshold and sell_threshold symmetrically.
REGIME_THRESHOLD_MULT: Dict[str, float] = {
    "TRENDING_CALM":     1.0,    # baseline — use configured thresholds as-is
    "TRENDING_VOLATILE": 1.3,    # more selective in noisy trends
    "RANGING":           0.85,   # slightly easier to fire in clear ranging markets
    "CHOPPY_VOLATILE":   2.0,    # very hard to fire — almost always HOLD
    "UNKNOWN":           1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndicatorSnapshot:
    name: str
    signal: int        # +1, 0, -1
    value: float
    label: str
    weight: float


@dataclass
class AggregateResult:
    symbol: str
    timestamp: str
    regime: str                             # current market regime label
    score: float                            # regime-adjusted weighted average ∈ [-1, +1]
    final_signal: str                       # BUY / SELL / HOLD
    confirmed: bool                         # passed anti-flicker check
    indicators: List[IndicatorSnapshot]
    buy_count: int
    sell_count: int
    hold_count: int
    price: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["indicators"] = [asdict(i) for i in self.indicators]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class SignalAggregator:
    """
    Runs all active indicators on a DataFrame, produces a regime-adjusted
    weighted score, and applies anti-flicker confirmation.
    """

    def __init__(
        self,
        indicators: Optional[List[BaseIndicator]] = None,
        buy_threshold: float = None,
        sell_threshold: float = None,
        confirm_bars: int = None,
        log_file: str = None,
        weight_overrides: Optional[Dict[str, float]] = None,
        regime_weight_mult: Optional[Dict[str, Dict[str, float]]] = None,
        regime_threshold_mult: Optional[Dict[str, float]] = None,
        disable_logging: bool = False,
    ) -> None:
        self._indicators = indicators if indicators is not None else ALL_INDICATORS
        self._buy_th = buy_threshold if buy_threshold is not None else cfg.buy_threshold
        self._sell_th = sell_threshold if sell_threshold is not None else cfg.sell_threshold
        self._confirm = confirm_bars if confirm_bars is not None else cfg.confirm_bars

        # Per-indicator weight overrides (keyed by indicator display name)
        self._weight_overrides: Dict[str, float] = weight_overrides or {}

        # Custom regime multiplier tables (fall back to module-level constants)
        self._regime_weight_mult = regime_weight_mult if regime_weight_mult is not None else REGIME_WEIGHT_MULT
        self._regime_threshold_mult = regime_threshold_mult if regime_threshold_mult is not None else REGIME_THRESHOLD_MULT

        # Regime detector (reused across calls)
        self._regime = RegimeDetector()

        # Per-symbol: deque of recent raw signals for anti-flicker
        self._history: Dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self._confirm)
        )

        # Logging — skip entirely when disable_logging is True (e.g. during optimization)
        self._disable_logging = disable_logging
        self._log_fh = None

        if not disable_logging:
            log_path = log_file if log_file is not None else cfg.log_file
            if log_file is None:
                # Default config path: resolve relative to project root and validate
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                resolved = os.path.realpath(os.path.join(project_root, log_path))
                if not resolved.startswith(project_root):
                    raise ValueError(f"Log path escapes project directory: {log_path}")
            else:
                resolved = os.path.abspath(log_path)
            os.makedirs(os.path.dirname(resolved) if os.path.dirname(resolved) else ".", exist_ok=True)
            self._log_fh = open(resolved, "a", buffering=1)  # line-buffered

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, symbol: str, df: pd.DataFrame) -> AggregateResult:
        """
        Detect regime, run all indicators with regime-adjusted weights,
        aggregate score, apply anti-flicker, log, and return AggregateResult.
        """
        # 1. Market regime
        regime = self._regime.detect(df)
        weight_mult_map = self._regime_weight_mult.get(regime, self._regime_weight_mult.get("UNKNOWN", {}))
        threshold_mult = self._regime_threshold_mult.get(regime, 1.0)

        # 2. Run indicators
        snapshots: List[IndicatorSnapshot] = []
        weighted_sum = 0.0
        total_weight = 0.0
        buy_count = sell_count = hold_count = 0

        for ind in self._indicators:
            result: SignalResult = ind.compute(df)
            snapshots.append(
                IndicatorSnapshot(
                    name=ind.name,
                    signal=result.signal,
                    value=result.value,
                    label=result.label,
                    weight=ind.weight,
                )
            )
            # Apply per-indicator weight override, then regime-based category multiplier
            base_weight = self._weight_overrides.get(ind.name, ind.weight)
            category = INDICATOR_CATEGORY.get(ind.name, "momentum")
            effective_weight = base_weight * weight_mult_map.get(category, 1.0)
            weighted_sum += result.signal * effective_weight
            total_weight += effective_weight

            if result.signal == 1:
                buy_count += 1
            elif result.signal == -1:
                sell_count += 1
            else:
                hold_count += 1

        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        raw_signal = self._classify(score, threshold_mult)

        # 3. Anti-flicker: require confirm_bars consecutive consistent signals
        history = self._history[symbol]
        history.append(raw_signal)
        confirmed = len(history) == self._confirm and all(s == raw_signal for s in history)

        price = float(df["close"].iloc[-1]) if len(df) else float("nan")

        result_obj = AggregateResult(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            regime=regime,
            score=round(score, 4),
            final_signal=raw_signal if confirmed else "HOLD",
            confirmed=confirmed,
            indicators=snapshots,
            buy_count=buy_count,
            sell_count=sell_count,
            hold_count=hold_count,
            price=price,
        )

        if confirmed:
            self._log(result_obj)

        return result_obj

    def close(self) -> None:
        """Flush and close the log file."""
        if self._log_fh is None:
            return
        try:
            self._log_fh.flush()
            os.fsync(self._log_fh.fileno())
            self._log_fh.close()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify(self, score: float, threshold_mult: float = 1.0) -> str:
        buy_th = self._buy_th * threshold_mult
        sell_th = self._sell_th * threshold_mult
        if score > buy_th:
            return "BUY"
        elif score < sell_th:
            return "SELL"
        else:
            return "HOLD"

    def _log(self, result: AggregateResult) -> None:
        if self._log_fh is None:
            return
        try:
            line = json.dumps(result.to_dict(), default=str)
            self._log_fh.write(line + "\n")
        except Exception as exc:
            logger.warning("Failed to write signal log: %s", exc)

    @staticmethod
    def build_weight_overrides(
        category_multipliers: Dict[str, float],
        indicators: Optional[List[BaseIndicator]] = None,
    ) -> Dict[str, float]:
        """
        Build per-indicator weight overrides by scaling base weights with
        category multipliers.  Reduces the 11-indicator search space to
        3-5 category dimensions.

        Parameters
        ----------
        category_multipliers : dict
            Mapping of category name (trend/momentum/volatility/volume/sr)
            to a scaling factor applied to all indicators in that category.
        indicators : list, optional
            Indicator list to use.  Defaults to ALL_INDICATORS.

        Returns
        -------
        Dict[str, float]
            Mapping of indicator display name → adjusted weight.
        """
        inds = indicators if indicators is not None else ALL_INDICATORS
        overrides: Dict[str, float] = {}
        for ind in inds:
            cat = INDICATOR_CATEGORY.get(ind.name, "momentum")
            mult = category_multipliers.get(cat, 1.0)
            overrides[ind.name] = ind.weight * mult
        return overrides
