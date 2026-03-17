"""Walk-forward optimizer for the factor-based pipeline.

Grid-searches over entry/exit thresholds and max holding period,
using rolling train/test windows to avoid overfitting.

Parameter grid (default, 4×3×4 = 48 combinations):
  buy_threshold:        [0.15, 0.20, 0.25, 0.30]
  exit_long_threshold:  [-0.10, -0.05, 0.0]
  max_holding_bars:     [80, 120, 160, 200]

sell_threshold = -buy_threshold
exit_short_threshold = -exit_long_threshold

IC-based factor weights are recomputed per window (use_ic_weights=True).
External data (sentiment/macro/fundamental) is fetched once for the full
date range and sliced per window to avoid redundant API calls.
"""
from __future__ import annotations

import itertools
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.engine import WARMUP_BARS
from src.backtest.factor_engine import FactorBacktestEngine
from src.backtest.metrics import compute_metrics, Metrics
from src.backtest.optimizer import (
    PENALTY,
    OptimizationResult,
    WindowResult,
    objective_sharpe,
    objective_sharpe_trades,
    OBJECTIVES,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FACTOR_GRID: Dict[str, List] = {
    "buy_threshold":       [0.15, 0.20, 0.25, 0.30],
    "exit_long_threshold": [-0.10, -0.05, 0.0],
    "max_holding_bars":    [80, 120, 160, 200],
}


def build_factor_grid(grid: Optional[Dict[str, List]] = None) -> List[Dict[str, Any]]:
    g = grid or DEFAULT_FACTOR_GRID
    keys = sorted(g.keys())
    combos = list(itertools.product(*(g[k] for k in keys)))
    return [dict(zip(keys, vals)) for vals in combos]


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class FactorWalkForwardOptimizer:
    """Rolling walk-forward optimizer for the factor pipeline.

    Parameters
    ----------
    symbol : str
    data : pd.DataFrame
        Full OHLCV with DatetimeIndex.
    train_months : int
        Training window length (default 6).
    test_months : int
        Out-of-sample test window length (default 2).
    step_months : int
        Rolling step size (defaults to test_months).
    objective : str
        "sharpe" or "sharpe_trades".
    grid : dict, optional
        Custom parameter grid.
    initial_capital : float
    slippage_bps : float
    min_trades : int
        Minimum trades for a valid result (default 3).
    use_external_data : bool
        Fetch external data once for the full range (default True).
    use_ic_weights : bool
        Recompute IC weights per window (default True).
    """

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        train_months: int = 6,
        test_months: int = 2,
        step_months: Optional[int] = None,
        objective: str = "sharpe",
        grid: Optional[Dict[str, List]] = None,
        initial_capital: float = 100_000.0,
        slippage_bps: float = 5.0,
        min_trades: int = 3,
        use_external_data: bool = True,
        use_ic_weights: bool = True,
    ) -> None:
        self.symbol = symbol
        self.data = data
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months or test_months
        self.objective_name = objective
        self.objective_fn = OBJECTIVES.get(objective, objective_sharpe)
        self.param_grid = build_factor_grid(grid)
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.min_trades = min_trades
        self.use_external_data = use_external_data
        self.use_ic_weights = use_ic_weights

    def run(self) -> OptimizationResult:
        """Execute the full walk-forward optimization."""
        windows = self._make_windows()
        if not windows:
            raise ValueError(
                f"Insufficient data for walk-forward: need at least "
                f"{self.train_months + self.test_months} months."
            )

        # Load external data ONCE for the full date range
        external_lookup: dict = {}
        if self.use_external_data:
            try:
                from src.data.external_loader import ExternalDataLoader
                start_str = self.data.index[0].strftime("%Y-%m-%d")
                end_str = self.data.index[-1].strftime("%Y-%m-%d")
                external_lookup = ExternalDataLoader().load_aligned(
                    self.symbol, start_str, end_str
                )
            except Exception as exc:
                logger.warning("External data load failed: %s", exc)

        logger.info(
            "Factor Walk-Forward: %d windows | %d param combos | IC=%s",
            len(windows), len(self.param_grid), self.use_ic_weights,
        )

        window_results: List[WindowResult] = []
        for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            logger.info(
                "Window %d/%d: train %s→%s  test %s→%s",
                i + 1, len(windows),
                tr_s.date(), tr_e.date(), te_s.date(), te_e.date(),
            )
            train_data = self.data.loc[tr_s:tr_e]
            test_data = self.data.loc[te_s:te_e]

            wr = self._optimize_window(train_data, test_data, i, external_lookup)
            window_results.append(wr)

            logger.info(
                "  IS  Sharpe=%.3f ret=%+.2f%% trades=%d"
                "  OOS Sharpe=%.3f ret=%+.2f%% trades=%d",
                wr.is_sharpe, wr.is_return_pct, wr.is_trades,
                wr.oos_sharpe, wr.oos_return_pct, wr.oos_trades,
            )

        # Aggregate
        valid = [w for w in window_results if w.oos_objective > PENALTY]
        oos_sharpes = [w.oos_sharpe for w in valid]
        oos_returns = [w.oos_return_pct for w in valid]
        oos_trades = sum(w.oos_trades for w in window_results)

        oos_equities = [
            w.oos_equity for w in window_results
            if w.oos_equity is not None and len(w.oos_equity) > 0
        ]
        combined_equity = pd.concat(oos_equities) if oos_equities else None
        stable = self._find_stable_params(window_results)

        return OptimizationResult(
            symbol=self.symbol,
            objective=self.objective_name,
            total_windows=len(windows),
            windows=window_results,
            agg_oos_sharpe=float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
            agg_oos_return_pct=float(np.mean(oos_returns)) if oos_returns else 0.0,
            agg_oos_trades=oos_trades,
            stable_params=stable,
            oos_equity=combined_equity,
            param_grid_size=len(self.param_grid),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_windows(
        self,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        start = self.data.index[0]
        end = self.data.index[-1]
        windows = []
        cursor = start
        while True:
            train_start = cursor
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            if test_end > end:
                break
            windows.append((train_start, train_end, test_start, test_end))
            cursor += pd.DateOffset(months=self.step_months)
        return windows

    def _run_single(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        external_lookup: dict,
    ) -> Tuple[Dict[str, Any], Metrics]:
        """Run one backtest with given params on given data slice."""
        buy_thr = params["buy_threshold"]
        exit_long = params["exit_long_threshold"]
        engine = FactorBacktestEngine(
            symbol=self.symbol,
            data=data,
            initial_capital=self.initial_capital,
            slippage_bps=self.slippage_bps,
            buy_threshold=buy_thr,
            sell_threshold=-buy_thr,
            exit_long_threshold=exit_long,
            exit_short_threshold=-exit_long,
            max_holding_bars=params["max_holding_bars"],
            use_ic_weights=self.use_ic_weights,
            use_external_data=False,
            external_lookup=external_lookup,
        )
        try:
            result = engine.run()
        except ValueError:
            return params, Metrics()

        metrics = compute_metrics(
            trades=result.trades,
            equity_curve=result.equity_curve,
            initial_capital=self.initial_capital,
            cash_flows=engine.portfolio.cash_flows,
        )
        return params, metrics

    def _optimize_window(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        window_idx: int,
        external_lookup: dict,
    ) -> WindowResult:
        if len(train_data) < WARMUP_BARS + 10:
            logger.warning("Window %d: train data too short (%d bars)", window_idx, len(train_data))
            return WindowResult(
                window_idx=window_idx,
                train_start="", train_end="", test_start="", test_end="",
                best_params=self.param_grid[0],
            )

        best_obj = PENALTY
        best_params = self.param_grid[0]
        best_metrics = Metrics()

        for params in self.param_grid:
            _, metrics = self._run_single(train_data, params, external_lookup)
            obj = self.objective_fn(metrics, self.min_trades)
            if obj > best_obj:
                best_obj = obj
                best_params = params
                best_metrics = metrics

        # Evaluate best params on OOS test set
        oos_metrics = Metrics()
        oos_equity = None
        if len(test_data) > WARMUP_BARS + 10:
            _, oos_metrics = self._run_single(test_data, best_params, external_lookup)
            # Re-run to get equity curve
            buy_thr = best_params["buy_threshold"]
            exit_long = best_params["exit_long_threshold"]
            oos_engine = FactorBacktestEngine(
                symbol=self.symbol,
                data=test_data,
                initial_capital=self.initial_capital,
                slippage_bps=self.slippage_bps,
                buy_threshold=buy_thr,
                sell_threshold=-buy_thr,
                exit_long_threshold=exit_long,
                exit_short_threshold=-exit_long,
                max_holding_bars=best_params["max_holding_bars"],
                use_ic_weights=self.use_ic_weights,
                use_external_data=False,
                external_lookup=external_lookup,
            )
            try:
                oos_result = oos_engine.run()
                oos_equity = oos_result.equity_curve
                oos_metrics = compute_metrics(
                    trades=oos_result.trades,
                    equity_curve=oos_result.equity_curve,
                    initial_capital=self.initial_capital,
                    cash_flows=oos_engine.portfolio.cash_flows,
                )
            except ValueError:
                pass

        oos_obj = self.objective_fn(oos_metrics, self.min_trades)

        return WindowResult(
            window_idx=window_idx,
            train_start=str(train_data.index[0]) if len(train_data) else "",
            train_end=str(train_data.index[-1]) if len(train_data) else "",
            test_start=str(test_data.index[0]) if len(test_data) else "",
            test_end=str(test_data.index[-1]) if len(test_data) else "",
            best_params=best_params,
            is_sharpe=best_metrics.sharpe_ratio,
            is_return_pct=best_metrics.total_return_pct,
            is_trades=best_metrics.total_trades,
            is_objective=best_obj,
            oos_sharpe=oos_metrics.sharpe_ratio,
            oos_return_pct=oos_metrics.total_return_pct,
            oos_trades=oos_metrics.total_trades,
            oos_objective=oos_obj,
            oos_equity=oos_equity,
        )

    @staticmethod
    def _find_stable_params(windows: List[WindowResult]) -> Dict[str, Any]:
        if not windows:
            return {}
        all_keys = windows[0].best_params.keys()
        stable = {}
        for key in all_keys:
            values = [w.best_params[key] for w in windows]
            stable[key] = Counter(values).most_common(1)[0][0]
        return stable
