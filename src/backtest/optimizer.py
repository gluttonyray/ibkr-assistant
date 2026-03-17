"""
Walk-Forward parameter optimizer.

Grid-searches over signal thresholds, confirmation bars, and category
weight multipliers, using rolling train/test windows to avoid overfitting.

Usage (programmatic):
    from src.backtest.optimizer import WalkForwardOptimizer
    opt = WalkForwardOptimizer(symbol="AAPL", data=df)
    results = opt.run()

See also: run_optimize.py for CLI usage.
"""
from __future__ import annotations

import itertools
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult, WARMUP_BARS
from src.backtest.metrics import compute_metrics, Metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid definition
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_GRID: Dict[str, List] = {
    "buy_threshold":        [0.30, 0.35, 0.40, 0.45, 0.50],
    "confirm_bars":         [3, 5, 7],
    "trend_weight_mult":    [0.8, 1.0, 1.5],
    "momentum_weight_mult": [0.5, 1.0, 1.2],
    "volume_weight_mult":   [0.8, 1.0, 1.2],
}


def build_param_grid(grid: Optional[Dict[str, List]] = None) -> List[Dict[str, Any]]:
    """Expand a grid dict into a list of parameter combinations."""
    g = grid or DEFAULT_GRID
    keys = sorted(g.keys())
    combos = list(itertools.product(*(g[k] for k in keys)))
    return [dict(zip(keys, vals)) for vals in combos]


def params_to_aggregator_kwargs(params: Dict[str, Any]) -> dict:
    """Convert optimizer params into kwargs for SignalAggregator."""
    from src.indicators.aggregator import SignalAggregator

    category_mults = {
        "trend":      params.get("trend_weight_mult", 1.0),
        "momentum":   params.get("momentum_weight_mult", 1.0),
        "volume":     params.get("volume_weight_mult", 1.0),
        "volatility": 1.0,
        "sr":         1.0,
    }
    weight_overrides = SignalAggregator.build_weight_overrides(category_mults)

    return {
        "buy_threshold":  params.get("buy_threshold", 0.40),
        "sell_threshold": -params.get("buy_threshold", 0.40),
        "confirm_bars":   params.get("confirm_bars", 5),
        "weight_overrides": weight_overrides,
        "disable_logging": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Objective functions
# ─────────────────────────────────────────────────────────────────────────────

PENALTY = -999.0


def objective_sharpe(metrics: Metrics, min_trades: int = 5) -> float:
    """Maximize Sharpe ratio; penalize if too few trades."""
    if metrics.total_trades < min_trades:
        return PENALTY
    return metrics.sharpe_ratio


def objective_sharpe_trades(metrics: Metrics, min_trades: int = 5) -> float:
    """Sharpe * sqrt(trades) — balances quality and frequency."""
    if metrics.total_trades < min_trades:
        return PENALTY
    return metrics.sharpe_ratio * math.sqrt(metrics.total_trades)


OBJECTIVES = {
    "sharpe": objective_sharpe,
    "sharpe_trades": objective_sharpe_trades,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    # In-sample metrics
    is_sharpe: float = 0.0
    is_return_pct: float = 0.0
    is_trades: int = 0
    is_objective: float = 0.0
    # Out-of-sample metrics
    oos_sharpe: float = 0.0
    oos_return_pct: float = 0.0
    oos_trades: int = 0
    oos_objective: float = 0.0
    # OOS equity curve (for concatenation)
    oos_equity: Optional[pd.Series] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("oos_equity", None)
        return d


@dataclass
class OptimizationResult:
    """Aggregated walk-forward optimization output."""
    symbol: str
    objective: str
    total_windows: int
    windows: List[WindowResult]
    # Aggregate OOS metrics
    agg_oos_sharpe: float = 0.0
    agg_oos_return_pct: float = 0.0
    agg_oos_trades: int = 0
    # Most stable params (most frequently selected)
    stable_params: Dict[str, Any] = field(default_factory=dict)
    # Full OOS equity curve (concatenated)
    oos_equity: Optional[pd.Series] = field(default=None, repr=False)
    param_grid_size: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Single backtest runner (picklable for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _run_single(
    symbol: str,
    data: pd.DataFrame,
    params: Dict[str, Any],
    initial_capital: float,
    position_size: int,
    slippage_bps: float,
) -> Tuple[Dict[str, Any], float, Metrics]:
    """
    Run one backtest with given params.  Returns (params, objective_value, metrics).
    Designed to be called in a subprocess.
    """
    agg_kw = params_to_aggregator_kwargs(params)
    engine = BacktestEngine(
        symbol=symbol,
        data=data,
        initial_capital=initial_capital,
        position_size=position_size,
        slippage_bps=slippage_bps,
        aggregator_kwargs=agg_kw,
    )
    try:
        result = engine.run()
    except ValueError:
        # Insufficient data
        return params, PENALTY, Metrics()

    metrics = compute_metrics(
        trades=result.trades,
        equity_curve=result.equity_curve,
        initial_capital=initial_capital,
    )
    return params, 0.0, metrics  # objective computed by caller


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardOptimizer:
    """
    Rolling walk-forward optimizer.

    Parameters
    ----------
    symbol : str
    data : pd.DataFrame
        Full OHLCV with DatetimeIndex.
    train_months : int
        Training window length.
    test_months : int
        Out-of-sample test window length.
    step_months : int
        Step size for rolling windows (defaults to test_months).
    objective : str
        Objective function name ('sharpe' or 'sharpe_trades').
    grid : dict, optional
        Custom parameter grid.
    initial_capital : float
    position_size : int
    slippage_bps : float
    max_workers : int, optional
        Number of parallel processes (defaults to CPU count - 1).
    min_trades : int
        Minimum trades for a valid parameter set.
    """

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        train_months: int = 6,
        test_months: int = 2,
        step_months: int = None,
        objective: str = "sharpe",
        grid: Optional[Dict[str, List]] = None,
        initial_capital: float = 100_000.0,
        position_size: int = 100,
        slippage_bps: float = 5.0,
        max_workers: int = None,
        min_trades: int = 5,
    ) -> None:
        self.symbol = symbol
        self.data = data
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months or test_months
        self.objective_name = objective
        self.objective_fn = OBJECTIVES.get(objective, objective_sharpe)
        self.param_grid = build_param_grid(grid)
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.slippage_bps = slippage_bps
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
        self.min_trades = min_trades

    def _make_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
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

    def _slice_data(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Slice data for a window, inclusive of start, exclusive of end."""
        return self.data.loc[start:end].iloc[:-1] if len(self.data.loc[start:end]) > 0 else self.data.loc[start:end]

    def _optimize_window(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        window_idx: int,
    ) -> WindowResult:
        """Run grid search on train_data, evaluate best on test_data."""
        best_obj = PENALTY
        best_params = self.param_grid[0] if self.param_grid else {}
        best_metrics = Metrics()

        # Grid search over training set (parallel)
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = []
            for params in self.param_grid:
                f = pool.submit(
                    _run_single,
                    self.symbol,
                    train_data,
                    params,
                    self.initial_capital,
                    self.position_size,
                    self.slippage_bps,
                )
                futures.append((params, f))

            for params, f in futures:
                try:
                    _, _, metrics = f.result(timeout=300)
                    obj = self.objective_fn(metrics, self.min_trades)
                    results.append((params, obj, metrics))
                except Exception as exc:
                    logger.warning("Window %d: param set failed: %s", window_idx, exc)
                    results.append((params, PENALTY, Metrics()))

        # Find best
        for params, obj, metrics in results:
            if obj > best_obj:
                best_obj = obj
                best_params = params
                best_metrics = metrics

        # Evaluate best params on OOS test set
        oos_metrics = Metrics()
        oos_equity = None
        if len(test_data) > WARMUP_BARS + 10:
            agg_kw = params_to_aggregator_kwargs(best_params)
            engine = BacktestEngine(
                symbol=self.symbol,
                data=test_data,
                initial_capital=self.initial_capital,
                position_size=self.position_size,
                slippage_bps=self.slippage_bps,
                aggregator_kwargs=agg_kw,
            )
            try:
                oos_result = engine.run()
                oos_metrics = compute_metrics(
                    trades=oos_result.trades,
                    equity_curve=oos_result.equity_curve,
                    initial_capital=self.initial_capital,
                )
                oos_equity = oos_result.equity_curve
            except ValueError:
                pass

        oos_obj = self.objective_fn(oos_metrics, self.min_trades)

        return WindowResult(
            window_idx=window_idx,
            train_start=str(train_data.index[0]),
            train_end=str(train_data.index[-1]),
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

    def run(self) -> OptimizationResult:
        """Execute the full walk-forward optimization."""
        windows = self._make_windows()
        if not windows:
            raise ValueError(
                f"Insufficient data for walk-forward: need at least "
                f"{self.train_months + self.test_months} months, "
                f"have {(self.data.index[-1] - self.data.index[0]).days / 30:.0f} months."
            )

        logger.info(
            "Walk-Forward: %d windows | %d param combos | %d workers",
            len(windows), len(self.param_grid), self.max_workers,
        )

        window_results: List[WindowResult] = []
        for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            logger.info("Window %d/%d: train %s→%s, test %s→%s", i + 1, len(windows), tr_s.date(), tr_e.date(), te_s.date(), te_e.date())

            train_data = self.data.loc[tr_s:tr_e]
            test_data = self.data.loc[te_s:te_e]

            wr = self._optimize_window(train_data, test_data, i)
            window_results.append(wr)

            logger.info(
                "  IS Sharpe=%.3f ret=%.2f%% trades=%d | OOS Sharpe=%.3f ret=%.2f%% trades=%d",
                wr.is_sharpe, wr.is_return_pct, wr.is_trades,
                wr.oos_sharpe, wr.oos_return_pct, wr.oos_trades,
            )

        # Aggregate OOS results
        oos_sharpes = [w.oos_sharpe for w in window_results if w.oos_objective > PENALTY]
        oos_returns = [w.oos_return_pct for w in window_results if w.oos_objective > PENALTY]
        oos_trades = sum(w.oos_trades for w in window_results)

        # Concatenate OOS equity curves
        oos_equities = [w.oos_equity for w in window_results if w.oos_equity is not None and len(w.oos_equity) > 0]
        combined_equity = pd.concat(oos_equities) if oos_equities else None

        # Find most stable params (mode of each param across windows)
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

    @staticmethod
    def _find_stable_params(windows: List[WindowResult]) -> Dict[str, Any]:
        """Find the most frequently selected value for each parameter."""
        from collections import Counter
        if not windows:
            return {}

        all_keys = windows[0].best_params.keys()
        stable = {}
        for key in all_keys:
            values = [w.best_params[key] for w in windows]
            counter = Counter(values)
            stable[key] = counter.most_common(1)[0][0]
        return stable
