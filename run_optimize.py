"""
CLI entry point for Walk-Forward parameter optimization.

Usage:
    # From IBKR (recommended — auto-downloads and caches as Parquet)
    .venv/bin/python run_optimize.py --symbol AAPL --download
    .venv/bin/python run_optimize.py --symbol AAPL --download --duration "2 Y" --bar-size "15 min"

    # From local file
    .venv/bin/python run_optimize.py --symbol AAPL --data data/AAPL_15min.csv

    # Custom optimization params
    .venv/bin/python run_optimize.py --symbol AAPL --download \
        --train-months 6 --test-months 2 --objective sharpe_trades
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("optimize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IBKR Signal Assistant — Walk-Forward Optimizer")
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL)")

    # Data source: either --data or --download
    data_grp = p.add_mutually_exclusive_group(required=True)
    data_grp.add_argument("--data", help="Path to CSV or Parquet file")
    data_grp.add_argument("--download", action="store_true",
                          help="Download from IBKR (auto-cached as Parquet in data/)")

    # IBKR download options
    p.add_argument("--duration", default="2 Y",
                   help="IBKR duration string (default: '2 Y')")
    p.add_argument("--bar-size", default="15 min",
                   help="Bar size: '1 min', '5 min', '15 min', '1 hour', '1 day' (default: '15 min')")

    # Walk-Forward options
    p.add_argument("--train-months", type=int, default=6, help="Training window months (default: 6)")
    p.add_argument("--test-months", type=int, default=2, help="Test window months (default: 2)")
    p.add_argument("--step-months", type=int, default=None, help="Step size months (default: same as test)")
    p.add_argument("--objective", default="sharpe", choices=["sharpe", "sharpe_trades"],
                    help="Objective function (default: sharpe)")
    p.add_argument("--capital", type=float, default=100_000, help="Initial capital (default: 100000)")
    p.add_argument("--position-size", type=int, default=100, help="Shares per trade (default: 100)")
    p.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps (default: 5)")
    p.add_argument("--workers", type=int, default=None, help="Max parallel workers (default: CPU-1)")
    p.add_argument("--min-trades", type=int, default=5, help="Min trades for valid params (default: 5)")
    p.add_argument("--output", default=None, help="Output HTML report path")
    return p.parse_args()


async def run_async(args: argparse.Namespace) -> None:
    from src.backtest.data_loader import from_csv, from_parquet, from_ibkr
    from src.backtest.optimizer import WalkForwardOptimizer
    from src.backtest.optimization_report import generate_optimization_report

    # Load data
    if args.download:
        data = await from_ibkr(args.symbol, duration=args.duration, bar_size=args.bar_size)
    elif args.data:
        if args.data.endswith(".parquet"):
            data = from_parquet(args.data)
        else:
            data = from_csv(args.data)

    logger.info("Data loaded: %d bars from %s to %s", len(data), data.index[0], data.index[-1])

    # Run optimizer
    optimizer = WalkForwardOptimizer(
        symbol=args.symbol,
        data=data,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        objective=args.objective,
        initial_capital=args.capital,
        position_size=args.position_size,
        slippage_bps=args.slippage,
        max_workers=args.workers,
        min_trades=args.min_trades,
    )
    result = optimizer.run()

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {args.symbol} Walk-Forward Optimization Results")
    print(f"{'='*60}")
    print(f"  Windows:          {result.total_windows}")
    print(f"  Grid Size:        {result.param_grid_size}")
    print(f"  Avg OOS Sharpe:   {result.agg_oos_sharpe:.3f}")
    print(f"  Avg OOS Return:   {result.agg_oos_return_pct:+.2f}%")
    print(f"  Total OOS Trades: {result.agg_oos_trades}")
    print(f"\n  Most Stable Parameters:")
    for k, v in result.stable_params.items():
        print(f"    {k:25s} = {v}")
    print(f"{'='*60}\n")

    # Generate report
    output = args.output or f"optimization_report_{args.symbol}.html"
    report_path = generate_optimization_report(result, output_path=output)
    print(f"Report saved to: {report_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
