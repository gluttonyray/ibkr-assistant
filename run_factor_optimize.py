"""CLI entry point for factor walk-forward optimization.

Usage::

    # From local CSV file
    .venv/bin/python run_factor_optimize.py --symbol AAPL --data data/AAPL_15_mins_2Y.csv

    # Download from IBKR
    .venv/bin/python run_factor_optimize.py --symbol AAPL --download

    # Custom windows and objective
    .venv/bin/python run_factor_optimize.py --symbol AAPL --data data/AAPL_15_mins_2Y.csv \\
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
logger = logging.getLogger("factor_optimize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IBKR Signal Assistant — Factor Walk-Forward Optimizer")
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument("--data", help="Path to CSV or Parquet file")
    p.add_argument("--download", action="store_true", help="Download from IBKR")
    p.add_argument("--duration", default="2 Y", help="IBKR duration (default: '2 Y')")
    p.add_argument("--bar-size", default="15 min", help="Bar size (default: '15 min')")
    p.add_argument("--train-months", type=int, default=6, help="Training window in months (default: 6)")
    p.add_argument("--test-months", type=int, default=2, help="Test window in months (default: 2)")
    p.add_argument("--objective", default="sharpe_trades",
                   choices=["sharpe", "sharpe_trades"], help="Objective function (default: sharpe_trades)")
    p.add_argument("--capital", type=float, default=100_000, help="Initial capital (default: 100000)")
    p.add_argument("--slippage", type=float, default=5.0, help="Slippage bps (default: 5)")
    p.add_argument("--min-trades", type=int, default=3, help="Min trades for valid result (default: 3)")
    p.add_argument("--no-ic-weights", action="store_true", help="Disable IC-based factor weighting")
    p.add_argument("--no-external", action="store_true",
                   help="Skip external data (VIX/macro/fundamentals)")
    p.add_argument("--output", default=None, help="Output HTML report path")
    return p.parse_args()


async def run_async(args: argparse.Namespace) -> None:
    from src.backtest.data_loader import from_csv, from_parquet, from_ibkr
    from src.backtest.factor_optimizer import FactorWalkForwardOptimizer
    from src.backtest.optimization_report import generate_optimization_report

    # Load data
    if args.download:
        data = await from_ibkr(args.symbol, duration=args.duration, bar_size=args.bar_size)
    elif args.data:
        path = args.data
        data = from_parquet(path) if path.endswith(".parquet") else from_csv(path)
    else:
        print("ERROR: Provide --data <file> or --download", file=sys.stderr)
        sys.exit(1)

    logger.info("Data loaded: %d bars from %s to %s", len(data), data.index[0], data.index[-1])

    optimizer = FactorWalkForwardOptimizer(
        symbol=args.symbol,
        data=data,
        train_months=args.train_months,
        test_months=args.test_months,
        objective=args.objective,
        initial_capital=args.capital,
        slippage_bps=args.slippage,
        min_trades=args.min_trades,
        use_ic_weights=not args.no_ic_weights,
        use_external_data=not args.no_external,
    )
    result = optimizer.run()

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {args.symbol} Factor Walk-Forward Results  ({result.total_windows} windows)")
    print(f"{'='*65}")
    print(f"  Agg OOS Sharpe:   {result.agg_oos_sharpe:+.3f}")
    print(f"  Agg OOS Return:   {result.agg_oos_return_pct:+.2f}%")
    print(f"  Total OOS Trades: {result.agg_oos_trades}")
    print(f"  Grid size:        {result.param_grid_size} combinations")
    print()
    print("  Most Stable Parameters:")
    for k, v in sorted(result.stable_params.items()):
        print(f"    {k} = {v}")
    print()
    print(f"  {'Win#':<5} {'IS Sharpe':>10} {'IS Ret%':>9} {'IS #':>5}"
          f"  {'OOS Sharpe':>10} {'OOS Ret%':>9} {'OOS #':>6}")
    print(f"  {'-'*4:<5} {'-'*9:>10} {'-'*8:>9} {'-'*4:>5}"
          f"  {'-'*9:>10} {'-'*8:>9} {'-'*5:>6}")
    for w in result.windows:
        print(
            f"  {w.window_idx+1:<5}"
            f" {w.is_sharpe:>+10.3f} {w.is_return_pct:>+9.2f}% {w.is_trades:>5}"
            f"  {w.oos_sharpe:>+10.3f} {w.oos_return_pct:>+9.2f}% {w.oos_trades:>6}"
        )
    print(f"{'='*65}\n")

    # ── Apply stable params suggestion ────────────────────────────────
    sp = result.stable_params
    if sp:
        exit_long = sp.get("exit_long_threshold", -0.05)
        print("  Suggested .env settings:")
        print(f"    BUY_THRESHOLD={sp.get('buy_threshold', 0.20)}")
        print(f"    SELL_THRESHOLD=-{sp.get('buy_threshold', 0.20)}")
        print(f"    # exit_long_threshold={exit_long}  max_holding_bars={sp.get('max_holding_bars', 150)}")
        print()

    # ── HTML report ───────────────────────────────────────────────────
    output = args.output or f"factor_optimization_{args.symbol}.html"
    try:
        report_path = generate_optimization_report(result, output_path=output)
        print(f"Report saved to: {report_path}")
    except Exception as exc:
        logger.warning("Could not generate HTML report: %s", exc)


def main() -> None:
    args = parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
