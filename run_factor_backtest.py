"""CLI entry point for factor-based backtesting.

Usage::

    # From local CSV file
    .venv/bin/python run_factor_backtest.py --symbol AAPL --data data/AAPL_15min.csv

    # Download from IBKR
    .venv/bin/python run_factor_backtest.py --symbol AAPL --download

    # Custom params
    .venv/bin/python run_factor_backtest.py --symbol SPY --download \\
        --capital 50000 --slippage 3 --buy-threshold 0.25
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
logger = logging.getLogger("factor_backtest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IBKR Signal Assistant — Factor Backtester")
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument("--data", help="Path to CSV or Parquet file")
    p.add_argument("--download", action="store_true", help="Download from IBKR")
    p.add_argument("--duration", default="2 Y", help="IBKR duration string (default: '2 Y')")
    p.add_argument("--bar-size", default="15 min", help="Bar size (default: '15 min')")
    p.add_argument("--capital", type=float, default=100_000, help="Initial capital (default: 100000)")
    p.add_argument("--slippage", type=float, default=5.0, help="Slippage bps (default: 5)")
    p.add_argument("--buy-threshold", type=float, default=0.2, help="Alpha BUY threshold (default: 0.2)")
    p.add_argument("--sell-threshold", type=float, default=-0.2, help="Alpha SELL threshold (default: -0.2)")
    p.add_argument("--exit-long-threshold", type=float, default=-0.05,
                   help="Exit long when score drops below this (default: -0.05)")
    p.add_argument("--max-holding-bars", type=int, default=150,
                   help="Force-close after this many bars (default: 150)")
    p.add_argument("--no-ic-weights", action="store_true",
                   help="Disable IC-based factor weighting")
    p.add_argument("--no-external", action="store_true",
                   help="Skip external data (VIX/macro/fundamentals) — use technical factors only")
    p.add_argument("--output", default=None, help="Output HTML path")
    return p.parse_args()


async def run_async(args: argparse.Namespace) -> None:
    from src.backtest.data_loader import from_csv, from_parquet, from_ibkr
    from src.backtest.factor_engine import FactorBacktestEngine
    from src.backtest.metrics import compute_metrics
    from src.backtest.report import generate_report

    # Load data
    if args.download:
        data = await from_ibkr(args.symbol, duration=args.duration, bar_size=args.bar_size)
    elif args.data:
        path = args.data
        if path.endswith(".parquet"):
            data = from_parquet(path)
        else:
            data = from_csv(path)
    else:
        print("ERROR: Provide --data <file> or --download", file=sys.stderr)
        sys.exit(1)

    logger.info("Data loaded: %d bars from %s to %s", len(data), data.index[0], data.index[-1])

    # Run factor backtest
    engine = FactorBacktestEngine(
        symbol=args.symbol,
        data=data,
        initial_capital=args.capital,
        slippage_bps=args.slippage,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        exit_long_threshold=args.exit_long_threshold,
        exit_short_threshold=-args.exit_long_threshold,
        max_holding_bars=args.max_holding_bars,
        use_ic_weights=not args.no_ic_weights,
        use_external_data=not args.no_external,
    )
    result = engine.run()

    # Compute metrics
    metrics = compute_metrics(
        trades=result.trades,
        equity_curve=result.equity_curve,
        initial_capital=result.initial_capital,
        cash_flows=engine.portfolio.cash_flows,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {args.symbol} Factor Backtest Results")
    print(f"{'='*60}")
    print(f"  Total Return:   {metrics.total_return_pct:+.2f}%")
    print(f"  CAGR:           {metrics.cagr_pct:+.2f}%")
    print(f"  Sharpe:         {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino:        {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown:   {metrics.drawdown.max_dd_pct:.2f}%")
    print(f"  Win Rate:       {metrics.win_rate:.1f}%")
    print(f"  Profit Factor:  {metrics.profit_factor:.2f}")
    print(f"  Total Trades:   {metrics.total_trades}")
    print(f"  Total Costs:    ${metrics.costs.total_costs:.2f}")
    print(f"  Cost Drag:      {metrics.costs.cost_drag_pct:.2f}%")
    print(f"  Factors used:   {len(engine._factors)}")
    print(f"{'='*60}\n")

    # Generate HTML report
    output = args.output or f"factor_backtest_{args.symbol}.html"
    report_path = generate_report(result, metrics, output_path=output)
    print(f"Report saved to: {report_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
