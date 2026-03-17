"""
CLI entry point for backtesting.

Usage:
    # From IBKR (recommended — auto-downloads and caches as Parquet)
    .venv/bin/python run_backtest.py --symbol AAPL --download
    .venv/bin/python run_backtest.py --symbol AAPL --download --duration "2 Y" --bar-size "15 min"

    # From local file
    .venv/bin/python run_backtest.py --symbol AAPL --data data/AAPL_15min.csv

    # Custom params
    .venv/bin/python run_backtest.py --symbol SPY --download \
        --capital 50000 --position-size 200 --slippage 3
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
logger = logging.getLogger("backtest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IBKR Signal Assistant — Backtester")
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument("--data", help="Path to CSV or Parquet file")
    p.add_argument("--download", action="store_true", help="Download from IBKR")
    p.add_argument("--duration", default="2 Y", help="IBKR duration string (default: '2 Y')")
    p.add_argument("--bar-size", default="15 min", help="Bar size (default: '15 min')")
    p.add_argument("--capital", type=float, default=100_000, help="Initial capital (default: 100000)")
    p.add_argument("--position-size", type=int, default=100, help="Shares per trade (default: 100)")
    p.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps (default: 5)")
    p.add_argument("--output", default=None, help="Output HTML path")
    return p.parse_args()


async def run_async(args: argparse.Namespace) -> None:
    from src.backtest.data_loader import from_csv, from_parquet, from_ibkr
    from src.backtest.engine import BacktestEngine
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

    # Run backtest
    engine = BacktestEngine(
        symbol=args.symbol,
        data=data,
        initial_capital=args.capital,
        position_size=args.position_size,
        slippage_bps=args.slippage,
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
    print(f"  {args.symbol} Backtest Results")
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
    print(f"{'='*60}\n")

    # Generate report
    output = args.output or f"backtest_report_{args.symbol}.html"
    report_path = generate_report(result, metrics, output_path=output)
    print(f"Report saved to: {report_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
