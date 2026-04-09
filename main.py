#!/usr/bin/env python3
"""IBKR Signal Assistant — 新系统入口点。

运行模式：
  信号模式（auto_trade=false）：计算并打印信号，不下单
  实盘模式（auto_trade=true）：通过 IBKR TWS 自动执行交易

用法：
  cd new/
  .venv/bin/python main.py                         # 使用默认配置
  .venv/bin/python main.py --config configs/app.yaml
  .venv/bin/python main.py --backtest --data data/ES_15min.csv
  .venv/bin/python main.py --backtest --data data/ES_15min.csv --model models/layer1/v2_20260408/
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 确保 src 在 import 路径中
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quant.config.loader import load_config
from quant.config.schema import AppConfig
from quant.instrument.registry import InstrumentRegistry


def setup_logging(cfg: AppConfig) -> None:
    log_dir = Path(cfg.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(cfg.log_file),
        ],
    )


async def run_live(cfg: AppConfig, registry: InstrumentRegistry) -> None:
    """实盘/信号模式主循环。"""
    from quant.data.sources.ibkr_margin import IBKRMarginRefresher
    from quant.execution.ibkr_executor import IBKRExecutor
    from quant.strategy.layer1.layer1 import Layer1
    from quant.strategy.layer2.layer2 import Layer2
    from quant.strategy.layer2.composer import DefaultComposer
    from quant.risk.position_sizer import ATRPositionSizer
    from quant.risk.risk_engine import RiskEngine

    logger = logging.getLogger(__name__)
    hkd_rate = cfg.risk.default_hkd_usd_rate

    layer1 = Layer1(cfg.strategy.layer1)
    layer2 = Layer2(cfg.strategy.layer2)
    composer = DefaultComposer(cfg.strategy)
    risk_engine = RiskEngine(cfg.risk, hkd_usd_rate=hkd_rate)
    sizer = ATRPositionSizer(cfg.risk, hkd_usd_rate=hkd_rate)

    executor = IBKRExecutor(cfg.ibkr, registry)

    # 启动时动态拉取最新保证金（有缓存用缓存，TTL 6 小时）
    refresher = IBKRMarginRefresher(
        host=cfg.ibkr.host,
        port=cfg.ibkr.port,
        client_id=cfg.ibkr.client_id + 10,  # 避免与主连接 ID 冲突
    )
    refresher.refresh(registry)

    mode = "实盘" if cfg.strategy.auto_trade else "信号"
    logger.info(f"启动 {mode}模式，品种：{cfg.ibkr.symbols}")

    if cfg.strategy.auto_trade:
        await executor.connect()

    try:
        # 实际行情数据通过 IBKR data_feed 获取（此处为框架骨架）
        # TODO: 接入 IBKRHistoricalSource 初始化 warmup，然后订阅实时 bar
        logger.info("主循环就绪（接入实时数据源后完整运行）")
        while True:
            await asyncio.sleep(cfg.refresh_interval)
    finally:
        if cfg.strategy.auto_trade:
            await executor.disconnect()


def run_backtest(
    cfg: AppConfig,
    registry: InstrumentRegistry,
    data_path: str,
    model_path: str | None = None,
) -> None:
    """从 CSV/Parquet 文件运行回测。"""
    import polars as pl
    from quant.engine.backtest import BacktestEngine
    from quant.strategy.layer1.layer1 import Layer1
    from quant.strategy.layer2.layer2 import Layer2
    from quant.strategy.layer2.composer import DefaultComposer
    from quant.risk.position_sizer import ATRPositionSizer
    from quant.risk.risk_engine import RiskEngine
    from quant.core.types import Bar
    from datetime import timezone

    logger = logging.getLogger(__name__)
    logger.info(f"回测数据：{data_path}")

    # 加载数据
    p = Path(data_path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(p)
    else:
        df = pl.read_csv(p, try_parse_dates=True)

    # 推断品种（从文件名），未在 registry 中的品种自动创建通用股票合约
    from quant.core.types import Currency, Instrument, InstrumentType
    symbol = p.stem.split("_")[0].upper()
    try:
        inst = registry.get(symbol)
    except KeyError:
        logger.info(
            f"品种 {symbol} 不在 instruments.yaml 中，自动创建通用股票合约（multiplier=1, tick=0.01）"
        )
        inst = Instrument(
            symbol=symbol,
            instrument_type=InstrumentType.EQUITY,
            exchange="SMART",
            currency=Currency.USD,
            multiplier=1.0,
            tick_size=0.01,
            margin_initial=0.0,
            margin_maintenance=0.0,
        )

    bars: list[Bar] = []
    for row in df.iter_rows(named=True):
        ts = (row.get("event_time") or row.get("timestamp")
              or row.get("date") or row.get("datetime"))
        if ts is None:
            continue
        if hasattr(ts, "replace"):
            ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
        bars.append(Bar(
            instrument=inst,
            timestamp=ts,
            open=float(row.get("open", 0)),
            high=float(row.get("high", 0)),
            low=float(row.get("low", 0)),
            close=float(row.get("close", 0)),
            volume=float(row.get("volume", 0)),
        ))

    hkd_rate = cfg.risk.default_hkd_usd_rate
    risk_engine = RiskEngine(cfg.risk, hkd_usd_rate=hkd_rate)

    layer1 = Layer1(cfg.strategy.layer1)
    if model_path is not None:
        logger.info(f"加载预训练模型：{model_path}")
        layer1.load_model(model_path)

    engine = BacktestEngine(
        config=cfg,
        registry=registry,
        layer1=layer1,
        layer2=Layer2(cfg.strategy.layer2),
        composer=DefaultComposer(cfg.strategy),
        pre_risk=risk_engine.pre_trade,
        post_risk=risk_engine.post_trade,
        sizer=ATRPositionSizer(cfg.risk, hkd_usd_rate=hkd_rate),
        hkd_usd_rate=hkd_rate,
    )

    result = engine.run({symbol: bars})
    m = result.metrics
    logger.info(
        f"\n{'='*50}\n"
        f"回测结果 — {symbol}\n"
        f"  总收益率：{m.total_return:.2%}\n"
        f"  年化收益：{m.annualized_return:.2%}\n"
        f"  Sharpe：{m.sharpe_ratio:.2f}\n"
        f"  Sortino：{m.sortino_ratio:.2f}\n"
        f"  Calmar：{m.calmar_ratio:.2f}\n"
        f"  最大回撤：{m.max_drawdown:.2%}\n"
        f"  胜率：{m.win_rate:.2%}\n"
        f"  总交易次数：{m.total_trades}\n"
        f"{'='*50}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR Signal Assistant")
    parser.add_argument("--config", default="configs/app.yaml", help="配置文件路径")
    parser.add_argument("--backtest", action="store_true", help="运行回测模式")
    parser.add_argument("--data", help="回测数据文件路径（CSV 或 Parquet）")
    parser.add_argument(
        "--model",
        default=None,
        help="预训练 Layer1 模型目录路径（可选，含 meta.json + 模型文件）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    registry = InstrumentRegistry.from_yaml(cfg.instruments_path)

    if args.backtest:
        if not args.data:
            print("错误：回测模式需要 --data 参数")
            sys.exit(1)
        run_backtest(cfg, registry, args.data, args.model)
    else:
        asyncio.run(run_live(cfg, registry))


if __name__ == "__main__":
    main()
