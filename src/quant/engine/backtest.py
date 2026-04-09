"""事件驱动的回测引擎，严格保证无前视偏差。

每个时间戳的 K 线处理顺序：
  1. 以当前 K 线开盘价（+ 滑点）成交待处理订单
  2. 对所有持仓进行盯市估值
  3. 交易后风控检查（止损、追踪止损、最大持仓期）
  4. 若仍在预热期则跳过
  5. Layer 1 计算（基于窗口数据，无前视）
  6. Layer 2 + 信号合成
  7. 交易前风控门 + 仓位计算 -> 为下一根 K 线生成待处理订单
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from quant.config.schema import AppConfig
from quant.core.events import OrderEvent
from quant.core.risk import (
    PositionSizerProtocol,
    PostTradeRiskGate,
    PreTradeRiskGate,
)
from quant.core.strategy import (
    Layer1Strategy,
    Layer2Strategy,
    SignalComposer,
)
from quant.core.types import (
    Bar,
    Fill,
    Side,
    Signal,
    SignalType,
)
from quant.engine.base import BaseEngine
from quant.instrument.registry import InstrumentRegistry
from quant.portfolio.book import PortfolioBook, TradeRecord
from quant.portfolio.cost_model import FuturesCostModel
from quant.portfolio.metrics import BacktestMetrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测输出的容器。"""

    metrics: BacktestMetrics
    equity_curve: list[float]
    trades: list[TradeRecord]
    signals: list[Signal]
    config: AppConfig


class BacktestEngine(BaseEngine):
    """逐 K 线回测引擎，保证无前视偏差。

    Parameters
    ----------
    config : AppConfig
        完整应用配置。
    registry : InstrumentRegistry
        合约规格注册表。
    layer1 : Layer1Strategy
        宏观/状态策略。
    layer2 : Layer2Strategy
        Alpha 因子策略。
    composer : SignalComposer
        信号合成器（L1 + L2 -> 最终 Signal）。
    pre_risk : PreTradeRiskGate
        交易前风控检查。
    post_risk : PostTradeRiskGate
        交易后风控检查（止损、追踪止损等）。
    sizer : PositionSizerProtocol
        仓位计算器。
    hkd_usd_rate : float or None
        港币合约的汇率（回退到配置默认值）。
    """

    def __init__(
        self,
        config: AppConfig,
        registry: InstrumentRegistry,
        layer1: Layer1Strategy,
        layer2: Layer2Strategy,
        composer: SignalComposer,
        pre_risk: PreTradeRiskGate,
        post_risk: PostTradeRiskGate,
        sizer: PositionSizerProtocol,
        hkd_usd_rate: float | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._layer1 = layer1
        self._layer2 = layer2
        self._composer = composer
        self._pre_risk = pre_risk
        self._post_risk = post_risk
        self._sizer = sizer
        self._hkd_usd_rate = hkd_usd_rate or config.risk.default_hkd_usd_rate

        self._cost_model = FuturesCostModel(config.cost)
        self._book: PortfolioBook | None = None
        self._data: dict[str, list[Bar]] = {}
        self._pending_orders: dict[str, tuple[Signal, OrderEvent]] = {}
        self._recorded_signals: list[Signal] = []
        self._result: BacktestResult | None = None

    def run(
        self,
        data: dict[str, list[Bar]],
    ) -> BacktestResult:
        """在提供的 K 线数据上执行回测。

        Parameters
        ----------
        data : dict[str, list[Bar]]
            以合约代码为键的交易 Universe K 线数据，每个列表须按时间戳排序。

        Returns
        -------
        BacktestResult
        """
        self._data = data
        super().run()
        assert self._result is not None
        return self._result

    # -- BaseEngine 接口 --------------------------------------------------

    def _setup(self) -> None:
        self._book = PortfolioBook(self._config.backtest.initial_capital)
        self._pending_orders = {}
        self._recorded_signals = []

    def _on_bar(self, bar_index: int, bars: dict[str, Bar]) -> None:
        # 未直接使用；逻辑在 _run_loop 中实现
        pass

    def _on_fill(self, fill: Fill) -> None:
        assert self._book is not None
        self._book.on_fill(fill, self._cost_model, self._hkd_usd_rate)

    def _teardown(self) -> None:
        assert self._book is not None
        metrics = compute_metrics(
            self._book.equity_curve,
            self._book.trade_records,
        )
        self._result = BacktestResult(
            metrics=metrics,
            equity_curve=list(self._book.equity_curve),
            trades=list(self._book.trade_records),
            signals=list(self._recorded_signals),
            config=self._config,
        )

    def _run_loop(self) -> None:
        assert self._book is not None
        book = self._book
        config = self._config
        data = self._data
        registry = self._registry

        # 构建对齐的时间戳索引（所有合约时间戳的并集，已排序）
        all_timestamps: set[datetime] = set()
        # 构建索引：symbol -> {timestamp: Bar}
        bar_index_map: dict[str, dict[datetime, Bar]] = {}
        for sym, bars in data.items():
            bar_index_map[sym] = {}
            for bar in bars:
                all_timestamps.add(bar.timestamp)
                bar_index_map[sym][bar.timestamp] = bar

        aligned_timestamps = sorted(all_timestamps)
        logger.info(
            "Backtest: %d timestamps, %d trading symbols, warmup=%d",
            len(aligned_timestamps),
            len(data),
            config.backtest.warmup_bars,
        )

        for bar_index, timestamp in enumerate(aligned_timestamps):
            # 收集当前时间戳的各合约 K 线
            current_bars: dict[str, Bar] = {}
            for sym in data:
                bar = bar_index_map[sym].get(timestamp)
                if bar is not None:
                    current_bars[sym] = bar

            if not current_bars:
                continue

            # 步骤 1：以当前 K 线开盘价成交待处理订单
            self._process_pending_orders(current_bars, timestamp)

            # 步骤 2：盯市估值
            book.mark_to_market(current_bars, self._hkd_usd_rate)

            # 步骤 3：交易后风控检查
            self._process_post_trade_risk(current_bars, timestamp)

            # 步骤 4：预热期跳过
            if bar_index < config.backtest.warmup_bars:
                continue

            # 步骤 5：Layer 1 计算（基于窗口数据，无前视）
            # 构建交易品种的推理窗口
            lookback = config.strategy.layer1.bar_lookback_bars
            windows: dict = {}
            for inst in registry.all():
                sym = inst.symbol
                if sym in data:
                    # 只使用不超过当前 bar_index 的 K 线
                    sym_bars = data[sym]
                    # 找出时间戳 <= timestamp 的 K 线数量
                    end_idx = _find_bar_end_index(sym_bars, timestamp)
                    start_idx = max(0, end_idx - lookback)
                    windows[inst] = sym_bars[start_idx:end_idx]

            layer1_results = self._layer1.compute(windows)

            # 步骤 6：各合约的 Layer 2 + 信号合成
            for inst in registry.all():
                if inst not in layer1_results:
                    continue
                sym = inst.symbol
                if sym not in current_bars:
                    continue
                window = windows.get(inst, [])
                if len(window) < 20:
                    continue

                l2 = self._layer2.compute(inst, window, layer1_results[inst])
                signal = self._composer.compose(
                    inst, layer1_results[inst], l2, timestamp
                )

                # 步骤 7：交易前风控 + 仓位计算 -> 为下一 bar 生成待处理订单
                if signal.signal_type in (SignalType.HOLD,):
                    continue

                # 检查是否为已有持仓的退出信号
                pos = book.get_position(inst)
                is_exit = signal.signal_type in (
                    SignalType.LONG_EXIT,
                    SignalType.SHORT_EXIT,
                )

                if is_exit:
                    if pos is not None:
                        exit_side = Side.SELL if pos.is_long else Side.BUY
                        self._pending_orders[sym] = (
                            signal,
                            OrderEvent(
                                instrument=inst,
                                side=exit_side,
                                qty=abs(pos.qty),
                                order_type="MARKET",
                                timestamp=timestamp,
                            ),
                        )
                    continue

                # 入场信号
                atr = _compute_atr(window[-15:]) if len(window) >= 15 else inst.tick_size * 10
                proposed_qty = self._sizer.compute_size(
                    signal, inst, book.snapshot(), atr
                )
                approved, qty, reason = self._pre_risk.check(
                    signal, proposed_qty, book.snapshot()
                )
                if approved and qty > 0:
                    side = (
                        Side.BUY
                        if signal.signal_type == SignalType.LONG_ENTRY
                        else Side.SELL
                    )
                    self._pending_orders[sym] = (
                        signal,
                        OrderEvent(
                            instrument=inst,
                            side=side,
                            qty=qty,
                            order_type="MARKET",
                            timestamp=timestamp,
                        ),
                    )

    # -- 内部辅助方法 ------------------------------------------------------

    def _process_pending_orders(
        self,
        current_bars: dict[str, Bar],
        timestamp: datetime,
    ) -> None:
        """以当前 K 线开盘价 + 滑点成交待处理订单。"""
        assert self._book is not None
        filled_symbols: list[str] = []

        for sym, (signal, order) in list(self._pending_orders.items()):
            if sym not in current_bars:
                continue

            bar = current_bars[sym]
            inst = self._registry.get(sym)
            exc = self._config.cost.for_exchange(inst.exchange)
            slippage_offset = inst.tick_size * exc.slippage_ticks

            if order.side == Side.BUY:
                fill_price = bar.open + slippage_offset
            else:
                fill_price = bar.open - slippage_offset

            commission, exchange_fees, slippage_cost = self._cost_model.calculate(
                inst, order.qty, fill_price
            )

            fill = Fill(
                instrument=inst,
                side=order.side,
                qty=order.qty,
                price=fill_price,
                timestamp=timestamp,
                commission=commission,
                exchange_fees=exchange_fees,
                slippage_cost=slippage_cost,
            )

            self._on_fill(fill)
            self._recorded_signals.append(signal)
            filled_symbols.append(sym)

        for sym in filled_symbols:
            del self._pending_orders[sym]

    def _process_post_trade_risk(
        self,
        current_bars: dict[str, Bar],
        timestamp: datetime,
    ) -> None:
        """执行交易后风控检查，并将触发的持仓加入退出队列。"""
        assert self._book is not None
        book = self._book

        exit_list = self._post_risk.check(
            book.positions(), current_bars, book.snapshot()
        )

        for inst, _reason in exit_list:
            sym = inst.symbol
            if sym in book.positions():
                pos = book.get_position(inst)
                if pos is None:
                    continue
                exit_side = Side.SELL if pos.is_long else Side.BUY
                sig_type = (
                    SignalType.LONG_EXIT if pos.is_long else SignalType.SHORT_EXIT
                )
                self._pending_orders[sym] = (
                    Signal(
                        instrument=inst,
                        signal_type=sig_type,
                        score=0.0,
                        regime="",
                        layer1_score=0.0,
                        layer2_score=0.0,
                        mii=0.0,
                        timestamp=timestamp,
                    ),
                    OrderEvent(
                        instrument=inst,
                        side=exit_side,
                        qty=abs(pos.qty),
                        order_type="MARKET",
                        timestamp=timestamp,
                    ),
                )


def _find_bar_end_index(bars: list[Bar], timestamp: datetime) -> int:
    """二分查找第一个超过 *timestamp* 的 K 线索引（不含）。

    返回时间戳 <= target 的 K 线数量，确保无前视偏差。
    """
    lo, hi = 0, len(bars)
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid].timestamp <= timestamp:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _compute_atr(bars: list[Bar], period: int = 14) -> float:
    """根据最近若干根 K 线计算平均真实波幅（ATR）。"""
    if len(bars) < 2:
        return bars[0].high - bars[0].low if bars else 1.0
    trs: list[float] = []
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        trs.append(tr)
    return float(np.mean(trs[-period:]))
