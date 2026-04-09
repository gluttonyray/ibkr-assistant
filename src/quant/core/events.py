"""流经事件总线的领域事件。

设计原则：
  - 所有事件均不可变（frozen=True, slots=True）。
  - 事件只携带数据，不包含任何修改状态的方法。
  - 本模块只从标准库 + quant.core.types 导入。

事件层级：
  BarEvent      -- 新的 OHLCV K 线可用
  SignalEvent   -- 策略发出信号
  OrderEvent    -- 策略发出的订单请求（风控检查前）
  FillEvent     -- 订单已执行（实盘或模拟）
  RiskEvent     -- 风控门触发状态变更
  SessionEvent  -- 交易时段边界穿越
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from quant.core.types import Bar, Fill, Instrument, Side, Signal

# ---------------------------------------------------------------------------
# 行情数据
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BarEvent:
    """一根已收盘的 OHLCV K 线可用。

    ``bar.instrument`` 已包含合约规格，此处不重复存储。
    ``bar_index`` 是回测引擎用于可重现性的顺序 K 线计数器。
    """
    bar: Bar
    bar_index: int = 0          # 顺序计数器；实盘模式下为 0


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """交易时段边界已穿越。"""
    event_type: Literal["SESSION_OPEN", "SESSION_CLOSE", "MAINTENANCE_BREAK"]
    instrument: Instrument
    session_label: str          # "Globex"、"Morning"、"T+1"
    timestamp: datetime


# ---------------------------------------------------------------------------
# 策略输出
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SignalEvent:
    """策略管线发出了一个综合信号。"""
    signal: Signal


# ---------------------------------------------------------------------------
# 订单生命周期
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OrderEvent:
    """来自策略的订单请求（风控检查前）。

    order_type:
        "MARKET"           -- 以下一个可用价格成交
        "LIMIT"            -- 被动限价订单
        "LIMIT_AGGRESSIVE" -- 偏移 n 个 tick 的限价单（L2 分层退出）
    """
    instrument: Instrument
    side: Side
    qty: int
    order_type: Literal["MARKET", "LIMIT", "LIMIT_AGGRESSIVE"]
    limit_price: float | None = None
    timestamp: datetime = datetime.min   # 由引擎在创建时设置
    signal_ref: Signal | None = None     # 来源信号（用于审计）
    order_id: str = ""                   # 由 Executor 提交后设置


@dataclass(frozen=True, slots=True)
class FillEvent:
    """订单已执行（实盘成交或回测模拟）。"""
    fill: Fill
    order: OrderEvent


@dataclass(frozen=True, slots=True)
class CancelEvent:
    """订单已被取消（由系统或券商）。"""
    order_id: str
    reason: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# 风控事件
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RiskEvent:
    """风控门触发了一次状态变更。

    CIRCUIT_BREAKER_OPEN  -- 触及回撤限制；不允许新建仓
    CIRCUIT_BREAKER_CLOSED -- 冷却期结束；恢复交易
    MAX_DRAWDOWN_HIT      -- 峰值到谷值超过 max_drawdown_pct
    DAILY_LOSS_LIMIT_HIT  -- 当日盈亏低于 max_daily_loss_usd
    FORCE_CLOSE           -- 必须立即平仓（分层退出）
    POSITION_LIMIT_HIT    -- 触及相关组或单品种持仓上限
    ORDER_REJECTED        -- 交易前检查拦截了一笔订单
    """
    event_type: Literal[
        "CIRCUIT_BREAKER_OPEN",
        "CIRCUIT_BREAKER_CLOSED",
        "MAX_DRAWDOWN_HIT",
        "DAILY_LOSS_LIMIT_HIT",
        "FORCE_CLOSE",
        "POSITION_LIMIT_HIT",
        "ORDER_REJECTED",
    ]
    message: str
    timestamp: datetime
    instrument: Instrument | None = None    # 仅在事件与特定合约相关时设置
