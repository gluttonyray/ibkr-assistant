"""量化交易系统全局共享的核心领域类型。

设计原则：
  - 本模块只允许从标准库 + typing 导入。
  - 所有值对象使用 ``frozen=True, slots=True``（不可变 + 紧凑内存）。
  - Position 是唯一的可变 dataclass（逐 bar 更新）。
  - PortfolioSnapshot 是传递给风控门的不可变只读视图。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any

# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------

class Currency(StrEnum):
    """支持的结算货币。"""
    USD = "USD"
    HKD = "HKD"


class InstrumentType(StrEnum):
    """资产类别判别器。"""
    FUTURE = "FUTURE"
    EQUITY = "EQUITY"  # 保留字段，暂未启用


class Side(StrEnum):
    """订单 / 成交方向。"""
    BUY = "BUY"
    SELL = "SELL"


class SignalType(StrEnum):
    """策略管线发出的离散信号动作。"""
    LONG_ENTRY = auto()
    SHORT_ENTRY = auto()
    LONG_EXIT = auto()
    SHORT_EXIT = auto()
    HOLD = auto()


# ---------------------------------------------------------------------------
# 合约规格
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TradingWindow:
    """单个合约的交易时段窗口。

    时间使用合约本地时区，以便 ``zoneinfo`` 正确处理夏令时转换。

    Parameters
    ----------
    start : str
        交易时段开始时间，格式 ``"HH:MM"``，采用 *timezone* 时区。
    end : str
        交易时段结束时间，格式 ``"HH:MM"``，采用 *timezone* 时区。
        跨午夜时段（如 T+1）时 *end* 可早于 *start*。
    timezone : str
        IANA 时区名称（如 ``"America/Chicago"``）。
    label : str
        人类可读标签（如 ``"Globex"``、``"Morning"``、``"T+1"``）。
    """
    start: str
    end: str
    timezone: str
    label: str = ""


@dataclass(frozen=True, slots=True)
class Instrument:
    """从 ``instruments.yaml`` 加载的不可变合约规格。

    Parameters
    ----------
    symbol : str
        根代码（如 ``"ES"``、``"HSI"``）。
    instrument_type : InstrumentType
        当前仅支持 ``FUTURE``。
    exchange : str
        主要交易所（``"CME"``、``"HKEX"``、``"CBOT"``）。
    currency : Currency
        结算 / 保证金货币。
    multiplier : float
        合约乘数（ES=50，HSI=50，MHI=10）。
    tick_size : float
        最小报价单位（ES=0.25，HSI=1）。
    margin_initial : float
        每张合约初始保证金，单位为 *currency*。
    margin_maintenance : float
        每张合约维持保证金，单位为 *currency*。
    sessions : tuple[TradingWindow, ...]
        有序的交易时段列表，使用 tuple 以支持哈希。
    """
    symbol: str
    instrument_type: InstrumentType
    exchange: str
    currency: Currency
    multiplier: float
    tick_size: float
    margin_initial: float
    margin_maintenance: float
    sessions: tuple[TradingWindow, ...] = ()

    @property
    def tick_value(self) -> float:
        """每张合约每跳一个最小价格单位的货币价值。"""
        return self.tick_size * self.multiplier

    def notional(self, price: float, qty: int = 1) -> float:
        """*qty* 张合约在 *price* 价格下的完整名义价值。"""
        return price * self.multiplier * qty


# ---------------------------------------------------------------------------
# 行情数据
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Bar:
    """单根 OHLCV K 线 —— 系统统一数据单元。

    Parameters
    ----------
    instrument : Instrument
        该 K 线所属合约。
    timestamp : datetime
        K 线*收盘*时间（UTC）。
    open, high, low, close : float
        价格字段。
    volume : float
        成交量（期货为合约手数）。
    bar_duration_secs : int
        K 线时长，单位秒（900 = 15 分钟）。
    """
    instrument: Instrument
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_duration_secs: int = 900


# ---------------------------------------------------------------------------
# 策略输出
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Signal:
    """策略管线发出的综合信号。

    合成得分公式::

        S_combined = w_L1 * S_L1 * MII + w_L2 * S_L2

    当 MII 趋近于 0（动量耗尽）时，Layer 1 贡献淡出至中性；
    Layer 2 Alpha 独立运行。

    Parameters
    ----------
    instrument : Instrument
        目标合约。
    signal_type : SignalType
        离散动作。
    score : float
        合成信号强度，范围 ``[-1, +1]``。
    regime : str
        当前市场状态标签。
    layer1_score : float
        Layer 1（宏观/状态）原始得分（MII 缩放前）。
    layer2_score : float
        Layer 2（Alpha）原始得分。
    mii : float
        动量强度指数，范围 ``[0, 1]``。
    timestamp : datetime
        信号生成时间（UTC）。
    metadata : dict
        每个信号的诊断信息（因子分解等）。
    """
    instrument: Instrument
    signal_type: SignalType
    score: float
    regime: str
    layer1_score: float
    layer2_score: float
    mii: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Fill:
    """单笔成交记录。

    Parameters
    ----------
    instrument : Instrument
        成交合约。
    side : Side
        买入或卖出。
    qty : int
        成交合约数量。
    price : float
        成交价格。
    timestamp : datetime
        成交时间（UTC）。
    commission : float
        券商佣金（单位：instrument.currency）。
    exchange_fees : float
        交易所 / 结算费用（单位：instrument.currency）。
    slippage_cost : float
        估算的滑点成本（仅回测使用；实盘为 0）。
    """
    instrument: Instrument
    side: Side
    qty: int
    price: float
    timestamp: datetime
    commission: float
    exchange_fees: float
    slippage_cost: float = 0.0


# ---------------------------------------------------------------------------
# 持仓跟踪（可变）
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Position:
    """单个合约的实时持仓状态。

    核心类型中唯一的可变 dataclass。
    由组合账本在每根 K 线和每次成交时更新。
    """
    instrument: Instrument
    qty: int                        # +多头 / -空头 / 0 平仓
    avg_entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0    # 单位：instrument.currency
    unrealized_pnl_usd: float = 0.0
    high_watermark: float = 0.0    # 用于多头追踪止损
    low_watermark: float = 0.0     # 用于空头追踪止损
    atr_at_entry: float = 0.0
    entry_score: float = 0.0
    entry_regime: str = ""
    holding_bars: int = 0

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @property
    def direction(self) -> int:
        """+1 多头，-1 空头，0 平仓。"""
        return (self.qty > 0) - (self.qty < 0)


# ---------------------------------------------------------------------------
# 组合快照（不可变只读视图）
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """组合状态的不可变时点快照。

    传递给风控门和仓位计算器，确保它们无法意外修改账本。
    """
    total_equity_usd: float
    cash_usd: float
    positions: dict[str, Position]   # 以合约代码为键
    peak_equity_usd: float
    daily_pnl_usd: float
    daily_trade_count: int
    timestamp: datetime
