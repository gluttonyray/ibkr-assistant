"""Core abstractions -- zero external dependencies.

This package contains Protocol interfaces, domain types, and event
definitions.  Nothing here may import outside stdlib + typing.
"""
from quant.core.bus import EventBus
from quant.core.clock import TradingClock
from quant.core.data_feed import DataFeed
from quant.core.events import (
    BarEvent,
    CancelEvent,
    FillEvent,
    OrderEvent,
    RiskEvent,
    SessionEvent,
    SignalEvent,
)
from quant.core.execution import Executor
from quant.core.portfolio import PortfolioView
from quant.core.risk import (
    PositionSizerProtocol,
    PostTradeRiskGate,
    PreTradeRiskGate,
)
from quant.core.strategy import (
    Layer1Result,
    Layer1Strategy,
    Layer2Result,
    Layer2Strategy,
    SignalComposer,
)
from quant.core.types import (
    Bar,
    Currency,
    Fill,
    Instrument,
    InstrumentType,
    PortfolioSnapshot,
    Position,
    Side,
    Signal,
    SignalType,
    TradingWindow,
)

__all__ = [
    # Types
    "Bar",
    "Currency",
    "Fill",
    "Instrument",
    "InstrumentType",
    "Position",
    "PortfolioSnapshot",
    "Side",
    "Signal",
    "SignalType",
    "TradingWindow",
    # Events
    "BarEvent",
    "CancelEvent",
    "FillEvent",
    "OrderEvent",
    "RiskEvent",
    "SessionEvent",
    "SignalEvent",
    # Protocols
    "DataFeed",
    "EventBus",
    "Executor",
    "Layer1Strategy",
    "Layer2Strategy",
    "PortfolioView",
    "PositionSizerProtocol",
    "PostTradeRiskGate",
    "PreTradeRiskGate",
    "SignalComposer",
    "TradingClock",
    # Result types
    "Layer1Result",
    "Layer2Result",
]
