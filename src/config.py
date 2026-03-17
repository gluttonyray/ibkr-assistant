"""
Central configuration loaded from .env / environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes")


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


@dataclass
class Config:
    # IBKR connection
    ibkr_host: str = field(default_factory=lambda: os.getenv("IBKR_HOST", "127.0.0.1"))
    ibkr_port: int = field(default_factory=lambda: _int("IBKR_PORT", 7497))
    ibkr_client_id: int = field(default_factory=lambda: _int("IBKR_CLIENT_ID", 1))

    # Symbols
    symbols: List[str] = field(
        default_factory=lambda: [
            s.strip().upper()
            for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
            if s.strip()
        ]
    )

    # Bars
    bar_size: str = field(default_factory=lambda: os.getenv("BAR_SIZE", "15 min"))
    history_bars: int = field(default_factory=lambda: _int("HISTORY_BARS", 500))

    # Thresholds
    buy_threshold: float = field(default_factory=lambda: _float("BUY_THRESHOLD", 0.4))
    sell_threshold: float = field(default_factory=lambda: _float("SELL_THRESHOLD", -0.4))

    # Anti-flicker
    confirm_bars: int = field(default_factory=lambda: _int("CONFIRM_BARS", 5))

    # Dashboard
    refresh_interval: int = field(default_factory=lambda: _int("REFRESH_INTERVAL", 15))

    # Opening-hour filter (minutes after NYSE 09:30 ET where no orders are placed)
    open_filter_minutes: int = field(default_factory=lambda: _int("OPEN_FILTER_MINUTES", 30))

    # Auto-trading
    auto_trade: bool = field(default_factory=lambda: _bool("AUTO_TRADE", False))
    atr_limit_offset: float = field(default_factory=lambda: _float("ATR_LIMIT_OFFSET", 0.5))
    max_position_size: int = field(default_factory=lambda: _int("MAX_POSITION_SIZE", 100))
    daily_loss_limit: float = field(default_factory=lambda: _float("DAILY_LOSS_LIMIT", 500.0))
    order_timeout: int = field(default_factory=lambda: _int("ORDER_TIMEOUT", 60))
    max_orders_per_day: int = field(default_factory=lambda: _int("MAX_ORDERS_PER_DAY", 20))

    # Logging
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/signals.log"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ── Factor system ────────────────────────────────────────────────────────
    factor_mode: bool = field(default_factory=lambda: _bool("FACTOR_MODE", False))

    # External data
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))

    # Risk management (used by FactorBacktestEngine and live factor mode)
    risk_per_trade: float = field(default_factory=lambda: _float("RISK_PER_TRADE", 0.02))
    max_position_pct: float = field(default_factory=lambda: _float("MAX_POSITION_PCT", 0.20))
    stop_loss_atr: float = field(default_factory=lambda: _float("STOP_LOSS_ATR", 2.0))
    take_profit_atr: float = field(default_factory=lambda: _float("TAKE_PROFIT_ATR", 4.0))
    trailing_stop_atr: float = field(default_factory=lambda: _float("TRAILING_STOP_ATR", 1.5))
    max_drawdown_pct: float = field(default_factory=lambda: _float("MAX_DRAWDOWN_PCT", 0.10))


# Singleton
cfg = Config()
