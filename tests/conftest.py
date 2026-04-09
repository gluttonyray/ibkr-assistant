from __future__ import annotations
import os
import pytest
import random

# macOS: 预加载 libomp，防止 LightGBM 在 pytest 多次训练时 segfault
import ctypes as _ctypes
_LIBOMP = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
if os.path.exists(_LIBOMP):
    try:
        _ctypes.CDLL(_LIBOMP)
    except OSError:
        pass
from datetime import datetime, timezone
from quant.core.types import (
    Bar, Currency, Instrument, InstrumentType, TradingWindow,
)
from quant.config.schema import AppConfig

UTC = timezone.utc


@pytest.fixture
def es_instrument() -> Instrument:
    return Instrument(
        symbol="ES", instrument_type=InstrumentType.FUTURE,
        exchange="CME", currency=Currency.USD,
        multiplier=50.0, tick_size=0.25,
        margin_initial=15200.0, margin_maintenance=13800.0,
        sessions=(TradingWindow(start="17:00", end="16:00", timezone="America/Chicago", label="Globex"),),
    )


@pytest.fixture
def hsi_instrument() -> Instrument:
    return Instrument(
        symbol="HSI", instrument_type=InstrumentType.FUTURE,
        exchange="HKEX", currency=Currency.HKD,
        multiplier=50.0, tick_size=1.0,
        margin_initial=132180.0, margin_maintenance=105744.0,
        sessions=(
            TradingWindow(start="09:15", end="12:00", timezone="Asia/Hong_Kong", label="Morning"),
            TradingWindow(start="17:15", end="03:00", timezone="Asia/Hong_Kong", label="T+1"),
        ),
    )


@pytest.fixture
def default_config() -> AppConfig:
    return AppConfig()


def make_bars(instrument: Instrument, n: int, start_price: float = 4000.0) -> list[Bar]:
    """生成 n 根测试 K 线，价格围绕 start_price 做随机游走。"""
    rng = random.Random(42)
    bars = []
    price = start_price
    base_time = datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
    for i in range(n):
        change = rng.gauss(0, price * 0.001)
        close = max(price + change, 1.0)
        high = close + abs(rng.gauss(0, price * 0.0005))
        low = close - abs(rng.gauss(0, price * 0.0005))
        bars.append(Bar(
            instrument=instrument,
            timestamp=base_time.replace(
                minute=(i * 15) % 60,
                hour=((i * 15) // 60) % 24,
            ),
            open=price, high=high, low=low, close=close,
            volume=float(rng.randint(1000, 10000)),
        ))
        price = close
    return bars
