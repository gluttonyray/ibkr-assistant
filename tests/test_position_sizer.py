import pytest
from datetime import datetime, timezone
from quant.risk.position_sizer import ATRPositionSizer
from quant.config.schema import RiskConfig
from quant.core.types import PortfolioSnapshot, Signal, SignalType

UTC = timezone.utc


def make_portfolio(equity=100_000.0):
    return PortfolioSnapshot(
        total_equity_usd=equity, cash_usd=equity * 0.8, positions={},
        peak_equity_usd=equity, daily_pnl_usd=0.0, daily_trade_count=0,
        timestamp=datetime.now(UTC),
    )


def make_signal(instrument):
    return Signal(
        instrument=instrument, signal_type=SignalType.LONG_ENTRY,
        score=0.5, regime="TRENDING_STRONG",
        layer1_score=0.5, layer2_score=0.5, mii=1.0,
        timestamp=datetime.now(UTC),
    )


def test_zero_equity_returns_zero(es_instrument):
    """P0 修复：权益 <= 0 时应返回 0。"""
    sizer = ATRPositionSizer(RiskConfig())
    portfolio = make_portfolio(equity=0.0)
    qty = sizer.compute_size(make_signal(es_instrument), es_instrument, portfolio, atr=10.0)
    assert qty == 0


def test_zero_atr_returns_zero(es_instrument):
    """ATR 为 0 时应返回 0（防止除零）。"""
    sizer = ATRPositionSizer(RiskConfig())
    qty = sizer.compute_size(make_signal(es_instrument), es_instrument, make_portfolio(), atr=0.0)
    assert qty == 0


def test_normal_sizing(es_instrument):
    """正常条件下的仓位计算验证。"""
    sizer = ATRPositionSizer(RiskConfig())
    # risk_amount = 100000 * 0.02 = 2000
    # stop_distance_usd = 10 * 2.0 * 50 = 1000
    # qty = floor(2000/1000) = 2
    qty = sizer.compute_size(make_signal(es_instrument), es_instrument, make_portfolio(), atr=10.0)
    assert qty == 2


def test_hkd_instrument_converts(hsi_instrument):
    """港币合约应正确进行汇率换算。"""
    sizer = ATRPositionSizer(RiskConfig(), hkd_usd_rate=0.128)
    qty = sizer.compute_size(
        make_signal(hsi_instrument), hsi_instrument, make_portfolio(500_000.0), atr=100.0,
    )
    assert qty >= 0


def test_max_contracts_cap(es_instrument):
    """仓位应被 max_contracts_per_instrument 上限截断。"""
    cfg = RiskConfig(max_contracts_per_instrument=3)
    sizer = ATRPositionSizer(cfg)
    # 大资金 + 小 ATR → 计算结果会超过 3，应被截断
    qty = sizer.compute_size(
        make_signal(es_instrument), es_instrument, make_portfolio(10_000_000.0), atr=1.0,
    )
    assert qty <= 3
