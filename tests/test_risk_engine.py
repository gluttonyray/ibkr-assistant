import pytest
from datetime import datetime, timezone
from quant.risk.risk_engine import RiskEngine
from quant.config.schema import RiskConfig
from quant.core.types import PortfolioSnapshot, Signal, SignalType

UTC = timezone.utc


def make_portfolio(equity=100_000.0, cash=80_000.0, daily_pnl=0.0, daily_trades=0, peak=100_000.0):
    return PortfolioSnapshot(
        total_equity_usd=equity,
        cash_usd=cash,
        positions={},
        peak_equity_usd=peak,
        daily_pnl_usd=daily_pnl,
        daily_trade_count=daily_trades,
        timestamp=datetime.now(UTC),
    )


def make_signal(instrument, score=0.5, signal_type=SignalType.LONG_ENTRY):
    return Signal(
        instrument=instrument, signal_type=signal_type,
        score=score, regime="TRENDING_STRONG",
        layer1_score=score, layer2_score=score, mii=1.0,
        timestamp=datetime.now(UTC),
    )


def test_circuit_breaker_blocks_trades(es_instrument):
    """熔断器打开时，交易前检查应拦截所有新订单。"""
    engine = RiskEngine(RiskConfig())
    engine._circuit_breaker_open = True
    sig = make_signal(es_instrument)
    approved, qty, reason = engine.pre_trade_check(sig, 2, make_portfolio())
    assert not approved
    assert "Circuit breaker" in reason


def test_daily_loss_limit(es_instrument):
    """当日亏损超过限额时，交易前检查应拦截新订单。"""
    engine = RiskEngine(RiskConfig())
    sig = make_signal(es_instrument)
    portfolio = make_portfolio(daily_pnl=-6000.0)  # 超过 max_daily_loss_usd=5000
    approved, qty, reason = engine.pre_trade_check(sig, 2, portfolio)
    assert not approved
    assert "loss" in reason.lower()


def test_circuit_breaker_recovers(es_instrument):
    """冷却期结束后，熔断器应自动恢复（P0 修复验证）。"""
    cfg = RiskConfig(circuit_breaker_cooldown_bars=5)
    engine = RiskEngine(cfg)
    engine._circuit_breaker_open = True
    engine._cb_triggered_bar = 0
    # 模拟经过 cooldown_bars 根 K 线
    for i in range(6):
        engine.tick(i)
    assert not engine._circuit_breaker_open


def test_margin_check_rejects_if_insufficient(es_instrument):
    """现金不足以支付保证金时，交易前检查应拒绝订单。"""
    engine = RiskEngine(RiskConfig())
    sig = make_signal(es_instrument)
    # 现金仅 1000，保证金需要 15200 → 拒绝
    portfolio = make_portfolio(cash=1000.0)
    approved, qty, reason = engine.pre_trade_check(sig, 1, portfolio)
    assert not approved


def test_normal_trade_approved(es_instrument):
    """正常条件下，交易前检查应批准订单。"""
    engine = RiskEngine(RiskConfig())
    sig = make_signal(es_instrument)
    approved, qty, reason = engine.pre_trade_check(sig, 1, make_portfolio())
    assert approved
    assert qty == 1
