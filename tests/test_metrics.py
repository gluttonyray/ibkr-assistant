import pytest
from datetime import datetime, timezone
from quant.portfolio.metrics import compute_metrics
from quant.portfolio.book import TradeRecord

UTC = timezone.utc


def test_flat_equity_returns_zero_drawdown():
    """水平权益曲线的回撤应接近 0。"""
    equity = [100_000.0] * 100
    m = compute_metrics(equity, [])
    assert m.max_drawdown == pytest.approx(0.0, abs=0.001)
    assert m.total_return == pytest.approx(0.0, abs=0.001)
    # Sharpe 为负，因为超额收益 = 0 - rf_per_bar < 0
    assert m.sharpe_ratio < 0


def test_monotone_rising_equity():
    """单调上升的权益曲线：总收益 > 0，回撤 = 0，Sharpe > 0。"""
    equity = [100_000.0 + i * 100 for i in range(200)]
    m = compute_metrics(equity, [])
    assert m.total_return > 0
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-6)
    assert m.sharpe_ratio > 0


def test_drawdown_calculation():
    """验证最大回撤计算：先涨后跌 10%。"""
    # 先涨后跌 10%
    equity = [100_000.0] * 50 + [110_000.0] * 50 + [99_000.0] * 50
    m = compute_metrics(equity, [])
    # 峰值 = 110000，谷值 = 99000，回撤 = (110000-99000)/110000
    assert m.max_drawdown == pytest.approx(1 - 99000 / 110000, rel=0.01)


def test_win_rate_with_trades(es_instrument):
    """含交易记录时，胜率和交易计数应正确计算。"""
    now = datetime.now(UTC)
    trades = [
        TradeRecord(
            instrument=es_instrument, entry_time=now, exit_time=now,
            entry_price=4000.0, exit_price=4100.0, qty=1, direction=1,
            pnl=5000.0, pnl_usd=5000.0, commission=2.0, slippage_cost=12.5,
            entry_score=0.5, exit_reason="take_profit",
            regime_at_entry="TRENDING_STRONG", holding_bars=10,
        ),
        TradeRecord(
            instrument=es_instrument, entry_time=now, exit_time=now,
            entry_price=4000.0, exit_price=3900.0, qty=1, direction=1,
            pnl=-5000.0, pnl_usd=-5000.0, commission=2.0, slippage_cost=12.5,
            entry_score=0.3, exit_reason="stop_loss",
            regime_at_entry="RANGING", holding_bars=5,
        ),
    ]
    m = compute_metrics([100_000.0] * 100, trades)
    assert m.win_rate == pytest.approx(0.5)
    assert m.total_trades == 2
