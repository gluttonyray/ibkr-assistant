import pytest
from datetime import datetime, timezone
from quant.portfolio.book import PortfolioBook
from quant.portfolio.cost_model import FuturesCostModel
from quant.config.schema import CostConfig
from quant.core.types import Fill, Side
from tests.conftest import make_bars

UTC = timezone.utc


def test_initial_state(es_instrument):
    """验证初始账本状态：权益等于现金，无持仓。"""
    book = PortfolioBook(100_000.0)
    assert book.equity_usd == pytest.approx(100_000.0)
    assert book.cash_usd == pytest.approx(100_000.0)
    assert book.positions() == {}


def test_open_long_reduces_cash(es_instrument):
    """开多头仓位后，现金应减少（保证金 + 费用）。"""
    book = PortfolioBook(100_000.0)
    cost_model = FuturesCostModel(CostConfig())
    fill = Fill(
        instrument=es_instrument, side=Side.BUY, qty=1,
        price=4000.0, timestamp=datetime.now(UTC),
        commission=0.85, exchange_fees=1.30, slippage_cost=12.50,
    )
    book.on_fill(fill, cost_model, hkd_usd_rate=0.128)
    # 现金应减少：保证金 + 费用
    assert book.cash_usd < 100_000.0
    assert es_instrument.symbol in book.positions()


def test_mtm_updates_unrealized_pnl(es_instrument):
    """盯市估值应正确更新未实现盈亏。"""
    book = PortfolioBook(100_000.0)
    cost_model = FuturesCostModel(CostConfig())
    fill = Fill(
        instrument=es_instrument, side=Side.BUY, qty=1,
        price=4000.0, timestamp=datetime.now(UTC),
        commission=0.85, exchange_fees=1.30, slippage_cost=12.50,
    )
    book.on_fill(fill, cost_model, hkd_usd_rate=0.128)
    bars = make_bars(es_instrument, 5, start_price=4100.0)
    bar_dict = {es_instrument.symbol: bars[-1]}
    book.mark_to_market(bar_dict, hkd_usd_rate=0.128)
    pos = book.get_position(es_instrument)
    assert pos is not None
    # 价格从 4000 涨至约 4100，多头 1 张，未实现盈亏应大于 0
    assert pos.unrealized_pnl > 0


def test_hkd_margin_converts_to_usd(hsi_instrument):
    """港币保证金应按汇率正确折算为美元。"""
    book = PortfolioBook(200_000.0)
    cost_model = FuturesCostModel(CostConfig())
    hkd_rate = 0.128
    fill = Fill(
        instrument=hsi_instrument, side=Side.BUY, qty=1,
        price=20000.0, timestamp=datetime.now(UTC),
        commission=20.0, exchange_fees=10.0, slippage_cost=50.0,
    )
    before = book.cash_usd
    book.on_fill(fill, cost_model, hkd_usd_rate=hkd_rate)
    # 保证金 132180 港币 × 0.128 + 费用（美元）
    margin_usd = 132180.0 * hkd_rate
    assert book.cash_usd < before
    assert book.cash_usd == pytest.approx(
        before - margin_usd - (20.0 + 10.0 + 50.0) * hkd_rate, rel=0.01
    )
