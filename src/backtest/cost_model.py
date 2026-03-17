"""
IBKR cost model: commissions, regulatory fees, slippage, and margin interest.

Implements IBKR US Tiered pricing as of 2024:
  - Commission: $0.005/share, min $1.00, max 1% of trade value
  - SEC fee (sell only): $8.00 per $1M (current rate)
  - TAF fee (sell only): $0.000166/share, max $8.30
  - FINRA fee: $0.00278/share (sell-side only)
  - Slippage: configurable basis points
  - Margin interest: annualized rate / 360 * days held * notional
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillCosts:
    """Itemised costs for a single fill."""
    commission: float
    sec_fee: float        # sell only
    taf_fee: float        # sell only
    finra_fee: float      # sell only
    slippage: float
    total: float

    @staticmethod
    def zero() -> "FillCosts":
        return FillCosts(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class CostModel:
    """
    Calculates realistic trading costs for IBKR US equities.

    Parameters
    ----------
    slippage_bps : float
        Slippage in basis points per fill (default 5 = 0.05%).
    margin_rate : float
        Annual margin interest rate (default 6.83% — IBKR current USD rate).
    """

    # IBKR tiered commission
    COMM_PER_SHARE: float = 0.005
    COMM_MIN: float = 1.00
    COMM_MAX_PCT: float = 0.01   # 1% of trade value

    # Regulatory (sell-side only)
    SEC_RATE: float = 8.00 / 1_000_000     # $8 per $1M
    TAF_PER_SHARE: float = 0.000166
    TAF_MAX: float = 8.30
    FINRA_PER_SHARE: float = 0.00278

    def __init__(
        self,
        slippage_bps: float = 5.0,
        margin_rate: float = 0.0683,
    ) -> None:
        self.slippage_bps = slippage_bps
        self.margin_rate = margin_rate

    def fill_cost(self, price: float, qty: int, side: str) -> FillCosts:
        """
        Calculate all-in costs for a single fill.

        Parameters
        ----------
        price : float
            Fill price per share.
        qty : int
            Number of shares (always positive).
        side : str
            "BUY" or "SELL".
        """
        qty = abs(qty)
        notional = price * qty
        is_sell = side.upper() == "SELL"

        # Commission
        raw_comm = self.COMM_PER_SHARE * qty
        commission = max(self.COMM_MIN, min(raw_comm, self.COMM_MAX_PCT * notional))

        # Regulatory (sell only)
        sec_fee = round(notional * self.SEC_RATE, 2) if is_sell else 0.0
        taf_fee = min(self.TAF_PER_SHARE * qty, self.TAF_MAX) if is_sell else 0.0
        finra_fee = self.FINRA_PER_SHARE * qty if is_sell else 0.0

        # Slippage
        slippage = notional * self.slippage_bps / 10_000

        total = commission + sec_fee + taf_fee + finra_fee + slippage

        return FillCosts(
            commission=round(commission, 4),
            sec_fee=round(sec_fee, 4),
            taf_fee=round(taf_fee, 4),
            finra_fee=round(finra_fee, 4),
            slippage=round(slippage, 4),
            total=round(total, 4),
        )

    def margin_interest(self, notional: float, days: int) -> float:
        """Daily-compounded margin interest for *days* held."""
        if days <= 0 or notional <= 0:
            return 0.0
        return round(notional * self.margin_rate / 360 * days, 4)

    def limit_order_fills(
        self, limit_price: float, side: str, next_high: float, next_low: float
    ) -> bool:
        """
        Simulate whether a limit order fills based on next bar's range.

        BUY limit fills if next bar's low <= limit_price.
        SELL limit fills if next bar's high >= limit_price.
        """
        if side.upper() == "BUY":
            return next_low <= limit_price
        else:
            return next_high >= limit_price
