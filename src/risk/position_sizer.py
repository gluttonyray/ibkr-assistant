"""ATR-based position sizing with conviction scaling.

Formula:
    risk_budget   = equity × max_risk_per_trade
    risk_per_share = stop_atr_multiple × ATR(14)
    base_shares   = risk_budget / risk_per_share
    scaled_shares = base_shares × |alpha_score|   (conviction scaling)
    max_shares    = equity × max_position_pct / price
    final_shares  = min(scaled_shares, max_shares)
"""
from __future__ import annotations


class PositionSizer:
    """Compute position size in shares from risk parameters.

    Parameters
    ----------
    max_risk_per_trade : float
        Maximum fraction of equity to risk on a single trade (default 2%).
    max_position_pct : float
        Maximum single-position size as fraction of equity (default 20%).
    stop_atr_multiple : float
        Stop-loss distance in ATR units (default 2.0).
    min_shares : int
        Minimum shares returned when a non-zero size is computed (default 1).
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.02,
        max_position_pct: float = 0.20,
        stop_atr_multiple: float = 2.0,
        min_shares: int = 1,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.stop_atr_multiple = stop_atr_multiple
        self.min_shares = min_shares

    def compute_size(
        self,
        alpha_score: float,
        price: float,
        atr: float,
        equity: float,
    ) -> int:
        """Return number of shares to trade.

        Parameters
        ----------
        alpha_score : float
            Alpha signal in [-1, +1]. Magnitude = conviction.
        price : float
            Current market price per share.
        atr : float
            Current ATR(14) value.
        equity : float
            Current portfolio equity.

        Returns
        -------
        int
            Shares to trade (always non-negative; direction comes from signal).
        """
        if price <= 0.0 or atr <= 0.0 or equity <= 0.0:
            return self.min_shares

        abs_score = abs(alpha_score)
        if abs_score < 1e-6:
            return 0

        risk_budget = equity * self.max_risk_per_trade
        risk_per_share = self.stop_atr_multiple * atr
        if risk_per_share < 1e-6:
            return self.min_shares

        base_shares = risk_budget / risk_per_share
        scaled_shares = base_shares * abs_score
        max_shares = equity * self.max_position_pct / price

        result = int(min(scaled_shares, max_shares))
        return max(result, self.min_shares)
