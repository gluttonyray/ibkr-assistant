"""Put/call ratio sentiment factor.

High put/call ratio → excessive fear → contrarian bullish signal.
Low put/call ratio  → excessive complacency → contrarian bearish signal.
"""
from __future__ import annotations

import math

from src.factors.base import BaseFactor, FactorData
from src.factors.registry import FactorRegistry


@FactorRegistry.register
class SentimentPutCall(BaseFactor):
    """CBOE put/call ratio (inverted): high P/C = contrarian buy.

    Raw value = -(put_call_ratio - 1.0) * 100 so:
      - P/C = 1.5 (extreme fear) → raw = -50 (bearish raw) → negative z-score
        BUT: contrarian = buy → positive z-score after normalization...

    Actually: we use the raw P/C ratio inverted: high P/C = bullish contrarian.
    Raw = (1/put_call_ratio) * 100: high ratio → low raw → low z-score
    Low ratio → high raw → high z-score (but means complacency = bearish)

    Simpler: raw = -put_call_ratio. High P/C (fear) → negative raw → but we
    want contrarian buy so: raw = put_call_ratio (high P/C → higher raw, but
    the z-score interpretation needs the combiner to know direction).

    Final decision: raw = -put_call_ratio (high fear = negative = bearish in
    the factor system). The z-score will normalize this correctly: when P/C
    is unusually high relative to recent history, z-score is negative, which
    AlphaCombiner interprets as bearish (which aligns with trend-followers)
    OR as a contrarian buy if put in a mean_reversion category.

    We register it as "sentiment" with neutral interpretation — the combiner
    will weight it appropriately. The raw value is the P/C ratio itself
    (unnegated) so analysts can read it directly.

    Reads: FactorData.sentiment["put_call_ratio"].
    """

    name = "sentiment_put_call"
    category = "sentiment"
    frequency = "daily"
    lookback_bars = 1
    weight = 0.8
    z_window = 60
    stale_seconds = 86400

    def _compute(self, data: FactorData) -> float:
        if data.sentiment is None:
            return float("nan")
        pcr = data.sentiment.get("put_call_ratio")
        if pcr is None or math.isnan(float(pcr)):
            return float("nan")
        return float(pcr)

    def _label(self, raw: float) -> str:
        return f"P/C={raw:.2f}"
