"""External data loader for factor backtesting.

Fetches and aligns sentiment, macro, and fundamental data to a daily date
lookup dictionary. Each entry maps a date to:
  {"sentiment": {...}, "macro": {...}, "fundamental": {...}}

Used by FactorBacktestEngine to populate FactorData at each bar.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional

import pandas as pd

from src.config import cfg

logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    """Return float or None if missing/NaN."""
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


class ExternalDataLoader:
    """Fetch and align all external factor data for a backtest date range.

    Parameters
    ----------
    fred_api_key : str, optional
        FRED API key. Falls back to FRED_API_KEY env var.
    """

    def __init__(self, fred_api_key: Optional[str] = None) -> None:
        self.fred_api_key = fred_api_key or cfg.fred_api_key

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load_aligned(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> Dict[date, dict]:
        """Fetch all external data and return a date → data dict.

        Parameters
        ----------
        symbol : str
            Ticker for fundamental data (e.g. "AAPL").
        start, end : str
            Date range in "YYYY-MM-DD" format.

        Returns
        -------
        dict
            Keys are ``datetime.date`` objects. Values are::

                {
                    "sentiment": {"vix": float|None, "vix3m": float|None,
                                  "put_call_ratio": float|None},
                    "macro":     {"spread": float|None, "fed_rate": float|None,
                                  "dxy": float|None},
                    "fundamental": {"trailing_pe": float|None, ...},
                }
        """
        logger.info("Loading external data for %s [%s → %s]", symbol, start, end)

        # ── Sentiment ────────────────────────────────────────────────
        sentiment_df = self._load_sentiment(start, end)

        # ── Macro ────────────────────────────────────────────────────
        macro_df = self._load_macro(start, end)

        # ── Fundamental (single snapshot — forward-filled across dates) ──
        fundamental = self._load_fundamental(symbol)

        # ── Align to business-day index and forward-fill ─────────────
        bdays = pd.bdate_range(start, end)
        daily = pd.DataFrame(index=bdays)

        for col in sentiment_df.columns:
            daily[col] = sentiment_df[col].reindex(bdays).ffill()
        for col in macro_df.columns:
            daily[col] = macro_df[col].reindex(bdays).ffill()

        # ── Build date lookup ─────────────────────────────────────────
        lookup: Dict[date, dict] = {}
        for ts, row in daily.iterrows():
            d = ts.date()
            lookup[d] = {
                "sentiment": {
                    "vix": _safe_float(row.get("vix")),
                    "vix3m": _safe_float(row.get("vix3m")),
                    "put_call_ratio": _safe_float(row.get("put_call_ratio")),
                },
                "macro": {
                    "spread": _safe_float(row.get("spread")),
                    "fed_rate": _safe_float(row.get("fed_rate")),
                    "dxy": _safe_float(row.get("dxy")),
                },
                "fundamental": fundamental,
            }

        n_sentiment = sum(1 for v in lookup.values() if v["sentiment"]["vix"] is not None)
        n_macro = sum(1 for v in lookup.values() if v["macro"]["spread"] is not None)
        logger.info(
            "External data loaded: %d dates | sentiment coverage=%.0f%% | macro coverage=%.0f%%",
            len(lookup),
            100 * n_sentiment / max(len(lookup), 1),
            100 * n_macro / max(len(lookup), 1),
        )
        return lookup

    # ------------------------------------------------------------------
    # Internal fetchers
    # ------------------------------------------------------------------

    def _load_sentiment(self, start: str, end: str) -> pd.DataFrame:
        from src.data.yahoo import fetch_vix, fetch_vix3m, fetch_put_call_ratio
        df = pd.DataFrame()
        try:
            df["vix"] = fetch_vix(start, end)
        except Exception as exc:
            logger.warning("VIX fetch failed: %s", exc)
        try:
            df["vix3m"] = fetch_vix3m(start, end)
        except Exception as exc:
            logger.warning("VIX3M fetch failed: %s", exc)
        try:
            df["put_call_ratio"] = fetch_put_call_ratio(start, end)
        except Exception as exc:
            logger.warning("Put/call ratio fetch failed: %s", exc)
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    def _load_macro(self, start: str, end: str) -> pd.DataFrame:
        from src.data.yahoo import fetch_dxy
        df = pd.DataFrame()
        # Yield curve (FRED)
        if self.fred_api_key:
            try:
                from src.data.fred import fetch_yield_curve, fetch_fed_rate
                yc = fetch_yield_curve(start, end, api_key=self.fred_api_key)
                if not yc.empty:
                    yc.index = pd.to_datetime(yc.index).tz_localize(None)
                    df["spread"] = yc["spread"]
                fr = fetch_fed_rate(start, end, api_key=self.fred_api_key)
                if not fr.empty:
                    fr.index = pd.to_datetime(fr.index).tz_localize(None)
                    df["fed_rate"] = fr
            except Exception as exc:
                logger.warning("FRED fetch failed: %s", exc)
        else:
            logger.warning("FRED_API_KEY not set — macro factors (yield curve, fed rate) disabled")
        # DXY (Yahoo)
        try:
            dxy = fetch_dxy(start, end)
            if not dxy.empty:
                dxy.index = pd.to_datetime(dxy.index).tz_localize(None)
                df["dxy"] = dxy
        except Exception as exc:
            logger.warning("DXY fetch failed: %s", exc)
        return df

    def _load_fundamental(self, symbol: str) -> dict:
        try:
            from src.data.yahoo import fetch_fundamentals
            data = fetch_fundamentals(symbol)
            logger.info("Fundamentals for %s: PE=%.1f PB=%.2f rev_growth=%.2f%%",
                symbol,
                data.get("trailing_pe") or 0,
                data.get("price_to_book") or 0,
                (data.get("revenue_growth") or 0) * 100,
            )
            return data
        except Exception as exc:
            logger.warning("Fundamentals fetch failed for %s: %s", symbol, exc)
            return {}
