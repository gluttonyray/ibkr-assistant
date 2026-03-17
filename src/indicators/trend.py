"""
Trend indicators:
  - EMA cross (9/21/55/200)
  - SMA cross (20/50/200)
  - MACD (12,26,9)
  - ADX (14) + DI lines
  - Parabolic SAR
  - Ichimoku (9,26,52)
  - Supertrend (10,3)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    import talib
    _TALIB = True
except ImportError:
    _TALIB = False

from src.indicators.base import BaseIndicator, SignalResult


# ─────────────────────────────────────────────────────────────────────────────
# EMA Cross
# ─────────────────────────────────────────────────────────────────────────────

class EMAcross(BaseIndicator):
    """
    Computes four EMAs (9, 21, 55, 200) and scores the cross alignment.
    Bullish stack: EMA9 > EMA21 > EMA55 > EMA200 → +1
    Bearish stack: EMA9 < EMA21 < EMA55 < EMA200 → -1
    Otherwise a partial score ∈ {-1, -0.33, +0.33, +1} → rounded.
    """
    weight = 1.5
    name = "EMA Cross"
    min_bars = 205

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        c = df["close"]
        e9 = ta.ema(c, 9).iloc[-1]
        e21 = ta.ema(c, 21).iloc[-1]
        e55 = ta.ema(c, 55).iloc[-1]
        e200 = ta.ema(c, 200).iloc[-1]

        checks = [e9 > e21, e21 > e55, e55 > e200]
        bullish = sum(checks)
        bearish = sum(not x for x in checks)

        if bullish == 3:
            sig, lbl = 1, f"full bull stack ({e9:.2f}>{e21:.2f}>{e55:.2f}>{e200:.2f})"
        elif bearish == 3:
            sig, lbl = -1, f"full bear stack ({e9:.2f}<{e21:.2f}<{e55:.2f}<{e200:.2f})"
        elif bullish >= 2:
            sig, lbl = 1, f"partial bull ({bullish}/3)"
        elif bearish >= 2:
            sig, lbl = -1, f"partial bear ({bearish}/3)"
        else:
            sig, lbl = 0, "mixed"

        return SignalResult(sig, round(e9, 4), lbl)


# ─────────────────────────────────────────────────────────────────────────────
# SMA Cross
# ─────────────────────────────────────────────────────────────────────────────

class SMAcross(BaseIndicator):
    """SMA 20/50/200 golden/death cross."""
    weight = 1.0
    name = "SMA Cross"
    min_bars = 205

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        c = df["close"]
        s20 = ta.sma(c, 20).iloc[-1]
        s50 = ta.sma(c, 50).iloc[-1]
        s200 = ta.sma(c, 200).iloc[-1]

        checks = [s20 > s50, s50 > s200, c.iloc[-1] > s200]
        bullish = sum(checks)
        bearish = sum(not x for x in checks)

        if bullish >= 3:
            sig, lbl = 1, f"golden ({s20:.2f}>{s50:.2f}>{s200:.2f})"
        elif bearish >= 3:
            sig, lbl = -1, f"death ({s20:.2f}<{s50:.2f}<{s200:.2f})"
        elif bullish >= 2:
            sig, lbl = 1, f"partial bull ({bullish}/3)"
        elif bearish >= 2:
            sig, lbl = -1, f"partial bear ({bearish}/3)"
        else:
            sig, lbl = 0, "mixed"

        return SignalResult(sig, round(s50, 4), lbl)


# ─────────────────────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────────────────────

class MACD(BaseIndicator):
    """MACD(12,26,9): signal line cross + histogram direction."""
    weight = 1.5
    name = "MACD"
    min_bars = 35

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty:
            return SignalResult(0, float("nan"), "no data")

        # pandas-ta column names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        line = macd_df.filter(like="MACD_").iloc[:, 0]
        hist = macd_df.filter(like="MACDh_").iloc[:, 0]
        sig_line = macd_df.filter(like="MACDs_").iloc[:, 0]

        macd_val = self._last(line)
        hist_val = self._last(hist)
        sig_val = self._last(sig_line)

        prev_hist = hist.dropna().iloc[-2] if len(hist.dropna()) >= 2 else hist_val

        if macd_val > sig_val and hist_val > 0 and hist_val > prev_hist:
            sig, lbl = 1, f"bullish hist={hist_val:.4f}"
        elif macd_val < sig_val and hist_val < 0 and hist_val < prev_hist:
            sig, lbl = -1, f"bearish hist={hist_val:.4f}"
        elif macd_val > sig_val:
            sig, lbl = 1, f"above signal ({macd_val:.4f}>{sig_val:.4f})"
        elif macd_val < sig_val:
            sig, lbl = -1, f"below signal ({macd_val:.4f}<{sig_val:.4f})"
        else:
            sig, lbl = 0, "at signal"

        return SignalResult(sig, round(macd_val, 4), lbl)


# ─────────────────────────────────────────────────────────────────────────────
# ADX + DI
# ─────────────────────────────────────────────────────────────────────────────

class ADXSignal(BaseIndicator):
    """ADX(14): trend strength + DI cross direction."""
    weight = 1.2
    name = "ADX+DI"
    min_bars = 30

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is None or adx_df.empty:
            return SignalResult(0, float("nan"), "no data")

        adx_col = adx_df.filter(like="ADX_").iloc[:, 0]
        dmp_col = adx_df.filter(like="DMP_").iloc[:, 0]
        dmn_col = adx_df.filter(like="DMN_").iloc[:, 0]

        adx = self._last(adx_col)
        dmp = self._last(dmp_col)
        dmn = self._last(dmn_col)

        if adx < 20:
            return SignalResult(0, round(adx, 2), f"weak trend (ADX={adx:.1f})")

        if dmp > dmn:
            sig, lbl = 1, f"+DI>{-dmn:.1f} ADX={adx:.1f}"
        elif dmn > dmp:
            sig, lbl = -1, f"-DI>{dmp:.1f} ADX={adx:.1f}"
        else:
            sig, lbl = 0, f"equal DI ADX={adx:.1f}"

        return SignalResult(sig, round(adx, 2), lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Parabolic SAR
# ─────────────────────────────────────────────────────────────────────────────

class ParabolicSAR(BaseIndicator):
    """Price above SAR → buy; below → sell."""
    weight = 1.0
    name = "Parabolic SAR"
    min_bars = 10

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        if _TALIB:
            sar = talib.SAR(df["high"].values, df["low"].values, acceleration=0.02, maximum=0.2)
            sar_val = float(sar[-1])
        else:
            psar = ta.psar(df["high"], df["low"], df["close"])
            if psar is None or psar.empty:
                return SignalResult(0, float("nan"), "no data")
            # pandas-ta returns PSARl (long), PSARs (short), PSARaf, PSARr
            long_col = psar.filter(like="PSARl_")
            short_col = psar.filter(like="PSARs_")
            # SAR value: whichever is not NaN in the last row
            row = psar.iloc[-1]
            long_vals = long_col.iloc[-1].dropna() if not long_col.empty else pd.Series(dtype=float)
            short_vals = short_col.iloc[-1].dropna() if not short_col.empty else pd.Series(dtype=float)
            if not long_vals.empty:
                sar_val = float(long_vals.iloc[0])
            elif not short_vals.empty:
                sar_val = float(short_vals.iloc[0])
            else:
                return SignalResult(0, float("nan"), "no SAR value")

        price = df["close"].iloc[-1]
        if price > sar_val:
            return SignalResult(1, round(sar_val, 4), f"price {price:.2f} > SAR {sar_val:.2f}")
        else:
            return SignalResult(-1, round(sar_val, 4), f"price {price:.2f} < SAR {sar_val:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Ichimoku
# ─────────────────────────────────────────────────────────────────────────────

class Ichimoku(BaseIndicator):
    """
    Ichimoku cloud (9,26,52).
    Score based on: price vs cloud, tenkan/kijun cross, kumo color.
    """
    weight = 1.3
    name = "Ichimoku"
    min_bars = 60

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        ich = ta.ichimoku(df["high"], df["low"], df["close"], tenkan=9, kijun=26, senkou=52)
        if ich is None or (isinstance(ich, tuple) and ich[0] is None):
            return SignalResult(0, float("nan"), "no data")

        # pandas-ta returns a tuple (span_df, signal_df) or a single df
        if isinstance(ich, tuple):
            ich_df = pd.concat([ich[0], ich[1]], axis=1)
        else:
            ich_df = ich

        price = df["close"].iloc[-1]

        # Column names vary by version — use filter
        tenkan = self._last(ich_df.filter(like="ITS_").iloc[:, 0]) if not ich_df.filter(like="ITS_").empty else float("nan")
        kijun = self._last(ich_df.filter(like="IKS_").iloc[:, 0]) if not ich_df.filter(like="IKS_").empty else float("nan")
        span_a = self._last(ich_df.filter(like="ISA_").iloc[:, 0]) if not ich_df.filter(like="ISA_").empty else float("nan")
        span_b = self._last(ich_df.filter(like="ISB_").iloc[:, 0]) if not ich_df.filter(like="ISB_").empty else float("nan")

        if any(math.isnan(v) for v in [tenkan, kijun, span_a, span_b]):
            return SignalResult(0, float("nan"), "insufficient Ichimoku data")

        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        bullish_cloud = span_a >= span_b

        score = 0
        # Price vs cloud
        if price > cloud_top:
            score += 2
        elif price < cloud_bot:
            score -= 2
        # Tenkan / Kijun
        if tenkan > kijun:
            score += 1
        elif tenkan < kijun:
            score -= 1
        # Cloud color
        if bullish_cloud:
            score += 1
        else:
            score -= 1

        if score >= 3:
            sig, lbl = 1, f"bullish (score={score})"
        elif score <= -3:
            sig, lbl = -1, f"bearish (score={score})"
        else:
            sig, lbl = 0, f"neutral (score={score})"

        return SignalResult(sig, round(tenkan, 4), lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend
# ─────────────────────────────────────────────────────────────────────────────

class Supertrend(BaseIndicator):
    """Supertrend(10,3): direction column."""
    weight = 1.2
    name = "Supertrend"
    min_bars = 20

    def _compute(self, df: pd.DataFrame) -> SignalResult:
        st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3)
        if st is None or st.empty:
            return SignalResult(0, float("nan"), "no data")

        # Direction column: SUPERTd_10_3.0
        dir_col = st.filter(like="SUPERTd_")
        val_col = st.filter(like="SUPERT_").iloc[:, 0] if not st.filter(like="SUPERT_").empty else None

        if dir_col.empty:
            return SignalResult(0, float("nan"), "no direction col")

        direction = self._last(dir_col.iloc[:, 0])
        val = self._last(val_col) if val_col is not None else float("nan")

        if direction == 1:
            return SignalResult(1, round(val, 4), f"uptrend (dir=1)")
        elif direction == -1:
            return SignalResult(-1, round(val, 4), f"downtrend (dir=-1)")
        else:
            return SignalResult(0, round(val, 4), "neutral")
