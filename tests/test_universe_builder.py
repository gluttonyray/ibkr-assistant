"""UniverseBuilder 单元测试。

使用 mock HTTP 响应，不依赖网络。
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quant.data.universe.ticker_universe import AssetClass
from quant.data.universe.universe_builder import (
    UniverseBuilder,
    _is_valid_us_ticker,
)

# ── 模拟数据 ──────────────────────────────────────────────────────────────

# NASDAQ 官方格式：| 分隔，Symbol 第 0 列，ETF 第 6 列
_MOCK_NASDAQ_DATA = """\
Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N
GOOG|Alphabet Inc. - Class C Capital Stock|Q|N|N|100|N|N
AMZN|Amazon.com Inc. - Common Stock|Q|N|N|100|N|N
TSLA|Tesla Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
TQQQ|ProShares UltraPro QQQ|G|N|N|100|Y|N
BAD SPACE|Bad Ticker With Space|Q|N|N|100|N|N
TOOLNG|Too Long Ticker Symbol|Q|N|N|100|N|N
WARRX|Some Warrant Issue|Q|N|N|100|N|N
TE$T|Test Symbol|Q|N|N|100|N|N
^NDX|Index Symbol|Q|N|N|100|N|N
NVDA|NVIDIA Corporation|Q|N|N|100|N|N
META|Meta Platforms Inc.|Q|N|N|100|N|N
NFLX|Netflix Inc.|Q|N|N|100|N|N
File Creation Time: 2026-04-01|||||
"""

# NYSE/AMEX 格式：ACT Symbol 第 0 列，ETF 第 4 列
_MOCK_OTHER_DATA = """\
ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
JPM|JPMorgan Chase & Co.|N|JPM|N|100|N|JPM
BAC|Bank of America Corporation|N|BAC|N|100|N|BAC
WMT|Walmart Inc.|N|WMT|N|100|N|WMT
V|Visa Inc. - Class A|N|V|N|100|N|V
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
DIA|SPDR Dow Jones Industrial Average ETF|P|DIA|Y|100|N|DIA
BRK A|Berkshire Hathaway|N|BRK A|N|100|N|BRK A
ABCDEF|Foreign Private Issue|A|ABCDEF|N|100|N|ABCDEF
UNITX|Some Unit Issue|N|UNITX|N|100|N|UNITX
GS|Goldman Sachs|N|GS|N|100|N|GS
MA|Mastercard Inc.|N|MA|N|100|N|MA
HD|Home Depot Inc.|N|HD|N|100|N|HD
File Creation Time: 2026-04-01|||||
"""


def _mock_urlopen(req, timeout=30):
    """根据 URL 返回对应的 mock 响应。"""
    url = req.full_url if hasattr(req, "full_url") else str(req)
    if "nasdaqlisted" in url:
        data = _MOCK_NASDAQ_DATA.encode("utf-8")
    elif "otherlisted" in url:
        data = _MOCK_OTHER_DATA.encode("utf-8")
    else:
        raise ValueError(f"未知 URL: {url}")

    resp = MagicMock()
    resp.read.return_value = data
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── 过滤规则测试 ──────────────────────────────────────────────────────────


class TestTickerFilter:
    """测试品种代码过滤规则。"""

    def test_valid_tickers(self):
        """正常品种代码应通过。"""
        for symbol in ["AAPL", "MSFT", "A", "GOOG", "SPY"]:
            assert _is_valid_us_ticker(symbol), f"{symbol} 应为合法品种"

    def test_reject_space(self):
        """含空格的品种代码应被排除。"""
        assert not _is_valid_us_ticker("BRK A")
        assert not _is_valid_us_ticker("BF B")

    def test_reject_caret(self):
        """含 ^ 的品种代码应被排除。"""
        assert not _is_valid_us_ticker("^NDX")
        assert not _is_valid_us_ticker("^GSPC")

    def test_reject_dollar(self):
        """含 $ 的测试品种应被排除。"""
        assert not _is_valid_us_ticker("TE$T")

    def test_reject_long_ticker(self):
        """长度 > 5 的品种代码应被排除（外国私募）。"""
        assert not _is_valid_us_ticker("ABCDEF")
        assert _is_valid_us_ticker("ABCDE")  # 恰好 5 位应保留

    def test_reject_warrant_unit_right(self):
        """含 warr/unit/right 的品种应被排除。"""
        assert not _is_valid_us_ticker("WARRX")
        assert not _is_valid_us_ticker("UNITX")
        assert not _is_valid_us_ticker("RIGHT")

    def test_allow_futures(self):
        """=F 结尾的期货连续合约应保留。"""
        assert _is_valid_us_ticker("ES=F")
        assert _is_valid_us_ticker("GC=F")
        # 即使长度 > 5 的期货也保留
        assert _is_valid_us_ticker("LONGFUTURE=F")

    def test_reject_empty(self):
        """空字符串应被排除。"""
        assert not _is_valid_us_ticker("")
        assert not _is_valid_us_ticker("  ")


# ── UniverseBuilder 测试 ──────────────────────────────────────────────────


class TestUniverseBuilder:
    """测试 UniverseBuilder 的 build() 方法。"""

    @patch("quant.data.universe.universe_builder.urlopen", side_effect=_mock_urlopen)
    def test_build_returns_sufficient_tickers(self, mock_url):
        """build() 应返回 >= 50 品种（含 mock 股票 + 默认期货/港股/加密货币）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(cache_dir=tmpdir)
            universe = builder.build()

            # mock 数据中有效美股约 12 只 + 默认 ETF ~40 + 期货 16 + 加密 50 + 港股 23
            assert universe.total_count >= 50, (
                f"品种总数 {universe.total_count} 应 >= 50"
            )

    @patch("quant.data.universe.universe_builder.urlopen", side_effect=_mock_urlopen)
    def test_filter_rules_applied(self, mock_url):
        """过滤规则应正确排除非法品种。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(cache_dir=tmpdir)
            universe = builder.build(
                include_futures=False,
                include_crypto=False,
                include_hk=False,
            )

            all_tickers = universe.get_tickers()

            # 应包含的有效品种
            assert "AAPL" in all_tickers
            assert "MSFT" in all_tickers
            assert "JPM" in all_tickers
            assert "GS" in all_tickers

            # 应排除的品种
            assert "BAD SPACE" not in all_tickers  # 含空格
            assert "^NDX" not in all_tickers  # 含 ^
            assert "TE$T" not in all_tickers  # 含 $
            assert "TOOLNG" not in all_tickers  # 长度 > 5
            assert "WARRX" not in all_tickers  # 含 warr
            assert "UNITX" not in all_tickers  # 含 unit
            assert "ABCDEF" not in all_tickers  # 长度 > 5
            assert "BRK A" not in all_tickers  # 含空格

    @patch("quant.data.universe.universe_builder.urlopen", side_effect=_mock_urlopen)
    def test_cache_hit_no_http(self, mock_url):
        """缓存命中时不应重复发送 HTTP 请求。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(cache_dir=tmpdir)

            # 第一次构建：触发 HTTP 请求
            builder.build()
            first_call_count = mock_url.call_count
            assert first_call_count == 2  # nasdaqlisted + otherlisted

            # 第二次构建：应命中缓存，不触发 HTTP
            builder.build()
            assert mock_url.call_count == first_call_count, (
                "缓存命中时不应发送额外 HTTP 请求"
            )

    @patch("quant.data.universe.universe_builder.urlopen", side_effect=_mock_urlopen)
    def test_etf_classification(self, mock_url):
        """ETF 应被正确分类到 US_ETFS。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(cache_dir=tmpdir)
            universe = builder.build(
                include_futures=False,
                include_crypto=False,
                include_hk=False,
            )

            etfs = universe.get_tickers(AssetClass.US_ETFS)
            stocks = universe.get_tickers(AssetClass.US_STOCKS)

            # mock 数据中的 ETF
            assert "QQQ" in etfs
            assert "TQQQ" in etfs
            assert "SPY" in etfs
            assert "DIA" in etfs

            # mock 数据中的个股不应在 ETF 列表
            assert "AAPL" not in etfs
            assert "AAPL" in stocks

    @patch("quant.data.universe.universe_builder.urlopen", side_effect=_mock_urlopen)
    def test_max_stocks_limit(self, mock_url):
        """max_stocks 应限制个股数量。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(cache_dir=tmpdir)
            universe = builder.build(
                include_etfs=False,
                include_futures=False,
                include_crypto=False,
                include_hk=False,
                max_stocks=3,
            )

            stocks = universe.get_tickers(AssetClass.US_STOCKS)
            assert len(stocks) == 3

    @patch("quant.data.universe.universe_builder.urlopen", side_effect=_mock_urlopen)
    def test_selective_asset_classes(self, mock_url):
        """应支持按资产类别选择性构建。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(cache_dir=tmpdir)

            # 仅加密货币
            universe = builder.build(
                include_stocks=False,
                include_etfs=False,
                include_futures=False,
                include_crypto=True,
                include_hk=False,
            )

            assert universe.count(AssetClass.US_STOCKS) == 0
            assert universe.count(AssetClass.US_ETFS) == 0
            assert universe.count(AssetClass.CRYPTO) > 0
            # 不应有任何 HTTP 请求（股票/ETF 未请求）
            assert mock_url.call_count == 0

    @patch("quant.data.universe.universe_builder.urlopen", side_effect=_mock_urlopen)
    def test_cache_expiry_refetch(self, mock_url):
        """缓存过期后应重新下载。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(cache_dir=tmpdir, cache_ttl_days=0)

            # 第一次构建
            builder.build(
                include_futures=False,
                include_crypto=False,
                include_hk=False,
            )
            assert mock_url.call_count == 2

            # TTL=0 意味着缓存立即过期，第二次应重新下载
            builder.build(
                include_futures=False,
                include_crypto=False,
                include_hk=False,
            )
            assert mock_url.call_count == 4
