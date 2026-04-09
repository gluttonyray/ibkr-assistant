"""InstrumentRegistry —— 加载并查询合约规格。

一次性读取 ``configs/instruments.yaml``，构建不可变的 ``Instrument``
对象，并提供按代码、交易所或货币查询的接口。

用法::

    registry = InstrumentRegistry.from_yaml("configs/instruments.yaml")
    es = registry.get("ES")
    hk_instruments = registry.by_currency(Currency.HKD)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from quant.core.types import (
    Currency,
    Instrument,
    InstrumentType,
    TradingWindow,
)


class InstrumentRegistry:
    """可交易合约的不可变注册表。

    Parameters
    ----------
    instruments : dict[str, Instrument]
        以合约代码为键的预构建合约映射表。
    """

    def __init__(self, instruments: dict[str, Instrument]) -> None:
        self._instruments = dict(instruments)

    # -- 构建 -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> InstrumentRegistry:
        """从 YAML 文件加载合约规格。

        Parameters
        ----------
        path : str or Path
            合约 YAML 文件路径。

        Raises
        ------
        FileNotFoundError
            若文件不存在。
        ValueError
            若必填字段缺失或无效。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Instruments file not found: {path}")

        with open(path) as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict) or "instruments" not in raw:
            raise ValueError(f"Expected top-level 'instruments' key in {path}")

        instruments: dict[str, Instrument] = {}
        for symbol, spec in raw["instruments"].items():
            instruments[symbol] = _parse_instrument(symbol, spec)

        return cls(instruments)

    # -- 查询 ------------------------------------------------------------

    def get(self, symbol: str) -> Instrument:
        """按代码返回合约；若不存在则抛出 KeyError。"""
        try:
            return self._instruments[symbol]
        except KeyError:
            available = ", ".join(sorted(self._instruments))
            raise KeyError(
                f"Unknown instrument '{symbol}'. Available: {available}"
            ) from None

    def all(self) -> list[Instrument]:
        """按插入顺序返回所有合约。"""
        return list(self._instruments.values())

    def symbols(self) -> list[str]:
        """返回所有合约代码名称。"""
        return list(self._instruments.keys())

    def by_exchange(self, exchange: str) -> list[Instrument]:
        """返回在 *exchange* 交易所上市的合约。"""
        return [i for i in self._instruments.values() if i.exchange == exchange]

    def by_currency(self, currency: Currency) -> list[Instrument]:
        """返回以 *currency* 结算的合约。"""
        return [i for i in self._instruments.values() if i.currency == currency]

    def update_margin(
        self,
        symbol: str,
        margin_initial: float,
        margin_maintenance: float,
    ) -> None:
        """用 IBKR 动态查询结果更新指定品种的保证金。

        由于 Instrument 是 frozen dataclass，使用 dataclasses.replace() 创建新实例。
        """
        if symbol not in self._instruments:
            raise KeyError(f"Unknown instrument: {symbol}")
        self._instruments[symbol] = dataclasses.replace(
            self._instruments[symbol],
            margin_initial=margin_initial,
            margin_maintenance=margin_maintenance,
        )

    def __len__(self) -> int:
        return len(self._instruments)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._instruments

    def __iter__(self):
        return iter(self._instruments.values())

    def __repr__(self) -> str:
        syms = ", ".join(self._instruments.keys())
        return f"InstrumentRegistry([{syms}])"


# ---------------------------------------------------------------------------
# YAML 解析辅助函数
# ---------------------------------------------------------------------------

def _parse_instrument(symbol: str, spec: dict) -> Instrument:
    """从 YAML 解析单个合约条目。"""
    _require_keys(symbol, spec, [
        "type", "exchange", "currency", "multiplier",
        "tick_size", "margin_initial", "margin_maintenance", "sessions",
    ])

    sessions = tuple(
        _parse_session(s) for s in spec["sessions"]
    )

    return Instrument(
        symbol=symbol,
        instrument_type=InstrumentType(spec["type"]),
        exchange=spec["exchange"],
        currency=Currency(spec["currency"]),
        multiplier=float(spec["multiplier"]),
        tick_size=float(spec["tick_size"]),
        margin_initial=float(spec["margin_initial"]),
        margin_maintenance=float(spec["margin_maintenance"]),
        sessions=sessions,
    )


def _parse_session(raw: dict) -> TradingWindow:
    """解析单个交易时段条目。"""
    return TradingWindow(
        start=raw["start"],
        end=raw["end"],
        timezone=raw["timezone"],
        label=raw.get("label", ""),
    )


def _require_keys(symbol: str, spec: dict, keys: list[str]) -> None:
    """若有必填键缺失则抛出 ValueError。"""
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(
            f"Instrument '{symbol}' is missing required fields: {missing}"
        )
