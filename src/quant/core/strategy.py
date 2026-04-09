"""策略协议 —— Layer 1 与 Layer 2 接口。

Layer 1（宏观/状态）：接收所有合约的 K 线窗口，
输出每个合约的状态分类和 Beta 得分。

Layer 2（Alpha）：接收单个合约的 K 线窗口及 Layer 1 结果，
输出带因子分解的 Alpha 得分。

SignalComposer：使用公式  S = w_L1 * S_L1 * MII + w_L2 * S_L2
将 Layer 1 + Layer 2 合并为最终的 Signal。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from quant.core.types import Bar, Instrument, Signal


@dataclass(frozen=True, slots=True)
class Layer1Result:
    """单个合约的 Layer 1 输出。"""
    regime: str                 # 如 "TRENDING_CALM"
    beta_score: float           # [-1, +1] 宏观方向偏差
    mii: float                  # [0, 1] 动量强度指数
    confidence: float           # [0, 1]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Layer2Result:
    """单个合约的 Layer 2 输出。"""
    alpha_score: float          # [-1, +1]
    factor_details: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Layer1Strategy(Protocol):
    """宏观/状态层 —— 跨资产分析。"""

    def compute(
        self,
        windows: dict[Instrument, list[Bar]],
        external_data: dict[str, Any] | None = None,
    ) -> dict[Instrument, Layer1Result]:
        """计算所有合约的状态和宏观得分。"""
        ...


@runtime_checkable
class Layer2Strategy(Protocol):
    """Alpha 因子层 —— 单合约分析。"""

    def compute(
        self,
        instrument: Instrument,
        window: list[Bar],
        layer1: Layer1Result,
    ) -> Layer2Result:
        """计算单个合约的 Alpha 得分。"""
        ...


@runtime_checkable
class SignalComposer(Protocol):
    """将 Layer 1 + Layer 2 合并为最终的交易信号。"""

    def compose(
        self,
        instrument: Instrument,
        layer1: Layer1Result,
        layer2: Layer2Result,
        timestamp: Any,
    ) -> Signal:
        """从两层结果生成合并后的 Signal。"""
        ...
