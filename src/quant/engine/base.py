"""回测与实盘交易循环的抽象基类引擎。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from quant.core.types import Bar, Fill


class BaseEngine(ABC):
    """提供 setup / loop / teardown 生命周期的抽象引擎。"""

    @abstractmethod
    def _setup(self) -> None: ...

    @abstractmethod
    def _on_bar(self, bar_index: int, bars: dict[str, Bar]) -> None: ...

    @abstractmethod
    def _on_fill(self, fill: Fill) -> None: ...

    @abstractmethod
    def _teardown(self) -> None: ...

    @abstractmethod
    def _run_loop(self) -> None: ...

    def run(self) -> None:
        """执行完整的引擎生命周期。"""
        self._setup()
        self._run_loop()
        self._teardown()
