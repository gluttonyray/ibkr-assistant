"""IBKR 历史数据速率限制器。

IBKR 对历史数据请求实施严格的节流限制：

按合约：
    - 相同请求（相同合约 + K 线周期）之间最少间隔 10 秒

全局：
    - 每分钟最多 6 个请求（任意合约）
    - 10 分钟内最多 60 个请求

违反节流限制会触发临时封禁（1-10 分钟）。本模块实现了
令牌桶速率限制器 + 指数退避以保持在限制范围内。

用法::

    limiter = IBKRRateLimiter()
    await limiter.acquire("ES")     # 阻塞直到可安全请求
    # ... 发送 IBKR 请求 ...
    await limiter.acquire("NQ")     # 遵守全局限制
"""
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class IBKRRateLimiter:
    """IBKR 历史数据 API 的异步速率限制器。

    同时执行按合约和全局速率限制，避免节流违规。

    Parameters
    ----------
    per_contract_interval : float
        同一合约请求之间的最小间隔秒数。
    global_max_per_minute : int
        60 秒滑动窗口内最大请求数（任意合约）。
    backoff_base : float
        节流违规后指数退避的基础延迟（秒）。
    backoff_max : float
        最大退避延迟（秒）。
    """

    def __init__(
        self,
        per_contract_interval: float = 10.0,
        global_max_per_minute: int = 6,
        backoff_base: float = 15.0,
        backoff_max: float = 300.0,
    ) -> None:
        self._per_contract_interval = per_contract_interval
        self._global_max_per_minute = global_max_per_minute
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

        # 按合约：上次请求时间戳
        self._last_request: dict[str, float] = {}

        # 全局：滑动窗口内的请求时间戳
        self._global_requests: list[float] = []

        # 退避状态
        self._consecutive_violations = 0

        # 串行化访问
        self._lock = asyncio.Lock()

    async def acquire(self, symbol: str) -> None:
        """等待直到可以安全地对 *symbol* 发起请求。

        Parameters
        ----------
        symbol : str
            合约品种（例如 ``"ES"``、``"HSI"``）。
        """
        async with self._lock:
            now = time.monotonic()

            # 1. 按合约节流
            last = self._last_request.get(symbol, 0.0)
            per_contract_wait = max(0.0, self._per_contract_interval - (now - last))

            # 2. 全局节流 -- 滑动窗口
            self._global_requests = [
                t for t in self._global_requests if now - t < 60.0
            ]
            global_wait = 0.0
            if len(self._global_requests) >= self._global_max_per_minute:
                oldest = self._global_requests[0]
                global_wait = 60.0 - (now - oldest) + 0.1  # 小缓冲

            wait = max(per_contract_wait, global_wait)
            if wait > 0:
                logger.debug(
                    "Rate limiter: waiting %.1fs for %s (per_contract=%.1f, global=%.1f)",
                    wait, symbol, per_contract_wait, global_wait,
                )
                await asyncio.sleep(wait)

            # 记录本次请求
            now = time.monotonic()
            self._last_request[symbol] = now
            self._global_requests.append(now)

    def report_pacing_violation(self) -> float:
        """上报 IBKR 节流违规并计算退避延迟。

        当 IBKR 返回节流违规错误时调用此方法。

        Returns
        -------
        float
            建议的退避延迟（秒）。
        """
        self._consecutive_violations += 1
        delay = min(
            self._backoff_base * (2 ** (self._consecutive_violations - 1)),
            self._backoff_max,
        )
        logger.warning(
            "Pacing violation #%d, backing off %.1fs",
            self._consecutive_violations, delay,
        )
        return delay

    def report_success(self) -> None:
        """上报请求成功，重置退避状态。"""
        if self._consecutive_violations > 0:
            logger.debug("Resetting backoff after %d violations", self._consecutive_violations)
            self._consecutive_violations = 0

    @property
    def consecutive_violations(self) -> int:
        """连续节流违规次数。"""
        return self._consecutive_violations
