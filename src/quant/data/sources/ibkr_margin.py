"""从 IBKR TWS/Gateway 动态拉取最新合约保证金。

使用 ``whatIfOrder`` API 提交假单（不实际成交），直接获取交易所
当前的 SPAN 初始保证金和维持保证金，确保数值与盘中实时一致。

特性：
- 本地 JSON 缓存，默认 TTL 6 小时（避免重复连接）
- TTL 内复用缓存，TTL 过期或强制刷新时重连 IBKR
- 失败时保留 instruments.yaml 静态值（不中断启动流程）
- 自动跳过不支持 whatIfOrder 的品种

用法::

    refresher = IBKRMarginRefresher(host="127.0.0.1", port=7497)
    refresher.refresh(registry)           # 有缓存用缓存，过期才连 IBKR
    refresher.refresh(registry, force=True)  # 强制重新查询
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from quant.instrument.registry import InstrumentRegistry

logger = logging.getLogger(__name__)

_CACHE_DEFAULT = Path("data/margin_cache.json")


class IBKRMarginRefresher:
    """从 IBKR 动态拉取保证金并更新 InstrumentRegistry。

    Parameters
    ----------
    host : str
        TWS/Gateway 主机地址。
    port : int
        TWS/Gateway 端口（Paper: 7497，Live: 7496，Gateway: 4001）。
    client_id : int
        IBKR 客户端 ID，需与其他连接区分（默认 20）。
    ttl_hours : float
        缓存有效期（小时），默认 6 小时。
    cache_path : Path or str
        本地 JSON 缓存文件路径。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 20,
        ttl_hours: float = 6.0,
        cache_path: Path | str = _CACHE_DEFAULT,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ttl_seconds = ttl_hours * 3600
        self._cache_path = Path(cache_path)

    def refresh(
        self,
        registry: InstrumentRegistry,
        force: bool = False,
    ) -> dict[str, tuple[float, float]]:
        """查询并更新 registry 中所有期货合约的保证金。

        Parameters
        ----------
        registry : InstrumentRegistry
            待更新的注册表。
        force : bool
            为 True 时忽略缓存，强制重连 IBKR 查询。

        Returns
        -------
        dict[str, tuple[float, float]]
            {symbol: (margin_initial, margin_maintenance)} 映射。
            仅包含本次成功更新的品种。
        """
        # 尝试读取有效缓存
        if not force:
            cached = self._load_cache()
            if cached is not None:
                logger.info(
                    "使用保证金缓存（%d 个品种，TTL %.1f 小时）",
                    len(cached),
                    self._ttl_seconds / 3600,
                )
                self._apply_to_registry(cached, registry)
                return cached

        # 缓存失效或强制刷新，连接 IBKR 查询
        logger.info("连接 IBKR 查询最新保证金：%s:%d", self._host, self._port)
        try:
            result = self._fetch_from_ibkr(registry)
        except Exception as exc:
            logger.warning(
                "保证金动态查询失败（%s），保留 instruments.yaml 静态值：%s",
                type(exc).__name__,
                exc,
            )
            return {}

        if result:
            self._save_cache(result)
            self._apply_to_registry(result, registry)
            logger.info("保证金已更新：%d 个品种，缓存至 %s", len(result), self._cache_path)

        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _fetch_from_ibkr(
        self,
        registry: InstrumentRegistry,
    ) -> dict[str, tuple[float, float]]:
        """通过 ib_insync whatIfOrder 查询每手保证金。"""
        try:
            from ib_insync import IB, Future, MarketOrder
        except ImportError:
            raise ImportError(
                "ib_insync 未安装，请运行：pip install ib-insync"
            )

        ib = IB()
        ib.connect(self._host, self._port, clientId=self._client_id)
        logger.info("IBKR 已连接（client_id=%d）", self._client_id)

        # 交易所代码映射（IBKR 使用 HKFE 而非 HKEX）
        _EXCHANGE_MAP = {"HKEX": "HKFE"}

        result: dict[str, tuple[float, float]] = {}
        try:
            for inst in registry:
                from quant.core.types import InstrumentType
                if inst.instrument_type != InstrumentType.FUTURE:
                    continue  # 只查期货

                exchange = _EXCHANGE_MAP.get(inst.exchange, inst.exchange)
                contract = Future(
                    symbol=inst.symbol,
                    exchange=exchange,
                    currency=inst.currency.value,
                )
                try:
                    qualified = ib.qualifyContracts(contract)
                    if not qualified:
                        logger.warning("%s：合约未能确认，跳过", inst.symbol)
                        continue
                    contract = qualified[0]
                except Exception as exc:
                    logger.warning("%s：qualifyContracts 失败（%s），跳过", inst.symbol, exc)
                    continue

                order = MarketOrder("BUY", 1)
                try:
                    state = ib.whatIfOrder(contract, order)
                except Exception as exc:
                    logger.warning("%s：whatIfOrder 失败（%s），跳过", inst.symbol, exc)
                    continue

                # whatIfOrder 返回字符串或浮点，统一转换
                init_margin = _parse_margin(state.initMarginChange)
                maint_margin = _parse_margin(state.maintMarginChange)

                if init_margin > 0 and maint_margin > 0:
                    result[inst.symbol] = (init_margin, maint_margin)
                    logger.info(
                        "%s：初始保证金 %.0f，维持保证金 %.0f（%s）",
                        inst.symbol,
                        init_margin,
                        maint_margin,
                        inst.currency.value,
                    )
                else:
                    logger.warning(
                        "%s：保证金数值无效（init=%.0f, maint=%.0f），跳过",
                        inst.symbol,
                        init_margin,
                        maint_margin,
                    )
        finally:
            ib.disconnect()
            logger.info("IBKR 已断开")

        return result

    def _apply_to_registry(
        self,
        margins: dict[str, tuple[float, float]],
        registry: InstrumentRegistry,
    ) -> None:
        """将查询结果写入 registry。"""
        for symbol, (init, maint) in margins.items():
            try:
                registry.update_margin(symbol, init, maint)
            except KeyError:
                pass  # 品种可能已从 registry 移除

    def _load_cache(self) -> dict[str, tuple[float, float]] | None:
        """读取本地缓存，TTL 过期或文件不存在返回 None。"""
        if not self._cache_path.exists():
            return None
        try:
            with open(self._cache_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        # 检查时效
        saved_at_str = data.get("saved_at", "")
        try:
            saved_at = datetime.fromisoformat(saved_at_str)
        except ValueError:
            return None

        age = (datetime.now(timezone.utc) - saved_at).total_seconds()
        if age > self._ttl_seconds:
            logger.info(
                "保证金缓存已过期（%.1f 小时 > TTL %.1f 小时）",
                age / 3600,
                self._ttl_seconds / 3600,
            )
            return None

        margins: dict[str, tuple[float, float]] = {}
        for sym, pair in data.get("margins", {}).items():
            margins[sym] = (float(pair[0]), float(pair[1]))
        return margins if margins else None

    def _save_cache(self, margins: dict[str, tuple[float, float]]) -> None:
        """将查询结果写入本地 JSON 缓存。"""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "ttl_hours": self._ttl_seconds / 3600,
            "margins": {sym: list(pair) for sym, pair in margins.items()},
        }
        with open(self._cache_path, "w") as fh:
            json.dump(payload, fh, indent=2)


def _parse_margin(value: object) -> float:
    """将 whatIfOrder 返回的保证金字段转换为浮点。

    ib_insync 可能返回字符串（如 ``"15200"``）或浮点，
    缺失或无效时返回 0.0。
    """
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
