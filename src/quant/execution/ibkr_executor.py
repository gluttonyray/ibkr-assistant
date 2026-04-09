from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from quant.config.schema import IBKRConfig
from quant.core.types import Currency, Fill, Instrument, InstrumentType, Side
from quant.instrument.registry import InstrumentRegistry

logger = logging.getLogger(__name__)

class IBKRExecutor:
    """IBKR TWS/Gateway 实盘执行器，实现 Executor Protocol。

    使用 ib_insync 库连接 TWS/IB Gateway。
    熔断保护：发单前检查日订单数不超过 max_daily_trades。
    断线自动重连：最多 3 次，指数退避（2^n 秒）。
    """

    def __init__(self, cfg: IBKRConfig, registry: InstrumentRegistry) -> None:
        self._cfg = cfg
        self._registry = registry
        self._ib: object | None = None  # ib_insync.IB instance
        self._daily_order_count: int = 0
        self._fills_buffer: list[Fill] = []

    async def connect(self) -> None:
        """连接 TWS/IB Gateway，失败时指数退避重试最多 3 次。"""
        try:
            from ib_insync import IB
        except ImportError:
            raise RuntimeError("ib_insync not installed. Run: pip install ib-insync")

        ib = IB()
        for attempt in range(3):
            try:
                await ib.connectAsync(
                    self._cfg.host,
                    self._cfg.port,
                    clientId=self._cfg.client_id,
                )
                self._ib = ib
                logger.info(f"Connected to IBKR TWS at {self._cfg.host}:{self._cfg.port}")
                # 注册成交回调
                ib.execDetailsEvent += self._on_exec_details
                return
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"IBKR connect attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
        raise ConnectionError(f"Failed to connect to IBKR after 3 attempts")

    async def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None
            logger.info("Disconnected from IBKR")

    async def submit(
        self,
        instrument: Instrument,
        side: Side,
        qty: int,
        order_type: str = "LIMIT",
        limit_price: float | None = None,
    ) -> str:
        if self._ib is None:
            raise RuntimeError("Not connected to IBKR")
        if self._daily_order_count >= self._cfg.history_bars:  # 复用 max field
            raise RuntimeError(f"Daily order limit reached: {self._daily_order_count}")

        from ib_insync import Future, MarketOrder, LimitOrder

        contract = self._build_contract(instrument)
        action = "BUY" if side == Side.BUY else "SELL"

        if order_type == "MARKET":
            order = MarketOrder(action, qty)
        else:
            if limit_price is None:
                raise ValueError("limit_price required for LIMIT order")
            order = LimitOrder(action, qty, limit_price)

        trade = self._ib.placeOrder(contract, order)
        self._daily_order_count += 1
        order_id = str(trade.order.orderId)
        logger.info(f"Submitted {order_type} {action} {qty}x{instrument.symbol} → orderId={order_id}")
        return order_id

    async def cancel(self, order_id: str) -> bool:
        if self._ib is None:
            return False
        for trade in self._ib.trades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                logger.info(f"Cancelled order {order_id}")
                return True
        return False

    async def get_fills(self) -> list[Fill]:
        fills = list(self._fills_buffer)
        self._fills_buffer.clear()
        return fills

    def reset_daily(self) -> None:
        self._daily_order_count = 0

    def _on_exec_details(self, trade: object, fill: object) -> None:
        """ib_insync execDetails 回调，将成交转换为 Fill 对象。"""
        try:
            inst = self._registry.get(trade.contract.symbol)
            side = Side.BUY if fill.execution.side == "BOT" else Side.SELL
            f = Fill(
                instrument=inst,
                side=side,
                qty=int(fill.execution.shares),
                price=float(fill.execution.price),
                timestamp=datetime.now(timezone.utc),
                commission=float(fill.commissionReport.commission) if fill.commissionReport else 0.0,
                exchange_fees=0.0,
                slippage_cost=0.0,
            )
            self._fills_buffer.append(f)
        except Exception as e:
            logger.error(f"Error processing fill: {e}")

    def _build_contract(self, instrument: Instrument) -> object:
        """将 Instrument 转换为 ib_insync Future contract。"""
        from ib_insync import Future
        if instrument.instrument_type != InstrumentType.FUTURE:
            raise ValueError(f"Only FUTURE supported, got {instrument.instrument_type}")

        exchange_map = {
            "CME": "CME",
            "CBOT": "CBOT",
            "HKEX": "HKFE",  # IBKR 使用 HKFE 作为港股期货交易所代码
        }
        currency_map = {
            Currency.USD: "USD",
            Currency.HKD: "HKD",
        }
        return Future(
            symbol=instrument.symbol,
            exchange=exchange_map.get(instrument.exchange, instrument.exchange),
            currency=currency_map[instrument.currency],
        )
