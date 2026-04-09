"""跨时区品种的交易日计算。

规则：
    CME/CBOT Globex：
        交易时段 17:00 CT -> 16:00 CT（下一个日历日）。
        周日 17:00 CT 的 K 线归属于 *周一* 的交易日。
        周一 15:59 CT 的 K 线归属于 *周一* 的交易日。
        分界点：17:00 CT 之后的 K 线 -> 下一个日历日。

    HKEX：
        三个交易时段 -- 早盘（09:15-12:00）、午盘（13:00-16:30）、
        T+1 夜盘（17:15-次日 03:00）。
        早盘 + 午盘 -> 同一日历日。
        T+1 时段（17:15-03:00）：归属于 *下一个* 交易日。
        分界点：17:15 HKT 之后的 K 线 -> 下一个日历日。

这些函数基于 ``quant.core.types`` 中的 ``Instrument`` 和
``TradingWindow``，因此与 ``instruments.yaml`` 保持一致。
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from quant.core.types import Instrument, TradingWindow


def compute_trading_day(ts: datetime, instrument: Instrument) -> date:
    """判断时间戳所属的交易日。

    Parameters
    ----------
    ts : datetime
        K 线时间戳（应为时区感知，最好是 UTC）。
    instrument : Instrument
        合约规格，包含交易所和交易时段。

    Returns
    -------
    date
        该 K 线所属的交易日。
    """
    if instrument.exchange in ("CME", "CBOT"):
        return _trading_day_globex(ts, instrument)
    elif instrument.exchange == "HKEX":
        return _trading_day_hkex(ts, instrument)
    else:
        # 回退方案：使用 UTC 日历日期
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        return ts.date()


def determine_session(ts: datetime, instrument: Instrument) -> TradingWindow | None:
    """查找时间戳落入的交易时段。

    Parameters
    ----------
    ts : datetime
        待检查的时间戳（时区感知）。
    instrument : Instrument
        合约规格，包含交易时段。

    Returns
    -------
    TradingWindow or None
        匹配的交易时段，若不在任何时段内则返回 None。
    """
    for session in instrument.sessions:
        if _in_session(ts, session):
            return session
    return None


def compute_bars_per_day(instrument: Instrument, bar_duration_secs: int = 900) -> int:
    """动态计算每个交易日的 K 线数量。

    汇总所有交易时段的总分钟数，除以 K 线时长。

    Parameters
    ----------
    instrument : Instrument
        合约规格，包含交易时段。
    bar_duration_secs : int
        K 线宽度（秒），默认 900 = 15 分钟。

    Returns
    -------
    int
        每个交易日预期的 K 线数量。
    """
    total_minutes = 0
    for session in instrument.sessions:
        total_minutes += _session_minutes(session)

    bar_minutes = bar_duration_secs / 60
    return int(total_minutes / bar_minutes)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _trading_day_globex(ts: datetime, instrument: Instrument) -> date:
    """CME/CBOT Globex：交易时段从 17:00 CT 开始，到次日 16:00 CT 结束。"""
    session = instrument.sessions[0]  # Globex 只有一个交易时段
    tz = ZoneInfo(session.timezone)

    local = ts.astimezone(tz) if ts.tzinfo else ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)

    # Globex 起始时间为 17:00 CT。17:00 之后的 K 线归属于
    # 下一个日历日的交易时段。
    cutoff = time(17, 0)
    if local.time() >= cutoff:
        return (local + timedelta(days=1)).date()
    else:
        return local.date()


def _trading_day_hkex(ts: datetime, instrument: Instrument) -> date:
    """HKEX：早盘+午盘 = 当天；T+1（17:15-03:00）= 次日。"""
    # 所有 HKEX 交易时段使用相同时区
    tz = ZoneInfo(instrument.sessions[0].timezone)
    local = ts.astimezone(tz) if ts.tzinfo else ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)

    local_time = local.time()

    # 凌晨 00:00-03:00 -- 这是前一个日历日 T+1 夜盘的尾段。
    # 它归属于 *当天* 的交易日（T+1 时段在昨天 17:15 后开始，
    # 映射到今天）。
    t1_end = time(3, 0)
    t1_start = time(17, 15)

    if local_time < t1_end:
        # 处于 T+1 的凌晨部分 -> 交易日 = 今天
        return local.date()
    elif local_time >= t1_start:
        # 处于 T+1 的傍晚部分 -> 交易日 = 明天
        return (local + timedelta(days=1)).date()
    else:
        # 早盘（09:15-12:00）或午盘（13:00-16:30）-> 今天
        return local.date()


def _in_session(ts: datetime, session: TradingWindow) -> bool:
    """检查 *ts* 是否在交易时段窗口内。"""
    tz = ZoneInfo(session.timezone)
    local = ts.astimezone(tz) if ts.tzinfo else ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)

    start = time.fromisoformat(session.start)
    end = time.fromisoformat(session.end)
    local_time = local.time()

    if start <= end:
        # 正常时段（例如 09:15 -> 12:00）
        return start <= local_time < end
    else:
        # 跨午夜时段（例如 17:15 -> 03:00 或 17:00 -> 16:00）
        return local_time >= start or local_time < end


def _session_minutes(session: TradingWindow) -> int:
    """计算交易时段的总分钟数。"""
    start = time.fromisoformat(session.start)
    end = time.fromisoformat(session.end)

    start_mins = start.hour * 60 + start.minute
    end_mins = end.hour * 60 + end.minute

    if end_mins > start_mins:
        return end_mins - start_mins
    else:
        # 跨午夜：今天剩余分钟 + 午夜后的分钟
        return (24 * 60 - start_mins) + end_mins
