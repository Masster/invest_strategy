"""Торговый календарь (§4–5).

Внутри всё хранится в UTC; правила применяются по времени МСК (UTC+3, без перехода на летнее время
с 2014 г.). Расписание не зашито как константа: окна торговли и граница торгового дня задаются
конфигурацией, а список торговых дат выводится из фактических данных основной сессии.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

MSK = "Europe/Moscow"


@dataclass(frozen=True)
class TradingWindow:
    """Одна торгуемая сессия внутри торгового дня (UTC)."""
    open_utc: pd.Timestamp
    close_utc: pd.Timestamp

    def entries_allowed(self, ts: pd.Timestamp, cooldown_min: int, stop_before_close_min: int) -> bool:
        return (self.open_utc + pd.Timedelta(minutes=cooldown_min) <= ts
                < self.close_utc - pd.Timedelta(minutes=stop_before_close_min))

    def force_close_at(self, force_before_close_min: int) -> pd.Timestamp:
        return self.close_utc - pd.Timedelta(minutes=force_before_close_min)


def to_msk(ts_utc: pd.Series | pd.DatetimeIndex):
    return ts_utc.tz_convert(MSK)


def assign_trading_day(ts_utc: pd.DatetimeIndex, main_dates: list[date], boundary_msk: time) -> np.ndarray:
    """Каждой свече сопоставляет торговый день FORTS.

    Свеча относится к первому торговому дню D из `main_dates`, для которого
    ts < D + boundary (МСК). Так вечерняя сессия и сессии выходного дня уходят
    в следующий торговый день (допущение A-02).
    """
    msk = ts_utc.tz_convert(MSK)
    ends = pd.DatetimeIndex(
        [pd.Timestamp(datetime.combine(d, boundary_msk), tz=MSK) for d in sorted(main_dates)]
    )
    idx = ends.searchsorted(msk, side="right")
    days = np.array(sorted(main_dates) + [None], dtype=object)
    return days[idx]


def main_session_dates(ts_utc: pd.DatetimeIndex, windows: list[tuple[time, time]]) -> list[date]:
    """Даты, в которые была хотя бы одна свеча внутри торговых окон (основная сессия)."""
    msk = ts_utc.tz_convert(MSK)
    t = msk.time
    mask = np.zeros(len(msk), dtype=bool)
    for o, c in windows:
        mask |= (t >= o) & (t < c)
    wd = msk.weekday < 5
    return sorted(set(msk[mask & wd].date))


def windows_for_day(d: date, windows: list[tuple[time, time]]) -> list[TradingWindow]:
    out = []
    for o, c in windows:
        ou = pd.Timestamp(datetime.combine(d, o), tz=MSK).tz_convert("UTC")
        cu = pd.Timestamp(datetime.combine(d, c), tz=MSK).tz_convert("UTC")
        out.append(TradingWindow(ou, cu))
    return out


def iso_week_key(d: date) -> tuple[int, int]:
    y, w, _ = d.isocalendar()
    return (y, w)


def trading_days_between(dates: list[date], a: date, b: date) -> int:
    """Число торговых дней в (a, b] по известному календарю."""
    arr = np.array(sorted(dates), dtype=object)
    return int(((arr > a) & (arr <= b)).sum())


__all__ = [
    "MSK", "TradingWindow", "assign_trading_day", "main_session_dates", "windows_for_day",
    "iso_week_key", "trading_days_between", "to_msk", "timedelta",
]
