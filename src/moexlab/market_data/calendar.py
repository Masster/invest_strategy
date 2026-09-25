"""Торговый календарь FORTS / фондового рынка.

Внутри всё хранится в UTC, правила применяются по МСК (UTC+3; перехода на летнее время нет с 2014 г.).
Расписание не зашито константой (спецификация R-Breaker §4): торговые окна и граница торгового дня
задаются конфигурацией, а сами торговые даты выводятся из фактических данных.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time

import numpy as np
import pandas as pd

MSK = "Europe/Moscow"


def parse_hhmm(s: str) -> time:
    hh, mm = str(s).split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class SessionConfig:
    """Какие части торгового дня торгуются стратегией.

    windows_msk      — список (open, close) по МСК; вход разрешён только внутри окна;
    day_boundary_msk — всё, что позже этой отметки, относится к следующему торговому дню
                       (на FORTS вечерняя сессия открывает следующий торговый день).
    """
    windows_msk: tuple[tuple[time, time], ...] = ((time(10, 0), time(18, 50)),)
    day_boundary_msk: time = time(19, 0)
    opening_cooldown_min: int = 15
    stop_entries_before_close_min: int = 20
    force_close_before_close_min: int = 15

    @classmethod
    def from_dict(cls, d: dict) -> "SessionConfig":
        return cls(
            windows_msk=tuple((parse_hhmm(o), parse_hhmm(c)) for o, c in d.get("windows_msk", [["10:00", "18:50"]])),
            day_boundary_msk=parse_hhmm(d.get("day_boundary_msk", "19:00")),
            opening_cooldown_min=int(d.get("opening_cooldown_min", 15)),
            stop_entries_before_close_min=int(d.get("stop_entries_before_close_min", 20)),
            force_close_before_close_min=int(d.get("force_close_before_close_min", 15)),
        )


def main_session_dates(ts_utc: pd.DatetimeIndex, windows: tuple[tuple[time, time], ...]) -> list[date]:
    """Будние даты, в которые была хотя бы одна свеча внутри торговых окон."""
    msk = ts_utc.tz_convert(MSK)
    t = np.array(msk.time)
    mask = np.zeros(len(msk), dtype=bool)
    for o, c in windows:
        mask |= (t >= o) & (t < c)
    mask &= msk.weekday < 5
    return sorted(set(msk[mask].date))


def assign_trading_day(ts_utc: pd.DatetimeIndex, main_dates: list[date], boundary_msk: time) -> np.ndarray:
    """Каждой свече ставит в соответствие торговый день.

    Свеча относится к первому дню D из main_dates, для которого ts < D + boundary (МСК):
    вечерняя сессия и сессии выходных уходят в следующий торговый день.
    Свечи после последнего известного дня получают None.
    """
    days = sorted(main_dates)
    ends = pd.DatetimeIndex([pd.Timestamp(datetime.combine(d, boundary_msk), tz=MSK) for d in days])
    idx = ends.searchsorted(ts_utc.tz_convert(MSK), side="right")
    lookup = np.array(days + [None], dtype=object)
    return lookup[idx]


def window_bounds_utc(d: date, windows: tuple[tuple[time, time], ...]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for o, c in windows:
        out.append((pd.Timestamp(datetime.combine(d, o), tz=MSK).tz_convert("UTC"),
                    pd.Timestamp(datetime.combine(d, c), tz=MSK).tz_convert("UTC")))
    return out


def annotate_sessions(df: pd.DataFrame, cfg: SessionConfig, bar_minutes: int) -> pd.DataFrame:
    """Добавляет к свечам служебные колонки расписания.

    Индекс df — время НАЧАЛА свечи (UTC). Колонки:
      trading_day   — торговый день (date);
      in_window     — свеча целиком внутри торгового окна;
      entry_ok      — после закрытия этой свечи можно выставить вход на следующую свечу
                      (следующая свеча начинается не раньше open+cooldown и заканчивается не позже
                       close-stop_entries);
      force_exit    — свеча, на открытии которой обязан начаться принудительный выход
                      (первая свеча с началом >= close - force_close_before_close);
      last_in_day   — последняя свеча торгового дня в данных.
    Всё рассчитывается только из собственного времени свечи и заранее известного расписания.
    """
    df = df.copy()
    idx = df.index
    dates = main_session_dates(idx, cfg.windows_msk)
    df["trading_day"] = assign_trading_day(idx, dates, cfg.day_boundary_msk)
    df = df[df["trading_day"].notna()]
    idx = df.index
    bar = pd.Timedelta(minutes=bar_minutes)
    in_window = np.zeros(len(df), dtype=bool)
    entry_ok = np.zeros(len(df), dtype=bool)
    force_exit = np.zeros(len(df), dtype=bool)
    msk = idx.tz_convert(MSK)
    td = df["trading_day"].to_numpy()
    for d in np.unique(td):
        sel = np.where(td == d)[0]
        for o, c in window_bounds_utc(d, cfg.windows_msk):
            ts = idx[sel]
            inw = (ts >= o) & (ts + bar <= c)
            in_window[sel] |= inw
            # вход исполняется на СЛЕДУЮЩЕЙ свече: её начало = ts + bar
            nxt = ts + bar
            ok = (nxt >= o + pd.Timedelta(minutes=cfg.opening_cooldown_min)) & \
                 (nxt < c - pd.Timedelta(minutes=cfg.stop_entries_before_close_min)) & inw
            entry_ok[sel] |= ok
            fc = c - pd.Timedelta(minutes=cfg.force_close_before_close_min)
            after = np.where((ts >= fc) & (ts < c))[0]
            if len(after):
                force_exit[sel[after[0]]] = True
    df["in_window"] = in_window
    df["entry_ok"] = entry_ok
    df["force_exit"] = force_exit
    nxt_day = np.append(td[1:], None)
    df["last_in_day"] = td != nxt_day
    _ = msk
    return df


def iso_week(d: date) -> tuple[int, int]:
    y, w, _ = d.isocalendar()
    return (y, w)
