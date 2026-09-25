"""Подготовка внутридневных данных из локальной БД (moexlab.storage.db) для исследования.

* Режимы расписания FORTS (research/sources.md §6):
    - до ЕТС: вечерняя сессия (с 19:05) открывает СЛЕДУЮЩИЙ торговый день => граница дня 19:00 МСК;
    - с 23.03.2026 (ЕТС): вечерняя сессия относится к ТЕКУЩЕМУ дню => граница дня 23:59 МСК.
  Переключение режима — с вечерней сессии пятницы 20.03.2026 19:00 МСК (она уже относится к дню 23.03).
* Акции TQBR: торговый день = календарный день (граница 23:59), окно стратегии — основная сессия.
* Окно стратегии — основная сессия: до ЕТС 10:00–18:50, с ЕТС 10:00–19:00 МСК (допущение A-01).
* Сессии выходного дня попадают в ближайший следующий будний торговый день (H/L/C, объём), но входов в них нет.
* Поток активной серии: выбор серии по объёмам завершённых дней (спецификация §6), сигнальные цены —
  форвард-корректировка перекладок; признаки R-Breaker — по собственной истории каждой серии (§7).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from ..contracts.series import RollConfig, SeriesInfo, select_active_series
from ..instruments.spec import InstrumentSpec
from ..market_data.bars import stitch_active
from ..market_data.calendar import MSK, SessionConfig, annotate_sessions

ETS_SWITCH = pd.Timestamp("2026-03-20 19:00", tz=MSK)
MAIN_WINDOW = ((time(10, 0), time(18, 50)),)
FUT_PRE_ETS = SessionConfig(windows_msk=MAIN_WINDOW, day_boundary_msk=time(19, 0))
FUT_ETS = SessionConfig(windows_msk=((time(10, 0), time(19, 0)),), day_boundary_msk=time(23, 59))
EQ_SESSION = SessionConfig(windows_msk=((time(10, 0), time(18, 40)),), day_boundary_msk=time(23, 59))

# группы риска (спецификация §44)
GROUPS = {"Si": "fx", "CR": "fx", "RI": "equity_index", "MX": "equity_index", "SR": "equity", "GZ": "equity",
          "RN": "equity", "GD": "commodity", "BR": "commodity"}


def annotate_futures(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Разметка торговых дней/окон с учётом смены режима (ЕТС)."""
    if df.empty:
        return df
    pre = df[df.index < ETS_SWITCH.tz_convert("UTC")]
    post = df[df.index >= ETS_SWITCH.tz_convert("UTC")]
    parts = []
    if len(pre):
        parts.append(annotate_sessions(pre, FUT_PRE_ETS, bar_minutes))
    if len(post):
        parts.append(annotate_sessions(post, FUT_ETS, bar_minutes))
    return pd.concat(parts).sort_index()


def annotate_equity(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    return annotate_sessions(df, EQ_SESSION, bar_minutes) if len(df) else df


@dataclass
class IntradayItem:
    code: str
    kind: str
    stream: pd.DataFrame                 # минутный поток активной серии (реальные + сигнальные цены)
    series: dict[str, pd.DataFrame]      # полная минутная история каждой серии (с разметкой)
    active: pd.Series                    # торговый день -> secid
    spec: InstrumentSpec
    issues: list[str]


def _spec_from_row(code: str, kind: str, r: pd.Series, group: str) -> InstrumentSpec:
    tick = float(r["tick"])
    if kind == "future":
        tv = float(r["tick_value"]) if np.isfinite(r["tick_value"]) and r["tick_value"] > 0 else tick
        return InstrumentSpec(code, "future", tick=tick, tick_value=tv, group=group, spread_ticks=1.0,
                              slippage_ticks=1.0, stop_extra_ticks=1.0, margin_fraction=0.15,
                              source="T-Invest InstrumentsService (tick, tick value); spread/slippage A-05")
    lot = int(r["lot"])
    return InstrumentSpec(code, "equity", tick=tick, tick_value=tick * lot, lot=lot, group="equity",
                          spread_ticks=1.0, slippage_ticks=1.0, stop_extra_ticks=1.0, margin_fraction=1.0,
                          source="T-Invest ShareBy (tick, lot); spread/slippage A-05")


def load_future(db, base: str, start=None, end=None, roll: RollConfig | None = None) -> IntradayItem | None:
    """Минутный поток активной серии базового актива из БД."""
    roll = roll or RollConfig(early_roll_requires_spread=False)   # спреда в истории нет (A-08)
    ins = db.instruments(kind="future", base=base)
    series, infos, issues, rows = {}, [], [], []
    for _, r in ins.iterrows():
        df = db.candles(r["ticker"], "1m", start, end)
        if df.empty:
            continue
        a = annotate_futures(df, 1)
        a["secid"] = r["ticker"]
        series[r["ticker"]] = a
        meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else {}
        last_trade = pd.Timestamp(meta.get("last_trade") or r["expiration"])
        exp = last_trade.tz_convert(MSK).date() if last_trade.tzinfo else last_trade.date()
        infos.append(SeriesInfo(secid=r["ticker"], base=base, expiration=exp, first_trade=a.index[0].date()))
        g = a.groupby("trading_day")["volume"].sum()
        rows.append(pd.DataFrame({"trading_day": g.index, "secid": r["ticker"], "volume": g.to_numpy()}))
    if not series:
        return None
    daily = pd.concat(rows, ignore_index=True)
    tdays = sorted(set(daily["trading_day"]))
    active = select_active_series(infos, daily, tdays, roll)
    # серия должна иметь свечи в этот день; иначе день пропускается (нет данных)
    have = daily.groupby("secid")["trading_day"].apply(set).to_dict()
    missing = [d for d, s in active.items() if d not in have.get(s, set())]
    if missing:
        issues.append(f"{base}: {len(missing)} days without data for active series (skipped)")
        active = active.drop(missing)
    stream = stitch_active(series, active)
    spec = _spec_from_row(base, "future", ins.set_index("ticker").loc[stream["secid"].iloc[-1]], GROUPS.get(base, "other"))
    return IntradayItem(base, "future", stream, series, active, spec, issues)


def load_equity(db, ticker: str, start=None, end=None) -> IntradayItem | None:
    df = db.candles(ticker, "1m", start, end)
    if df.empty:
        return None
    a = annotate_equity(df, 1)
    a["secid"] = ticker
    for c in ("open", "high", "low", "close"):
        a[f"a_{c}"] = a[c]
    a["adj_shift"] = 0.0
    r = db.instruments(kind="equity").set_index("ticker").loc[ticker]
    active = pd.Series(ticker, index=sorted(set(a["trading_day"])))
    return IntradayItem(ticker, "equity", a, {ticker: a}, active, _spec_from_row(ticker, "equity", r, "equity"), [])


def resample_stream(stream: pd.DataFrame, minutes: int, kind: str) -> pd.DataFrame:
    """N-минутные свечи потока: ведро не пересекает смену серии и торгового дня; сигнальные цены сохраняются."""
    b = stream.index.floor(f"{minutes}min")
    g = stream.groupby([stream["secid"], stream["trading_day"], b], sort=False)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"), a_open=("a_open", "first"), a_high=("a_high", "max"),
                a_low=("a_low", "min"), a_close=("a_close", "last"), adj_shift=("adj_shift", "last"))
    out = out.reset_index()
    ts = out.columns[2]
    out = out.set_index(ts).sort_index()
    out.index.name = "ts"
    keep = out.drop(columns=["trading_day"])
    ann = annotate_futures(keep, minutes) if kind == "future" else annotate_equity(keep, minutes)
    return ann


def daily_from_stream(stream: pd.DataFrame) -> pd.DataFrame:
    """Дневные свечи потока по торговым дням (реальные и сигнальные цены); индекс — UTC-полночь дня."""
    g = stream.groupby("trading_day", sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"), a_open=("a_open", "first"), a_high=("a_high", "max"),
                a_low=("a_low", "min"), a_close=("a_close", "last"), adj_shift=("adj_shift", "last"),
                secid=("secid", "last"))
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize("UTC")
    out.index.name = "ts"
    out["trading_day"] = out.index.date
    out["entry_ok"] = True
    out["force_exit"] = False
    out["last_in_day"] = True
    out["in_window"] = True
    return out


def trading_days_between(stream: pd.DataFrame, start: date | str, end: date | str) -> pd.DataFrame:
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    td = pd.to_datetime(stream["trading_day"]).dt.date
    return stream[((td >= s) & (td <= e)).to_numpy()]
