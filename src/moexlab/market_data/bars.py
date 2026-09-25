"""Агрегация свечей без пересечения границ серий и торговых дней."""
from __future__ import annotations

import numpy as np
import pandas as pd


def resample_intraday(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Минутные свечи -> N-минутные. Ведро не пересекает смену серии и торгового дня.

    Индекс результата — начало ведра (UTC). Вход должен содержать trading_day (см. annotate_sessions)
    или будет агрегирован только по secid.
    """
    keys = [k for k in ("secid", "trading_day") if k in df.columns]
    bucket = df.index.floor(f"{minutes}min")
    g = df.groupby(keys + [bucket], sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"))
    out = out.reset_index()
    ts_col = out.columns[len(keys)]
    out = out.set_index(ts_col).sort_index()
    out.index.name = "ts"
    return out[["open", "high", "low", "close", "volume"] + keys]


def daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Дневные свечи по торговым дням (включая все сессии дня); индекс — UTC-полночь торгового дня."""
    keys = [k for k in ("secid", "trading_day") if k in df.columns]
    g = df.groupby(keys, sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum")).reset_index()
    out.index = pd.DatetimeIndex(pd.to_datetime(out["trading_day"])).tz_localize("UTC")
    out.index.name = "ts"
    return out.sort_index()


def annotate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Служебные колонки для дневных свечей: решение на закрытии дня, исполнение на открытии следующего."""
    df = df.copy()
    if "trading_day" not in df:
        df["trading_day"] = df.index.date
    df["entry_ok"] = True
    df["force_exit"] = False
    df["last_in_day"] = True
    df["in_window"] = True
    return df


def stitch_active(bars_by_secid: dict[str, pd.DataFrame], active: pd.Series) -> pd.DataFrame:
    """Поток активной серии с реальными ценами (для исполнения) и форвард-скорректированными (для сигналов).

    active: торговый день -> secid. Реальные open/high/low/close берутся из активной серии.
    Колонки a_open/a_high/a_low/a_close — «сигнальные» цены: при перекладке с серии A на B
    все БУДУЩИЕ цены сдвигаются на разницу close_A - close_B на последней свече перед перекладкой
    (известна в момент перекладки => корректировка каузальна). adj_shift = a_close - close.
    Склеенный ряд используется только для индикаторов, никогда для цен исполнения.
    """
    parts = []
    order = []
    for d, sec in active.sort_index().items():
        if not order or order[-1][0] != sec:
            order.append((sec, [d]))
        else:
            order[-1][1].append(d)
    shift = 0.0
    prev_sec = None
    for sec, days in order:
        b = bars_by_secid.get(sec)
        if b is None:
            continue
        sel = b[b["trading_day"].isin(set(days))].copy()
        if sel.empty:
            continue
        if prev_sec is not None:
            pb = bars_by_secid[prev_sec]
            t0 = sel.index[0]
            old = pb[pb.index < t0]
            new = b[b.index < t0]
            if len(old) and len(new):
                shift += float(old["close"].iloc[-1]) - float(new["close"].iloc[-1])
        sel["secid"] = sec
        for c in ("open", "high", "low", "close"):
            sel[f"a_{c}"] = sel[c] + shift
        sel["adj_shift"] = shift
        parts.append(sel)
        prev_sec = sec
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="first")]


def signal_view(df: pd.DataFrame) -> pd.DataFrame:
    """Свечи для расчёта индикаторов: скорректированные цены, если есть, иначе исходные."""
    if "a_close" not in df:
        return df
    v = df.copy()
    for c in ("open", "high", "low", "close"):
        v[c] = df[f"a_{c}"]
    return v


def back_adjusted_close(stream: pd.DataFrame) -> pd.Series:
    """Склеенный (разностно скорректированный) ряд — ТОЛЬКО для вспомогательной аналитики."""
    c = stream["close"].to_numpy(float).copy()
    sec = stream["secid"].to_numpy()
    o = stream["open"].to_numpy(float)
    adj = np.zeros(len(c))
    for i in range(1, len(c)):
        if sec[i] != sec[i - 1]:
            adj[:i] += o[i] - c[i - 1]
    return pd.Series(c + adj, index=stream.index)
