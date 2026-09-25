"""Гипотеза H_ON: «ночной» дрейф — покупка в конце вечерней сессии, продажа утром следующего дня.

Точное исполнение по минутным свечам (без заглядывания):
  * вход: рыночная покупка на ЗАКРЫТИИ последней минутной свечи дня с началом >= entry_mod (МСК),
    цена = close + ½ спреда + проскальзывание (последняя сделка может быть по bid — поэтому платим спред);
  * выход: рыночная продажа на ОТКРЫТИИ первой свечи следующего торгового дня с началом >= exit_mod
    (exit_mod = 0 — первая свеча дня: для акций — аукцион открытия/первая сделка утренней сессии),
    цена = open − ½ спреда − проскальзывание;
  * комиссия — с обеих сторон; перекладка фьючерса: ночь через смену серии пропускается.
Решение «входить ли сегодня» может зависеть только от данных до момента входа (фильтры ниже).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .screen import SPREAD_TICKS


def nights(code: str, entry_mod: int = 1425, exit_mod: int = 0, end=D.DEV_END, start=D.FIRST_DAY,
           tariff: str = "PREMIUM", slip_mult: float = 1.0, comm_mult: float = 1.0, side: int = 1) -> pd.DataFrame:
    d = D.normalized(code)
    dd = pd.to_datetime(d["day"]).dt.date
    d = d[((dd >= start) & (dd <= end)).to_numpy()].reset_index(drop=True)
    sp = D.spec(code)
    tick = sp["tick"]
    hs = 0.5 * SPREAD_TICKS.get(code, 1.0) * tick
    slip = tick * slip_mult
    comm = D.COMMISSION[tariff][sp["kind"]] * comm_mult
    g = d.groupby("day", sort=True)
    rows = []
    days = list(g.groups.keys())
    idx = g.indices
    for j in range(len(days) - 1):
        a = d.iloc[idx[days[j]]]
        b = d.iloc[idx[days[j + 1]]]
        ea = a[a["mod"] >= entry_mod] if entry_mod > 0 else a
        if ea.empty:
            continue
        e = ea.iloc[-1]                                  # последняя свеча дня (после entry_mod)
        xb = b[b["mod"] >= exit_mod] if exit_mod > 0 else b
        if xb.empty:
            continue
        x = xb.iloc[0]
        if e["secid"] != x["secid"]:
            continue                                     # ночь через перекладку не торгуем
        pe = e["close"] + side * (hs + slip)
        px = x["open"] - side * (hs + slip)
        gross = side * (x["open"] / e["close"] - 1)
        net = side * (px / pe - 1) - comm * (1 + px / pe)
        rows.append(dict(day=days[j], next_day=days[j + 1], entry_ts=e["ts"], exit_ts=x["ts"], entry_px=pe,
                         exit_px=px, gross=gross, net=net, weekday=pd.Timestamp(days[j]).weekday()))
    return pd.DataFrame(rows)
