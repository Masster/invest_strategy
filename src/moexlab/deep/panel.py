"""Панельный дневной бэктестер портфельных стратегий (21 инструмент).

Данные: дневные бары нормализованного слоя (фьючерсы — мультипликативно склеенные цены активной серии,
т.е. доходности = доходности удерживаемой серии с перекладкой на закрытии; акции — total return).
Тайминг: веса w_t решаются на закрытии дня t (данные <= t), исполняются на ОТКРЫТИИ дня t+1.
Доходность дня t+1 = w_{t-1}·r_gap(t+1) + w_t·r_intraday(t+1), где r_gap = open/prev_close − 1,
r_intraday = close/open − 1. Издержки: |w_t − w_{t-1}| × (комиссия + ½спреда + проскальзывание) по open t+1,
+ круг издержек на перекладке фьючерса для удерживаемого веса.
Доходность — в долях капитала; w — доля номинала на капитал (сумма |w| = валовое плечо).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .screen import SPREAD_TICKS


def load_panel(codes=None, end=D.DEV_END, start=D.FIRST_DAY, tariff="TRADER", slip_mult=1.0, comm_mult=1.0):
    codes = codes or D.ALL_CODES
    frames = {}
    for c in codes:
        b = D.bars(c, 1440)
        d = pd.to_datetime(b["day"]).dt.date
        b = b[((d >= start) & (d <= end)).to_numpy()]
        adj = b["adj"].to_numpy(float)
        frames[c] = pd.DataFrame({"o": b["open"].to_numpy() * adj, "h": b["high"].to_numpy() * adj,
                                  "l": b["low"].to_numpy() * adj, "c": b["close"].to_numpy() * adj,
                                  "v": b["volume"].to_numpy(), "roll": b["new_series"].to_numpy(),
                                  "real_c": b["close"].to_numpy()},
                                 index=pd.to_datetime(b["day"]).dt.strftime("%Y%m%d").astype(int).to_numpy())
    days = sorted(set().union(*[set(f.index) for f in frames.values()]))
    P = {}
    for k in ("o", "h", "l", "c", "v", "roll", "real_c"):
        P[k] = pd.DataFrame({c: frames[c][k] for c in codes}, index=days).sort_index()
    P["traded"] = P["c"].notna()
    # на днях без торгов инструмента (праздники валютного рынка) — цена переносится, доходность 0
    for k in ("o", "h", "l", "c", "real_c"):
        P[k] = P[k].ffill()
    P["o"] = P["o"].where(P["traded"], P["c"])
    P["roll"] = P["roll"].fillna(False).astype(bool)
    prev_c = P["c"].shift(1)
    P["r_gap"] = (P["o"] / prev_c - 1).fillna(0.0)
    P["r_intra"] = (P["c"] / P["o"] - 1).fillna(0.0)
    P["r_cc"] = (P["c"] / prev_c - 1).fillna(0.0)
    cost = {}
    for c in codes:
        sp = D.spec(c)
        ticks = SPREAD_TICKS.get(c, 1.0) * 0.5 + 1.0 * slip_mult
        tick_frac = (sp["tick"] * ticks / P["real_c"][c]).fillna(0.0)
        cost[c] = D.COMMISSION[tariff][sp["kind"]] * comm_mult + tick_frac
    P["cost"] = pd.DataFrame(cost)
    P["codes"] = list(codes)
    P["days"] = np.array(days)
    return P


FIN_RATE = 0.00068   # плата за перенос непокрытой позиции, в календарный день (research/sources.md §7)


def backtest(P, W: pd.DataFrame, max_gross: float | None = None, fin_rate: float = FIN_RATE):
    """W — веса на закрытии дня (индекс дни, колонки инструменты). Возвращает дневную доходность и оборот."""
    W = W.reindex(index=P["c"].index, columns=P["codes"]).fillna(0.0)
    if max_gross is not None:
        g = W.abs().sum(1)
        W = W.mul(np.minimum(1.0, max_gross / g.replace(0, np.inf)), axis=0)
    # запрет открывать/менять позицию по инструменту, который не торгуется в день исполнения
    Wd = W.shift(1).fillna(0.0)            # веса, действующие с открытия дня (решены вчера)
    traded = P["traded"]
    # если инструмент не торгуется в день исполнения, держим старый вес
    Wexec = Wd.copy()
    prev = np.zeros(len(P["codes"]))
    arr = Wd.to_numpy().copy()
    tr = traded.to_numpy()
    for i in range(len(arr)):
        arr[i] = np.where(tr[i], arr[i], prev)
        prev = arr[i]
    Wexec = pd.DataFrame(arr, index=Wd.index, columns=Wd.columns)
    Wold = Wexec.shift(1).fillna(0.0)
    turn = (Wexec - Wold).abs()
    roll_cost = (Wold.abs() * P["roll"].astype(float) * 2.0)   # перекладка: закрыть старую + открыть новую серию
    # финансирование: шорт акций и лонг акций сверх 1× капитала оплачиваются за каждую ночь (календарные дни)
    eq_cols = [c for c in P["codes"] if c in D.EQUITIES]
    days_idx = pd.to_datetime(pd.Series(P["c"].index.astype(str)), format="%Y%m%d")
    cal_gap = days_idx.diff().dt.days.fillna(1).to_numpy()
    if eq_cols:
        short_eq = (-Wold[eq_cols].clip(upper=0)).sum(1)
        long_eq_over = (Wold[eq_cols].clip(lower=0).sum(1) - 1.0).clip(lower=0)
        fin = fin_rate * (short_eq + long_eq_over) * cal_gap
    else:
        fin = 0.0
    ret = (Wold * P["r_gap"]).sum(1) + (Wexec * P["r_intra"]).sum(1) - ((turn + roll_cost) * P["cost"]).sum(1) - fin
    gross = Wexec.abs().sum(1)
    return ret, turn.sum(1), gross, Wexec


# ---------------------------------------------------------------- сигналы (все каузальны: только данные <= t)
def vol(P, n=20):
    return P["r_cc"].rolling(n, min_periods=max(5, n // 2)).std()


def w_tsmom(P, n=20, target=0.01, vn=20, cap=3.0, cls=None):
    s = np.sign(P["c"] / P["c"].shift(n) - 1)
    return _scale(P, s, target, vn, cap, cls)


def w_ema(P, f=10, s=50, target=0.01, vn=20, cap=3.0, cls=None):
    ef = P["c"].ewm(span=f, adjust=False).mean()
    es = P["c"].ewm(span=s, adjust=False).mean()
    return _scale(P, np.sign(ef - es), target, vn, cap, cls)


def w_breakout(P, n=20, target=0.01, vn=20, cap=3.0, cls=None):
    hi = P["h"].rolling(n).max().shift(1)
    lo = P["l"].rolling(n).min().shift(1)
    sig = pd.DataFrame(np.nan, index=P["c"].index, columns=P["codes"])
    sig[P["c"] > hi] = 1.0
    sig[P["c"] < lo] = -1.0
    sig = sig.ffill().fillna(0.0)
    return _scale(P, sig, target, vn, cap, cls)


def w_xs(P, n=20, k=3, skip=0, target=0.01, vn=20, cap=3.0, cls=None, reverse=False, hold=1):
    """Кросс-секционный моментум (reverse=True — разворот): long top-k / short bottom-k по доходности за n дней."""
    r = P["c"].shift(skip) / P["c"].shift(skip + n) - 1
    r = _mask_cls(P, r, cls)
    rk = r.rank(axis=1, ascending=False)
    cnt = r.notna().sum(1)
    s = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    s[rk <= k] = 1.0
    s[rk.gt(cnt - k, axis=0) & r.notna()] = -1.0
    if reverse:
        s = -s
    if hold > 1:
        # ребалансировка раз в hold дней (по номеру дня — известен заранее)
        keep = np.arange(len(s)) % hold == 0
        s = s.where(pd.Series(keep, index=s.index), np.nan).ffill().fillna(0.0)
    return _scale(P, s, target, vn, cap, cls)


def w_carry(P, target=0.01, vn=20, cap=3.0, codes=("Si", "CR")):
    s = pd.DataFrame(0.0, index=P["c"].index, columns=P["codes"])
    for c in codes:
        s[c] = -1.0
    return _scale(P, s, target, vn, cap, None)


def w_buyhold(P, target=0.01, vn=20, cap=3.0, cls=None, side=1.0):
    s = pd.DataFrame(side, index=P["c"].index, columns=P["codes"])
    return _scale(P, s, target, vn, cap, cls)


def _mask_cls(P, X, cls):
    if cls is None:
        return X
    keep = [c for c in P["codes"] if (c in D.EQUITIES) == (cls == "EQ")]
    return X.where(pd.DataFrame({c: c in keep for c in P["codes"]}, index=[0]).reindex(X.index, method="ffill").fillna(False).to_numpy(), np.nan)


def _scale(P, s, target, vn, cap, cls):
    s = _mask_cls(P, pd.DataFrame(s, index=P["c"].index, columns=P["codes"]), cls).fillna(0.0)
    v = vol(P, vn)
    w = s * (target / v).clip(upper=cap)
    n_active = (s != 0).sum(1).replace(0, 1)
    return w.div(np.sqrt(n_active), axis=0).fillna(0.0)
