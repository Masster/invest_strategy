"""Семейства торговых гипотез: функции (ctx, params) -> сигналы для engine.simulate.

Соглашения:
  * все признаки — по сигнальным (скорректированным) ценам ctx.ao/ah/al/ac, только данные <= i;
  * long/short — булевы массивы «на закрытии i хотим позицию/вход»; state=True — сигнал-состояние
    (повторный вход после стопа только после сброса сигнала);
  * exit_long/exit_short — выход по сигналу (по open i+1);
  * mode 1 — стоп-заявка на уровне lvl_* (реальная цена), mode 2 — лимит;
  * intraday=True — семейство имеет смысл только внутри дня (движок закрывает позицию в конце дня).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import features as F


class Ctx:
    """Свечи одного инструмента/таймфрейма + кэш индикаторов."""

    def __init__(self, A: dict, tf: int, code: str):
        self.A = A
        self.tf = tf
        self.code = code
        self.o, self.h, self.l, self.c, self.v = A["o"], A["h"], A["l"], A["c"], A["v"]
        self.ao, self.ah, self.al, self.ac = A["ao"], A["ah"], A["al"], A["ac"]
        self.adj = self.ac / self.c
        self.n = len(self.c)
        self._cache = {}

    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # --- базовые индикаторы
    def ema(self, n):
        return self.get(("ema", n), lambda: F.ema(self.ac, n))

    def sma(self, n):
        return self.get(("sma", n), lambda: F.sma(self.ac, n))

    def std(self, n):
        return self.get(("std", n), lambda: F.rstd(self.ac, n))

    def atr(self, n=14):
        return self.get(("atr", n), lambda: F.atr(self.ah, self.al, self.ac, n))

    def atr_real(self, n=14):
        return self.get(("atr_real", n), lambda: self.atr(n) / self.adj)

    def hh(self, n):   # максимум high за n предыдущих свечей (без текущей)
        return self.get(("hh", n), lambda: F.shift(F.rmax(self.ah, n), 1))

    def ll(self, n):
        return self.get(("ll", n), lambda: F.shift(F.rmin(self.al, n), 1))

    def hh_incl(self, n):
        return self.get(("hhi", n), lambda: F.rmax(self.ah, n))

    def ll_incl(self, n):
        return self.get(("lli", n), lambda: F.rmin(self.al, n))

    def lr(self):
        return self.get("lr", lambda: F.logret(self.ac))

    def vol(self, n):
        return self.get(("vol", n), lambda: F.rstd(self.lr(), n))

    def roc(self, n):
        return self.get(("roc", n), lambda: F.roc(self.ac, n))

    def rsi(self, n):
        return self.get(("rsi", n), lambda: F.rsi(self.ac, n))

    def dmi(self, n):
        return self.get(("dmi", n), lambda: F.dmi(self.ah, self.al, self.ac, n))

    def vsma(self, n):
        return self.get(("vsma", n), lambda: F.shift(F.sma(self.v, n), 1))

    def vwap(self):
        return self.get("vwap", lambda: F.session_vwap(self.ah, self.al, self.ac, self.v, self.A["first_of_day"]))

    def daylv(self):
        return self.get("daylv", lambda: F.day_levels(self.ao, self.ah, self.al, self.ac, self.A["first_of_day"]))

    def bsd(self):
        return self.get("bsd", lambda: F.bars_since_day_start(self.A["first_of_day"]))

    def real(self, lvl):
        return lvl / self.adj


@dataclass
class Family:
    name: str
    group: str
    fn: object
    grid: dict
    tfs: tuple = (1, 2, 3, 5, 10, 15, 30, 60, 1440)
    intraday: bool = False       # только внутридневное удержание
    state: bool = True
    exits: tuple = ("X0", "X1", "X2", "X3", "X4", "X5")
    note: str = ""

    def params(self):
        keys = list(self.grid)
        for vals in itertools.product(*[self.grid[k] for k in keys]):
            yield dict(zip(keys, vals))


REGISTRY: dict[str, Family] = {}


def family(name, group, grid, **kw):
    def deco(fn):
        REGISTRY[name] = Family(name, group, fn, grid, **kw)
        return fn
    return deco


def _nz(x):
    return np.nan_to_num(x, nan=0.0)


def _b(x):
    return np.asarray(x, dtype=np.bool_)


# ============================================================ ТРЕНД
@family("T_DONCH", "trend", {"n": [10, 20, 40, 80, 160]})
def t_donch(x: Ctx, p):
    n = p["n"]
    up = x.ac > x.hh(n)
    dn = x.ac < x.ll(n)
    m = max(n // 2, 2)
    return dict(long=_b(up), short=_b(dn), exit_long=_b(x.ac < x.ll(m)), exit_short=_b(x.ac > x.hh(m)))


@family("T_DONCH_STOP", "trend", {"n": [10, 20, 40, 80, 160]}, state=False)
def t_donch_stop(x: Ctx, p):
    """Стоп-заявки на пробой канала n: сторона — ближайший к цене уровень."""
    n = p["n"]
    hi = x.hh_incl(n)
    lo = x.ll_incl(n)
    tick_l = np.ones(x.n, np.bool_)
    near_up = (hi - x.ac) <= (x.ac - lo)
    m = max(n // 2, 2)
    return dict(long=_b(tick_l & near_up & np.isfinite(hi)), short=_b(tick_l & ~near_up & np.isfinite(lo)), mode=1,
                lvl_long=x.real(hi) + 1e-9, lvl_short=x.real(lo) - 1e-9,
                exit_long=_b(x.ac < x.ll(m)), exit_short=_b(x.ac > x.hh(m)))


@family("T_EMA_X", "trend", {"f": [5, 10, 20, 50], "s": [20, 50, 100, 200, 400]})
def t_ema_x(x: Ctx, p):
    if p["f"] >= p["s"]:
        return None
    d = x.ema(p["f"]) - x.ema(p["s"])
    return dict(long=_b(d > 0), short=_b(d < 0))


@family("T_PRICE_MA", "trend", {"n": [20, 50, 100, 200, 400], "band": [0.0, 0.5]})
def t_price_ma(x: Ctx, p):
    e = x.ema(p["n"])
    b = p["band"] * x.atr(14)
    return dict(long=_b(x.ac > e + b), short=_b(x.ac < e - b), exit_long=_b(x.ac < e), exit_short=_b(x.ac > e))


@family("T_HMA", "trend", {"n": [16, 36, 64, 128, 256]})
def t_hma(x: Ctx, p):
    hm = x.get(("hma", p["n"]), lambda: F.hma(x.ac, p["n"]))
    s = hm - F.shift(hm, 1)
    return dict(long=_b(s > 0), short=_b(s < 0))


@family("T_SUPERTREND", "trend", {"n": [10, 20, 40], "m": [2.0, 3.0, 4.0]})
def t_supertrend(x: Ctx, p):
    d = x.get(("st", p["n"], p["m"]), lambda: F.supertrend(x.ah, x.al, x.ac, x.atr(p["n"]), p["m"]))
    return dict(long=_b(d > 0), short=_b(d < 0))


@family("T_ADX", "trend", {"n": [14, 28], "thr": [20, 25, 30]})
def t_adx(x: Ctx, p):
    pdi, ndi, adx = x.dmi(p["n"])
    strong = adx > p["thr"]
    lg = strong & (pdi > ndi)
    sh = strong & (ndi > pdi)
    return dict(long=_b(lg), short=_b(sh), exit_long=_b(pdi < ndi), exit_short=_b(ndi < pdi))


@family("T_MTF", "trend", {"n": [5, 10, 20, 40], "k": [4, 8]})
def t_mtf(x: Ctx, p):
    n, k = p["n"], p["k"]
    a, b, c = x.roc(n), x.roc(n * k), x.roc(n * k * k)
    return dict(long=_b((a > 0) & (b > 0) & (c > 0)), short=_b((a < 0) & (b < 0) & (c < 0)),
                exit_long=_b(b < 0), exit_short=_b(b > 0))


@family("T_KELTNER", "trend", {"n": [20, 50, 100], "k": [1.0, 2.0, 3.0]})
def t_keltner(x: Ctx, p):
    e = x.ema(p["n"])
    a = x.atr(p["n"])
    return dict(long=_b(x.ac > e + p["k"] * a), short=_b(x.ac < e - p["k"] * a),
                exit_long=_b(x.ac < e), exit_short=_b(x.ac > e))


# ============================================================ МОМЕНТУМ
@family("M_TSMOM", "momentum", {"n": [10, 20, 50, 100, 200, 400], "z": [0.0, 0.5, 1.0]})
def m_tsmom(x: Ctx, p):
    n = p["n"]
    r = x.roc(n)
    s = x.vol(max(n, 20)) * np.sqrt(n)
    return dict(long=_b(r > p["z"] * s), short=_b(r < -p["z"] * s))


@family("M_ACCEL", "momentum", {"n": [10, 20, 50, 100]})
def m_accel(x: Ctx, p):
    n = p["n"]
    r = x.roc(n)
    acc = r - F.shift(r, n)
    return dict(long=_b((r > 0) & (acc > 0)), short=_b((r < 0) & (acc < 0)),
                exit_long=_b(acc < 0), exit_short=_b(acc > 0))


@family("M_ROC_VOTE", "momentum", {"base": [5, 10, 20, 40], "need": [3, 4]})
def m_roc_vote(x: Ctx, p):
    b = p["base"]
    votes = np.zeros(x.n)
    for k in (1, 2, 4, 8):
        votes += np.sign(_nz(x.roc(b * k)))
    return dict(long=_b(votes >= p["need"]), short=_b(votes <= -p["need"]),
                exit_long=_b(votes <= 0), exit_short=_b(votes >= 0))


# ============================================================ ВОЗВРАТ К СРЕДНЕМУ
@family("R_BB", "meanrev", {"n": [10, 20, 50, 100], "k": [1.5, 2.0, 2.5, 3.0]}, state=True)
def r_bb(x: Ctx, p):
    m = x.sma(p["n"])
    s = x.std(p["n"])
    return dict(long=_b(x.ac < m - p["k"] * s), short=_b(x.ac > m + p["k"] * s),
                exit_long=_b(x.ac >= m), exit_short=_b(x.ac <= m))


@family("R_BB_LIMIT", "meanrev", {"n": [10, 20, 50], "k": [2.0, 2.5, 3.0]}, state=False)
def r_bb_limit(x: Ctx, p):
    """Лимитные заявки на полосах Боллинджера (вход откатом, без спреда)."""
    m = x.sma(p["n"])
    s = x.std(p["n"])
    lo = m - p["k"] * s
    hi = m + p["k"] * s
    near_lo = (x.ac - lo) <= (hi - x.ac)
    ok = np.isfinite(lo)
    return dict(long=_b(ok & near_lo), short=_b(ok & ~near_lo), mode=2, lvl_long=x.real(lo), lvl_short=x.real(hi),
                exit_long=_b(x.ac >= m), exit_short=_b(x.ac <= m))


@family("R_RSI", "meanrev", {"n": [2, 3, 5, 14], "lo": [5, 10, 20, 30]})
def r_rsi(x: Ctx, p):
    r = x.rsi(p["n"])
    lo = p["lo"]
    return dict(long=_b(r < lo), short=_b(r > 100 - lo), exit_long=_b(r > 50), exit_short=_b(r < 50))


@family("R_MA_DEV", "meanrev", {"n": [20, 50, 200], "k": [1.0, 2.0, 3.0]})
def r_ma_dev(x: Ctx, p):
    e = x.ema(p["n"])
    a = x.atr(p["n"])
    return dict(long=_b(x.ac < e - p["k"] * a), short=_b(x.ac > e + p["k"] * a),
                exit_long=_b(x.ac >= e), exit_short=_b(x.ac <= e))


@family("R_VWAP", "meanrev", {"k": [1.0, 2.0, 3.0], "n": [20, 60]}, tfs=(1, 2, 3, 5, 10, 15, 30), intraday=True)
def r_vwap(x: Ctx, p):
    vw = x.vwap()
    dev = x.ac - vw
    s = x.get(("vwdev_sd", p["n"]), lambda: F.rstd(_nz(dev), p["n"] * 10))
    bsd = x.bsd()
    ok = bsd >= p["n"] // 4
    return dict(long=_b(ok & (dev < -p["k"] * s)), short=_b(ok & (dev > p["k"] * s)),
                exit_long=_b(dev >= 0), exit_short=_b(dev <= 0))


@family("R_EXTREME", "meanrev", {"m": [1, 3, 5, 10], "z": [2.0, 3.0, 4.0]})
def r_extreme(x: Ctx, p):
    m = p["m"]
    r = x.roc(m)
    s = x.vol(100) * np.sqrt(m)
    return dict(long=_b(r < -p["z"] * s), short=_b(r > p["z"] * s),
                exit_long=_b(x.ac > x.ema(10)), exit_short=_b(x.ac < x.ema(10)))


@family("R_GAP_FADE", "meanrev", {"z": [0.5, 1.0, 2.0]}, tfs=(1, 5, 15, 60), intraday=True, state=False)
def r_gap_fade(x: Ctx, p):
    dopen, dhi, dlo, pc, ph, pl = x.daylv()
    first = x.A["first_of_day"]
    gap = dopen / pc - 1
    s = x.get("dvol", lambda: _daily_vol(x))
    sig_l = first & (gap < -p["z"] * s)
    sig_s = first & (gap > p["z"] * s)
    return dict(long=_b(sig_l), short=_b(sig_s), exit_long=_b(x.ac >= pc), exit_short=_b(x.ac <= pc))


@family("R_VOLSPIKE_FADE", "meanrev", {"k": [3.0, 5.0], "z": [2.0, 3.0]}, tfs=(1, 2, 3, 5, 10, 15, 30, 60))
def r_volspike_fade(x: Ctx, p):
    rv = x.v / x.vsma(100)
    r = x.lr()
    s = x.vol(100)
    return dict(long=_b((rv > p["k"]) & (r < -p["z"] * s)), short=_b((rv > p["k"]) & (r > p["z"] * s)),
                exit_long=_b(x.ac > x.ema(10)), exit_short=_b(x.ac < x.ema(10)))


def _daily_vol(x: Ctx):
    """σ дневной доходности по завершённым дням (каузально, на каждую свечу)."""
    first = x.A["first_of_day"]
    idx = np.where(first)[0]
    closes_prev = np.full(x.n, np.nan)
    out = np.full(x.n, np.nan)
    # закрытия завершённых дней
    ends = np.append(idx[1:] - 1, x.n - 1)
    dc = x.ac[ends]
    dr = np.full(len(dc), np.nan)
    dr[1:] = np.log(dc[1:] / dc[:-1])
    sd = np.full(len(dc), np.nan)
    for j in range(len(dc)):
        w = dr[max(1, j - 19):j + 1]
        w = w[np.isfinite(w)]
        if len(w) >= 5:
            sd[j] = w.std()
    # день j использует sd[j-1] (закрытые дни)
    for j, s0 in enumerate(idx):
        e = ends[j]
        out[s0:e + 1] = sd[j - 1] if j >= 1 else np.nan
    _ = closes_prev
    return out


# ============================================================ ВОЛАТИЛЬНОСТЬ
@family("V_SQUEEZE", "volatility", {"n": [20, 50], "pct": [0.1, 0.2], "lb": [200, 500]})
def v_squeeze(x: Ctx, p):
    n = p["n"]
    w = x.std(n) / x.sma(n)
    pr = x.get(("prank", n, p["lb"]), lambda: F.rolling_percentile_rank(w, p["lb"]))
    sq = F.shift(pr, 1) <= p["pct"]
    return dict(long=_b(sq & (x.ac > x.hh(n))), short=_b(sq & (x.ac < x.ll(n))),
                exit_long=_b(x.ac < x.sma(n)), exit_short=_b(x.ac > x.sma(n)))


@family("V_NR_BREAK", "volatility", {"n": [4, 7, 14]}, state=False)
def v_nr_break(x: Ctx, p):
    rng = x.ah - x.al
    nr = rng <= F.rmin(rng, p["n"])
    hi, lo = x.ah, x.al
    near_up = (hi - x.ac) <= (x.ac - lo)
    return dict(long=_b(nr & near_up), short=_b(nr & ~near_up), mode=1, lvl_long=x.real(hi) + 1e-9,
                lvl_short=x.real(lo) - 1e-9)


@family("V_RANGE_EXP", "volatility", {"k": [1.5, 2.0, 3.0], "fade": [0, 1]})
def v_range_exp(x: Ctx, p):
    a = F.shift(x.atr(20), 1)
    rng = x.ah - x.al
    pos = (x.ac - x.al) / np.where(rng > 0, rng, np.nan)
    big = rng > p["k"] * a
    up = big & (pos > 0.75)
    dn = big & (pos < 0.25)
    if p["fade"]:
        up, dn = dn, up
    return dict(long=_b(up), short=_b(dn), state=False)


@family("V_REGIME_TREND", "volatility", {"n": [20, 50, 100], "hi": [0, 1]})
def v_regime_trend(x: Ctx, p):
    """Тренд (EMA n vs 4n), только в режиме высокой (hi=1) или низкой (hi=0) волатильности (медиана за 1000 свечей)."""
    n = p["n"]
    vv = x.vol(n)
    med = x.get(("volmed", n), lambda: F.shift(_rolling_median(vv, 1000), 1))
    reg = (vv > med) if p["hi"] else (vv <= med)
    d = x.ema(n) - x.ema(4 * n)
    return dict(long=_b(reg & (d > 0)), short=_b(reg & (d < 0)), exit_long=_b(d < 0), exit_short=_b(d > 0))


def _rolling_median(v, n):
    out = np.full(len(v), np.nan)
    step = max(n // 20, 1)
    last = np.nan
    for i in range(len(v)):
        if i >= n and i % step == 0:
            w = v[i - n:i]
            last = np.nanmedian(w)
        out[i] = last
    return out


# ============================================================ ОБЪЁМ
@family("U_VOL_BREAK", "volume", {"n": [20, 50], "k": [1.5, 3.0]})
def u_vol_break(x: Ctx, p):
    rv = x.v / x.vsma(p["n"])
    return dict(long=_b((x.ac > x.hh(p["n"])) & (rv > p["k"])), short=_b((x.ac < x.ll(p["n"])) & (rv > p["k"])),
                exit_long=_b(x.ac < x.ema(p["n"])), exit_short=_b(x.ac > x.ema(p["n"])))


@family("U_SIGNED_VOL", "volume", {"n": [10, 30, 100], "thr": [0.2, 0.4], "fade": [0, 1]})
def u_signed_vol(x: Ctx, p):
    """Прокси order flow: доля знакового объёма sign(close-open)*volume за n свечей."""
    sv = np.sign(x.ac - x.ao) * x.v
    num = F.sma(sv, p["n"])
    den = F.sma(x.v, p["n"])
    imb = num / np.where(den > 0, den, np.nan)
    up, dn = imb > p["thr"], imb < -p["thr"]
    if p["fade"]:
        up, dn = dn, up
    return dict(long=_b(up), short=_b(dn), exit_long=_b(np.abs(_nz(imb)) < 0.05), exit_short=_b(np.abs(_nz(imb)) < 0.05))


@family("U_OBV", "volume", {"n": [20, 100]})
def u_obv(x: Ctx, p):
    obv = np.cumsum(np.sign(_nz(x.lr())) * x.v)
    e = F.ema(obv, p["n"])
    return dict(long=_b((obv > e) & (x.ac > x.ema(p["n"]))), short=_b((obv < e) & (x.ac < x.ema(p["n"]))))


@family("U_BAR_PRESSURE", "volume", {"n": [5, 20], "thr": [0.7, 0.8]})
def u_bar_pressure(x: Ctx, p):
    """Прокси давления: средняя позиция close в диапазоне свечи, взвешенная объёмом."""
    rng = x.ah - x.al
    pos = np.where(rng > 0, (x.ac - x.al) / np.where(rng > 0, rng, 1), 0.5)
    w = F.sma(pos * x.v, p["n"]) / np.where(F.sma(x.v, p["n"]) > 0, F.sma(x.v, p["n"]), np.nan)
    return dict(long=_b(w > p["thr"]), short=_b(w < 1 - p["thr"]),
                exit_long=_b(_nz(w) < 0.5), exit_short=_b(_nz(w) > 0.5))


# ============================================================ ВНУТРИДНЕВНЫЕ
@family("I_ORB", "intraday", {"start": [600, 420], "len": [15, 30, 60], "cut": [240, 480]},
        tfs=(1, 5, 15), intraday=True, state=False, exits=("X0", "X1", "X3", "X4"))
def i_orb(x: Ctx, p):
    """Пробой диапазона открытия (start — минута МСК: 600 = 10:00 основная, 420 = 07:00 утро/09:00 до 14.07)."""
    start = p["start"]
    mod = x.A["mod"]
    first = x.A["first_of_day"]
    # до 14.07.2026 фьючерсы открывались в 09:00: для start=420 берём первую свечу дня
    if start == 420:
        st = np.zeros(x.n, np.int64)
        cur = 0
        for i in range(x.n):
            if first[i]:
                cur = mod[i]
            st[i] = cur
        orh, orl = _or_dynamic(x, st, p["len"])
    else:
        orh, orl = F.opening_range(x.ah, x.al, mod, first, start, start + p["len"])
    end_mod = (start if start != 420 else 540) + p["len"] + p["cut"]
    ok = np.isfinite(orh) & (mod < end_mod)
    near_up = (orh - x.ac) <= (x.ac - orl)
    return dict(long=_b(ok & near_up & (x.ac < orh)), short=_b(ok & ~near_up & (x.ac > orl)), mode=1,
                lvl_long=x.real(orh) + 1e-9, lvl_short=x.real(orl) - 1e-9)


def _or_dynamic(x, st, ln):
    orh = np.full(x.n, np.nan)
    orl = np.full(x.n, np.nan)
    first = x.A["first_of_day"]
    mod = x.A["mod"]
    hi, lo = -np.inf, np.inf
    for i in range(x.n):
        if first[i]:
            hi, lo = -np.inf, np.inf
        if mod[i] < st[i] + ln:
            hi = max(hi, x.ah[i])
            lo = min(lo, x.al[i])
        elif np.isfinite(hi):
            orh[i], orl[i] = hi, lo
    return orh, orl


@family("I_PDAY_BRK", "intraday", {"hold": [0, 1]}, tfs=(1, 5, 15, 60), state=False, exits=("X0", "X1", "X3", "X4"))
def i_pday_brk(x: Ctx, p):
    """Стоп-заявки на пробой high/low предыдущего дня (hold=0 — закрытие в конце дня)."""
    dopen, dhi, dlo, pc, ph, pl = x.daylv()
    fresh_up = dhi <= ph
    fresh_dn = dlo >= pl
    near_up = (ph - x.ac) <= (x.ac - pl)
    return dict(long=_b(fresh_up & near_up & np.isfinite(ph)), short=_b(fresh_dn & ~near_up & np.isfinite(pl)),
                mode=1, lvl_long=x.real(ph) + 1e-9, lvl_short=x.real(pl) - 1e-9, eod=(p["hold"] == 0))


@family("I_TOD", "seasonal", {"at": list(range(420, 1410, 30)), "hold": [30, 60, 120, 240], "side": [1, -1]},
        tfs=(1,), intraday=True, state=False, exits=("X0",))
def i_tod(x: Ctx, p):
    """Сезонность времени дня: вход по open свечи, начинающейся в минуту `at` МСК, выход через hold минут."""
    mod = x.A["mod"]
    nxt = np.append(mod[1:], -1)
    same_day = np.append(~x.A["first_of_day"][1:], False)
    trig = (nxt == p["at"]) & same_day
    # выход: через hold минут по open (в минутах МСК)
    t_exit = p["at"] + p["hold"]
    ex = (np.append(mod[1:], 10 ** 6) >= t_exit) & (mod < t_exit)
    if p["side"] > 0:
        return dict(long=_b(trig), exit_long=_b(ex), dir=1)
    return dict(short=_b(trig), exit_short=_b(ex), dir=-1)


@family("I_OVERNIGHT", "seasonal", {"exit_after": [0, 30, 60, 180], "side": [1, -1]}, tfs=(1,), state=False,
        exits=("X0",))
def i_overnight(x: Ctx, p):
    """Перенос через ночь: вход на open последней минутной свечи дня, выход через exit_after минут после открытия."""
    last = x.A["last_of_day"]
    pre_last = np.append(last[1:], False) & ~last
    first = x.A["first_of_day"]
    bsd = x.bsd()
    ex = bsd >= max(p["exit_after"], 0)
    ex = ex & ~last
    if p["exit_after"] == 0:
        ex = first.copy()
    if p["side"] > 0:
        return dict(long=_b(pre_last), exit_long=_b(ex), dir=1)
    return dict(short=_b(pre_last), exit_short=_b(ex), dir=-1)


@family("I_DOW", "seasonal", {"dow": [0, 1, 2, 3, 4], "side": [1, -1], "part": [0, 1]}, tfs=(1440,), state=False,
        exits=("X0",))
def i_dow(x: Ctx, p):
    """День недели: part=0 — удержание с close предыдущего дня до close дня dow; part=1 — open→close дня dow."""
    day = x.A["day"]
    dow = np.array([(np.datetime64(f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}").astype("datetime64[D]")
                     .astype(np.int64) - 4) % 7 for d in day])
    nxt = np.append(dow[1:], -1)
    if p["part"] == 0:
        trig = np.append(dow[1:] == p["dow"], False)   # вход на open следующего дня... см. ниже
        # удержание close(D-1)->close(D) приближаем входом по open дня D (нет входа по close)
        sig = trig
        ex = dow == p["dow"]
        eod = False
    else:
        sig = nxt == p["dow"]
        ex = dow == p["dow"]
        eod = True
    if p["side"] > 0:
        return dict(long=_b(sig), exit_long=_b(ex), dir=1, eod=eod)
    return dict(short=_b(sig), exit_short=_b(ex), dir=-1, eod=eod)


# ============================================================ CARRY
@family("C_CARRY", "carry", {"filt": [0, 50, 200]}, tfs=(1440,), exits=("X0", "X2", "X4"))
def c_carry(x: Ctx, p):
    """Постоянный шорт (контанго валютных/индексных фьючерсов), опционально только ниже EMA(filt)."""
    if p["filt"] == 0:
        sh = np.ones(x.n, np.bool_)
    else:
        sh = x.ac < x.ema(p["filt"])
    return dict(short=_b(sh), dir=-1)


def build(fam: Family, ctx: Ctx, params: dict):
    sig = fam.fn(ctx, params)
    if sig is None:
        return None
    sig.setdefault("state", fam.state)
    return sig
