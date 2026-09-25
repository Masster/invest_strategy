"""Индикаторы (Numba/NumPy). Все функции каузальны: значение в i использует только данные <= i.
Проверяется тестом tests/causality/test_deep_causality.py (обрезка ряда не меняет прошлые значения)."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def ema(x, n):
    out = np.empty_like(x)
    a = 2.0 / (n + 1.0)
    s = x[0]
    for i in range(len(x)):
        v = x[i]
        if np.isnan(v):
            out[i] = s
            continue
        s = v if i == 0 else a * v + (1 - a) * s
        out[i] = s
    return out


@njit(cache=True)
def wilder(x, n):
    out = np.empty_like(x)
    s = x[0]
    for i in range(len(x)):
        s = x[i] if i == 0 else s + (x[i] - s) / n
        out[i] = s
    return out


@njit(cache=True)
def sma(x, n):
    out = np.full(len(x), np.nan)
    s = 0.0
    for i in range(len(x)):
        s += x[i]
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


@njit(cache=True)
def rstd(x, n):
    out = np.full(len(x), np.nan)
    s = 0.0
    q = 0.0
    for i in range(len(x)):
        s += x[i]
        q += x[i] * x[i]
        if i >= n:
            s -= x[i - n]
            q -= x[i - n] * x[i - n]
        if i >= n - 1:
            m = s / n
            v = q / n - m * m
            out[i] = np.sqrt(v) if v > 0 else 0.0
    return out


@njit(cache=True)
def rmax(x, n):
    """Максимум x[i-n+1..i] (монотонная очередь)."""
    m = len(x)
    out = np.empty(m)
    dq = np.empty(m, np.int64)
    hd = 0
    tl = 0
    for i in range(m):
        while tl > hd and x[dq[tl - 1]] <= x[i]:
            tl -= 1
        dq[tl] = i
        tl += 1
        if dq[hd] <= i - n:
            hd += 1
        out[i] = x[dq[hd]]
    return out


@njit(cache=True)
def rmin(x, n):
    m = len(x)
    out = np.empty(m)
    dq = np.empty(m, np.int64)
    hd = 0
    tl = 0
    for i in range(m):
        while tl > hd and x[dq[tl - 1]] >= x[i]:
            tl -= 1
        dq[tl] = i
        tl += 1
        if dq[hd] <= i - n:
            hd += 1
        out[i] = x[dq[hd]]
    return out


def shift(x, k=1, fill=np.nan):
    out = np.empty_like(x, dtype=float)
    if k <= 0:
        return x.astype(float)
    out[:k] = fill
    out[k:] = x[:-k]
    return out


@njit(cache=True)
def atr(h, l, c, n):
    tr = np.empty(len(h))
    tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        a = h[i] - l[i]
        b = abs(h[i] - c[i - 1])
        d = abs(l[i] - c[i - 1])
        tr[i] = max(a, max(b, d))
    return wilder(tr, n)


@njit(cache=True)
def rsi(c, n):
    up = np.zeros(len(c))
    dn = np.zeros(len(c))
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        if d > 0:
            up[i] = d
        else:
            dn[i] = -d
    au = wilder(up, n)
    ad = wilder(dn, n)
    out = np.empty(len(c))
    for i in range(len(c)):
        out[i] = 100.0 if ad[i] == 0 else 100.0 - 100.0 / (1.0 + au[i] / ad[i])
    return out


@njit(cache=True)
def dmi(h, l, c, n):
    m = len(h)
    pdm = np.zeros(m)
    ndm = np.zeros(m)
    tr = np.zeros(m)
    for i in range(1, m):
        u = h[i] - h[i - 1]
        d = l[i - 1] - l[i]
        pdm[i] = u if (u > d and u > 0) else 0.0
        ndm[i] = d if (d > u and d > 0) else 0.0
        tr[i] = max(h[i] - l[i], max(abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    atr_ = wilder(tr, n)
    pdi = np.zeros(m)
    ndi = np.zeros(m)
    dx = np.zeros(m)
    ps = wilder(pdm, n)
    ns = wilder(ndm, n)
    for i in range(m):
        if atr_[i] > 0:
            pdi[i] = 100 * ps[i] / atr_[i]
            ndi[i] = 100 * ns[i] / atr_[i]
        s = pdi[i] + ndi[i]
        dx[i] = 100 * abs(pdi[i] - ndi[i]) / s if s > 0 else 0.0
    adx = wilder(dx, n)
    return pdi, ndi, adx


@njit(cache=True)
def supertrend(h, l, c, a, mult):
    """Направление SuperTrend (+1/-1) по ATR-массиву a."""
    m = len(c)
    d = np.ones(m)
    ub = np.zeros(m)
    lb = np.zeros(m)
    for i in range(m):
        mid = 0.5 * (h[i] + l[i])
        bu = mid + mult * a[i]
        bl = mid - mult * a[i]
        if i == 0:
            ub[i] = bu
            lb[i] = bl
            continue
        ub[i] = bu if (bu < ub[i - 1] or c[i - 1] > ub[i - 1]) else ub[i - 1]
        lb[i] = bl if (bl > lb[i - 1] or c[i - 1] < lb[i - 1]) else lb[i - 1]
        if d[i - 1] > 0:
            d[i] = -1.0 if c[i] < lb[i] else 1.0
        else:
            d[i] = 1.0 if c[i] > ub[i] else -1.0
    return d


@njit(cache=True)
def wma(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    den = n * (n + 1) / 2.0
    for i in range(n - 1, m):
        s = 0.0
        for k in range(n):
            s += x[i - k] * (n - k)
        out[i] = s / den
    return out


def hma(x, n):
    n2 = max(int(n / 2), 1)
    ns = max(int(np.sqrt(n)), 1)
    a = wma(x, n2)
    b = wma(x, n)
    d = 2 * a - b
    d = np.where(np.isnan(d), x, d)
    return wma(d, ns)


@njit(cache=True)
def session_vwap(h, l, c, v, first_of_day):
    m = len(c)
    out = np.empty(m)
    pv = 0.0
    vv = 0.0
    for i in range(m):
        if first_of_day[i]:
            pv = 0.0
            vv = 0.0
        tp = (h[i] + l[i] + c[i]) / 3.0
        pv += tp * v[i]
        vv += v[i]
        out[i] = pv / vv if vv > 0 else c[i]
    return out


@njit(cache=True)
def day_levels(o, h, l, c, first_of_day):
    """Для каждой свечи: open дня, high/low дня по i, close/high/low предыдущего дня."""
    m = len(c)
    dopen = np.empty(m)
    dhi = np.empty(m)
    dlo = np.empty(m)
    pc = np.full(m, np.nan)
    ph = np.full(m, np.nan)
    pl = np.full(m, np.nan)
    co = 0.0
    chi = 0.0
    clo = 0.0
    lc = np.nan
    lh = np.nan
    ll = np.nan
    for i in range(m):
        if first_of_day[i]:
            if i > 0:
                lc = c[i - 1]
                lh = chi
                ll = clo
            co = o[i]
            chi = h[i]
            clo = l[i]
        else:
            chi = max(chi, h[i])
            clo = min(clo, l[i])
        dopen[i] = co
        dhi[i] = chi
        dlo[i] = clo
        pc[i] = lc
        ph[i] = lh
        pl[i] = ll
    return dopen, dhi, dlo, pc, ph, pl


@njit(cache=True)
def opening_range(h, l, mod, first_of_day, start_mod, end_mod):
    """High/low диапазона [start_mod, end_mod) текущего дня; NaN, пока диапазон не сформирован
    (значение доступно на свечах, начинающихся не раньше end_mod)."""
    m = len(h)
    orh = np.full(m, np.nan)
    orl = np.full(m, np.nan)
    hi = -1e300
    lo = 1e300
    have = False
    for i in range(m):
        if first_of_day[i]:
            hi = -1e300
            lo = 1e300
            have = False
        if mod[i] >= start_mod and mod[i] < end_mod:
            hi = max(hi, h[i])
            lo = min(lo, l[i])
            have = True
        if have and mod[i] >= end_mod:
            orh[i] = hi
            orl[i] = lo
    return orh, orl


@njit(cache=True)
def bars_since_day_start(first_of_day):
    m = len(first_of_day)
    out = np.zeros(m, np.int64)
    k = 0
    for i in range(m):
        if first_of_day[i]:
            k = 0
        out[i] = k
        k += 1
    return out


def logret(x):
    r = np.zeros(len(x))
    r[1:] = np.log(x[1:] / x[:-1])
    return r


def roc(x, n):
    out = np.full(len(x), np.nan)
    out[n:] = x[n:] / x[:-n] - 1.0
    return out


def rolling_percentile_rank(x, n):
    """Доля значений из последних n (включая текущее), не превосходящих текущее."""
    return _prank(np.asarray(x, float), int(n))


@njit(cache=True)
def _prank(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        cnt = 0
        v = x[i]
        if np.isnan(v):
            continue
        for k in range(i - n + 1, i + 1):
            if x[k] <= v:
                cnt += 1
        out[i] = cnt / n
    return out
