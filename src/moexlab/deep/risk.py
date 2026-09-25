"""Плечо, сложный процент, просадки, Monte Carlo (блочный бутстреп торговых дней), риск разорения."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import metrics as M

RUIN = 0.10          # разорение = капитал упал до 10% начального (потеря 90%)


def equity_path(r: np.ndarray, lev: float = 1.0, start: float = 1.0) -> np.ndarray:
    g = np.clip(1.0 + lev * r, 0.0, None)
    eq = start * np.cumprod(g)
    # после нуля капитал остаётся нулём
    z = np.where(eq <= 0)[0]
    if len(z):
        eq[z[0]:] = 0.0
    return eq


def path_stats(r: np.ndarray, cal: np.ndarray, lev: float = 1.0) -> dict:
    eq = equity_path(r, lev)
    pk = np.maximum.accumulate(np.append(1.0, eq))[1:]
    dd = 1 - eq / pk
    mk = M.month_keys(cal)
    months = np.unique(mk)
    prev = 1.0
    mret = []
    for m in months:
        e = eq[mk == m][-1]
        mret.append(e / prev - 1 if prev > 0 else -1.0)
        prev = e
    mret = np.array(mret)
    return dict(final=float(eq[-1]), total=float(eq[-1] - 1), mdd=float(dd.max()), mret=mret,
                pmr=float((mret > 0).mean()), m_min=float(mret.min()), m_median=float(np.median(mret)),
                m_mean=float(mret.mean()), m_max=float(mret.max()), m_std=float(mret.std()),
                ruin=bool(eq.min() <= RUIN))


@njit(parallel=True, cache=True)
def _mc(r, n_paths, horizon, block, lev, seed, month_len):
    n = len(r)
    n_m = horizon // month_len
    mdd = np.empty(n_paths)
    final = np.empty(n_paths)
    minm = np.empty(n_paths)
    pmr = np.empty(n_paths)
    medm = np.empty(n_paths)
    ruin = np.zeros(n_paths, np.bool_)
    for p in prange(n_paths):
        np.random.seed(seed + p)
        eq = 1.0
        pk = 1.0
        md = 0.0
        t = 0
        mstart = 1.0
        mr = np.empty(n_m)
        k = 0
        while t < horizon:
            # стационарный бутстреп: блок случайной длины со средним block
            s = np.random.randint(0, n)
            L = 1 + np.random.geometric(1.0 / block) - 1
            for j in range(L):
                if t >= horizon:
                    break
                g = 1.0 + lev * r[(s + j) % n]
                if g < 0:
                    g = 0.0
                eq *= g
                if eq > pk:
                    pk = eq
                d = 1.0 - eq / pk if pk > 0 else 1.0
                if d > md:
                    md = d
                if eq <= 0.1:
                    ruin[p] = True
                t += 1
                if t % month_len == 0 and k < n_m:
                    mr[k] = eq / mstart - 1.0 if mstart > 0 else -1.0
                    mstart = eq
                    k += 1
        mdd[p] = md
        final[p] = eq
        minm[p] = mr[:k].min() if k > 0 else 0.0
        cnt = 0
        for q in range(k):
            if mr[q] > 0:
                cnt += 1
        pmr[p] = cnt / k if k > 0 else 0.0
        srt = np.sort(mr[:k])
        medm[p] = srt[k // 2] if k > 0 else 0.0
    return mdd, final, minm, pmr, medm, ruin


def monte_carlo(r: np.ndarray, lev: float = 1.0, n_paths: int = 20000, horizon: int = 252, block: float = 5.0,
                seed: int = 12345, month_len: int = 21) -> dict:
    """Блочный бутстреп дневных доходностей (стационарный, средняя длина блока `block` дней)."""
    mdd, final, minm, pmr, medm, ruin = _mc(np.asarray(r, float), n_paths, horizon, block, lev, seed, month_len)
    out = dict(n_paths=n_paths, horizon_days=horizon, block=block, lev=lev, seed=seed,
               final_median=float(np.median(final)), final_p05=float(np.quantile(final, 0.05)),
               final_p95=float(np.quantile(final, 0.95)), p_loss=float((final < 1).mean()),
               mdd_median=float(np.median(mdd)), mdd_p95=float(np.quantile(mdd, 0.95)),
               monthly_median=float(np.median(final) ** (month_len / horizon) - 1),
               min_month_median=float(np.median(minm)), pmr_median=float(np.median(pmr)),
               p_all_months_positive=float((pmr >= 0.999).mean()), p_ruin=float(ruin.mean()))
    for x in (0.1, 0.2, 0.3, 0.5, 0.7):
        out[f"p_dd_gt_{int(x * 100)}"] = float((mdd > x).mean())
    return out
