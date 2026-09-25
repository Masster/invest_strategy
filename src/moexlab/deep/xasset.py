"""Межинструментальные гипотезы: lead-lag (сигнал инструмента A для торговли B) и парный спред.

Выравнивание: для каждой свечи B берётся последняя свеча A с ключом (день, ведро) <= ключа свечи B.
Одинаковый ключ = свечи за один и тот же интервал, закрываются одновременно => A известна на закрытии B.
Никаких будущих данных: as-of соединение «назад».
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import data as D
from . import engine as E
from . import features as F
from . import metrics as M
from .screen import SCREEN, calendar, cost_model, load_arrays


def _key(A, tf):
    if tf >= 1440:
        return A["day"].astype(np.int64) * 10000
    return A["day"].astype(np.int64) * 10000 + (A["mod"] // tf).astype(np.int64)


def align_to(B: dict, A: dict, tf: int) -> np.ndarray:
    """Индекс свечи A (или -1) для каждой свечи B (as-of назад)."""
    kb, ka = _key(B, tf), _key(A, tf)
    j = np.searchsorted(ka, kb, side="right") - 1
    return j


def leadlag_signals(A, B, tf, n, z, mode):
    """mode=follow: B следует за движением A (если B отстал); mode=fade: против движения B относительно A."""
    j = align_to(B, A, tf)
    ok = j >= 0
    ra = F.roc(A["ac"], n)
    va = F.rstd(F.logret(A["ac"]), 100) * np.sqrt(n)
    rb = F.roc(B["ac"], n)
    vb = F.rstd(F.logret(B["ac"]), 100) * np.sqrt(n)
    ra_b = np.where(ok, ra[np.clip(j, 0, None)], np.nan)
    va_b = np.where(ok, va[np.clip(j, 0, None)], np.nan)
    # A должна быть «свежей»: день совпадает
    same_day = np.where(ok, A["day"][np.clip(j, 0, None)] == B["day"], False)
    za = ra_b / va_b
    zb = rb / vb
    if mode == "follow":
        lg = same_day & (za > z) & (zb < za - 0.5 * z)
        sh = same_day & (za < -z) & (zb > za + 0.5 * z)
    else:  # divergence fade: B ушёл сильно дальше A — ставим на сближение
        lg = same_day & (zb - za < -z)
        sh = same_day & (zb - za > z)
    return np.asarray(lg, np.bool_), np.asarray(sh, np.bool_)


LEADERS = ["MX", "RI", "BR", "Si", "GD", "SR", "SBER"]
LL_GRID = dict(n=[1, 3, 6, 12], z=[1.0, 2.0], mode=["follow", "fade"], hold=[3, 12])


def run_leadlag_job(b_code: str, tf: int, tag="x1", end=D.DEV_END):
    out_dir = SCREEN / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"LL_{b_code}_{tf}.parquet"
    if fp.exists():
        return {"skipped": True}
    t0 = time.time()
    cal = calendar(end)
    B = load_arrays(b_code, tf, end=end, cal=cal)
    costs = cost_model(b_code, B)
    is_eq = b_code in D.EQUITIES
    mk = M.month_keys(cal)
    months = np.unique(mk)
    rows, rets = [], []
    for a_code in LEADERS:
        if a_code == b_code:
            continue
        A = load_arrays(a_code, tf, end=end, cal=cal)
        for n in LL_GRID["n"]:
            for z in LL_GRID["z"]:
                for mode in LL_GRID["mode"]:
                    lg, sh = leadlag_signals(A, B, tf, n, z, mode)
                    sig = dict(long=lg, short=sh)
                    for hold in LL_GRID["hold"]:
                        for dr in (1, -1, 0):
                            r = E.run(B, sig, time_bars=hold, eod_exit=True if (is_eq or tf < 1440) else False,
                                      state_mode=True, direction=dr, costs=costs)
                            rets.append(r[0].astype(np.float32))
                            st = M.trade_stats(r[5])
                            rows.append(dict(code=b_code, tf=tf, family="X_LEADLAG", group="cross",
                                             params=json.dumps(dict(a=a_code, n=n, z=z, mode=mode, hold=hold)),
                                             exit="T", eod=True, direction=dr, trades=len(r[5]), win=st["win"],
                                             pf=st["pf"], avg_trade=st["avg_trade"],
                                             hold_bars=float(np.mean(r[3] - r[2] + 1)) if len(r[5]) else 0.0,
                                             expo_days=float((r[1] > 0).mean()), long_share=np.nan))
    R = np.vstack(rets)
    mret = M.monthly(R.astype(float), mk, months)
    s = M.summary_from_daily(R.astype(float), mret)
    df = pd.DataFrame(rows)
    for k, v in s.items():
        df[k] = v
    for jx, m in enumerate(months):
        df[f"m{m}"] = mret[:, jx]
    df["score"] = M.score_stability(mret)
    df["row"] = np.arange(len(df))
    np.save(out_dir / f"LL_{b_code}_{tf}.npy", R)
    df.to_parquet(fp, index=False)
    return {"code": b_code, "tf": tf, "n": len(df), "sec": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ пары (две ноги)
PAIRS = [("Si", "CR"), ("MX", "RI"), ("SBER", "SR"), ("GAZP", "GZ"), ("ROSN", "RN"), ("BR", "RN"), ("LKOH", "ROSN"),
         ("ROSN", "TATN"), ("GD", "PLZL"), ("SBER", "VTBR"), ("MX", "SBER"), ("MX", "LKOH"), ("MX", "GAZP"),
         ("NVTK", "GAZP"), ("GMKN", "ALRS"), ("LKOH", "TATN"), ("SR", "MX"), ("BR", "ROSN")]
PAIR_GRID = dict(w=[100, 300, 1000], zin=[2.0, 2.5, 3.0], zout=[0.0, 0.5], hold=[0, 50])


def pair_signals(A, B, tf, w, zin, zout):
    """Спред доходностей: s = cumsum(rB − β·rA), β — скользящая регрессия по окну w; z-score спреда по окну w."""
    j = align_to(B, A, tf)
    ok = j >= 0
    jj = np.clip(j, 0, None)
    a_c = np.where(ok, A["ac"][jj], np.nan)
    rb = F.logret(B["ac"])
    ra = np.zeros(len(a_c))
    ra[1:] = np.log(a_c[1:] / a_c[:-1])
    ra = np.nan_to_num(ra)
    rb[B["new_series"]] = 0.0
    cov = F.sma(ra * rb, w) - F.sma(ra, w) * F.sma(rb, w)
    var = F.sma(ra * ra, w) - F.sma(ra, w) ** 2
    beta = np.where(var > 0, cov / np.where(var > 0, var, 1), 0.0)
    beta = np.clip(np.nan_to_num(F.shift(beta, 1)), -3, 3)
    s = np.cumsum(rb - beta * ra)
    m = F.sma(s, w)
    sd = F.rstd(s, w)
    zz = (s - m) / np.where(sd > 0, sd, np.nan)
    long_b = zz < -zin
    short_b = zz > zin
    exit_l = zz > -zout
    exit_s = zz < zout
    return (np.asarray(long_b, np.bool_), np.asarray(short_b, np.bool_), np.asarray(exit_l, np.bool_),
            np.asarray(exit_s, np.bool_), beta, j)


def run_pair_job(a_code, b_code, tf, tag="x1", end=D.DEV_END):
    """Две ноги: B торгуется по сигналу спреда, A — противоположно с весом |β| (β на момент сигнала ≈ среднее)."""
    out_dir = SCREEN / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"PAIR_{a_code}-{b_code}_{tf}.parquet"
    if fp.exists():
        return {"skipped": True}
    t0 = time.time()
    cal = calendar(end)
    A = load_arrays(a_code, tf, end=end, cal=cal)
    B = load_arrays(b_code, tf, end=end, cal=cal)
    ca, cb = cost_model(a_code, A), cost_model(b_code, B)
    eod = (a_code in D.EQUITIES) or (b_code in D.EQUITIES) or tf < 1440
    mk = M.month_keys(cal)
    months = np.unique(mk)
    rows, rets = [], []
    for w in PAIR_GRID["w"]:
        for zin in PAIR_GRID["zin"]:
            for zout in PAIR_GRID["zout"]:
                lb, sb, xl, xs, beta, j = pair_signals(A, B, tf, w, zin, zout)
                bmed = float(np.nanmedian(np.abs(beta[w:]))) if len(beta) > w else 1.0
                # сигналы для ноги A на её собственной сетке: состояние B переносим на свечи A (as-of назад)
                ja = align_to(A, B, tf)
                oka = ja >= 0
                jja = np.clip(ja, 0, None)
                sameday = oka & (B["day"][jja] == A["day"])
                la = sameday & sb[jja]
                sa = sameday & lb[jja]
                xla = ~sameday | xs[jja]
                xsa = ~sameday | xl[jja]
                for hold in PAIR_GRID["hold"]:
                    rB = E.run(B, dict(long=lb, short=sb, exit_long=xl, exit_short=xs), time_bars=hold,
                               eod_exit=eod, state_mode=True, direction=0, costs=cb)
                    rA = E.run(A, dict(long=la, short=sa, exit_long=xla, exit_short=xsa), time_bars=hold,
                               eod_exit=eod, state_mode=True, direction=0, costs=ca)
                    day_ret = (rB[0] + bmed * rA[0]) / (1 + bmed)
                    rets.append(day_ret.astype(np.float32))
                    tr = np.concatenate([rB[5], rA[5]])
                    st = M.trade_stats(tr)
                    rows.append(dict(code=f"{a_code}-{b_code}", tf=tf, family="X_PAIR", group="cross",
                                     params=json.dumps(dict(w=w, zin=zin, zout=zout, hold=hold, beta=round(bmed, 3))),
                                     exit="S", eod=eod, direction=0, trades=len(rB[5]), win=st["win"], pf=st["pf"],
                                     avg_trade=st["avg_trade"], hold_bars=float(np.mean(rB[3] - rB[2] + 1)) if len(rB[5]) else 0.0,
                                     expo_days=float((rB[1] > 0).mean()), long_share=np.nan))
    R = np.vstack(rets)
    mret = M.monthly(R.astype(float), mk, months)
    s = M.summary_from_daily(R.astype(float), mret)
    df = pd.DataFrame(rows)
    for k, v in s.items():
        df[k] = v
    for jx, m in enumerate(months):
        df[f"m{m}"] = mret[:, jx]
    df["score"] = M.score_stability(mret)
    df["row"] = np.arange(len(df))
    np.save(out_dir / f"PAIR_{a_code}-{b_code}_{tf}.npy", R)
    df.to_parquet(fp, index=False)
    return {"pair": f"{a_code}-{b_code}", "tf": tf, "n": len(df), "sec": round(time.time() - t0, 1)}
