"""Walk-forward отбор по результатам скрининга.

Главный тест (§70): в начале каждого тестового месяца M мы знаем только дни < M. Процедура выбирает
конфигурации по обучающему окну и формирует портфель; доходность месяца M — честный out-of-sample.
Процедура целиком (критерий, число стратегий, фильтры) фиксируется ДО просмотра OOS-месяцев
и сравнивается с другими процедурами только по OOS; окончательный выбор проверяется на FINAL HOLDOUT.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M

SCREEN = D.CACHE / "screen"


def load_pool(tag="s1", codes=None, tfs=None, filt=None):
    metas, mats = [], []
    for fp in sorted((SCREEN / tag).glob("*.parquet")):
        code, tf = fp.stem.rsplit("_", 1)
        if codes and code not in codes:
            continue
        if tfs and int(tf) not in tfs:
            continue
        m = pd.read_parquet(fp)
        R = np.load(fp.with_suffix(".npy"))
        if filt is not None:
            keep = filt(m).to_numpy()
            m, R = m[keep], R[keep]
        metas.append(m)
        mats.append(R)
    meta = pd.concat(metas, ignore_index=True)
    R = np.vstack(mats)
    meta["uid"] = (meta["code"] + "|" + meta["tf"].astype(str) + "|" + meta["family"] + "|" + meta["params"] + "|" +
                   meta["exit"] + "|" + meta["eod"].astype(str) + "|" + meta["direction"].astype(str))
    meta["group_key"] = (meta["code"] + "|" + meta["tf"].astype(str) + "|" + meta["family"] + "|" + meta["exit"] + "|" +
                         meta["eod"].astype(str) + "|" + meta["direction"].astype(str))
    return meta, R


def month_starts(cal):
    mk = M.month_keys(cal)
    months = np.unique(mk)
    return months, mk


@dataclass
class Procedure:
    name: str
    criterion: str = "sharpe"          # sharpe | score | pmr | floor | sortino | calmar
    k: int = 10                        # число стратегий в портфеле
    min_active_days: int = 20
    min_train_months: int = 3
    max_corr: float = 0.6
    max_per_code: int = 3
    max_per_family: int = 4
    group_smooth: float = 0.0          # вес медианы группы параметров (устойчивые области)
    lookback_months: int | None = None  # None — расширяющееся окно
    require_positive: bool = True
    min_pmr: float = 0.0
    families: tuple | None = None
    groups: tuple | None = None
    tfs: tuple | None = None
    codes: tuple | None = None
    target_vol: float = 0.01          # целевая дневная волатильность портфеля (для сравнения процедур)
    extra: dict = field(default_factory=dict)


def _crit(Rtr, mtr, crit):
    mu = Rtr.mean(1)
    sd = Rtr.std(1)
    sd_safe = np.where(sd > 0, sd, np.inf)
    sharpe = mu / sd_safe * np.sqrt(252)
    if crit == "sharpe":
        return sharpe
    if crit == "sortino":
        neg = np.sqrt((np.minimum(Rtr, 0) ** 2).mean(1))
        return mu / np.where(neg > 0, neg, np.inf) * np.sqrt(252)
    if crit == "score":
        return M.score_stability(mtr)
    msd = mtr.std(1)
    z = mtr / np.where(msd > 0, msd, np.inf)[:, None]
    if crit == "pmr":
        return (mtr > 0).mean(1) + 0.01 * np.clip(sharpe, -10, 10)
    if crit == "floor":
        # максимум минимального месяца при нормировке к одной дневной волатильности
        zz = mtr / np.where(sd > 0, sd, np.inf)[:, None] * 0.01
        return zz.min(1) + 0.001 * np.clip(sharpe, -10, 10)
    if crit == "calmar":
        mdd = M.max_dd(Rtr / np.where(sd > 0, sd, np.inf)[:, None])
        return mu / sd_safe / np.where(mdd > 0, mdd, np.inf) * 252
    raise ValueError(crit)


_CACHE = {}


def _fold_stats(R, cal, train_mask):
    key = (id(R), R.shape, train_mask.tobytes())
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()
    cols = np.where(train_mask)[0]
    mk = M.month_keys(cal[train_mask])
    months = np.unique(mk)
    n = R.shape[0]
    mtr = np.zeros((n, len(months)))
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    neg2 = np.zeros(n)
    active = np.zeros(n, np.int64)
    step = 200000
    for a in range(0, n, step):
        X = R[a:a + step][:, cols].astype(np.float64)
        s1[a:a + step] = X.sum(1)
        s2[a:a + step] = (X * X).sum(1)
        neg2[a:a + step] = (np.minimum(X, 0) ** 2).sum(1)
        active[a:a + step] = (X != 0).sum(1)
        for j, m in enumerate(months):
            mtr[a:a + step, j] = X[:, mk == m].sum(1)
    T = len(cols)
    mu = s1 / T
    sd = np.sqrt(np.maximum(s2 / T - mu * mu, 0))
    st = dict(mu=mu, sd=sd, dsd=np.sqrt(neg2 / T), mtr=mtr, active=active, total=s1, cols=cols, crit={})
    _CACHE[key] = st
    return st


def _crit2(st, crit, R):
    if crit in st["crit"]:
        return st["crit"][crit]
    mu, sd, mtr = st["mu"], st["sd"], st["mtr"]
    sd_safe = np.where(sd > 0, sd, np.inf)
    sharpe = mu / sd_safe * np.sqrt(252)
    if crit == "sharpe":
        v = sharpe
    elif crit == "sortino":
        v = mu / np.where(st["dsd"] > 0, st["dsd"], np.inf) * np.sqrt(252)
    elif crit == "score":
        v = M.score_stability(mtr)
    elif crit == "pmr":
        v = (mtr > 0).mean(1) + 0.01 * np.clip(sharpe, -10, 10)
    elif crit == "floor":
        v = (mtr / sd_safe[:, None] * 0.01).min(1) + 0.001 * np.clip(sharpe, -10, 10)
    elif crit == "calmar":
        # просадка по месячной кривой (быстро), нормированная на дневную волатильность
        cum = np.cumsum(mtr / sd_safe[:, None], 1)
        pk = np.maximum.accumulate(np.concatenate([np.zeros((len(cum), 1)), cum], 1), 1)[:, 1:]
        mdd = (pk - cum).max(1)
        v = (mu / sd_safe) * 252 / np.where(mdd > 0, mdd, np.inf)
    else:
        raise ValueError(crit)
    v = np.where(np.isfinite(v), v, -np.inf)
    st["crit"][crit] = v
    return v


def select(meta, R, cal, train_mask, proc: Procedure, cand_mask=None):
    """Индексы выбранных конфигураций и их веса (равный риск по обучающей волатильности)."""
    st = _fold_stats(R, cal, train_mask)
    mtr = st["mtr"]
    ok = st["active"] >= proc.min_active_days
    if cand_mask is not None:
        ok &= cand_mask
    crit = _crit2(st, proc.criterion, R).copy()
    if proc.group_smooth > 0:
        g = pd.Series(np.where(ok, crit, np.nan)).groupby(meta["group_key"].to_numpy()).transform("median").to_numpy()
        crit = (1 - proc.group_smooth) * crit + proc.group_smooth * np.nan_to_num(g, nan=-np.inf)
    if proc.require_positive:
        ok &= st["total"] > 0
    if proc.min_pmr > 0:
        ok &= (mtr > 0).mean(1) >= proc.min_pmr
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return np.array([], int), np.array([])
    order = idx[np.argsort(-crit[idx])][:max(proc.k * 50, 200)]
    chosen, per_code, per_fam, zs = [], {}, {}, []
    sd = st["sd"]
    cols = st["cols"]
    for i in order:
        c, f = meta.at[i, "code"], meta.at[i, "family"]
        if per_code.get(c, 0) >= proc.max_per_code or per_fam.get(f, 0) >= proc.max_per_family:
            continue
        ri = R[i, cols].astype(np.float64)
        z = ri - ri.mean()
        nz = np.sqrt((z * z).sum())
        if nz <= 0:
            continue
        z = z / nz
        if zs and max(float(z @ q) for q in zs) > proc.max_corr:
            continue
        chosen.append(i)
        zs.append(z)
        per_code[c] = per_code.get(c, 0) + 1
        per_fam[f] = per_fam.get(f, 0) + 1
        if len(chosen) >= proc.k:
            break
    chosen = np.array(chosen, int)
    w = 1.0 / np.where(sd[chosen] > 0, sd[chosen], np.inf)
    w = w / w.sum() if w.sum() > 0 else w
    return chosen, w


def run_wf(meta, R, cal, proc: Procedure, first_test_month=202601, last_test_month=202606, verbose=False):
    months, mk = month_starts(cal)
    cand = np.ones(len(meta), bool)
    for attr, col in (("families", "family"), ("groups", "group"), ("tfs", "tf"), ("codes", "code")):
        v = getattr(proc, attr)
        if v:
            cand &= meta[col].isin(v).to_numpy()
    oos = np.zeros(len(cal))
    log = []
    for m in months:
        if m < first_test_month or m > last_test_month:
            continue
        train = mk < m
        if proc.lookback_months:
            tm = [x for x in months if x < m][-proc.lookback_months:]
            train = np.isin(mk, tm)
        test = mk == m
        idx, w = select(meta, R, cal, train, proc, cand)
        if len(idx) == 0:
            log.append(dict(month=m, n=0))
            continue
        # масштаб к целевой дневной волатильности по обучающему окну (без знания теста)
        port_tr = (R[idx][:, train].astype(float) * w[:, None]).sum(0)
        s = port_tr.std()
        scale = proc.target_vol / s if s > 0 else 0.0
        port_te = (R[idx][:, test].astype(float) * w[:, None]).sum(0) * scale
        oos[test] = port_te
        log.append(dict(month=m, n=len(idx), scale=scale, oos=port_te.sum(), picks=[meta.at[i, "uid"] for i in idx],
                        weights=(w * scale).tolist()))
        if verbose:
            print(m, len(idx), round(port_te.sum(), 4), flush=True)
    return oos, log


def oos_summary(cal, oos, first=202601, last=202606):
    mk = M.month_keys(cal)
    sel = (mk >= first) & (mk <= last)
    r = oos[sel]
    months = np.unique(mk[sel])
    mret = np.array([oos[mk == m].sum() for m in months])
    sd = r.std()
    return dict(months=len(months), total=float(r.sum()), sharpe=float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                pmr=float((mret > 0).mean()), m_min=float(mret.min()), m_mean=float(mret.mean()),
                m_median=float(np.median(mret)), mdd=float(M.max_dd(r[None])[0]),
                monthly=dict(zip(months.tolist(), np.round(mret, 4).tolist())))
