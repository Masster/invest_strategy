"""ML-гипотеза 2: LightGBM (классификатор направления) на часовых барах, общий по 21 инструменту.
Цель: знак доходности следующих H=4 свечей (open i+1 -> open i+1+H). Вход только при уверенности
|p−0.5| > thr (отсекает сделки, которые не окупают издержки). Переобучение ежемесячно, зазор 5 свечей.
Исполнение — Numba-движок (консервативно), выход по времени H свечей или в конце дня.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import data as D  # noqa: E402
from moexlab.deep import engine as E  # noqa: E402
from moexlab.deep import features as FT  # noqa: E402
from moexlab.deep import metrics as M  # noqa: E402
from moexlab.deep import screen as S  # noqa: E402

OUT = D.ROOT / "research/ml"
SEED = 20260925
TF, H = 60, 4
COST_MODE = "TRADER"


def feats(A):
    c = A["ac"]
    lr = FT.logret(c)
    lr[A["new_series"]] = 0.0
    vol = FT.rstd(lr, 50)
    X = {}
    for n in (1, 2, 4, 8, 16, 32):
        X[f"r{n}"] = (c / FT.shift(c, n) - 1) / (vol * np.sqrt(n))
    X["vol"] = vol
    X["volr"] = FT.rstd(lr, 8) / vol
    X["mod"] = A["mod"].astype(float)
    X["bsd"] = FT.bars_since_day_start(A["first_of_day"]).astype(float)
    X["clv"] = np.where(A["ah"] > A["al"], (c - A["al"]) / np.maximum(A["ah"] - A["al"], 1e-12), 0.5)
    lv = np.log(A["v"] + 1)
    X["vz"] = (lv - FT.sma(lv, 50)) / FT.rstd(lv, 50)
    rs = FT.rsi(c, 14)
    X["rsi"] = rs
    y = np.full(len(c), np.nan)
    o = A["ao"]
    y[:-H - 1] = np.log(o[1 + H:] / o[1:-H]) if len(o) > H + 1 else np.nan
    # цель не должна пересекать смену дня/серии
    day = A["day"]
    same = np.zeros(len(c), bool)
    same[:-H - 1] = day[1 + H:] == day[:-H - 1]
    y[~same] = np.nan
    return pd.DataFrame(X), y


def main(end=D.DEV_END, test_months=range(202601, 202607), label="dev"):
    cal = S.calendar(end)
    data = {}
    for code in D.ALL_CODES:
        A = S.load_arrays(code, TF, end=end, cal=cal)
        X, y = feats(A)
        X["is_eq"] = float(code in D.EQUITIES)
        data[code] = (A, X, y)
    mk_bar = {c: M.month_keys(data[c][0]["day"]) for c in data}
    out_rows = {}
    for thr in (0.03, 0.06, 0.1, 0.15, 0.2):
        out_rows[thr] = np.zeros(len(cal))
    ic_log = []
    for m in test_months:
        Xs, ys = [], []
        for c, (A, X, y) in data.items():
            tr = np.where(mk_bar[c] < m)[0]
            tr = tr[:-5] if len(tr) > 5 else tr[:0]
            Xs.append(X.iloc[tr])
            ys.append(y[tr])
        Xtr = pd.concat(Xs)
        ytr = np.concatenate(ys)
        ok = np.isfinite(ytr) & np.isfinite(Xtr.to_numpy()).all(1) & (ytr != 0)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=16, min_child_samples=200,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                 random_state=SEED, verbose=-1)
        clf.fit(Xtr[ok], (ytr[ok] > 0).astype(int))
        for c, (A, X, y) in data.items():
            te = mk_bar[c] == m
            if not te.any():
                continue
            p = np.full(len(y), 0.5)
            Xm = X[te]
            okm = np.isfinite(Xm.to_numpy()).all(1)
            pp = np.full(te.sum(), 0.5)
            pp[okm] = clf.predict_proba(Xm[okm])[:, 1]
            p[te] = pp
            yy = y[te]
            k = np.isfinite(yy) & okm
            if k.sum() > 20:
                ic_log.append(dict(month=m, code=c, ic=float(pd.Series(pp[k]).corr(pd.Series(yy[k]), method="spearman"))))
            costs = S.cost_model(c, A, tariff="PREMIUM" if COST_MODE == "PREMIUM" else "TRADER")
            if COST_MODE == "ZERO":
                costs = {k: 0.0 for k in costs}
            for thr in out_rows:
                lg = te & (p > 0.5 + thr)
                sh = te & (p < 0.5 - thr)
                r = E.run(A, dict(long=lg, short=sh), time_bars=H, eod_exit=True, costs=costs)
                out_rows[thr] += r[0] / len(D.ALL_CODES)
    mk = M.month_keys(cal)
    res = {}
    for thr, r in out_rows.items():
        sel = np.isin(mk, list(test_months))
        mret = {int(mm): float(r[mk == mm].sum()) for mm in test_months}
        sd = r[sel].std()
        res[str(thr)] = dict(sharpe=float(r[sel].mean() / sd * np.sqrt(252)) if sd > 0 else 0.0, total=float(r[sel].sum()),
                             monthly=mret)
    icdf = pd.DataFrame(ic_log)
    out = dict(label=label, tf=TF, horizon=H, results=res, ic_mean=float(icdf["ic"].mean()),
               ic_by_month=icdf.groupby("month")["ic"].mean().round(4).to_dict(), seed=SEED)
    label = f"{label}_H{H}_{COST_MODE}"
    out["label"] = label
    (OUT / f"ml_hourly_{label}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        H = int(sys.argv[1])
        COST_MODE = sys.argv[2]
    main()
