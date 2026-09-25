"""ML-гипотеза: LightGBM предсказывает доходность следующего дня (open t+1 -> open t+2, нормированную
на волатильность) по признакам, известным на закрытии дня t. Переобучение ежемесячно, обучение — только
на днях, ЦЕЛЬ которых полностью наблюдаема до начала тестового месяца (зазор 2 дня против утечки).

Портфель: w = clip(pred / std(pred_train), −2, 2) × target_vol / vol20 (направленный) и его
кросс-секционно демеанированная версия (рыночно-нейтральный). Издержки и финансирование — panel.backtest.
Выход: research/ml/ml_panel_oos.csv (помесячно), data/cache/screen/ml1/*.npy (для walk-forward/мета).
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
from moexlab.deep import metrics as M  # noqa: E402
from moexlab.deep import panel as PN  # noqa: E402

OUT = D.ROOT / "research/ml"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260925


def features(P):
    c, o, h, l, v = P["c"], P["o"], P["h"], P["l"], P["v"]
    r1 = P["r_cc"]
    vol20 = r1.rolling(20, min_periods=10).std()
    F = {}
    for n in (1, 3, 5, 10, 20, 60):
        F[f"mom{n}"] = (c / c.shift(n) - 1) / (vol20 * np.sqrt(n))
    F["vol20"] = vol20
    F["volratio"] = r1.rolling(5).std() / r1.rolling(60, min_periods=20).std()
    F["gap"] = P["r_gap"] / vol20
    F["intra"] = P["r_intra"] / vol20
    F["hi20"] = (c / h.rolling(20).max() - 1) / vol20
    F["lo20"] = (c / l.rolling(20).min() - 1) / vol20
    F["range"] = (h - l) / c / vol20
    F["clv"] = ((c - l) / (h - l)).where(h > l, 0.5)
    F["vz"] = (np.log(v + 1) - np.log(v + 1).rolling(20).mean()) / np.log(v + 1).rolling(20).std()
    mkt = r1.mean(1)
    for n in (1, 5, 20):
        F[f"mkt{n}"] = pd.DataFrame({k: mkt.rolling(n).sum() for k in P["codes"]})
    dow = pd.to_datetime(pd.Series(c.index.astype(str)), format="%Y%m%d").dt.weekday.to_numpy()
    F["dow"] = pd.DataFrame({k: dow for k in P["codes"]}, index=c.index)
    F["is_eq"] = pd.DataFrame({k: float(k in D.EQUITIES) for k in P["codes"]}, index=c.index)
    # цель: open(t+1)->open(t+2) = r_intra(t+1) + r_gap(t+2), нормированная на vol20(t)
    tgt = (P["r_intra"].shift(-1) + P["r_gap"].shift(-2)) / vol20
    return F, tgt, vol20


def stack(F, tgt):
    cols = list(F)
    X = np.stack([F[k].to_numpy(float) for k in cols], -1)     # [days, codes, feats]
    y = tgt.to_numpy(float)
    return X, y, cols


def run(end=D.DEV_END, test_months=range(202601, 202607), label="dev"):
    P = PN.load_panel(end=end)
    F, tgt, vol20 = features(P)
    X, y, cols = stack(F, tgt)
    days = np.array(P["days"])
    mk = M.month_keys(days)
    preds = np.full(y.shape, np.nan)
    info = []
    for m in test_months:
        te = np.where(mk == m)[0]
        if len(te) == 0:
            continue
        first_te = te[0]
        tr = np.arange(0, max(first_te - 2, 0))          # цель дня t использует t+2 -> зазор 2 дня
        Xtr = X[tr].reshape(-1, X.shape[-1])
        ytr = y[tr].reshape(-1)
        ok = np.isfinite(ytr) & np.isfinite(Xtr).all(1)
        ytr_c = np.clip(ytr[ok], -4, 4)
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=60,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                  random_state=SEED, verbose=-1)
        model.fit(Xtr[ok], ytr_c)
        Xte = X[te].reshape(-1, X.shape[-1])
        p = np.full(len(Xte), np.nan)
        okt = np.isfinite(Xte).all(1)
        p[okt] = model.predict(Xte[okt])
        preds[te] = p.reshape(len(te), -1)
        ptr = model.predict(Xtr[ok])
        info.append(dict(month=m, n_train=int(ok.sum()), pred_std_train=float(ptr.std()),
                         ic_train=float(np.corrcoef(ptr, ytr_c)[0, 1])))
        info[-1]["scale"] = info[-1]["pred_std_train"]
    pred = pd.DataFrame(preds, index=P["c"].index, columns=P["codes"])
    scale = pd.Series({d["month"]: d["scale"] for d in info})
    sc = pd.Series(mk, index=pred.index).map(scale)
    z = pred.div(sc, axis=0).clip(-2, 2)
    target = 0.01
    base = (target / vol20).clip(upper=3.0)
    n_act = z.notna().sum(1).replace(0, 1)
    W_dir = (z * base).div(np.sqrt(n_act), axis=0).fillna(0.0)
    zn = z.sub(z.mean(1), axis=0)
    W_neu = (zn * base).div(np.sqrt(n_act), axis=0).fillna(0.0)
    zf = z.copy()
    zf.loc[:, [c for c in P["codes"] if c in D.EQUITIES]] = np.nan
    W_fut = (zf * base).div(np.sqrt(zf.notna().sum(1).replace(0, 1)), axis=0).fillna(0.0)
    res = {}
    for name, W in (("ML_DIR", W_dir), ("ML_NEUTRAL", W_neu), ("ML_FUT", W_fut)):
        r, turn, gross, _ = PN.backtest(P, W)
        r = r.to_numpy()
        sel = np.isin(mk, list(test_months))
        mret = {int(mm): float(r[mk == mm].sum()) for mm in test_months if (mk == mm).any()}
        sd = r[sel].std()
        res[name] = dict(sharpe=float(r[sel].mean() / sd * np.sqrt(252)) if sd > 0 else 0.0, total=float(r[sel].sum()),
                         pmr=float(np.mean([v > 0 for v in mret.values()])), monthly=mret,
                         turnover=float(turn[sel].mean()))
        np.save(OUT / f"{name}_{label}_daily.npy", r)
    # информационный коэффициент вне выборки
    yy = y.copy()
    ic = []
    for m in test_months:
        te = np.where(mk == m)[0]
        a, b = preds[te].ravel(), yy[te].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 30:
            ic.append(float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")))
    out = dict(label=label, info=info, oos_ic_by_month=ic, results=res, features=cols, seed=SEED)
    (OUT / f"ml_panel_{label}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    run()
