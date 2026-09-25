"""Карта предсказуемости: предсказывает ли результат на обучении результат следующего месяца?

Для каждой группы (семейство × таймфрейм × класс инструмента) и каждого тестового месяца M (2026-01…06):
  * rank-корреляция между Sharpe обучения (все дни < M) и доходностью месяца M по конфигурациям группы;
  * OOS-доходность верхнего дециля по обучению минус нижнего;
  * доходность «ансамбля без отбора» (среднее всех конфигураций группы, нормированных к равной волатильности).
Выход: research/wf/persistence_<tag>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import data as D  # noqa: E402
from moexlab.deep import metrics as M  # noqa: E402
from moexlab.deep import screen as S  # noqa: E402
from moexlab.deep import wf as W  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "s1"


def main():
    meta, R = W.load_pool(TAG)
    cal = S.calendar()
    mk = M.month_keys(cal)
    meta["cls"] = np.where(meta["code"].isin(D.EQUITIES), "EQ", np.where(meta["code"].str.contains("-"), "PAIR", "FUT"))
    rows = []
    for (fam, tf, cls), g in meta.groupby(["family", "tf", "cls"]):
        idx = g.index.to_numpy()
        X = R[idx].astype(np.float64)
        rec = dict(family=fam, tf=tf, cls=cls, configs=len(idx))
        ics, spreads, ens = [], [], []
        for m in range(202601, 202607):
            tr = mk < m
            te = mk == m
            Xtr = X[:, tr]
            sd = Xtr.std(1)
            sh = np.where(sd > 0, Xtr.mean(1) / np.where(sd > 0, sd, 1), np.nan)
            nxt = X[:, te].sum(1)
            nxt_n = np.where(sd > 0, nxt / np.where(sd > 0, sd, 1), np.nan)    # нормировано к волатильности обучения
            ok = np.isfinite(sh) & ((Xtr != 0).sum(1) >= 20)
            if ok.sum() >= 20:
                ics.append(spearmanr(sh[ok], nxt_n[ok]).statistic)
                q = np.quantile(sh[ok], [0.1, 0.9])
                spreads.append(np.nanmean(nxt_n[ok & (sh >= q[1])]) - np.nanmean(nxt_n[ok & (sh <= q[0])]))
            ens.append(np.nanmean(nxt_n[ok]) if ok.any() else np.nan)
        rec.update(ic_mean=np.nanmean(ics) if ics else np.nan, ic_pos_months=int(np.sum(np.array(ics) > 0)) if ics else 0,
                   top_minus_bottom=np.nanmean(spreads) if spreads else np.nan,
                   ens_oos_mean=np.nanmean(ens), ens_pos_months=int(np.sum(np.array(ens) > 0)))
        for j, m in enumerate(range(202601, 202607)):
            rec[f"ens_{m}"] = ens[j]
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("ic_mean", ascending=False)
    out = D.ROOT / "research/wf" / f"persistence_{TAG}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.head(40).round(3).to_string())
    print(df.sort_values("ens_oos_mean", ascending=False).head(30).round(3).to_string())


if __name__ == "__main__":
    main()
