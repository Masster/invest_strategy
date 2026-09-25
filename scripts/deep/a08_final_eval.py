"""Stage 5: оценка замороженных кандидатов (research/FROZEN_CANDIDATES.md) на всём периоде, включая FINAL HOLDOUT.

1. Проверка отсутствия утечки: доходности кандидатов на днях разработки, посчитанные на полных данных,
   совпадают с посчитанными на данных, обрезанных по DEV_END.
2. Кандидаты-процедуры (walk-forward) продолжают ежемесячный отбор в holdout-месяцы (обучение — всё до месяца).
3. Все кандидаты приводятся к целевой волатильности 10% годовых по прошлым 60 дням (каузально).
Выход: research/final/candidates_daily.parquet, research/final/candidates_summary.csv, *_monthly.csv, wf logs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import candidates as C  # noqa: E402
from moexlab.deep import data as D  # noqa: E402
from moexlab.deep import metrics as M  # noqa: E402
from moexlab.deep import panel as PN  # noqa: E402
from moexlab.deep import screen as S  # noqa: E402
from moexlab.deep import wf as W  # noqa: E402

OUT = D.ROOT / "research/final"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 0.10


def seg_stats(r, days, lo, hi):
    mk = M.month_keys(days)
    sel = (mk >= lo) & (mk <= hi)
    x = r[sel]
    months = np.unique(mk[sel])
    mret = np.array([r[mk == m].sum() for m in months])
    sd = x.std()
    return dict(months=len(months), total=float(x.sum()), sharpe=float(x.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                pmr=float((mret > 0).mean()) if len(mret) else np.nan, m_min=float(mret.min()) if len(mret) else np.nan,
                m_mean=float(mret.mean()) if len(mret) else np.nan, mdd=float(M.max_dd(x[None])[0]))


def main():
    Pf = PN.load_panel(end=D.LAST_DAY)
    Pd = PN.load_panel(end=D.DEV_END)
    days = np.array(Pf["days"])
    ndev = len(Pd["days"])
    raw, info = {}, {}
    fixed_f = C.fixed_candidates(Pf)
    fixed_d = C.fixed_candidates(Pd)
    leak = {}
    for k, (Wt, desc) in fixed_f.items():
        r, turn, gross, _ = C.returns(Pf, Wt)
        rd, *_ = C.returns(Pd, fixed_d[k][0])
        leak[k] = float(np.max(np.abs(r[:ndev] - rd)))
        raw[k] = r
        info[k] = dict(desc=desc, turnover=float(turn.mean()), gross=float(gross.mean()))
    print("leak check (max abs diff on dev days):", leak, flush=True)
    # walk-forward кандидаты
    meta, R = W.load_pool("p1full")
    cal = S.calendar(D.LAST_DAY)
    assert np.array_equal(cal, days)
    logs = {}
    for k, proc in C.FROZEN.items():
        if isinstance(proc, W.Procedure):
            oos, log = C.wf_candidate(meta, R, cal, proc, 202609)
            raw[k] = oos
            logs[k] = log
            info[k] = dict(desc=f"walk-forward процедура {proc.name} по библиотеке 423 портфельных стратегий "
                                f"(ежемесячный отбор по прошлому, 1-й тестовый месяц 2026-01)")
    (OUT / "wf_candidate_logs.json").write_text(json.dumps(logs, default=str, indent=1))
    # целевая волатильность
    vt = {}
    scales = {}
    for k, r in raw.items():
        vt[k], scales[k] = C.vol_target(r, TARGET)
    # мета-стратегия: равный риск трёх структурно разных источников (тренд, XS-моментум, carry),
    # веса = обратная прошлая волатильность (60 дней, с лагом), затем общий таргет 10%
    comps = ["C1_TREND_FUT_ENS", "C3_XSMOM_EQ_ENS", "C4_CARRY_FX"]
    X = np.stack([vt[c] for c in comps])
    sd = pd.DataFrame(X.T).rolling(60, min_periods=20).std().shift(1).to_numpy().T
    iw = np.where(sd > 0, 1 / sd, 0.0)
    wsum = iw.sum(0)
    wts = np.where(wsum > 0, iw / np.where(wsum > 0, wsum, 1), 0.0)
    meta_r = (wts * X).sum(0)
    vt["C9_META_RP"], scales["C9_META_RP"] = C.vol_target(meta_r, TARGET)
    info["C9_META_RP"] = dict(desc="мета-стратегия: риск-паритет C1+C3+C4 (веса по прошлой волатильности, лаг 1 день)")
    # сводка
    rows = []
    for k in vt:
        r = vt[k]
        row = dict(candidate=k, **{f"dev_{a}": b for a, b in seg_stats(r, days, 202510, 202606).items()},
                   **{f"wf_{a}": b for a, b in seg_stats(r, days, 202601, 202606).items()},
                   **{f"hold_{a}": b for a, b in seg_stats(r, days, 202607, 202609).items()},
                   **{f"year_{a}": b for a, b in seg_stats(r, days, 202510, 202609).items()},
                   avg_scale=float(np.mean(scales[k][scales[k] > 0])) if (scales[k] > 0).any() else 0.0,
                   desc=info.get(k, {}).get("desc", ""))
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "candidates_summary.csv", index=False)
    daily = pd.DataFrame({k: v for k, v in vt.items()}, index=days)
    daily_raw = pd.DataFrame({k: v for k, v in raw.items()}, index=days)
    daily.to_parquet(OUT / "candidates_daily_vt10.parquet")
    daily_raw.to_parquet(OUT / "candidates_daily_raw.parquet")
    mk = M.month_keys(days)
    mon = daily.groupby(mk).sum()
    mon.to_csv(OUT / "candidates_monthly_vt10.csv")
    (OUT / "leak_check.json").write_text(json.dumps(leak, indent=1))
    pd.set_option("display.width", 250)
    print(summ.drop(columns=["desc"]).round(3).T.to_string())
    print(mon.round(4).to_string())


if __name__ == "__main__":
    main()
