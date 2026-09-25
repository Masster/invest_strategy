"""Stage 3: walk-forward процедур отбора на всём пуле Stage 1 (+ межинструментальный пул).

Тестовые месяцы: 2026-01 … 2026-06 (обучение — все дни до начала месяца, минимум 3 месяца).
Выход: research/wf/<tag>_procedures.csv (OOS-метрики каждой процедуры), research/wf/<tag>_picks.json.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import data as D  # noqa: E402
from moexlab.deep import screen as S  # noqa: E402
from moexlab.deep import wf as W  # noqa: E402

OUT = D.ROOT / "research/wf"
OUT.mkdir(parents=True, exist_ok=True)


def procedures():
    P = []
    for crit in ("sharpe", "score", "pmr", "floor", "sortino", "calmar"):
        for k in (1, 5, 10, 20):
            P.append(W.Procedure(f"{crit}_k{k}", criterion=crit, k=k))
    for crit in ("sharpe", "score"):
        P.append(W.Procedure(f"{crit}_k10_smooth", criterion=crit, k=10, group_smooth=0.5))
        P.append(W.Procedure(f"{crit}_k10_lb3", criterion=crit, k=10, lookback_months=3))
        P.append(W.Procedure(f"{crit}_k10_pmr08", criterion=crit, k=10, min_pmr=0.8))
        P.append(W.Procedure(f"{crit}_k10_daily", criterion=crit, k=10, tfs=(1440,), min_active_days=10))
        P.append(W.Procedure(f"{crit}_k10_h60", criterion=crit, k=10, tfs=(30, 60, 1440)))
        P.append(W.Procedure(f"{crit}_k10_fut", criterion=crit, k=10, codes=tuple(D.FUT_BASES)))
        P.append(W.Procedure(f"{crit}_k10_eq", criterion=crit, k=10, codes=tuple(D.EQUITIES)))
        P.append(W.Procedure(f"{crit}_k10_trend", criterion=crit, k=10, groups=("trend", "momentum")))
        P.append(W.Procedure(f"{crit}_k10_mr", criterion=crit, k=10, groups=("meanrev",)))
        P.append(W.Procedure(f"{crit}_k10_vol", criterion=crit, k=10, groups=("volatility", "volume")))
        P.append(W.Procedure(f"{crit}_k10_intraday", criterion=crit, k=10, groups=("intraday", "seasonal")))
        P.append(W.Procedure(f"{crit}_k30_c05", criterion=crit, k=30, max_corr=0.5, max_per_code=5, max_per_family=8))
    return P


def main(tags=("s1",), out_tag="wf1", extra_filter=None):
    t0 = time.time()
    metas, mats = [], []
    for tag in tags:
        m, R = W.load_pool(tag, filt=extra_filter)
        metas.append(m)
        mats.append(R)
    meta = pd.concat(metas, ignore_index=True)
    R = np.vstack(mats)
    cal = S.calendar()
    print("pool", R.shape, round(time.time() - t0, 1), flush=True)
    rows, picks = [], {}
    for proc in procedures():
        oos, log = W.run_wf(meta, R, cal, proc)
        s = W.oos_summary(cal, oos)
        row = dict(procedure=proc.name, **{k: v for k, v in s.items() if k != "monthly"})
        row.update({f"m{k}": v for k, v in s["monthly"].items()})
        rows.append(row)
        picks[proc.name] = dict(proc=asdict(proc), log=log)
        print(proc.name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
        np.save(OUT / f"{out_tag}_{proc.name}_oos.npy", oos)
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(OUT / f"{out_tag}_procedures.csv", index=False)
    (OUT / f"{out_tag}_picks.json").write_text(json.dumps(picks, default=str, indent=1))
    print(df.to_string())


if __name__ == "__main__":
    tags = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("s1",)
    main(tags, sys.argv[2] if len(sys.argv) > 2 else "wf1")
