"""Проверка статистического конвейера на синтетике (не рыночные данные!).

NULL:    8 инструментов — случайное блуждание с GARCH-волатильностью, преимущества нет.
PLANTED: 8 инструментов — добавлен медленный скрытый дрейф (трендовость), должен обнаруживаться трендовыми семьями.

Ожидание: на NULL лучшая конфигурация in-sample выглядит «прибыльной», но DSR/SPA/PBO/FDR её отвергают,
walk-forward OOS ≈ 0 или < 0. На PLANTED трендовые семьи проходят проверки.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moexlab.instruments.spec import InstrumentSpec  # noqa: E402
from moexlab.market_data.bars import annotate_daily  # noqa: E402
from moexlab.market_data.synthetic import generate_daily_bars  # noqa: E402
from moexlab.research.grid import GRID_AXES, GRID_VERSION, daily_grid  # noqa: E402
from moexlab.research.runner import (Ledger, RunSettings, neighbors, returns_matrix, run_grid,  # noqa: E402
                                     summarize, walk_forward)
from moexlab.statistics.inference import (benjamini_hochberg, bootstrap_mean_pvalue,  # noqa: E402
                                          deflated_sharpe_ratio, hansen_spa, pbo_cscv, whites_reality_check)


def universe(kind: str, seed0: int):
    streams, specs = {}, {}
    for k in range(8):
        code = f"{kind[:1]}{k}"
        df = generate_daily_bars(1500, seed=seed0 + k, trend_persist=(0.05 if kind == "PLANTED" else 0.0),
                                 code=code)
        streams[code] = annotate_daily(df)
        specs[code] = InstrumentSpec(code, "future", tick=0.01, tick_value=0.01, spread_ticks=1, slippage_ticks=1,
                                     stop_extra_ticks=1, margin_fraction=0.1)
    return streams, specs


def analyse(kind: str, seed0: int, ledger: Ledger) -> dict:
    streams, specs = universe(kind, seed0)
    configs = daily_grid()
    rs = RunSettings(commission_fraction=0.00025)
    g = run_grid(configs, streams, specs, rs)
    res = g["results"]
    M = returns_matrix(res)
    fam = {cid: cfg.family for cid, (cfg, _) in res.items()}
    summ = pd.DataFrame([summarize(cfg, out) for cid, (cfg, out) in res.items()])
    sr = M.mean() / (M.std(ddof=1) + 1e-12)
    best = sr.idxmax()
    trial_sr = sr.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    dsr = deflated_sharpe_ratio(M[best].to_numpy(), len(M.columns), trial_sr)
    pv = np.array([bootstrap_mean_pvalue(M[c].to_numpy(), 500, 5.0, 1) for c in M.columns])
    rej = benjamini_hochberg(pv, 0.10)
    spa = hansen_spa(M.to_numpy(), 1000, 10.0, 2)
    rc = whites_reality_check(M.to_numpy(), 1000, 10.0, 3)
    pbo = pbo_cscv(M.to_numpy(), 12)
    nb = neighbors([cfg for cfg, _ in res.values()], GRID_AXES)
    last = str(M.index[-1].date())
    first_year = M.index[0].year
    wf, oos = walk_forward(M, fam, nb, list(range(first_year + 2, M.index[-1].year + 1)), str(M.index[0].date()), last)
    oos_sr = (oos.mean() / (oos.std(ddof=1) + 1e-12) * np.sqrt(252)).sort_values(ascending=False) if len(oos) else pd.Series()
    out = dict(
        kind=kind, n_configs=len(M.columns), errors=len(g["errors"]),
        best_in_sample=best, best_in_sample_sharpe_ann=float(sr[best] * np.sqrt(252)),
        dsr=dsr, fdr_discoveries=int(rej.sum()), fdr_discovered=[c for c, r in zip(M.columns, rej) if r][:10],
        spa_pvalue=spa["pvalue"], reality_check_pvalue=rc["pvalue"], pbo=pbo["pbo"],
        wf_oos_sharpe_by_family={k: round(float(v), 3) for k, v in oos_sr.items()},
    )
    ledger.add(hypothesis=f"Pipeline validation on synthetic {kind} universe", strategy="ALL (daily grid)",
               parameters=json.dumps({"grid": GRID_VERSION, "n_configs": len(M.columns)}),
               data=f"SYNTHETIC_{kind} 8x1500 daily bars seed0={seed0}", period="synthetic",
               result=json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'n_trials'})
                                  for k, v in out.items() if k not in ("fdr_discovered",)}, default=str)[:1500],
               conclusion="see research/reports/pipeline_validation.md", next_step="run on real data",
               seed=str(seed0), execution_model="ConservativeL1", cost_model="NORMAL comm=0.025%")
    summ.to_csv(ROOT / f"research/reports/pipeline_validation_{kind}_configs.csv", index=False)
    return out


def main():
    ledger = Ledger(ROOT / "research/experiments.csv")
    (ROOT / "research/reports").mkdir(parents=True, exist_ok=True)
    rows = [analyse("NULL", 100, ledger), analyse("PLANTED", 200, ledger)]
    with open(ROOT / "research/reports/pipeline_validation.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    for r in rows:
        print(json.dumps(r, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
