"""Собирает markdown-таблицы из reports/*.csv для вставки в FINAL_RESEARCH_REPORT.md (reports/_tables.md).
Числа в отчёте берутся только отсюда — никаких ручных цифр."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports"


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    parts = []
    if (R / "candidate_selection.csv").exists():
        cs = pd.read_csv(R / "candidate_selection.csv")
        cs["chosen"] = cs["chosen"].str.replace("|", "¦", regex=False)
        parts += ["## Отбор кандидатов (walk-forward, правила RESEARCH_PLAN §5)",
                  md(cs[["family", "wf_oos_sharpe", "wf_positive_year_share", "plateau_neighbors_sharpe",
                         "dev_expectancy_R", "dev_trades", "passed"]])]
    if (R / "walk_forward.csv").exists():
        wf = pd.read_csv(R / "walk_forward.csv")
        piv = wf.pivot(index="family", columns="test_year", values="test_sharpe")
        parts += ["## Walk-forward: Sharpe тестового года (параметры выбраны только по прошлым годам)",
                  md(piv.reset_index())]
    if (R / "multiple_testing.json").exists():
        mt = json.load(open(R / "multiple_testing.json"))
        rows = [dict(test="Hansen SPA (вся сетка)", value=mt["spa"]["pvalue"]),
                dict(test="White Reality Check (вся сетка)", value=mt["rc"]["pvalue"]),
                dict(test="PBO (CSCV, вся сетка)", value=mt["pbo"]["pbo"]),
                dict(test="DSR лучшей конфигурации", value=mt["dsr"]["dsr"]),
                dict(test="Эффективное число испытаний", value=mt["n_eff"])]
        parts += ["## Множественное тестирование", md(pd.DataFrame(rows))]
        ft = pd.DataFrame(mt["family_tests"]).T.reset_index().rename(columns={"index": "family"})
        parts += ["### По семействам (SPA p-value, PBO)", md(ft)]
    if (R / "strategy_comparison.csv").exists():
        sc = pd.read_csv(R / "strategy_comparison.csv")
        agg = sc.groupby("family").agg(configs=("strategy_id", "count"), best_sharpe=("sharpe_in_sample", "max"),
                                       median_sharpe=("sharpe_in_sample", "median"),
                                       median_expectancy_R=("net_expectancy_R", "median"),
                                       fdr_discoveries=("fdr10_discovery", "sum"),
                                       median_trades=("trades", "median")).reset_index()
        parts += ["## Все модели (dev, NORMAL): агрегат по семействам", md(agg.sort_values("median_sharpe", ascending=False))]
    if (R / "candidates.csv").exists():
        c = pd.read_csv(R / "candidates.csv")
        parts += ["## Кандидаты: dev / walk-forward / holdout / стресс",
                  md(c[["candidate", "trades", "in_sample_sharpe", "in_sample_expectancy_R", "walk_forward_sharpe",
                        "out_of_sample_holdout_sharpe", "holdout_expectancy_R", "holdout_trades",
                        "stress1_expectancy_R", "stress2_expectancy_R", "profit_factor", "drawdown"]])]
    if (R / "parameter_stability.csv").exists():
        ps = pd.read_csv(R / "parameter_stability.csv")
        for cand, g in ps.groupby("candidate"):
            st = g[g.test.isin(["costs", "commission", "latency_ms"])][["test", "case", "n_trades", "expectancy_R",
                                                                       "profit_factor", "sharpe", "max_drawdown"]]
            ex = g[g.test == "exit_variant"][["exit_variant", "n_trades", "expectancy_R", "profit_factor", "sharpe",
                                               "max_drawdown", "win_rate"]]
            parts += [f"### {cand}: издержки и задержка", md(st), f"### {cand}: варианты выхода", md(ex)]
    for f in sorted(R.glob("extremes_*.csv")):
        parts += [f"### Зависимость от экстремальных сделок: {f.stem[9:]}", md(pd.read_csv(f))]
    if (R / "monte_carlo.csv").exists():
        mc = pd.read_csv(R / "monte_carlo.csv")
        keep = ["candidate", "annual_return_p05", "annual_return_p50", "annual_return_p95", "maxdd_p50", "maxdd_p95",
                "p_dd_gt_10", "p_dd_gt_15", "p_dd_gt_20", "p_dd_gt_30", "p_loss"]
        parts += ["## Монте-Карло (10 000 сценариев, горизонт 1 год, блоки торговых дней, риск 0,5%/сделку)",
                  md(mc[[k for k in keep if k in mc.columns]])]
    if (R / "target_10pct_analysis.csv").exists():
        ta = pd.read_csv(R / "target_10pct_analysis.csv")
        keep = ["case", "feasible", "risk_per_trade", "fraction_of_kelly", "avg_leverage", "monthly_median",
                "worst_month", "cagr", "hist_maxdd", "mc_maxdd_median", "mc_maxdd_p95", "p_dd_gt_10", "p_dd_gt_20",
                "p_dd_gt_30", "p_dd_gt_50", "note"]
        parts += ["## Цель 10% в месяц: требуемый риск (портфель кандидатов, dev)", md(ta[[k for k in keep if k in ta.columns]])]
    if (R / "yearly_returns.csv").exists():
        y = pd.read_csv(R / "yearly_returns.csv")
        parts += ["## Доходность по годам (риск 0,5%/сделку, без сложного процента)", md(y, "{:.2%}")]
    if (R / "monthly_returns.csv").exists():
        m = pd.read_csv(R / "monthly_returns.csv")
        for cand, g in m.groupby("candidate"):
            t = g.pivot(index="year", columns="month", values="ret").reset_index()
            r = g["ret"]
            s = pd.DataFrame([dict(months=len(r), positive=int((r > 0).sum()), negative=int((r < 0).sum()),
                                   mean=r.mean(), median=r.median(), best=r.max(), worst=r.min())])
            parts += [f"### Месячная доходность: {cand}", md(t, "{:.2%}"), md(s, "{:.4f}")]
    (R / "_tables.md").write_text("\n\n".join(parts), encoding="utf-8")
    print(f"written {R / '_tables.md'} ({len(parts)} blocks)")


if __name__ == "__main__":
    main()
