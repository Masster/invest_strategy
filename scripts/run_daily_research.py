"""Основное исследование на реальных дневных данных фьючерсов MOEX (ISS, реальные серии).

Этапы (задача §52): данные -> сетка гипотез (dev) -> множественное тестирование -> walk-forward ->
отбор кандидатов по заранее заданным правилам -> стресс/задержка -> устойчивость параметров и выходов ->
экстремальные сделки -> Монте-Карло -> размер позиции / цель 10% -> FINAL HOLDOUT -> портфель.

Запуск:  python3 scripts/run_daily_research.py [--stage all|dev|final]
Результаты: reports/*, research/experiments.csv. Кэш прогонов: data/cache/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moexlab.instruments.spec import InstrumentSpec  # noqa: E402
from moexlab.market_data.iss_daily import build_daily_universe, build_equity_universe  # noqa: E402
from moexlab.reporting import plots  # noqa: E402
from moexlab.research.grid import GRID_AXES, GRID_VERSION, daily_grid  # noqa: E402
from moexlab.research.runner import (Ledger, RunSettings, make_config, neighbors, returns_matrix,  # noqa: E402
                                     robust_score, run_grid, run_single, summarize, walk_forward)
from moexlab.risk.sizing import average_leverage, target_analysis  # noqa: E402
from moexlab.statistics.inference import (benjamini_hochberg, block_bootstrap_paths,  # noqa: E402
                                          bootstrap_mean_pvalue, deflated_sharpe_ratio, effective_trials,
                                          hansen_spa, pbo_cscv, whites_reality_check)
from moexlab.statistics.metrics import drawdown_series, monthly_table, remove_extremes  # noqa: E402
from moexlab.strategies.base import ExitPolicy  # noqa: E402

import os
REPORTS = Path(os.environ.get("MOEXLAB_REPORTS", ROOT / "reports"))
CACHE = Path(os.environ.get("MOEXLAB_CACHE", ROOT / "data/cache"))
DATA = Path(os.environ.get("MOEXLAB_DATA", ROOT / "data/raw/iss_daily"))
ASOF = date(2026, 9, 25)
DEV_START = "2020-01-01"
DEV_END = "2025-09-30"          # всё, что позже, — FINAL HOLDOUT (не используется при разработке)
HOLDOUT_START = "2025-10-01"
WF_TEST_YEARS = [2022, 2023, 2024, 2025]

# Спецификации: шаг цены — ISS (research/sources.md §7). Стоимость шага приравнена шагу (point_value=1):
# все результаты считаются в R (прибыль/начальный риск) и в доходности при фиксированном риске на сделку,
# поэтому зависимость стоимости шага от курса USD/RUB на выводы о сигнале не влияет.
COMMISSION = 0.0004 if os.environ.get("MOEXLAB_UNIVERSE", "futures") == "equities" else 0.00025
TICKS = {"Si": 1.0, "RI": 10.0, "MX": 25.0, "GD": 0.1, "SR": 1.0, "GZ": 1.0, "CR": 0.001, "BR": 0.01}
GROUPS = {"Si": "fx", "CR": "fx", "RI": "equity_index", "MX": "equity_index", "SR": "equity", "GZ": "equity",
          "GD": "commodity", "BR": "commodity"}


UNIVERSE = os.environ.get("MOEXLAB_UNIVERSE", "futures")   # futures | equities
EQUITY_DATA = Path(os.environ.get("MOEXLAB_EQUITY_DATA", ROOT / "data/raw/iss_equity_daily"))


def equity_specs(uni):
    """Акции: издержки задаются в долях цены через «шаг» = 1 б.п. медианной цены (A-14):
    спред 2 б.п., проскальзывание 2 б.п., стоп +2 б.п.; комиссия 0,04% (Премиум, акции)."""
    out = {}
    for t, it in uni.items():
        tick = float(it.stream["close"].median()) * 1e-4
        out[t] = InstrumentSpec(t, "equity", tick=tick, tick_value=tick, group="equity", spread_ticks=2.0,
                                slippage_ticks=2.0, stop_extra_ticks=2.0, margin_fraction=1.0,
                                source="relative-cost model A-14")
    return out


def specs_for(bases):
    return {b: InstrumentSpec(b, "future", tick=TICKS[b], tick_value=TICKS[b], group=GROUPS[b], spread_ticks=1.0,
                              slippage_ticks=1.0, stop_extra_ticks=1.0, margin_fraction=0.15,
                              source="ISS securities (tick); spread/slippage = assumption A-05")
            for b in bases}


def cut(streams, start=None, end=None):
    out = {}
    for k, df in streams.items():
        m = pd.Series(True, index=df.index)
        if start:
            m &= df.index >= pd.Timestamp(start, tz="UTC")
        if end:
            m &= df.index <= pd.Timestamp(end, tz="UTC")
        out[k] = df[m.to_numpy()]
    return out


def ann_sharpe(x: pd.Series) -> float:
    return float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(252))


# ---------------------------------------------------------------------------
def stage_data():
    if UNIVERSE == "equities":
        uni = build_equity_universe(EQUITY_DATA)
    else:
        bases = [b for b in TICKS if (DATA / b).exists() and any((DATA / b).glob("*.csv"))]
        uni = build_daily_universe(DATA, bases, ASOF)
    lines = ["# Качество данных (дневные свечи реальных серий, MOEX ISS)", "",
             "DATA_QUALITY = DIRECT_ISS: свечи загружены напрямую из MOEX ISS (scripts/fetch_iss_daily.py);",
             "SHA-256 файлов — data/manifests/iss_daily_sha256.csv. Ниже — автоматические проверки.", ""]
    for b, it in uni.items():
        s = it.stream
        lines.append(f"## {b}: {len(s)} торговых дней {s.index[0].date()} → {s.index[-1].date()}, серий {len(it.series)}")
        rolls = (s["secid"] != s["secid"].shift()).sum() - 1
        lines.append(f"- перекладок: {rolls}; медианный объём активной серии: {s['volume'].median():,.0f}")
        low = (s["volume"] < 100).sum()
        lines.append(f"- дней с объёмом активной серии < 100 контрактов: {low}")
        for iss in it.issues[:15]:
            lines.append(f"- ⚠ {iss}")
        lines.append("")
    (REPORTS / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")
    return uni


def stage_dev(uni, ledger: Ledger, workers=None):
    specs = equity_specs(uni) if UNIVERSE == "equities" else specs_for(list(uni))
    streams = cut({b: it.stream for b, it in uni.items()}, DEV_START, DEV_END)
    configs = daily_grid(long_only=(UNIVERSE == "equities"))
    rs = RunSettings(commission_fraction=COMMISSION)
    g = run_grid(configs, streams, specs, rs, workers)
    res = g["results"]
    M = returns_matrix(res)
    fam = {cid: cfg.family for cid, (cfg, _) in res.items()}
    summ = pd.DataFrame([summarize(cfg, out) for cid, (cfg, out) in res.items()]).set_index("strategy_id")
    # --- множественное тестирование ---
    pv = pd.Series({c: bootstrap_mean_pvalue(M[c].to_numpy(), 1000, 5.0, 1) for c in M.columns})
    summ["p_boot"] = pv
    summ["fdr10_discovery"] = pd.Series(benjamini_hochberg(pv.to_numpy(), 0.10), index=pv.index)
    sr_daily = M.mean() / (M.std(ddof=1) + 1e-12)
    n_eff = effective_trials(M.to_numpy())
    best = sr_daily.idxmax()
    dsr = deflated_sharpe_ratio(M[best].to_numpy(), max(2, int(round(n_eff))), sr_daily.to_numpy())
    spa_all = hansen_spa(M.to_numpy(), 2000, 10.0, 2)
    rc_all = whites_reality_check(M.to_numpy(), 2000, 10.0, 3)
    pbo = pbo_cscv(M.to_numpy(), 12)
    fam_tests = {}
    for f in sorted(set(fam.values())):
        cols = [c for c in M.columns if fam[c] == f]
        fam_tests[f] = dict(spa=hansen_spa(M[cols].to_numpy(), 1000, 10.0, 4)["pvalue"],
                            pbo=pbo_cscv(M[cols].to_numpy(), 12)["pbo"] if len(cols) > 1 else np.nan)
    # --- walk-forward ---
    nb = neighbors([cfg for cfg, _ in res.values()], GRID_AXES)
    wf, oos = walk_forward(M, fam, nb, WF_TEST_YEARS, DEV_START, DEV_END)
    rscore = robust_score(M, nb)
    summ["robust_score_dev"] = rscore
    summ["sharpe_ann_dev"] = sr_daily * np.sqrt(252)
    dev = dict(configs=configs, res=res, M=M, fam=fam, summ=summ, nb=nb, wf=wf, oos=oos, n_eff=n_eff, dsr=dsr,
               spa_all=spa_all, rc_all=rc_all, pbo=pbo, fam_tests=fam_tests, specs=specs, errors=g["errors"])
    ledger.add(hypothesis="H-DAILY-GRID: хотя бы одно семейство дневных стратегий имеет положительное ожидание после издержек",
               strategy="ALL daily families", parameters=json.dumps({"grid": GRID_VERSION, "n_configs": len(configs)}),
               data=f"ISS daily real series {sorted(uni)} (DIRECT_ISS)", period=f"{DEV_START}..{DEV_END}",
               result=json.dumps(dict(best=best, best_sr_ann=float(sr_daily[best] * np.sqrt(252)),
                                      fdr=int(summ["fdr10_discovery"].sum()), spa=spa_all["pvalue"],
                                      rc=rc_all["pvalue"], pbo=pbo["pbo"], dsr=dsr["dsr"], n_eff=n_eff))[:1500],
               conclusion="см. reports/FINAL_RESEARCH_REPORT.md", next_step="walk-forward + отбор кандидатов",
               seed="1..4", execution_model="ConservativeL1 (daily bars, worst-case intrabar)",
               cost_model="NORMAL: comm 0.025%, spread 1 tick, slip 1 tick, stop +1 tick")
    return dev


def select_candidates(dev) -> list[dict]:
    """Заранее заданные правила отбора (записаны в research/RESEARCH_PLAN.md до запуска на реальных данных)."""
    wf, oos, M, fam, summ = dev["wf"], dev["oos"], dev["M"], dev["fam"], dev["summ"]
    out = []
    for f in sorted(set(fam.values())):
        if f.startswith("CONTROL") or f not in oos:
            continue
        r = oos[f]
        rows = wf[wf.family == f]
        sr = ann_sharpe(r)
        pos_years = float((rows.test_return > 0).mean()) if len(rows) else 0.0
        cols = [c for c in M.columns if fam[c] == f]
        best = summ.loc[cols, "robust_score_dev"].idxmax()
        nbs = dev["nb"].get(best, [])
        plateau = float(np.median([summ.loc[n, "sharpe_ann_dev"] for n in nbs])) if nbs else np.nan
        passed = sr > 0.3 and pos_years >= 0.6 and (np.isnan(plateau) or plateau > 0) and \
            summ.loc[best, "expectancy_R"] > 0
        out.append(dict(family=f, wf_oos_sharpe=sr, wf_positive_year_share=pos_years, chosen=best,
                        plateau_neighbors_sharpe=plateau, dev_expectancy_R=float(summ.loc[best, "expectancy_R"]),
                        dev_trades=int(summ.loc[best, "n_trades"]), passed=bool(passed)))
    return out


def stress_and_latency(cfg, streams, specs):
    rows = []
    for sc in ("ZERO", "NORMAL", "STRESS_1", "STRESS_2"):
        o = run_single(cfg, streams, specs, RunSettings(scenario=sc, commission_fraction=COMMISSION if sc != "ZERO" else 0.0))
        m = summarize(cfg, o)
        rows.append(dict(test="costs", case=sc, **{k: m.get(k) for k in ("n_trades", "expectancy_R", "profit_factor",
                                                                          "sharpe", "max_drawdown", "total_return")}))
    for comm in (0.0004, 0.0006):
        o = run_single(cfg, streams, specs, RunSettings(commission_fraction=comm))
        m = summarize(cfg, o)
        rows.append(dict(test="commission", case=f"{comm:.2%}", **{k: m.get(k) for k in (
            "n_trades", "expectancy_R", "profit_factor", "sharpe", "max_drawdown", "total_return")}))
    for lat in (0, 100, 250, 500, 1000, 2000):
        o = run_single(cfg, streams, specs, RunSettings(latency_ms=lat, commission_fraction=COMMISSION))
        m = summarize(cfg, o)
        rows.append(dict(test="latency_ms", case=str(lat), **{k: m.get(k) for k in (
            "n_trades", "expectancy_R", "profit_factor", "sharpe", "max_drawdown", "total_return")}))
    return pd.DataFrame(rows)


EXIT_VARIANTS = {
    "atr_trail": dict(initial="atr", trailing="atr"),
    "chandelier": dict(initial="atr", trailing="chandelier", trailing_lookback=22),
    "percent_trail": dict(initial="percent", initial_k=0.03, trailing="percent", trailing_k=0.05),
    "structural_trail": dict(initial="atr", trailing="structural", trailing_lookback=10),
    "no_trail": dict(initial="atr", trailing="none"),
    "rr1": dict(initial="atr", trailing="none", target_rr=1.0),
    "rr2": dict(initial="atr", trailing="none", target_rr=2.0),
    "rr3": dict(initial="atr", trailing="none", target_rr=3.0),
    "partial1_trail": dict(initial="atr", trailing="atr", partial_rr=1.0, partial_frac=0.5),
    "time20": dict(initial="atr", trailing="none", max_bars=20),
}


def exit_study(cfg, streams, specs):
    base = dict(cfg.exits)
    rows = []
    for name, ov in EXIT_VARIANTS.items():
        e = dict(base)
        e.update({k: None for k in ("target_rr", "partial_rr")})
        e["max_bars"] = base.get("max_bars")
        e.update(ov)
        c = make_config(cfg.family, dict(cfg.params), ExitPolicy(**e))
        m = summarize(c, run_single(c, streams, specs, RunSettings(commission_fraction=COMMISSION)))
        rows.append(dict(exit_variant=name, **{k: m.get(k) for k in ("n_trades", "expectancy_R", "profit_factor",
                                                                     "sharpe", "max_drawdown", "win_rate")}))
    # поверхность: начальный стоп × трейлинг (ATR)
    surf = []
    for ik in (1.0, 1.5, 2.0, 2.5, 3.0):
        for tk in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            e = dict(base)
            e.update(initial="atr", initial_k=ik, trailing="atr", trailing_k=tk, target_rr=None, partial_rr=None)
            c = make_config(cfg.family, dict(cfg.params), ExitPolicy(**e))
            m = summarize(c, run_single(c, streams, specs, RunSettings(commission_fraction=COMMISSION)))
            surf.append(dict(initial_k=ik, trailing_k=tk, sharpe=m.get("sharpe"), expectancy_R=m.get("expectancy_R"),
                             n_trades=m.get("n_trades")))
    return pd.DataFrame(rows), pd.DataFrame(surf)


def breakdowns(out):
    tr = out["trades"].copy()
    tr["year"] = pd.to_datetime(tr["trading_day"]).dt.year
    by = {}
    for key in ("code", "direction", "year", "tag", "exit_reason"):
        by[key] = tr.groupby(key)["R"].agg(["count", "mean", "sum"]).rename(columns={"mean": "mean_R", "sum": "sum_R"})
    return by


def main(stage: str = "all", workers=None):
    REPORTS.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(Path(os.environ.get("MOEXLAB_LEDGER", ROOT / "research/experiments.csv")))
    uni = stage_data()
    if not uni:
        print("NO DATA: data/raw/iss_daily is empty")
        return
    dev = stage_dev(uni, ledger, workers)
    with open(CACHE / "dev.pkl", "wb") as f:
        pickle.dump({k: v for k, v in dev.items() if k not in ("configs",)}, f)
    summ = dev["summ"]
    summ.to_csv(REPORTS / "strategy_comparison_dev_grid.csv")
    dev["wf"].to_csv(REPORTS / "walk_forward.csv", index=False)
    cands = select_candidates(dev)
    pd.DataFrame(cands).to_csv(REPORTS / "candidate_selection.csv", index=False)
    print(json.dumps(dict(n_eff=dev["n_eff"], dsr=dev["dsr"], spa=dev["spa_all"], rc=dev["rc_all"], pbo=dev["pbo"],
                          fam=dev["fam_tests"]), indent=1, default=str))
    print(pd.DataFrame(cands).to_string())
    json.dump(dict(n_eff=dev["n_eff"], dsr=dev["dsr"], spa=dev["spa_all"], rc=dev["rc_all"], pbo=dev["pbo"],
                   family_tests=dev["fam_tests"], errors=dev["errors"], candidates=cands),
              open(REPORTS / "multiple_testing.json", "w"), indent=2, default=str)
    if stage == "dev":
        return
    # ---------------- углублённый анализ кандидатов (на dev) ----------------
    specs = dev["specs"]
    dev_streams = cut({b: it.stream for b, it in uni.items()}, DEV_START, DEV_END)
    full_streams = cut({b: it.stream for b, it in uni.items()}, DEV_START, None)
    res = dev["res"]
    chosen = [c for c in cands if c["passed"]]
    # если ни один не прошёл — всё равно исследуем 3 лучших по WF как «лучшие из неудачных» (явно помечено)
    if not chosen:
        chosen = sorted(cands, key=lambda c: -c["wf_oos_sharpe"])[:3]
        for c in chosen:
            c["label"] = "BEST_OF_FAILED"
    else:
        for c in chosen:
            c["label"] = "CANDIDATE"
    comp_rows, final_rows, mc_rows, stab_rows, all_trades, eq_curves, dd_curves = [], [], [], [], [], {}, {}
    daily_by_cand = {}
    for k, c in enumerate(chosen):
        cfg, out = res[c["chosen"]]
        tag = f"{c['label']}_{chr(65 + k)}_{cfg.family}"
        st = stress_and_latency(cfg, dev_streams, specs)
        st.insert(0, "candidate", tag)
        exits, surf = exit_study(cfg, dev_streams, specs)
        exits.insert(0, "candidate", tag)
        surf.insert(0, "candidate", tag)
        stab_rows.append(pd.concat([st, exits.assign(test="exit_variant")], ignore_index=True))
        piv = surf.pivot(index="initial_k", columns="trailing_k", values="sharpe")
        plots.heat_surface(piv, REPORTS / f"parameter_surface_{tag}.png",
                           f"{tag}: Sharpe (dev) — начальный стоп × трейлинг, ATR", "трейлинг, ATR", "начальный стоп, ATR")
        surf.to_csv(REPORTS / f"parameter_surface_{tag}.csv", index=False)
        # сетка параметров семьи (плато)
        fam_rows = summ[summ.family == cfg.family][["params", "exits", "n_trades", "expectancy_R", "profit_factor",
                                                    "sharpe_ann_dev", "robust_score_dev", "max_drawdown"]]
        fam_rows.to_csv(REPORTS / f"parameter_grid_{tag}.csv")
        ext = remove_extremes(out["trades"])
        ext.insert(0, "candidate", tag)
        bd = breakdowns(out)
        for key, t in bd.items():
            t.to_csv(REPORTS / f"breakdown_{key}_{tag}.csv")
        mc = block_bootstrap_paths(out["daily"], 10000, 252, 5.0, 7)
        mc_rows.append(dict(candidate=tag, **{k: v for k, v in mc.items() if not k.startswith("_")}))
        plots.histogram(mc["_maxdd_samples"] * 100, REPORTS / f"monte_carlo_drawdowns_{tag}.png",
                        f"{tag}: распределение макс. просадки за 1 год (10 000 сценариев, риск 0,5%/сделку)",
                        "макс. просадка, %", {"медиана": mc["maxdd_p50"] * 100, "5% худших": mc["maxdd_p95"] * 100})
        # FINAL HOLDOUT: прогон на полном ряду, оценка только после HOLDOUT_START
        full = run_single(cfg, full_streams, specs, RunSettings(commission_fraction=COMMISSION))
        mh = summarize(cfg, full, start=HOLDOUT_START)
        md = summarize(cfg, out)
        s1 = run_single(cfg, full_streams, specs, RunSettings(scenario="STRESS_1", commission_fraction=COMMISSION))
        mh1 = summarize(cfg, s1, start=HOLDOUT_START)
        wf_r = dev["oos"][cfg.family] if cfg.family in dev["oos"] else pd.Series(dtype=float)
        comp_rows.append(dict(candidate=tag, strategy_id=cfg.id, family=cfg.family, instruments=",".join(specs),
                              parameters=json.dumps(dict(cfg.params)), exits=json.dumps(dict(cfg.exits)),
                              in_sample_sharpe=md.get("sharpe"), in_sample_expectancy_R=md.get("expectancy_R"),
                              walk_forward_sharpe=ann_sharpe(wf_r) if len(wf_r) else np.nan,
                              out_of_sample_holdout_sharpe=mh.get("sharpe"),
                              holdout_expectancy_R=mh.get("expectancy_R"), holdout_trades=mh.get("n_trades"),
                              holdout_stress1_expectancy_R=mh1.get("expectancy_R"),
                              stress1_expectancy_R=float(st[(st.test == "costs") & (st.case == "STRESS_1")].expectancy_R.iloc[0]),
                              stress2_expectancy_R=float(st[(st.test == "costs") & (st.case == "STRESS_2")].expectancy_R.iloc[0]),
                              trades=md.get("n_trades"), profit_factor=md.get("profit_factor"), sharpe=md.get("sharpe"),
                              drawdown=md.get("max_drawdown"), net_expectancy=md.get("expectancy_R")))
        tr = full["trades"].copy()
        tr.insert(0, "candidate", tag)
        all_trades.append(tr)
        eq = 1 + full["daily"].cumsum()
        eq_curves[tag] = eq
        dd_curves[tag] = drawdown_series(eq)
        daily_by_cand[tag] = full["daily"]
        ext.to_csv(REPORTS / f"extremes_{tag}.csv", index=False)
        ledger.add(hypothesis=f"Кандидат {tag}", strategy=cfg.family, parameters=cfg.id,
                   data="ISS daily real series", period=f"dev {DEV_START}..{DEV_END}; holdout {HOLDOUT_START}..",
                   result=json.dumps(comp_rows[-1], default=str)[:1500],
                   conclusion="holdout evaluated once", next_step="shadow trading if robust",
                   seed="7", execution_model="ConservativeL1", cost_model="NORMAL/STRESS_1/STRESS_2")
    comp = pd.DataFrame(comp_rows)
    # сводная таблица: все модели + кандидаты
    all_models = summ.reset_index()[["strategy_id", "family", "params", "exits", "n_trades", "profit_factor",
                                     "sharpe_ann_dev", "max_drawdown", "expectancy_R", "p_boot", "fdr10_discovery"]]
    all_models = all_models.rename(columns={"sharpe_ann_dev": "sharpe_in_sample", "expectancy_R": "net_expectancy_R",
                                            "n_trades": "trades", "max_drawdown": "drawdown"})
    all_models["instruments"] = ",".join(specs)
    all_models["in_sample"] = f"{DEV_START}..{DEV_END}"
    wf_sel = dev["wf"].groupby("selected")["test_year"].apply(lambda s: ",".join(map(str, s)))
    all_models["walk_forward"] = all_models["strategy_id"].map(wf_sel).fillna("")
    all_models["out_of_sample"] = all_models["strategy_id"].map(
        {r["strategy_id"]: r["out_of_sample_holdout_sharpe"] for r in comp_rows}).astype(float)
    all_models["stress"] = all_models["strategy_id"].map(
        {r["strategy_id"]: r["stress1_expectancy_R"] for r in comp_rows}).astype(float)
    all_models.to_csv(REPORTS / "strategy_comparison.csv", index=False)
    comp.to_csv(REPORTS / "candidates.csv", index=False)
    pd.concat(stab_rows, ignore_index=True).to_csv(REPORTS / "parameter_stability.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(REPORTS / "monte_carlo.csv", index=False)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades.to_csv(REPORTS / "trades.csv", index=False)
    E = pd.DataFrame(eq_curves)
    E.to_csv(REPORTS / "equity.csv")
    pd.DataFrame(dd_curves).to_csv(REPORTS / "drawdowns.csv")
    plots.line_chart({k: v - 1 for k, v in eq_curves.items()}, REPORTS / "equity_curve.png",
                     "Накопленная доходность при риске 0,5% капитала на сделку (без сложного процента); "
                     f"holdout с {HOLDOUT_START}", "доходность, %", pct=True)
    plots.drawdown_chart(dd_curves, REPORTS / "drawdown_curve.png", "Просадка (от максимума)")
    # портфель кандидатов (равный риск)
    D = pd.DataFrame(daily_by_cand).fillna(0.0)
    port = D.mean(axis=1) if len(D.columns) else pd.Series(dtype=float)
    corr = D.corr() if len(D.columns) > 1 else pd.DataFrame()
    corr.to_csv(REPORTS / "candidate_correlation.csv")
    # помесячно / по годам (портфель и кандидаты)
    monthly = {k: monthly_table(v) for k, v in daily_by_cand.items()}
    mrows = []
    for k, t in monthly.items():
        for y in t.index:
            for mth in t.columns:
                if pd.notna(t.loc[y, mth]):
                    mrows.append(dict(candidate=k, year=y, month=mth, ret=float(t.loc[y, mth])))
    pd.DataFrame(mrows).to_csv(REPORTS / "monthly_returns.csv", index=False)
    yr = {k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in daily_by_cand.items()}
    pd.DataFrame(yr).to_csv(REPORTS / "yearly_returns.csv")
    if len(port):
        plots.monthly_heatmap(monthly_table(port), REPORTS / "monthly_returns_heatmap.png",
                              "Портфель кандидатов: доходность по месяцам, % (риск 0,5%/сделку на каждого кандидата)")
        roll = port.rolling(126)
        plots.line_chart({"портфель": roll.mean() / roll.std() * np.sqrt(252)}, REPORTS / "rolling_sharpe.png",
                         "Скользящий Sharpe (126 дней), портфель кандидатов", "Sharpe")
        if not trades.empty:
            t2 = trades.sort_values("exit_ts")
            re = {k: g.set_index(pd.to_datetime(g["exit_ts"]))["R"].rolling(50).mean()
                  for k, g in t2.groupby("candidate")}
            plots.line_chart(re, REPORTS / "rolling_expectancy.png", "Скользящее ожидание, R (50 сделок)", "R")
        # анализ цели 10%: только на dev-части портфеля
        port_dev = port[port.index <= pd.Timestamp(DEV_END, tz=port.index.tz)]
        dev_trades = trades[pd.to_datetime(trades["trading_day"]) <= pd.Timestamp(DEV_END)]
        lev = average_leverage(dev_trades, port_dev.index, 0.005) / max(len(D.columns), 1)
        ta = target_analysis(port_dev, 0.005, lev)
        ta.to_csv(REPORTS / "target_10pct_analysis.csv", index=False)
    labels = [f"{r['family']}" for r in cands]
    plots.bar_chart(labels, [r["wf_oos_sharpe"] for r in cands], REPORTS / "strategy_comparison.png",
                    "Walk-forward OOS Sharpe по семействам (выбор параметров только по прошлому)", "Sharpe (годовой)")
    metrics = dict(dev=dict(n_configs=len(dev["M"].columns), n_eff=dev["n_eff"], dsr=dev["dsr"],
                            spa=dev["spa_all"], reality_check=dev["rc_all"], pbo=dev["pbo"]),
                   candidates=comp_rows, portfolio=dict(sharpe=ann_sharpe(port) if len(port) else None,
                                                        correlation=corr.round(3).to_dict() if len(corr) else {}))
    json.dump(metrics, open(REPORTS / "metrics.json", "w"), indent=2, default=str)
    print(comp.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "dev"])
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.stage, a.workers)
