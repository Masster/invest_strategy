"""Исследование на годе данных T-Invest (план v2, research/RESEARCH_PLAN.md): R-Breaker 2.1 по спецификации,
внутридневные семейства (5 мин), дневной свинг. Данные — только из локальной БД (moexlab.storage.db).

  python3 scripts/run_intraday_research.py [--universes RB_FUT ID_FUT ID_EQ DAY_FUT DAY_EQ] [--workers 4]
  python3 scripts/run_intraday_research.py --synthetic      # сухой прогон R-Breaker на синтетике

Этапы для каждой вселенной: dev-сетка -> множественное тестирование -> walk-forward по месяцам -> отбор по
заранее заданным правилам -> STRESS_1/2 -> FINAL HOLDOUT (один раз; кандидаты + заранее заявленная база
R-Breaker). Результаты: reports/year/*, журнал research/experiments.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moexlab.backtest.engine import Engine, InstrumentInput, RiskConfig  # noqa: E402
from moexlab.execution.models import SCENARIOS, CommissionModel, ConservativeL1ExecutionModel  # noqa: E402
from moexlab.instruments.spec import InstrumentSpec  # noqa: E402
from moexlab.market_data.calendar import SessionConfig, annotate_sessions  # noqa: E402
from moexlab.market_data.synthetic import generate_minute_bars  # noqa: E402
from moexlab.reporting import plots  # noqa: E402
from moexlab.research import grid_intraday as G  # noqa: E402
from moexlab.research.causality import check_causality  # noqa: E402
from moexlab.research.intraday import (daily_from_stream, load_equity, load_future,  # noqa: E402
                                       resample_stream)
from moexlab.research.runner import (Ledger, RunSettings, expand_grid, neighbors, returns_matrix,  # noqa: E402
                                     robust_score, run_grid, run_single, summarize, walk_forward_periods)
from moexlab.statistics.inference import (benjamini_hochberg, bootstrap_mean_pvalue,  # noqa: E402
                                          deflated_sharpe_ratio, effective_trials, hansen_spa, pbo_cscv,
                                          whites_reality_check)
from moexlab.statistics.metrics import trade_metrics  # noqa: E402
from moexlab.storage.db import open_db  # noqa: E402
from moexlab.strategies.rbreaker import RBreaker, series_features  # noqa: E402

REPORTS = Path(os.environ.get("MOEXLAB_REPORTS", ROOT / "reports")) / "year"
CACHE = ROOT / "data/cache/year"
DATA_START = "2025-09-25"
DEV_END = "2026-05-31"
HOLD_START, HOLD_END = "2026-06-01", "2026-09-25"
FOLDS = [(DATA_START, f"2026-{m:02d}-01", str((pd.Timestamp(f"2026-{m:02d}-01") + pd.offsets.MonthEnd(0)).date()))
         for m in range(1, 6)]
FUTURES = ["MX", "RN", "CR", "GD", "Si", "RI", "SR", "GZ", "BR"]
SPEC_SET = ["MX", "RN", "CR", "GD"]
EQUITIES = ["SBER", "GAZP", "LKOH", "ROSN", "GMKN", "NVTK", "TATN", "MGNT", "PLZL", "VTBR", "ALRS", "AFLT"]
ETS_DAY = pd.Timestamp("2026-03-23")

# для синтетического сухого прогона
SESSION_PRE2026 = SessionConfig()


def rb_grid():
    return expand_grid("RBREAKER_2_1", G.RB_AXES)


UNIVERSES = {
    "RB_FUT": dict(kind="future", bars=1, grid=rb_grid, axes={"RBREAKER_2_1": G.RB_AXES}, codes=FUTURES,
                   rs=dict(commission_fraction=0.00025, bar_seconds=60.0, risk_by_tag=(("REVERSAL", 0.0035),))),
    "ID_FUT": dict(kind="future", bars=G.ID_BAR_MIN, grid=G.intraday_grid, axes=G.ID_AXES, codes=FUTURES,
                   rs=dict(commission_fraction=0.00025, bar_seconds=300.0)),
    "ID_EQ": dict(kind="equity", bars=G.ID_BAR_MIN, grid=G.intraday_grid, axes=G.ID_AXES, codes=EQUITIES,
                  rs=dict(commission_fraction=0.0004, bar_seconds=300.0)),
    "DAY_FUT": dict(kind="future", bars="D", grid=G.day_grid, axes=G.DAY_AXES, codes=FUTURES,
                    rs=dict(commission_fraction=0.00025, bar_seconds=86400.0)),
    "DAY_EQ": dict(kind="equity", bars="D", grid=lambda: G.day_grid(long_only=True), axes=G.DAY_AXES,
                   codes=EQUITIES, rs=dict(commission_fraction=0.0004, bar_seconds=86400.0)),
}


# ---------------------------------------------------------------------------------------------- данные
def attach_rb_features(item) -> pd.DataFrame:
    st = item.stream.copy()
    for sec in st["secid"].unique():
        m = (st["secid"] == sec).to_numpy()
        f = series_features(item.series[sec])
        f = f[~f.index.duplicated(keep="first")]
        for c in f.columns:
            st.loc[m, c] = f[c].reindex(st.index[m]).to_numpy()
    return st


def equity_spec(code: str, stream: pd.DataFrame) -> InstrumentSpec:
    """A-14: издержки акций в долях цены — «шаг» = 1 б.п. медианной цены; спред 2, проскальзывание 2, стоп +2."""
    tick = float(stream["close"].median()) * 1e-4
    return InstrumentSpec(code, "equity", tick=tick, tick_value=tick, group="equity", spread_ticks=2.0,
                          slippage_ticks=2.0, stop_extra_ticks=2.0, margin_fraction=1.0,
                          source="relative-cost model A-14")


def load_universe(name: str, end: str) -> tuple[dict, dict, list[str]]:
    """Потоки вселенной, обрезанные по торговому дню <= end (данные позже в расчёт не попадают)."""
    u = UNIVERSES[name]
    key = CACHE / f"{name}_{end}.pkl"
    if key.exists():
        return pickle.load(open(key, "rb"))
    db = open_db(read_only=True)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    streams, specs, issues = {}, {}, []
    for code in u["codes"]:
        it = load_future(db, code, None, end_ts) if u["kind"] == "future" else load_equity(db, code, None, end_ts)
        if it is None or it.stream.empty:
            issues.append(f"{code}: no data")
            continue
        issues += it.issues
        td = pd.to_datetime(it.stream["trading_day"])
        it.stream = it.stream[(td <= pd.Timestamp(end)).to_numpy()]
        for s in list(it.series):
            tds = pd.to_datetime(it.series[s]["trading_day"])
            it.series[s] = it.series[s][(tds <= pd.Timestamp(end)).to_numpy()]
        if u["bars"] == 1:
            bars = attach_rb_features(it)
        elif u["bars"] == "D":
            bars = daily_from_stream(it.stream)
        else:
            bars = resample_stream(it.stream, u["bars"], u["kind"])
        streams[code] = bars
        specs[code] = it.spec if u["kind"] == "future" else equity_spec(code, it.stream)
    db.close()
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump((streams, specs, issues), open(key, "wb"))
    return streams, specs, issues


def data_quality(streams: dict, name: str) -> list[str]:
    lines = [f"## {name}"]
    for code, df in streams.items():
        days = pd.Series(df["trading_day"]).nunique()
        rolls = int((df["secid"] != df["secid"].shift()).sum() - 1) if "secid" in df else 0
        bad = ((df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))).sum()
        lines.append(f"- {code}: {len(df):,} свечей, {days} торговых дней {df['trading_day'].iloc[0]} → "
                     f"{df['trading_day'].iloc[-1]}, перекладок {rolls}, OHLC-нарушений {int(bad)}")
    return lines


# ---------------------------------------------------------------------------------------------- этапы
def ann_sharpe(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(x) > 2 else float("nan")


def dev_stage(name: str, workers: int, ledger: Ledger) -> dict:
    u = UNIVERSES[name]
    streams, specs, issues = load_universe(name, DEV_END)
    configs = u["grid"]()
    rs = RunSettings(**u["rs"])
    g = run_grid(configs, streams, specs, rs, workers)
    res = g["results"]
    M = returns_matrix(res)
    M = M[M.index <= pd.Timestamp(DEV_END, tz="UTC")]
    fam = {cid: cfg.family for cid, (cfg, _) in res.items()}
    summ = pd.DataFrame([summarize(cfg, out) for cid, (cfg, out) in res.items()]).set_index("strategy_id")
    pv = pd.Series({c: bootstrap_mean_pvalue(M[c].to_numpy(), 1000, 5.0, 1) for c in M.columns})
    summ["p_boot"] = pv
    summ["fdr10_discovery"] = pd.Series(benjamini_hochberg(pv.to_numpy(), 0.10), index=pv.index)
    sr = M.mean() / (M.std(ddof=1) + 1e-12)
    summ["sharpe_ann_dev"] = sr * np.sqrt(252)
    n_eff = effective_trials(M.to_numpy()) if M.shape[1] > 1 else 1.0
    best = sr.idxmax()
    dsr = deflated_sharpe_ratio(M[best].to_numpy(), max(2, int(round(n_eff))), sr.to_numpy())
    spa = hansen_spa(M.to_numpy(), 2000, 5.0, 2)
    rc = whites_reality_check(M.to_numpy(), 2000, 5.0, 3)
    pbo = pbo_cscv(M.to_numpy(), 8) if M.shape[1] > 1 else {"pbo": np.nan}
    nb = neighbors(configs, u["axes"])
    rscore = robust_score(M, nb, 20)
    summ["robust_score_dev"] = rscore
    wf, oos = walk_forward_periods(M, fam, nb, FOLDS, min_train_days=40, min_test_days=10, min_active_days=20)
    out = dict(name=name, configs=configs, res=res, M=M, fam=fam, summ=summ, nb=nb, wf=wf, oos=oos, n_eff=n_eff,
               dsr=dsr, spa=spa, rc=rc, pbo=pbo, specs=specs, errors=g["errors"], issues=issues,
               dq=data_quality(streams, name))
    ledger.add(hypothesis=f"{name}: хотя бы одно семейство имеет положительное ожидание после издержек (год T-Invest)",
               strategy=f"{name} grid", parameters=json.dumps({"grid": G.GRID_VERSION, "n_configs": len(configs)}),
               data=f"T-Invest 1m {sorted(streams)}", period=f"{DATA_START}..{DEV_END}",
               result=json.dumps(dict(best=best, best_sr_ann=float(sr[best] * np.sqrt(252)),
                                      fdr=int(summ["fdr10_discovery"].sum()), spa=spa["pvalue"], rc=rc["pvalue"],
                                      pbo=pbo["pbo"], dsr=dsr["dsr"], n_eff=n_eff, errors=len(g["errors"])),
                                 default=float)[:1500],
               conclusion="см. reports/FINAL_RESEARCH_REPORT.md", next_step="walk-forward + отбор",
               seed="1..3", execution_model="ConservativeL1 (bars, worst-case intrabar)",
               cost_model=json.dumps(u["rs"]))
    return out


def select(dev: dict) -> pd.DataFrame:
    """Правила отбора плана v2 (§v2.4)."""
    rows = []
    M, fam, summ, nb, wf, oos = dev["M"], dev["fam"], dev["summ"], dev["nb"], dev["wf"], dev["oos"]
    sr_dev = summ["sharpe_ann_dev"]
    for f in sorted(set(fam.values())):
        cols = [c for c in M.columns if fam[c] == f]
        rsc = summ.loc[cols, "robust_score_dev"]
        if not np.isfinite(rsc.max()):
            continue
        final = rsc.idxmax()
        w = wf[wf["family"] == f] if len(wf) else pd.DataFrame()
        oos_sr = ann_sharpe(oos[f]) if f in oos else float("nan")
        pos_share = float((w["test_return"] > 0).mean()) if len(w) else float("nan")
        nbs = [n for n in nb.get(final, []) if n in sr_dev.index]
        nb_med = float(np.median(sr_dev[nbs])) if nbs else float(sr_dev[final])
        exp_r = float(summ.loc[final, "expectancy_R"]) if "expectancy_R" in summ else float("nan")
        r = dict(family=f, final_config=final, wf_oos_sharpe=oos_sr, wf_months=len(w), wf_pos_share=pos_share,
                 neighbors_median_sharpe=nb_med, dev_sharpe=float(sr_dev[final]), dev_expectancy_R=exp_r,
                 dev_trades=int(summ.loc[final, "n_trades"]), p_boot=float(summ.loc[final, "p_boot"]))
        r["rule1_wf_sharpe"] = bool(oos_sr > 0.3)
        r["rule2_pos_months"] = bool(pos_share >= 0.6)
        r["rule3_plateau"] = bool(nb_med > 0)
        r["rule4_expectancy"] = bool(exp_r > 0)
        r["passed"] = r["rule1_wf_sharpe"] and r["rule2_pos_months"] and r["rule3_plateau"] and r["rule4_expectancy"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("wf_oos_sharpe", ascending=False)


def run_cfg(cfg, streams, specs, rs_kw, scenario="NORMAL", commission=None):
    kw = dict(rs_kw)
    if commission is not None:
        kw["commission_fraction"] = commission
    return run_single(cfg, streams, specs, RunSettings(scenario=scenario, **kw))


def seg(out: dict, start: str, end: str) -> dict:
    d = out["daily"]
    d = d[(d.index >= pd.Timestamp(start, tz="UTC")) & (d.index <= pd.Timestamp(end, tz="UTC"))]
    tr = out["trades"]
    if not tr.empty:
        td = pd.to_datetime(tr["trading_day"])
        tr = tr[(td >= pd.Timestamp(start)) & (td <= pd.Timestamp(end))]
    m = trade_metrics(tr)
    return dict(sharpe=ann_sharpe(d), total_return=float(d.sum()), days=len(d), trades=m.get("n_trades", 0),
                expectancy_R=m.get("expectancy_R", np.nan), gross_expectancy_R=m.get("gross_expectancy_R", np.nan),
                win_rate=m.get("win_rate", np.nan), profit_factor=m.get("profit_factor", np.nan))


def breakdown(tr: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame()
    g = tr.groupby(by)
    return pd.DataFrame({"trades": g.size(), "expectancy_R": g["R"].mean(), "gross_R": g["gross_R"].mean(),
                         "win_rate": g["R"].apply(lambda x: (x > 0).mean()), "sum_R": g["R"].sum()}).reset_index()


def main(universes: list[str], workers: int):
    REPORTS.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(ROOT / "research/experiments.csv")
    dq_all, stats_rows, sel_all, hold_rows = [], [], [], []
    for name in universes:
        u = UNIVERSES[name]
        print(f"=== {name}: dev", flush=True)
        dev = dev_stage(name, workers, ledger)
        dq_all += dev["dq"] + [f"- ⚠ {x}" for x in dev["issues"][:20]] + [""]
        dev["summ"].to_csv(REPORTS / f"{name}_dev_grid.csv")
        dev["wf"].to_csv(REPORTS / f"{name}_walk_forward.csv", index=False)
        stats_rows.append(dict(universe=name, n_configs=dev["M"].shape[1], n_eff=dev["n_eff"],
                               best=dev["M"].columns[int(np.argmax((dev["M"].mean() / (dev["M"].std() + 1e-12)).to_numpy()))],
                               dsr=dev["dsr"]["dsr"], spa_p=dev["spa"]["pvalue"], rc_p=dev["rc"]["pvalue"],
                               pbo=dev["pbo"]["pbo"], fdr10=int(dev["summ"]["fdr10_discovery"].sum()),
                               errors=len(dev["errors"])))
        sel = select(dev)
        sel.insert(0, "universe", name)
        # стресс издержек на dev для всех финальных конфигураций семейств (информативно; обязательно для прошедших)
        streams_dev, specs, _ = load_universe(name, DEV_END)
        cfg_by_id = {c.id: c for c in dev["configs"]}
        for i, r in sel.iterrows():
            cfg = cfg_by_id[r["final_config"]]
            for sc in ("STRESS_1", "STRESS_2"):
                o = run_cfg(cfg, streams_dev, specs, u["rs"], sc)
                s = seg(o, DATA_START, DEV_END)
                sel.loc[i, f"{sc.lower()}_expectancy_R"] = s["expectancy_R"]
                sel.loc[i, f"{sc.lower()}_sharpe"] = s["sharpe"]
        sel["candidate"] = sel["passed"] & (sel["stress_1_expectancy_R"] > 0)
        sel.to_csv(REPORTS / f"{name}_selection.csv", index=False)
        sel_all.append(sel)
        # FINAL HOLDOUT: кандидаты + заранее заявленная база R-Breaker (параметры спецификации)
        to_hold = list(sel.loc[sel["candidate"], "final_config"])
        if name == "RB_FUT":
            base_id = [c.id for c in dev["configs"] if dict(c.params) == G.RB_BASELINE][0]
            to_hold = list(dict.fromkeys([base_id] + to_hold))
        if to_hold:
            print(f"=== {name}: FINAL HOLDOUT ({len(to_hold)} configs)", flush=True)
            streams_full, specs_full, _ = load_universe(name, HOLD_END)
            for cid in to_hold:
                cfg = cfg_by_id[cid]
                for sc in ("NORMAL", "STRESS_1"):
                    o = run_cfg(cfg, streams_full, specs_full, u["rs"], sc)
                    h = seg(o, HOLD_START, HOLD_END)
                    d = seg(o, DATA_START, DEV_END)
                    hold_rows.append(dict(universe=name, config=cid, family=cfg.family, scenario=sc,
                                          is_candidate=cid in set(sel.loc[sel["candidate"], "final_config"]),
                                          **{f"dev_{k}": v for k, v in d.items()},
                                          **{f"holdout_{k}": v for k, v in h.items()}))
                    if sc == "NORMAL":
                        tr = o["trades"]
                        tag = f"{name}_{cfg.family}"
                        tr.to_csv(REPORTS / f"{tag}_trades_full.csv", index=False)
                        eq = o["daily"].cumsum()
                        plots.line_chart({cfg.family: eq}, REPORTS / f"{tag}_equity.png",
                                         f"{name} {cfg.family}: накопленная доходность (риск 0,5%/сделку), holdout с {HOLD_START}",
                                         "доходность", pct=True)
                        if name == "RB_FUT" and dict(cfg.params) == G.RB_BASELINE:
                            rb_baseline_report(o, streams_full, specs_full, u["rs"], cfg)
            ledger.add(hypothesis=f"{name}: FINAL HOLDOUT", strategy=",".join(sorted({cfg_by_id[c].family for c in to_hold})),
                       parameters=json.dumps(to_hold)[:1000], data="T-Invest 1m", period=f"{HOLD_START}..{HOLD_END}",
                       result=json.dumps([{k: r[k] for k in ("family", "scenario", "holdout_sharpe", "holdout_expectancy_R",
                                                               "holdout_trades")} for r in hold_rows if r["universe"] == name],
                                         default=float)[:1500],
                       conclusion="см. reports/FINAL_RESEARCH_REPORT.md", next_step="—", seed="—",
                       execution_model="ConservativeL1", cost_model="NORMAL/STRESS_1")
        if name == "RB_FUT":
            rb_surface(dev)
    pd.DataFrame(stats_rows).to_csv(REPORTS / "multiple_testing.csv", index=False)
    if sel_all:
        pd.concat(sel_all).to_csv(REPORTS / "selection_all.csv", index=False)
    pd.DataFrame(hold_rows).to_csv(REPORTS / "holdout.csv", index=False)
    (REPORTS / "data_quality.md").write_text("# Качество данных (T-Invest, минутные свечи)\n\n" + "\n".join(dq_all),
                                             encoding="utf-8")


def rb_surface(dev: dict):
    """Поверхность чувствительности R-Breaker (dev): Sharpe по сетке 3x3x3 — проверка плато."""
    rows = []
    for cid, (cfg, out) in dev["res"].items():
        p = dict(cfg.params)
        m = trade_metrics(out["trades"])
        rows.append(dict(**p, sharpe=float(dev["summ"].loc[cid, "sharpe_ann_dev"]), trades=m.get("n_trades", 0),
                         expectancy_R=m.get("expectancy_R", np.nan), gross_R=m.get("gross_expectancy_R", np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "RB_FUT_sensitivity.csv", index=False)
    piv = df[df["reversal_k"] == 0.07].pivot_table(index="setup_k", columns="breakout_k", values="expectancy_R")
    plots.heat_surface(piv, REPORTS / "RB_FUT_sensitivity.png", "R-Breaker 2.1, dev: ожидание R (reversal_k=0,07)",
                       "breakout_k", "setup_k")


def rb_baseline_report(o, streams, specs, rs_kw, cfg):
    """Раздельная статистика базовой версии (спецификация §76–82) за весь год (параметры заданы заранее)."""
    tr = o["trades"].copy()
    if tr.empty:
        return
    tr["period"] = np.where(pd.to_datetime(tr["trading_day"]) >= pd.Timestamp(HOLD_START), "holdout", "dev")
    tr["regime"] = np.where(pd.to_datetime(tr["trading_day"]) >= ETS_DAY, "ETS", "pre-ETS")
    tr["spec_set"] = tr["code"].isin(SPEC_SET)
    tr["month"] = pd.to_datetime(tr["trading_day"]).dt.to_period("M").astype(str)
    parts = {"code": ["code"], "tag": ["tag"], "direction": ["direction"], "tag_direction": ["tag", "direction"],
             "period": ["period"], "regime": ["regime"], "spec_set": ["spec_set"], "month": ["month"],
             "exit_reason": ["exit_reason"]}
    out = []
    for k, by in parts.items():
        b = breakdown(tr, by)
        b.insert(0, "breakdown", k)
        b["key"] = b[by].astype(str).agg("|".join, axis=1)
        out.append(b[["breakdown", "key", "trades", "expectancy_R", "gross_R", "win_rate", "sum_R"]])
    pd.concat(out).to_csv(REPORTS / "RB_FUT_baseline_breakdown.csv", index=False)
    # издержки: ZERO / NORMAL / STRESS / комиссия 0,04% и 0,06%
    rows = []
    for sc, com in (("ZERO", 0.0), ("NORMAL", None), ("STRESS_1", None), ("STRESS_2", None), ("NORMAL", 0.0004),
                    ("NORMAL", 0.0006)):
        x = run_cfg(cfg, streams, specs, rs_kw, sc, com)
        m = trade_metrics(x["trades"])
        rows.append(dict(scenario=sc, commission=com if com is not None else rs_kw["commission_fraction"],
                         trades=m.get("n_trades", 0), expectancy_R=m.get("expectancy_R"),
                         gross_R=m.get("gross_expectancy_R"), sharpe=ann_sharpe(x["daily"]),
                         total_return=float(x["daily"].sum())))
    pd.DataFrame(rows).to_csv(REPORTS / "RB_FUT_baseline_costs.csv", index=False)
    # вклад лучших сделок
    R = tr["R"].sort_values(ascending=False).to_numpy()
    ex = [dict(removed=k, expectancy_R=float(R[k:].mean()) if len(R) > k else np.nan) for k in (0, 1, 5, 10, 20)]
    pd.DataFrame(ex).to_csv(REPORTS / "RB_FUT_baseline_remove_best.csv", index=False)
    # по инструментам, одиночный прогон — дневная доходность
    D = o.get("daily_by_instrument")
    if D is not None and len(D):
        plots.line_chart({c: D[c].cumsum() for c in D.columns}, REPORTS / "RB_FUT_baseline_by_instrument.png",
                         "R-Breaker 2.1 (база): накопленная доходность по инструментам, риск 0,5%/сделку",
                         "доходность", pct=True)


def synthetic():
    raw = generate_minute_bars("2021-01-04", "2021-12-30", seed=5, daily_vol=0.015, tick=1.0)
    spec = InstrumentSpec("SYN", "future", tick=1.0, tick_value=1.0, group="equity")
    bars = annotate_sessions(raw, SESSION_PRE2026, 1)
    exe = ConservativeL1ExecutionModel(CommissionModel(0.00025), SCENARIOS["NORMAL"])
    risk = RiskConfig(risk_fraction=0.005, risk_fraction_by_tag={"BREAKOUT": 0.005, "REVERSAL": 0.0035})
    res = Engine([InstrumentInput(spec.code, spec, bars, RBreaker(), 60)], execution=exe, risk=risk).run()
    rep = check_causality(raw, lambda: RBreaker({"min_median_volume": 0}), spec, SESSION_PRE2026, 1, n_cuts=5)
    print("SYNTHETIC (not market evidence):", json.dumps(trade_metrics(res.trades), default=str, indent=1))
    print(rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--universes", nargs="+", default=list(UNIVERSES))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.synthetic:
        synthetic()
    else:
        main(a.universes, a.workers)
