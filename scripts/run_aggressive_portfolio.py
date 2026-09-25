"""Задача 2 (план v3): стратегия B — максимум роста капитала на фьючерсах при бюджете риска
P(просадка > 25% за 12 мес.) <= 5%, остановка при -30%. Стратегия A (EMA-cross, 0,5%) не меняется.

Данные читаются только из БД в режиме чтения (через кэш run_intraday_research.load_universe); запись в БД нет.
  python3 scripts/run_aggressive_portfolio.py
Результаты: reports/aggressive/*, журнал research/experiments.csv.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_intraday_research as Y  # noqa: E402
from moexlab.backtest.engine import Engine, InstrumentInput, RiskConfig  # noqa: E402
from moexlab.execution.models import SCENARIOS, CommissionModel, ConservativeL1ExecutionModel  # noqa: E402
from moexlab.reporting import plots  # noqa: E402
from moexlab.research.runner import Ledger  # noqa: E402
from moexlab.statistics.inference import stationary_bootstrap_indices  # noqa: E402
from moexlab.storage.db import open_db  # noqa: E402

OUT = ROOT / "reports/aggressive"
BASE_RISK = 0.005
DD_LIMIT, P_LIMIT, HALT_DD, REDUCE_DD = 0.25, 0.05, 0.30, 0.15
DAILY_LOSS, WEEKLY_LOSS, MAX_MARGIN = 0.05, 0.10, 0.60
HAIRCUT = 0.5
HORIZON, N_SIMS, BLOCK = 252, 4000, 10.0
CAPITALS = (300_000, 1_000_000, 3_000_000, 10_000_000)


# ------------------------------------------------------------------------------------------ единичные доходности
def candidates() -> dict:
    sel = pd.read_csv(ROOT / "reports/year/DAY_FUT_selection.csv")
    ids = sel.loc[sel["candidate"], ["family", "final_config"]]
    cfgs = {c.id: c for c in Y.G.day_grid()}
    return {r.family: cfgs[r.final_config] for r in ids.itertuples()}


def unit_returns(cands: dict, streams, specs) -> pd.DataFrame:
    """Дневные доходности каждого кандидата при риске 0,5%/сделку (без сложного процента), dev."""
    cols = {}
    for fam, cfg in cands.items():
        o = Y.run_cfg(cfg, streams, specs, Y.UNIVERSES["DAY_FUT"]["rs"])
        cols[fam] = o["daily"]
    D = pd.DataFrame(cols).fillna(0.0).sort_index()
    return D[D.index <= pd.Timestamp(Y.DEV_END, tz="UTC")]


def constructions(D: pd.DataFrame) -> dict[str, pd.Series]:
    inv = 1 / D.std()
    w = inv / inv.sum()
    return {"A_ema_only": D["TF_EMA_CROSS"],
            "B_equal_risk": D.mean(axis=1),
            "C_inverse_vol": (D * w).sum(axis=1)}, w


# ------------------------------------------------------------------------------------------ бутстреп
def boot_paths(r: np.ndarray, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = len(r)
    idx = np.empty((N_SIMS, HORIZON), dtype=int)
    for k in range(N_SIMS):
        ii = stationary_bootstrap_indices(T, BLOCK, rng)
        while len(ii) < HORIZON:
            ii = np.concatenate([ii, stationary_bootstrap_indices(T, BLOCK, rng)])
        idx[k] = ii[:HORIZON]
    return r[idx]


def sim(paths: np.ndarray, s: float) -> dict:
    """Сложный процент, риск s x 0,5%; правило «половина риска при просадке 15%» и остановка на 30% включены."""
    eq = np.ones(paths.shape[0])
    peak = eq.copy()
    mdd = np.zeros_like(eq)
    alive = np.ones_like(eq, dtype=bool)
    monthly = []
    m0 = eq.copy()
    for t in range(paths.shape[1]):
        dd = eq / peak - 1
        mult = np.where(dd <= -REDUCE_DD, 0.5, 1.0) * alive
        eq = eq * np.maximum(1 + s * mult * paths[:, t], 1e-9)
        peak = np.maximum(peak, eq)
        dd = eq / peak - 1
        mdd = np.minimum(mdd, dd)
        alive &= dd > -HALT_DD
        if (t + 1) % 21 == 0:
            monthly.append(eq / m0 - 1)
            m0 = eq.copy()
    monthly = np.array(monthly)
    return dict(scale=s, risk_per_trade=s * BASE_RISK, growth_12m_p50=float(np.median(eq) - 1),
                growth_12m_p05=float(np.quantile(eq, 0.05) - 1), growth_12m_p95=float(np.quantile(eq, 0.95) - 1),
                monthly_p50=float(np.median((np.median(eq)) ** (1 / 12) - 1)),
                p_dd_gt_25=float((mdd <= -DD_LIMIT).mean()), p_halt_30=float((~alive).mean()),
                maxdd_p50=float(np.median(mdd)), maxdd_p95=float(np.quantile(mdd, 0.05)),
                p_loss_12m=float((eq < 1).mean()), worst_month_p50=float(np.median(monthly.min(axis=0))))


def growth(r: np.ndarray, s: float) -> float:
    x = 1 + s * r
    return float(np.log(x).mean()) if (x > 0).all() else -np.inf


def kelly(r: np.ndarray) -> float:
    if r.mean() <= 0:
        return 0.0
    g = np.linspace(0.05, 60, 1200)
    return float(g[int(np.argmax([growth(r, s) for s in g]))])


def size(r: pd.Series, haircut: float) -> tuple[pd.DataFrame, float, float]:
    x = r.to_numpy(float)
    x = x - (1 - haircut) * x.mean() if haircut < 1 else x   # среднее уменьшено до haircut x исходного
    k = kelly(x)
    paths = boot_paths(x)
    grid = [s for s in np.round(np.arange(0.5, 20.01, 0.5), 2)]
    rows = [sim(paths, s) for s in grid]
    df = pd.DataFrame(rows)
    df["half_kelly"] = k / 2
    ok = df[(df["p_dd_gt_25"] <= P_LIMIT) & (df["scale"] <= k / 2 + 1e-9)]
    s_star = float(ok.loc[ok["growth_12m_p50"].idxmax(), "scale"]) if len(ok) else 0.0
    return df, s_star, k


# ------------------------------------------------------------------------------------------ движок
def margin_fractions(streams) -> dict[str, float]:
    """ГО / номинал по текущим ставкам T-Invest для последней серии в потоке."""
    db = open_db(read_only=True)
    ins = db.instruments(kind="future").set_index("ticker")
    db.close()
    out = {}
    for code, df in streams.items():
        sec = df["secid"].iloc[-1]
        r = ins.loc[sec]
        meta = json.loads(r["meta"])
        go = max(meta.get("initial_margin_buy") or 0, meta.get("initial_margin_sell") or 0)
        notional = float(df["close"].iloc[-1]) * r["tick_value"] / r["tick"]
        out[code] = float(go / notional) if go and notional > 0 else 0.15
    return out


def run_engine(cands: dict, fams: list[str], weights: dict, s: float, streams, specs, capital: float,
               scenario: str = "NORMAL") -> dict:
    mf = margin_fractions(streams)
    inputs, groups = [], {}
    by_tag = {}
    for fam in fams:
        cfg = cands[fam]
        for code, df in streams.items():
            key = f"{fam}:{code}"
            sp = replace(specs[code], code=key, margin_fraction=mf[code])
            st = cfg.build()
            tag_risk = s * BASE_RISK * weights[fam]
            st.name = key                                   # тег сделки = семейство; риск задаётся по тегу
            inputs.append(InstrumentInput(key, sp, df, _Tagged(st, fam), 86400.0))
            groups[key] = specs[code].group
            by_tag[fam] = tag_risk
    risk = RiskConfig(risk_fraction=s * BASE_RISK, risk_fraction_by_tag=by_tag, compounding=True, integer_qty=True,
                      max_margin_fraction=MAX_MARGIN, portfolio_max_open_risk=None, group_max_open_risk=None,
                      daily_loss_fraction=DAILY_LOSS, weekly_loss_fraction=WEEKLY_LOSS,
                      reduce_risk_drawdown=REDUCE_DD, halt_drawdown=HALT_DD,
                      max_entries_per_instrument_per_day=None, max_participation=0.10)
    exe = ConservativeL1ExecutionModel(CommissionModel(0.00025), SCENARIOS[scenario])
    res = Engine(inputs, execution=exe, risk=risk, initial_equity=capital, groups=groups, record_events=True).run()
    eq = res.daily_equity.copy()
    eq.index = pd.DatetimeIndex(pd.to_datetime(eq.index)).tz_localize("UTC")
    return dict(trades=res.trades, equity=eq, events=res.events)


class _Tagged:
    """Обёртка: все заявки стратегии помечаются тегом семейства (для риска по тегу), остальное — как у стратегии."""

    def __init__(self, st, tag):
        self._st, self._tag = st, tag
        self.name, self.family, self.exits = st.name, st.family, st.exits

    def __getattr__(self, a):
        return getattr(self._st, a)

    def on_bar_close(self, i, ctx):
        from moexlab.strategies.base import EntryOrder
        return [EntryOrder(o.direction, o.kind, o.price, self._tag, o.structural_stop, o.signal_id)
                for o in self._st.on_bar_close(i, ctx)]

    def exits_for(self, tag):
        return self._st.exits_for(tag)


def seg_stats(eq: pd.Series, start: str, end: str) -> dict:
    e = eq[(eq.index >= pd.Timestamp(start, tz="UTC")) & (eq.index <= pd.Timestamp(end, tz="UTC"))]
    prev = eq[eq.index < pd.Timestamp(start, tz="UTC")]
    base = float(prev.iloc[-1]) if len(prev) else float(e.iloc[0])
    e0 = pd.concat([pd.Series([base], index=[e.index[0] - pd.Timedelta(days=1)]), e])
    months = max(len(e) / 21.0, 1e-9)
    tot = float(e0.iloc[-1] / base - 1)
    r = e0.pct_change().dropna()
    return dict(return_total=tot, return_monthly_geo=float((1 + tot) ** (1 / months) - 1) if tot > -1 else -1.0,
                maxdd=float((e0 / e0.cummax() - 1).min()), sharpe=float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)),
                days=len(e))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates()
    streams_dev, specs, _ = Y.load_universe("DAY_FUT", Y.DEV_END)
    D = unit_returns(cands, streams_dev, specs)
    D.to_csv(OUT / "unit_returns_dev.csv")
    cons, w_inv = constructions(D)
    rows, curves, best = [], {}, None
    for name, r in cons.items():
        for hc in (HAIRCUT, 1.0):
            df, s_star, k = size(r, hc)
            df.insert(0, "construction", name)
            df.insert(1, "haircut", hc)
            curves[(name, hc)] = df
            pick = df[df["scale"] == s_star].iloc[0].to_dict() if s_star > 0 else dict(scale=0.0)
            rows.append(dict(construction=name, haircut=hc, dev_sharpe=Y.ann_sharpe(r), kelly=k, chosen_scale=s_star,
                             **{k2: pick.get(k2) for k2 in ("risk_per_trade", "growth_12m_p50", "growth_12m_p05",
                                                            "monthly_p50", "p_dd_gt_25", "p_halt_30", "maxdd_p50",
                                                            "maxdd_p95", "p_loss_12m")}))
            if hc == HAIRCUT and s_star > 0 and (best is None or pick["growth_12m_p50"] > best[2]["growth_12m_p50"]):
                best = (name, s_star, pick)
    pd.concat(curves.values()).to_csv(OUT / "sizing_curves.csv", index=False)
    choice = pd.DataFrame(rows)
    choice.to_csv(OUT / "sizing_choice.csv", index=False)
    print(choice.to_string())
    if best is None:
        print("no feasible construction")
        return
    name, s_star, _ = best
    fams = {"A_ema_only": ["TF_EMA_CROSS"], "B_equal_risk": list(D.columns), "C_inverse_vol": list(D.columns)}[name]
    weights = {"A_ema_only": {"TF_EMA_CROSS": 1.0},
               "B_equal_risk": {f: 1 / len(D.columns) for f in D.columns},
               "C_inverse_vol": w_inv.to_dict()}[name]
    # проверка в движке: весь год (dev + повторный просмотр holdout), сложный процент, целые контракты, реальное ГО
    streams_full, specs_full, _ = Y.load_universe("DAY_FUT", Y.HOLD_END)
    val, eqs = [], {}
    for cap in CAPITALS:
        for sc in ("NORMAL", "STRESS_1"):
            if cap != 1_000_000 and sc != "NORMAL":
                continue
            o = run_engine(cands, fams, weights, s_star, streams_full, specs_full, cap, sc)
            for seg, (a, b) in {"dev": (Y.DATA_START, Y.DEV_END), "holdout": (Y.HOLD_START, Y.HOLD_END),
                                "full": (Y.DATA_START, Y.HOLD_END)}.items():
                val.append(dict(strategy="B", construction=name, scale=s_star, capital=cap, scenario=sc, segment=seg,
                                trades=int(len(o["trades"])),
                                skipped_qty=sum(e.get("event") == "SKIP_QTY" for e in o["events"]),
                                **seg_stats(o["equity"], a, b)))
            if sc == "NORMAL":
                eqs[f"B, {cap:,.0f} ₽".replace(",", " ")] = o["equity"] / cap - 1
                if cap == 1_000_000:
                    o["trades"].to_csv(OUT / "strategy_B_trades.csv", index=False)
    # стратегия A для сравнения (как в задаче 1: 0,5%/сделку), тем же движком со сложным процентом и целыми контрактами
    for cap in CAPITALS:
        oa = run_engine(cands, ["TF_EMA_CROSS"], {"TF_EMA_CROSS": 1.0}, 1.0, streams_full, specs_full, cap)
        for seg, (a, b) in {"dev": (Y.DATA_START, Y.DEV_END), "holdout": (Y.HOLD_START, Y.HOLD_END),
                            "full": (Y.DATA_START, Y.HOLD_END)}.items():
            val.append(dict(strategy="A", construction="A_ema_only", scale=1.0, capital=cap, scenario="NORMAL",
                            segment=seg, trades=int(len(oa["trades"])),
                            skipped_qty=sum(e.get("event") == "SKIP_QTY" for e in oa["events"]),
                            **seg_stats(oa["equity"], a, b)))
        if cap == 3_000_000:
            eqs["A (EMA-cross, 0,5%), 3 000 000 ₽"] = oa["equity"] / cap - 1
    V = pd.DataFrame(val)
    V.to_csv(OUT / "validation.csv", index=False)
    print(V.to_string())
    plots.line_chart(eqs, OUT / "equity_A_vs_B.png",
                     f"Стратегии A и B: капитал со сложным процентом; holdout с {Y.HOLD_START} (повторный просмотр)",
                     "доходность", pct=True)
    dd = {k: (1 + v) / (1 + v).cummax() - 1 for k, v in eqs.items()}
    plots.drawdown_chart(dd, OUT / "drawdown_A_vs_B.png", "Просадка от максимума")
    Ledger(ROOT / "research/experiments.csv").add(
        hypothesis="Задача 2: стратегия B — максимум роста при P(DD>25%/12 мес) <= 5%, остановка на 30%",
        strategy=f"B={name}", parameters=json.dumps(dict(scale=s_star, risk_per_trade=s_star * BASE_RISK,
                                                         weights=weights, haircut=HAIRCUT)),
        data="T-Invest DAY_FUT (из минуток)", period=f"{Y.DATA_START}..{Y.HOLD_END}",
        result=json.dumps(V[V["capital"] == 1_000_000].round(4).to_dict("records"), default=float)[:1500],
        conclusion="см. reports/FINAL_RESEARCH_REPORT.md §11", next_step="SHADOW", seed="7",
        execution_model="ConservativeL1, compounding, integer contracts, real GO", cost_model="NORMAL/STRESS_1")


if __name__ == "__main__":
    main()
