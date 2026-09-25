"""Отчётные таблицы и графики: reports/*.csv, reports/figures/*.png, research/experiments_v2.csv."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import data as D  # noqa: E402
from moexlab.deep import metrics as M  # noqa: E402
from moexlab.deep import risk as RK  # noqa: E402

ROOT = D.ROOT
FIN = ROOT / "research/final"
REP = ROOT / "reports"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "lines.linewidth": 2.0, "legend.frameon": False})
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
DV = D.data_version()
MAIN = ["C1_TREND_FUT_ENS", "C5_TREND_FUT_LONGONLY", "C7_WF_PANEL_SORTINO_K1", "C9_META_RP", "C3_XSMOM_EQ_ENS"]


def to_dt(days):
    return pd.to_datetime(pd.Series(np.asarray(days).astype(str)), format="%Y%m%d").to_numpy()


def shade_holdout(ax):
    ax.axvspan(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-09-24"), color="#eda100", alpha=0.10, lw=0)
    ax.text(pd.Timestamp("2026-07-03"), ax.get_ylim()[1], "FINAL HOLDOUT", va="top", fontsize=8, color=INK2)


def pf(x):
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return float(g / l) if l > 0 else np.inf


def metrics_row(r, days):
    mk = M.month_keys(days)
    live = np.cumsum(r != 0) > 0
    months = np.unique(mk[live])
    mret = np.array([r[mk == m].sum() for m in months])
    sd = r[live].std()
    neg = np.sqrt((np.minimum(r[live], 0) ** 2).mean())
    mdd = float(M.max_dd(r[None])[0])
    ann = float(r[live].mean() * 252)
    return dict(avg_monthly_return=float(mret.mean()), median_monthly_return=float(np.median(mret)),
                min_monthly_return=float(mret.min()), max_monthly_return=float(mret.max()),
                monthly_return_std=float(mret.std()), profitable_month_ratio=float((mret > 0).mean()),
                longest_losing_month_streak=int(M._longest_losing(mret[None])[0]), annual_return=ann, max_drawdown=mdd,
                profit_factor=pf(r[live]), sharpe=float(r[live].mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                sortino=float(r[live].mean() / neg * np.sqrt(252)) if neg > 0 else 0.0,
                calmar=float(ann / mdd) if mdd > 0 else 0.0, months=len(months))


def rankings():
    daily = pd.read_parquet(FIN / "candidates_daily_vt10.parquet")
    summ = pd.read_csv(FIN / "candidates_summary.csv").set_index("candidate")
    days = daily.index.to_numpy()
    rows = []
    for c in daily.columns:
        r = daily[c].to_numpy()
        m = metrics_row(r, days)
        wf = summ.loc[c]
        rows.append(dict(strategy_id=c, family="candidate", instruments="FUT" if "FUT" in c or "FX" in c else "ALL/EQ",
                         timeframe="1d", trades=int((r != 0).sum()), **m,
                         walk_forward_result=f"OOS 2026-01..06: Sharpe {wf['wf_sharpe']:.2f}, {wf['wf_total']:+.1%}, прибыльных мес. {wf['wf_pmr']:.0%}",
                         holdout_result=f"2026-07..09: {wf['hold_total']:+.1%}, прибыльных мес. {wf['hold_pmr']:.0%}",
                         evidence="frozen candidate, vol-target 10%"))
    c10 = json.loads((ROOT / "research/ml/ml_hourly_full_H16_PREMIUM.json").read_text())["results"]["0.2"]["monthly"]
    mv = np.array(list(c10.values()))
    rows.append(dict(strategy_id="C10_ML_HOURLY_H16_PREMIUM", family="ml", instruments="ALL", timeframe="60m",
                     trades=np.nan, avg_monthly_return=mv.mean(), median_monthly_return=np.median(mv), min_monthly_return=mv.min(),
                     max_monthly_return=mv.max(), monthly_return_std=mv.std(), profitable_month_ratio=(mv > 0).mean(),
                     annual_return=mv.mean() * 12, max_drawdown=np.nan, profit_factor=np.nan, sharpe=np.nan, sortino=np.nan,
                     calmar=np.nan, months=len(mv), walk_forward_result="OOS 2026-01..06: Sharpe 1.27 (порог выбран на тех же месяцах)",
                     holdout_result=f"2026-07..09: {sum(mv[-3:]):+.1%}, 0/3 прибыльных", evidence="frozen candidate, 1× номинал"))
    # библиотека портфелей (полный год, без таргета — сырые веса ~1%/день на инструмент)
    lib = pd.read_parquet(D.CACHE / "screen/p1full/PANEL_1440.parquet")
    R = np.load(D.CACHE / "screen/p1full/PANEL_1440.npy").astype(float)
    for i, row in lib.iterrows():
        r = R[i]
        m = metrics_row(r, days)
        mk = M.month_keys(days)
        wfr = r[(mk >= 202601) & (mk <= 202606)]
        hr = r[mk >= 202607]
        rows.append(dict(strategy_id=f"LIB_{i:03d}", family=row["family"], instruments=row["code"], timeframe="1d",
                         trades=int(row["trades"]), **m,
                         walk_forward_result=f"(in-sample конфигурация) 2026-01..06: {wfr.sum():+.1%}",
                         holdout_result=f"2026-07..09: {hr.sum():+.1%}", evidence=f"library {row['params']}"))
    df = pd.DataFrame(rows)
    # композитная оценка стабильности (штраф за убыточные месяцы), нормировка к волатильности
    msd = df["monthly_return_std"].replace(0, np.nan)
    df["stability_score"] = (df["avg_monthly_return"] + 0.5 * df["median_monthly_return"] + 0.5 * df["min_monthly_return"]
                             - 0.5 * msd) / msd - 2.0 * (1 - df["profitable_month_ratio"])
    df = df.sort_values(["evidence", "stability_score"], ascending=[True, False])
    cols = ["rank", "strategy_id", "family", "instruments", "timeframe", "avg_monthly_return", "median_monthly_return",
            "min_monthly_return", "profitable_month_ratio", "annual_return", "max_drawdown", "profit_factor", "sharpe",
            "sortino", "calmar", "trades", "walk_forward_result", "holdout_result", "evidence"]
    a = df.sort_values("stability_score", ascending=False).reset_index(drop=True)
    a["rank"] = np.arange(1, len(a) + 1)
    a[cols].to_csv(REP / "strategy_ranking.csv", index=False)
    s = df.sort_values(["profitable_month_ratio", "min_monthly_return"], ascending=False).reset_index(drop=True)
    s["rank"] = np.arange(1, len(s) + 1)
    s[["rank", "strategy_id", "family", "instruments", "profitable_month_ratio", "min_monthly_return",
       "median_monthly_return", "monthly_return_std", "max_drawdown", "longest_losing_month_streak",
       "walk_forward_result", "holdout_result", "evidence"]].to_csv(REP / "stability_ranking.csv", index=False)
    r_ = df.sort_values("annual_return", ascending=False).reset_index(drop=True)
    r_["rank"] = np.arange(1, len(r_) + 1)
    r_[["rank", "strategy_id", "family", "instruments", "annual_return", "avg_monthly_return", "max_drawdown",
        "min_monthly_return", "sharpe", "profitable_month_ratio", "walk_forward_result", "holdout_result",
        "evidence"]].to_csv(REP / "return_ranking.csv", index=False)
    # помесячная таблица кандидатов (§36): год, месяц, капитал, доходность, сделки, win, PF, DD
    out = []
    for c in daily.columns:
        t = M.monthly_table(days, daily[c].to_numpy())
        t.insert(0, "strategy", c)
        g = pd.DataFrame({"r": daily[c].to_numpy(), "ym": M.month_keys(days)})
        t["active_days"] = g.groupby("ym")["r"].apply(lambda x: int((x != 0).sum())).to_numpy()
        t["win_rate_days"] = g.groupby("ym")["r"].apply(lambda x: float((x > 0).sum() / max(1, (x != 0).sum()))).to_numpy()
        t["profit_factor_days"] = g.groupby("ym")["r"].apply(lambda x: pf(x.to_numpy())).to_numpy()
        out.append(t)
    pd.concat(out).to_csv(REP / "monthly_metrics_candidates.csv", index=False)
    return df


def figures():
    daily = pd.read_parquet(FIN / "candidates_daily_vt10.parquet")
    days = daily.index.to_numpy()
    t = to_dt(days)
    # 1-2 equity
    for name, log in (("equity_curve.png", False), ("log_equity_curve.png", True)):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, c in enumerate(MAIN):
            eq = RK.equity_path(daily[c].to_numpy(), 1.0)
            ax.plot(t, eq, color=SERIES[i], label=c, lw=2.0 if i == 0 else 1.4)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel("капитал (старт = 1), таргет 10% годовых")
        ax.set_title("Кривые капитала финальных кандидатов" + (" (лог-шкала)" if log else ""), loc="left")
        ax.legend(loc="upper left", fontsize=8)
        shade_holdout(ax)
        fig.tight_layout()
        fig.savefig(FIG / name, dpi=130)
        plt.close(fig)
    # 3 drawdown
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, c in enumerate(MAIN[:3]):
        eq = RK.equity_path(daily[c].to_numpy(), 1.0)
        dd = eq / np.maximum.accumulate(eq) - 1
        ax.plot(t, 100 * dd, color=SERIES[i], label=c)
    ax.set_ylabel("просадка, %")
    ax.set_title("Просадки (таргет волатильности 10%)", loc="left")
    ax.legend(fontsize=8)
    shade_holdout(ax)
    fig.tight_layout()
    fig.savefig(FIG / "drawdown.png", dpi=130)
    plt.close(fig)
    # 4 heatmap
    mon = pd.read_csv(FIN / "candidates_monthly_vt10.csv", index_col=0)
    mon = mon.loc[mon.index >= 202511]
    fig, ax = plt.subplots(figsize=(11, 5))
    v = 100 * mon.T.to_numpy()
    lim = np.nanmax(np.abs(v))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3f2ee", "#2a78d6"])
    im = ax.imshow(v, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(mon.index)), [f"{m // 100}-{m % 100:02d}" for m in mon.index], rotation=45, fontsize=8)
    ax.set_yticks(range(len(mon.columns)), mon.columns, fontsize=8)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:+.1f}", ha="center", va="center", fontsize=7, color=INK)
    ax.grid(False)
    ax.axvline(len(mon.index) - 3.5, color=INK2, lw=1)
    ax.set_title("Месячная доходность, % (справа от линии — FINAL HOLDOUT)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "monthly_returns_heatmap.png", dpi=130)
    plt.close(fig)
    # 5 distribution: месяцы всех 423 библиотечных стратегий vs C1
    lib = pd.read_parquet(D.CACHE / "screen/p1full/PANEL_1440.parquet")
    mcols = [c for c in lib.columns if c.startswith("m20")]
    allm = 100 * lib[mcols].to_numpy().ravel()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(np.clip(allm, -30, 30), bins=60, color=SERIES[0], alpha=0.8, label="все месяцы 423 портфельных стратегий (сырые веса)")
    for x in 100 * mon["C1_TREND_FUT_ENS"].to_numpy():
        ax.axvline(x, color=SERIES[1], lw=1.5)
    ax.plot([], [], color=SERIES[1], label="месяцы C1 (таргет 10%)")
    ax.set_xlabel("месячная доходность, %")
    ax.set_title("Распределение месячных доходностей", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "monthly_returns_distribution.png", dpi=130)
    plt.close(fig)
    # 6-7 rolling
    r = pd.Series(daily["C1_TREND_FUT_ENS"].to_numpy(), index=t)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, 100 * r.rolling(63).sum(), color=SERIES[0], label="C1: доходность за 63 дня")
    ax.plot(t, 100 * r.rolling(21).sum(), color=SERIES[1], lw=1.2, label="C1: доходность за 21 день")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("%")
    ax.set_title("Скользящая доходность C1", loc="left")
    ax.legend(fontsize=8)
    shade_holdout(ax)
    fig.tight_layout()
    fig.savefig(FIG / "rolling_returns.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, c in enumerate(MAIN[:3]):
        x = pd.Series(daily[c].to_numpy(), index=t)
        rs = x.rolling(63).mean() / x.rolling(63).std() * np.sqrt(252)
        ax.plot(t, rs, color=SERIES[i], label=c)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_title("Скользящий Sharpe (63 дня)", loc="left")
    ax.legend(fontsize=8)
    shade_holdout(ax)
    fig.tight_layout()
    fig.savefig(FIG / "rolling_sharpe.png", dpi=130)
    plt.close(fig)
    # 8 comparison
    s = pd.read_csv(FIN / "candidates_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(s))
    for k, (col, lab) in enumerate((("dev_sharpe", "разработка"), ("wf_sharpe", "walk-forward 2026-01..06"),
                                    ("hold_sharpe", "holdout 2026-07..09"))):
        ax.barh(y + (k - 1) * 0.27, s[col], height=0.25, color=SERIES[k], label=lab)
    ax.set_yticks(y, s["candidate"], fontsize=8)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Sharpe (годовой)")
    ax.set_title("Сравнение кандидатов по периодам", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "strategy_comparison.png", dpi=130)
    plt.close(fig)
    # 9 risk-return: сетка плеча C1 + кандидаты
    lg = pd.read_csv(FIN / "leverage_grid.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(MAIN[:3]):
        g = lg[lg.candidate == c]
        ax.plot(100 * g["mc_mdd_median"], 100 * g["mc_monthly_median"], "-o", ms=5, color=SERIES[i], label=c)
    ax.set_xlabel("медианная макс. просадка за 12 мес. (Monte Carlo), %")
    ax.set_ylabel("медианная месячная доходность (Monte Carlo), %")
    ax.set_title("Доходность ↔ риск при разном плече (5…150% годовой волатильности)", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "risk_return_scatter.png", dpi=130)
    plt.close(fig)
    # 10 pareto: библиотека (месячная доходность / просадка / стабильность)
    rk = pd.read_csv(REP / "strategy_ranking.csv")
    lb = rk[rk["strategy_id"].str.startswith("LIB")]
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(100 * lb["max_drawdown"], 100 * lb["avg_monthly_return"], c=lb["profitable_month_ratio"], cmap="Blues",
                    s=14, edgecolors="none", vmin=0, vmax=1)
    pts = lb[["max_drawdown", "avg_monthly_return", "profitable_month_ratio"]].to_numpy()
    dom = np.array([not any((q[0] <= p[0]) & (q[1] >= p[1]) & (q[2] >= p[2]) & (q != p).any() for q in pts) for p in pts])
    pl = lb[dom].sort_values("max_drawdown")
    ax.plot(100 * pl["max_drawdown"], 100 * pl["avg_monthly_return"], "o", mfc="none", mec=SERIES[1], ms=8,
            label=f"Парето-фронт ({dom.sum()} стратегий)")
    ax.set_xlabel("макс. просадка за год, %")
    ax.set_ylabel("средняя месячная доходность, %")
    ax.set_title("Парето: доходность / просадка / доля прибыльных месяцев (цвет)", loc="left")
    fig.colorbar(sc, ax=ax, label="доля прибыльных месяцев")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pareto_front.png", dpi=130)
    plt.close(fig)
    pl.to_csv(REP / "pareto_front.csv", index=False)
    # 11 monte carlo fan (C1, таргет 10%)
    r1 = daily["C1_TREND_FUT_ENS"].to_numpy()
    r1 = r1[np.cumsum(r1 != 0) > 0]
    rng = np.random.default_rng(20260925)
    paths = np.empty((10000, 252))
    for p in range(10000):
        idx = []
        while len(idx) < 252:
            s0 = rng.integers(0, len(r1))
            L = rng.geometric(1 / 5)
            idx.extend(((s0 + np.arange(L)) % len(r1)).tolist())
        paths[p] = np.cumprod(1 + r1[idx[:252]])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(252)
    for lo, hi, a in ((5, 95, 0.15), (25, 75, 0.3)):
        ax.fill_between(x, np.percentile(paths, lo, 0), np.percentile(paths, hi, 0), color=SERIES[0], alpha=a, lw=0,
                        label=f"{lo}–{hi} перцентили")
    ax.plot(x, np.median(paths, 0), color=SERIES[0], label="медиана")
    ax.axhline(1, color=INK2, lw=1)
    ax.set_xlabel("торговых дней")
    ax.set_ylabel("капитал")
    ax.set_title("Monte Carlo C1 (блочный бутстреп дней, 10 000 путей для графика; в таблицах 50 000)", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "monte_carlo.png", dpi=130)
    plt.close(fig)
    # 12 parameter stability: Sharpe конфигураций трендового ансамбля по окну и vn
    L = pd.read_parquet(D.CACHE / "screen/p1full/PANEL_1440.parquet")
    L = L[(L["code"] == "FUT") & L["family"].isin(["P_TSMOM", "P_EMA", "P_BRK"])].copy()
    L["win"] = L["params"].map(lambda p: json.loads(p).get("n", json.loads(p).get("s")))
    L["vn"] = L["params"].map(lambda p: json.loads(p)["vn"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (fam, g) in enumerate(L.groupby("family")):
        for j, (vn, h) in enumerate(g.groupby("vn")):
            h = h.sort_values("win")
            ax.plot(h["win"], h["sharpe"], "-o", ms=5, color=SERIES[i], alpha=1.0 if vn == 60 else 0.55,
                    label=f"{fam}, vn={vn}")
    ax.set_xscale("log")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("окно тренда, дней (лог)")
    ax.set_ylabel("Sharpe за год (сырые веса)")
    ax.set_title("Устойчивость параметров трендового ансамбля на фьючерсах", loc="left")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "parameter_stability.png", dpi=130)
    plt.close(fig)


def experiments_journal():
    rows = []
    eid = 0

    def add(**kw):
        nonlocal eid
        eid += 1
        base = dict(experiment_id=f"V2-{eid:05d}", git_commit=COMMIT, data_version=DV, seed=20260925)
        base.update(kw)
        rows.append(base)
    for tag, stage, cost in (("s1", "Stage1 screening", "TRADER fut0.04%/eq0.05% +½spread+1tick"),
                             ("x1", "Stage1 cross-asset", "TRADER"), ("s2eq", "Stage1 equities overnight", "TRADER"),
                             ("g1", "Stage1 gross (zero cost) diagnostic", "ZERO")):
        f = ROOT / f"research/stage1/{tag}_all_configs.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        for (fam, tf, code), g in d.groupby(["family", "tf", "code"]):
            a = g[g["trades"] >= 10]
            best = a.sort_values("sharpe").iloc[-1] if len(a) else g.iloc[0]
            add(stage=stage, strategy_family=fam, strategy=f"{len(g)} configs (params×exits×dir×eod)", instrument=code,
                timeframe=str(tf), parameters=f"best IS: {best['params']} exit={best['exit']} dir={best['direction']}",
                train_period="2025-09-25..2026-06-30 (in-sample)", validation_period="see WF",
                trades=int(g["trades"].median()), **{"return": float(g["total"].median())},
                monthly_return=float(g["m_mean"].median()), profitable_month_ratio=float(g["pmr"].median()),
                max_drawdown=float(g["mdd"].median()), profit_factor=float(g["pf"].replace(np.inf, np.nan).median()),
                result=f"median Sharpe {g['sharpe'].median():.2f}; share positive {(a['total'] > 0).mean() if len(a) else 0:.1%}; best IS Sharpe {best['sharpe']:.2f}",
                cost_model=cost, execution_model="next-bar open, conservative OHLC")
    for tag in ("wf1", "wfp1"):
        d = pd.read_csv(ROOT / f"research/wf/{tag}_procedures.csv")
        for _, r in d.iterrows():
            add(stage="Stage3 walk-forward", strategy_family="WF procedure", strategy=r["procedure"],
                instrument="pool " + ("1.08M intraday/daily configs" if tag == "wf1" else "423 panel portfolios"),
                timeframe="mixed" if tag == "wf1" else "1d", parameters=r["procedure"],
                train_period="expanding from 2025-09-25", validation_period="2026-01..2026-06 monthly",
                trades=np.nan, **{"return": r["total"]}, monthly_return=r["m_mean"], profitable_month_ratio=r["pmr"],
                max_drawdown=r["mdd"], profit_factor=np.nan, result=f"OOS Sharpe {r['sharpe']:.2f}",
                cost_model="TRADER (+short fin. 0.068%/day for panel)", execution_model="as pool")
    s = pd.read_csv(FIN / "candidates_summary.csv")
    for _, r in s.iterrows():
        add(stage="Stage5 final holdout", strategy_family="candidate", strategy=r["candidate"], instrument="panel",
            timeframe="1d", parameters=r["desc"], train_period="2025-09-25..2026-06-30",
            validation_period="holdout 2026-07-01..2026-09-24", trades=np.nan, **{"return": r["hold_total"]},
            monthly_return=r["hold_m_mean"], profitable_month_ratio=r["hold_pmr"], max_drawdown=r["hold_mdd"],
            profit_factor=np.nan, result=f"WF Sharpe {r['wf_sharpe']:.2f}; holdout Sharpe {r['hold_sharpe']:.2f}; year {r['year_total']:+.1%}",
            cost_model="TRADER + fin", execution_model="close t -> open t+1, vol target 10%")
    for f in sorted((ROOT / "research/ml").glob("ml_*.json")):
        d = json.loads(f.read_text())
        for k, v in d["results"].items():
            add(stage="ML", strategy_family="ML LightGBM", strategy=f"{f.stem} {k}", instrument="21",
                timeframe="1d" if "panel" in f.stem else "60m", parameters=k, train_period="expanding monthly",
                validation_period=",".join(map(str, v["monthly"].keys())), trades=np.nan, **{"return": v["total"]},
                monthly_return=float(np.mean(list(v["monthly"].values()))),
                profitable_month_ratio=float(np.mean([x > 0 for x in v["monthly"].values()])), max_drawdown=np.nan,
                profit_factor=np.nan, result=f"Sharpe {v['sharpe']:.2f}", cost_model="per file label",
                execution_model="engine")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "research/experiments_v2.csv", index=False)
    return len(df)


if __name__ == "__main__":
    rankings()
    figures()
    n = experiments_journal()
    print("journal rows", n)
