"""RED TEAM для C1_TREND_FUT_ENS (и контрольно C5): попытка опровергнуть результат.

Все варианты считаются на полном году (2025-09-25 … 2026-09-24) с тем же таргетом волатильности 10%.
Выход: research/final/red_team.csv, research/final/red_team_stats.json
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
from moexlab.deep.panel_lib import library  # noqa: E402
from moexlab.statistics import inference as INF  # noqa: E402

OUT = D.ROOT / "research/final"
SEED = 20260925


def stats(r, days, label, **kw):
    mk = M.month_keys(days)
    live = np.cumsum(r != 0) > 0
    months = np.unique(mk[live])
    mret = np.array([r[mk == m].sum() for m in months])
    sd = r[live].std()
    hold = (mk >= 202607)
    return dict(variant=label, total=float(r.sum()), sharpe=float(r[live].mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                pmr=float((mret > 0).mean()), m_min=float(mret.min()), m_median=float(np.median(mret)),
                mdd=float(M.max_dd(r[None])[0]), holdout_total=float(r[hold].sum()), **kw)


def run_variant(P, Wt, **bt):
    r, *_ = PN.backtest(P, Wt, **bt) if bt else PN.backtest(P, Wt)
    rv, _ = C.vol_target(r.to_numpy(), 0.10)
    return rv


def main():
    rows = []
    P = PN.load_panel(end=D.LAST_DAY)
    days = np.array(P["days"])
    lib = library(P)
    base_W, n = C.avg_weights(P, C.TREND, {"FUT"}, lib)
    base = run_variant(P, base_W)
    rows.append(stats(base, days, "BASE C1 (Трейдер, спред 1 шаг, проскальзывание 1 шаг)"))
    # издержки
    for comm, slip, lab in ((1.0, 0.0, "проскальзывание ×0"), (1.0, 2.0, "проскальзывание ×2"), (1.0, 3.0, "проскальзывание ×3"),
                            (1.0, 5.0, "проскальзывание ×5"), (1.5, 1.0, "комиссия ×1,5"), (2.0, 1.0, "комиссия ×2"),
                            (2.0, 3.0, "комиссия ×2 + проскальзывание ×3"), (0.625, 1.0, "тариф Премиум (0,025%)")):
        Pc = PN.load_panel(end=D.LAST_DAY, slip_mult=slip, comm_mult=comm)
        rows.append(stats(run_variant(Pc, base_W), days, lab))
    # задержка исполнения на 1 и 2 дня
    for lag in (1, 2):
        rows.append(stats(run_variant(P, base_W.shift(lag).fillna(0.0)), days, f"исполнение на {lag} дн. позже"))
    # подмножества ансамбля (устойчивость к параметрам)
    for fam in ("P_TSMOM", "P_EMA", "P_BRK"):
        Wf, k = C.avg_weights(P, {fam}, {"FUT"}, lib)
        rows.append(stats(run_variant(P, Wf), days, f"только {fam} ({k} конф.)"))
    for vn in (20, 60):
        Ws = [fn() for fam, cn, p, fn in lib if fam in C.TREND and cn == "FUT" and p.get("vn") == vn]
        rows.append(stats(run_variant(P, sum(Ws) / len(Ws)), days, f"только vn={vn}"))
    for fast in (True, False):
        Ws = [fn() for fam, cn, p, fn in lib if fam in C.TREND and cn == "FUT" and
              ((p.get("n", p.get("s", 0)) <= 20) == fast)]
        rows.append(stats(run_variant(P, sum(Ws) / len(Ws)), days, "только быстрые (окно ≤ 20)" if fast else "только медленные (окно > 20)"))
    # исключение инструментов
    for c in D.FUT_BASES:
        Wx = base_W.copy()
        Wx[c] = 0.0
        rows.append(stats(run_variant(P, Wx), days, f"без {c}"))
    # только лонг / только шорт
    rows.append(stats(run_variant(P, base_W.clip(lower=0)), days, "только длинные позиции"))
    rows.append(stats(run_variant(P, base_W.clip(upper=0)), days, "только короткие позиции"))
    # удаление лучших дней
    order = np.argsort(-base)
    for k in (1, 5, 10):
        r2 = base.copy()
        r2[order[:k]] = 0.0
        rows.append(stats(r2, days, f"без {k} лучших дней"))
    # удаление лучших «сделок»: лучших эпизодов по инструменту (вклад инструмента за неделю)
    contrib = {}
    r_raw, turn, gross, Wx = PN.backtest(P, base_W)
    _, sc = C.vol_target(r_raw.to_numpy(), 0.10)
    per = (Wx.shift(1).fillna(0) * P["r_gap"] + Wx * P["r_intra"]).mul(sc, axis=0)
    wk = pd.to_datetime(pd.Series(days.astype(str)), format="%Y%m%d").dt.strftime("%G-%V").to_numpy()
    ep = per.groupby(wk).sum().stack().sort_values(ascending=False)
    for k in (1, 5, 10):
        r3 = base.copy()
        for (w, c), v in ep.head(k).items():
            r3[wk == w] -= per.loc[wk == w, c].to_numpy()
        rows.append(stats(r3, days, f"без {k} лучших эпизодов (инструмент×неделя)"))
    # периоды
    mk = M.month_keys(days)
    for lo, hi, lab in ((202510, 202603, "первая половина (окт–мар)"), (202604, 202609, "вторая половина (апр–сен)")):
        r4 = np.where((mk >= lo) & (mk <= hi), base, 0.0)
        rows.append(stats(r4, days, lab))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "red_team.csv", index=False)
    # статистика: бутстреп, перестановка (сдвиг сигналов по времени), DSR
    live = np.cumsum(base != 0) > 0
    x = base[live]
    p_boot = INF.bootstrap_mean_pvalue(x, n_boot=20000, mean_block=5.0, seed=SEED)
    lo, hi = INF.bootstrap_ci(x, stat=lambda v: v.mean() / v.std() * np.sqrt(252), n_boot=5000, seed=SEED)
    rng = np.random.default_rng(SEED)
    perm = []
    Wv = base_W.to_numpy()
    for _ in range(300):
        k = int(rng.integers(20, len(days) - 20))
        Wp = pd.DataFrame(np.roll(Wv, k, axis=0), index=base_W.index, columns=base_W.columns)
        rp = run_variant(P, Wp)
        lv = np.cumsum(rp != 0) > 0
        perm.append(rp[lv].mean() / rp[lv].std() * np.sqrt(252) if rp[lv].std() > 0 else 0.0)
    sh = x.mean() / x.std() * np.sqrt(252)
    p_perm = float((np.array(perm) >= sh).mean())
    # DSR: число испытаний = все протестированные конфигурации в сравнимой (дневной портфельной) группе
    lib_meta = pd.read_parquet(D.CACHE / "screen/p1full/PANEL_1440.parquet")
    trial_sh = lib_meta["sharpe"].to_numpy() / np.sqrt(252)
    dsr = INF.deflated_sharpe_ratio(x, n_trials=len(lib_meta), trial_sharpes=trial_sh)
    st = dict(sharpe_year=float(sh), bootstrap_p_mean_le_0=float(p_boot), sharpe_ci95=[float(lo), float(hi)],
              permutation_time_shift_p=p_perm, permutation_n=len(perm), dsr=dsr, n_trials_dsr=len(lib_meta), seed=SEED)
    (OUT / "red_team_stats.json").write_text(json.dumps(st, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    print(json.dumps(st, indent=1, default=float))


if __name__ == "__main__":
    main()
