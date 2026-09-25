"""Риск, плечо, Monte Carlo, целевые доходности и режимы риска для кандидатов (research/final/*).

База: дневные доходности кандидата при целевой волатильности 10% годовых (vt10, каузальный масштаб).
Плечо L: доходность дня = L × r_vt10 (линейно, издержки тоже масштабируются), сложный процент по дням.
Monte Carlo: стационарный блочный бутстреп торговых дней всего года (средний блок 5 дней), 50 000 путей × 252 дня.
Конвенция «риск на сделку»: риск позиции ≈ 2 дневных σ портфеля (стоп 2σ) => σ_дн = риск / 2.
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
from moexlab.deep import risk as RK  # noqa: E402

OUT = D.ROOT / "research/final"
N_MC = 50000
SEED = 20260925


def main():
    daily = pd.read_parquet(OUT / "candidates_daily_vt10.parquet")
    days = daily.index.to_numpy()
    rows, targets, modes = [], [], []
    for cand in daily.columns:
        r0 = daily[cand].to_numpy()
        live = np.cumsum(r0 != 0) > 0
        r = r0[live]
        cal = days[live]
        sd_d = r.std()
        for vol_ann in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.50):
            L = vol_ann / 0.10
            h = RK.path_stats(r, cal, L)
            mc = RK.monte_carlo(r, L, n_paths=N_MC, seed=SEED)
            risk_trade = 2 * sd_d * L
            rows.append(dict(candidate=cand, target_vol_ann=vol_ann, leverage_vs_vt10=L,
                             approx_risk_per_trade=risk_trade, hist_total=h["total"], hist_mdd=h["mdd"],
                             hist_m_median=h["m_median"], hist_m_min=h["m_min"], hist_m_max=h["m_max"],
                             hist_pmr=h["pmr"], hist_ruin=h["ruin"], **{f"mc_{k}": v for k, v in mc.items()
                                                                          if k not in ("lev", "seed", "n_paths", "horizon_days", "block")}))
        # риск на сделку (сетка задания)
        for rt in (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10):
            L = (rt / 2) / sd_d
            h = RK.path_stats(r, cal, L)
            mc = RK.monte_carlo(r, L, n_paths=N_MC, seed=SEED)
            modes.append(dict(candidate=cand, risk_per_trade=rt, leverage_vs_vt10=L, ann_vol=L * 0.10,
                              hist_m_mean=h["m_mean"], hist_m_median=h["m_median"], hist_m_min=h["m_min"],
                              hist_mdd=h["mdd"], hist_total=h["total"], mc_monthly_median=mc["monthly_median"],
                              mc_mdd_median=mc["mdd_median"], mc_mdd_p95=mc["mdd_p95"], p_dd10=mc["p_dd_gt_10"],
                              p_dd20=mc["p_dd_gt_20"], p_dd30=mc["p_dd_gt_30"], p_dd50=mc["p_dd_gt_50"],
                              p_dd70=mc["p_dd_gt_70"], p_ruin=mc["p_ruin"], p_loss_year=mc["p_loss"]))
        # целевые месячные доходности
        mu_m = r.mean() * 21
        for tgt in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00):
            if mu_m <= 0:
                targets.append(dict(candidate=cand, target_monthly=tgt, feasible=False,
                                    note="средняя доходность ≤ 0: цель недостижима ни при каком плече"))
                continue
            L = tgt / mu_m            # плечо относительно vt10, дающее такую СРЕДНЮЮ (арифм.) доходность
            h = RK.path_stats(r, cal, L)
            mc = RK.monte_carlo(r, L, n_paths=N_MC, seed=SEED)
            targets.append(dict(candidate=cand, target_monthly=tgt, required_leverage_vs_vt10=L,
                                required_ann_vol=0.10 * L, required_risk_per_trade=2 * sd_d * L,
                                hist_mdd=h["mdd"], hist_m_min=h["m_min"], hist_total=h["total"], hist_ruin=h["ruin"],
                                mc_monthly_median=mc["monthly_median"], mc_mdd_median=mc["mdd_median"],
                                mc_mdd_p95=mc["mdd_p95"], p_ruin=mc["p_ruin"], p_dd50=mc["p_dd_gt_50"],
                                feasible=bool(mc["monthly_median"] >= 0.5 * tgt and mc["p_ruin"] < 0.05)))
    pd.DataFrame(rows).to_csv(OUT / "leverage_grid.csv", index=False)
    pd.DataFrame(modes).to_csv(OUT / "risk_per_trade_grid.csv", index=False)
    pd.DataFrame(targets).to_csv(OUT / "return_targets.csv", index=False)
    pd.set_option("display.width", 250)
    m = pd.DataFrame(modes)
    print(m[m.candidate == "C1_TREND_FUT_ENS"].round(4).to_string())
    t = pd.DataFrame(targets)
    print(t[t.candidate == "C1_TREND_FUT_ENS"].round(4).to_string())


def contracts_check(cand="C1_TREND_FUT_ENS"):
    """Целые контракты и ГО: доля недостижимых позиций и загрузка ГО при капитале 1/3/10/30 млн ₽."""
    P = PN.load_panel(end=D.LAST_DAY)
    Wt, _ = C.fixed_candidates(P)[cand]
    r, turn, gross, Wx = C.returns(P, Wt)
    _, scale = C.vol_target(r, 0.10)
    Wsc = Wx.mul(scale, axis=0)      # масштаб дня t применяется к весам, действующим в день t
    rows = []
    specs = {c: D.spec(c) for c in P["codes"]}
    for cap in (1e6, 3e6, 10e6, 30e6):
        for lev in (1.0, 2.0, 4.0):
            W = Wsc * lev
            notional = pd.DataFrame({c: P["real_c"][c] / specs[c]["tick"] * specs[c]["tick_value"] * specs[c]["lot"]
                                     for c in P["codes"]})
            want = W * cap / notional
            got = want.round()
            eff = got * notional / cap
            err = (eff - W).abs().sum(1) / W.abs().sum(1).replace(0, np.nan)
            im = pd.Series({c: (specs[c].get("im") or 0.0) / (P["real_c"][c].iloc[-1] / specs[c]["tick"] *
                                                                 specs[c]["tick_value"]) for c in P["codes"]})
            margin = (eff.abs() * im).sum(1)
            r_eff, *_ = PN.backtest(P, eff.shift(-1).fillna(0.0))   # eff уже сдвинут: возвращаем к «решению»
            rows.append(dict(candidate=cand, capital=cap, lev=lev, weight_error_median=float(err.median()),
                             zero_position_share=float(((got == 0) & (W.abs() > 1e-6)).sum().sum() /
                                                       max(1, (W.abs() > 1e-6).sum().sum())),
                             margin_mean=float(margin.mean()), margin_max=float(margin.max()),
                             ret_total_integer=float(r_eff.sum()), ret_total_fractional=float((r * scale * lev).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "contracts_check.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
    contracts_check()
