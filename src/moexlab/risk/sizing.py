"""Размер позиции и анализ целевой доходности (задача §22, §35).

Исходные данные — дневные доходности стратегии при базовом риске base_risk на сделку (без сложного процента).
Масштабирование s означает риск s × base_risk на сделку. Приближение: доходность линейна по s
(не учитывает целочисленность контрактов, лимиты ГО и рост проскальзывания с объёмом — все они
делают реальный результат хуже).

Ключевой факт: при сложном проценте рост капитала g(s) = E[log(1 + s·r)] максимален при s* (критерий Келли);
при s > s* рост падает. Если целевой рост недостижим даже при s*, цель недостижима при любом плече.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..statistics.inference import block_bootstrap_paths


def growth_rate(r: np.ndarray, s: float) -> float:
    x = 1 + s * r
    if (x <= 0).any():
        return -np.inf
    return float(np.log(x).mean())


def kelly_scale(r: np.ndarray, s_max: float = 200.0) -> float:
    """Множитель риска, максимизирующий рост; 0 — если рост отрицателен при любом плече (нет преимущества)."""
    if np.mean(r) <= 0:
        return 0.0
    grid = np.linspace(0.01, s_max, 4000)
    g = np.array([growth_rate(r, s) for s in grid])
    return float(grid[int(np.nanargmax(g))])


def required_scale(r: np.ndarray, monthly_target: float, days_per_month: float = 21.0) -> float | None:
    """Минимальный множитель, дающий средний месячный рост >= target (по историческому ряду). None — недостижимо."""
    tgt = math.log(1 + monthly_target) / days_per_month
    s_star = kelly_scale(r)
    if s_star <= 0 or growth_rate(r, s_star) < tgt:
        return None
    lo, hi = 0.0, s_star
    for _ in range(60):
        mid = (lo + hi) / 2
        if growth_rate(r, mid) >= tgt:
            hi = mid
        else:
            lo = mid
    return hi


def kelly_ruin_probability(fraction_of_kelly: float, drawdown: float) -> float:
    """P(капитал когда-либо упадёт до (1-drawdown)) для ГБД при ставке c × Келли: x^(2/c − 1), x = 1 − drawdown."""
    c = fraction_of_kelly
    if c <= 0:
        return 0.0
    return float((1 - drawdown) ** (2 / c - 1))


def target_analysis(daily_r: pd.Series, base_risk: float, avg_notional_per_risk: float,
                    targets=(0.05, 0.075, 0.10, 0.15), n_sims: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Для каждой цели: требуемый риск на сделку, плечо, исторические и бутстреп-просадки.

    avg_notional_per_risk — средняя суммарная номинальная позиция на единицу капитала при base_risk
    (оценивается по сделкам: Σ risk × price / stop_distance по открытым позициям).
    """
    r = daily_r.to_numpy(float)
    s_star = kelly_scale(r)
    rows = []
    cases = [("conservative", 0.5), ("base", 1.0), ("elevated", 2.0), ("half_kelly", s_star / 2), ("kelly", s_star)]
    for t in targets:
        s = required_scale(r, t)
        cases.append((f"target_{t:.1%}/month", s))
    for name, s in cases:
        if s is not None and s <= 0:
            rows.append(dict(case=name, scale=0.0, feasible=False, note="ожидание <= 0: оптимальный риск = 0"))
            continue
        if s is None:
            rows.append(dict(case=name, scale=np.nan, risk_per_trade=np.nan, avg_leverage=np.nan,
                             feasible=False, note="недостижимо при любом плече (рост выше, чем у Келли)"))
            continue
        rs = s * r
        eq = np.cumprod(1 + rs)
        hist_dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        mc = block_bootstrap_paths(pd.Series(rs), n_sims=n_sims, mean_block=5.0, seed=seed)
        m = (pd.Series(1 + rs, index=daily_r.index).resample("ME").prod() - 1)
        rows.append(dict(
            case=name, scale=float(s), risk_per_trade=float(s * base_risk),
            fraction_of_kelly=float(s / s_star) if s_star > 0 else np.inf,
            avg_leverage=float(s * avg_notional_per_risk), feasible=True,
            monthly_median=float(m.median()), monthly_mean=float(m.mean()), worst_month=float(m.min()),
            cagr=float(eq[-1] ** (252 / len(rs)) - 1), hist_maxdd=hist_dd,
            mc_maxdd_median=mc["maxdd_p50"], mc_maxdd_p95=mc["maxdd_p95"],
            p_dd_gt_10=mc["p_dd_gt_10"], p_dd_gt_20=mc["p_dd_gt_20"], p_dd_gt_30=mc["p_dd_gt_30"],
            p_dd_gt_50=mc["p_dd_gt_50"], p_loss_horizon=mc["p_loss"],
            kelly_theory_p_dd50=kelly_ruin_probability(s / s_star, 0.5) if s_star > 0 else np.nan,
        ))
    return pd.DataFrame(rows)


def average_leverage(trades: pd.DataFrame, daily_index: pd.DatetimeIndex, base_risk: float) -> float:
    """Средняя суммарная номинальная экспозиция / капитал при риске base_risk на сделку."""
    if trades.empty:
        return 0.0
    lev = pd.Series(0.0, index=daily_index)
    for _, t in trades.iterrows():
        dist = abs(t["entry_price"] - t["initial_stop"])
        if dist <= 0:
            continue
        notional = base_risk * t["entry_price"] / dist
        a = pd.Timestamp(t["entry_ts"]).tz_localize(None).normalize() if pd.Timestamp(t["entry_ts"]).tzinfo \
            else pd.Timestamp(t["entry_ts"]).normalize()
        b = pd.Timestamp(t["exit_ts"]).tz_localize(None).normalize() if pd.Timestamp(t["exit_ts"]).tzinfo \
            else pd.Timestamp(t["exit_ts"]).normalize()
        if lev.index.tz is not None:
            a, b = a.tz_localize(lev.index.tz), b.tz_localize(lev.index.tz)
        lev[(lev.index >= a) & (lev.index <= b)] += notional
    return float(lev.mean())
