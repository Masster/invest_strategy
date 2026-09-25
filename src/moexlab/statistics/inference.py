"""Статистическая проверка с учётом множественного тестирования (задача §23, §26).

Реализовано:
  * stationary_bootstrap_indices — стационарный блочный бутстреп (Politis & Romano, 1994);
  * bootstrap_mean_pvalue       — односторонний p-value H0: mean <= 0 (бутстреп центрированного ряда);
  * benjamini_hochberg          — контроль FDR;
  * deflated_sharpe_ratio       — DSR (Bailey & López de Prado, 2014);
  * pbo_cscv                    — вероятность переобучения бэктеста, CSCV (Bailey et al., 2016);
  * whites_reality_check / hansen_spa — проверка лучшей стратегии среди N против нулевого бенчмарка;
  * block_bootstrap_paths       — Монте-Карло кривых капитала блоками торговых дней.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    starts = rng.random(n) < p
    rnd = rng.integers(0, n, size=n)
    for t in range(1, n):
        idx[t] = rnd[t] if starts[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_mean_pvalue(x: np.ndarray, n_boot: int = 2000, mean_block: float = 5.0, seed: int = 0) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = x.mean()
    xc = x - obs
    cnt = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(len(x), mean_block, rng)
        if xc[idx].mean() >= obs:
            cnt += 1
    return (cnt + 1) / (n_boot + 1)


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 2000, mean_block: float = 5.0, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = [stat(x[stationary_bootstrap_indices(len(x), mean_block, rng)]) for _ in range(n_boot)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Возвращает маску отвергнутых H0 при уровне FDR q."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    m = ok.sum()
    rej = np.zeros(len(p), dtype=bool)
    if m == 0:
        return rej
    order = np.argsort(np.where(ok, p, np.inf))
    thresh = q * (np.arange(1, len(p) + 1) / m)
    passed = np.where(np.where(ok, p, np.inf)[order] <= thresh)[0]
    if len(passed):
        rej[order[: passed.max() + 1]] = True
    return rej & ok


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe_ratio(sr: float, sr_ref: float, n: int, skew: float, kurt: float) -> float:
    """PSR: P(истинный SR > sr_ref) с поправкой на асимметрию и эксцесс (SR — непериодизированный)."""
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return float(stats.norm.cdf((sr - sr_ref) * math.sqrt(n - 1) / den))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] из N независимых испытаний с нулевым истинным SR (Bailey & López de Prado)."""
    if n_trials <= 1:
        return 0.0
    g = 0.5772156649
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(var_sr, 0.0)) * ((1 - g) * z1 + g * z2)


def effective_trials(M: np.ndarray) -> float:
    """Эффективное число независимых испытаний: participation ratio собственных значений корреляционной матрицы."""
    M = np.asarray(M, float)
    sd = M.std(axis=0)
    M = M[:, sd > 0]
    if M.shape[1] < 2:
        return float(M.shape[1])
    ev = np.clip(np.linalg.eigvalsh(np.corrcoef(M, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, trial_sharpes: np.ndarray) -> dict:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    sr = sharpe(r)
    v = float(np.nanvar(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, v)
    dsr = probabilistic_sharpe_ratio(sr, sr0, len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))
    return dict(sr=sr, sr_benchmark=sr0, dsr=dsr, n_trials=n_trials)


def pbo_cscv(M: np.ndarray, n_splits: int = 16, metric=None) -> dict:
    """CSCV. M: матрица T × N (периоды × конфигурации). Возвращает PBO и распределение логитов."""
    M = np.asarray(M, float)
    T, N = M.shape
    if N < 2:
        return dict(pbo=float("nan"), n_combinations=0)
    metric = metric or (lambda X: np.nanmean(X, axis=0) / (np.nanstd(X, axis=0, ddof=1) + 1e-12))
    S = n_splits - n_splits % 2
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for comb in itertools.combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in comb])
        te = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        ms_tr, ms_te = metric(M[tr]), metric(M[te])
        best = int(np.nanargmax(ms_tr))
        rank = stats.rankdata(ms_te)[best] / (N + 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return dict(pbo=float((logits <= 0).mean()), n_combinations=len(logits), logit_median=float(np.median(logits)))


def whites_reality_check(M: np.ndarray, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0) -> dict:
    """H0: ни одна из N стратегий не лучше нуля. M: T × N дневных доходностей (сверх бенчмарка)."""
    M = np.nan_to_num(np.asarray(M, float))
    T, N = M.shape
    rng = np.random.default_rng(seed)
    mu = M.mean(axis=0)
    stat = math.sqrt(T) * mu.max()
    cnt = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(T, mean_block, rng)
        vb = math.sqrt(T) * (M[idx].mean(axis=0) - mu).max()
        cnt += vb >= stat
    return dict(stat=float(stat), pvalue=float((cnt + 1) / (n_boot + 1)), best=int(mu.argmax()))


def hansen_spa(M: np.ndarray, n_boot: int = 2000, mean_block: float = 10.0, seed: int = 0) -> dict:
    """Studentized SPA (Hansen, 2005), consistent-вариант: плохие стратегии не раздувают p-value."""
    M = np.nan_to_num(np.asarray(M, float))
    T, N = M.shape
    rng = np.random.default_rng(seed)
    mu = M.mean(axis=0)
    boots = np.empty((n_boot, N))
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(T, mean_block, rng)
        boots[b] = M[idx].mean(axis=0)
    omega = np.sqrt(T) * boots.std(axis=0, ddof=1) + 1e-12
    t_obs = max(0.0, (math.sqrt(T) * mu / omega).max())
    thr = -np.sqrt(2 * math.log(math.log(T))) * omega / math.sqrt(T)
    # Hansen (2005): рецентрируются к нулю все модели, кроме заведомо плохих (mean <= -A_k);
    # у заведомо плохих сохраняется их отрицательное среднее, чтобы они не раздували p-value.
    mu_c = np.where(mu <= thr, mu, 0.0)
    tb = (np.sqrt(T) * (boots - mu + mu_c) / omega).max(axis=1)
    tb = np.maximum(tb, 0.0)
    return dict(stat=float(t_obs), pvalue=float(((tb >= t_obs).sum() + 1) / (n_boot + 1)))


def block_bootstrap_paths(daily_pnl: pd.Series, n_sims: int = 10000, horizon: int | None = None,
                          mean_block: float = 5.0, seed: int = 0, start_equity: float = 1.0,
                          as_returns: bool = True) -> dict:
    """Монте-Карло: стационарный бутстреп блоков торговых дней (сохраняет внутридневную зависимость сделок).

    daily_pnl — дневные доходности (as_returns=True) стратегии; возвращает распределения метрик.
    """
    x = np.asarray(daily_pnl, float)
    x = x[np.isfinite(x)]
    T = len(x)
    H = horizon or T
    rng = np.random.default_rng(seed)
    final, mdd, ann, worst_m, streak = [], [], [], [], []
    for _ in range(n_sims):
        idx = stationary_bootstrap_indices(T, mean_block, rng)[:H] if H <= T else \
            np.concatenate([stationary_bootstrap_indices(T, mean_block, rng) for _ in range(H // T + 1)])[:H]
        r = x[idx]
        eq = start_equity * np.cumprod(1 + r)
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        final.append(eq[-1])
        mdd.append(dd)
        ann.append(eq[-1] ** (252 / H) - 1)
        mo = [np.prod(1 + r[i:i + 21]) - 1 for i in range(0, H - 20, 21)]
        worst_m.append(min(mo) if mo else np.nan)
        s = c = 0
        for v in r:
            c = c + 1 if v < 0 else 0
            s = max(s, c)
        streak.append(s)
    mdd = np.array(mdd)
    out = dict(
        final_equity_p05=float(np.quantile(final, 0.05)), final_equity_p50=float(np.median(final)),
        final_equity_p95=float(np.quantile(final, 0.95)),
        annual_return_p05=float(np.quantile(ann, 0.05)), annual_return_p50=float(np.median(ann)),
        annual_return_p95=float(np.quantile(ann, 0.95)),
        maxdd_p50=float(np.median(mdd)), maxdd_p95=float(np.quantile(mdd, 0.05)), maxdd_p99=float(np.quantile(mdd, 0.01)),
        worst_month_p50=float(np.nanmedian(worst_m)),
        longest_losing_days_p50=float(np.median(streak)), longest_losing_days_p95=float(np.quantile(streak, 0.95)),
        p_loss=float((np.array(final) < start_equity).mean()),
    )
    for th in (0.10, 0.15, 0.20, 0.30, 0.50):
        out[f"p_dd_gt_{int(th * 100)}"] = float((mdd <= -th).mean())
    out["_maxdd_samples"] = mdd
    return out
