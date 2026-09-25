"""Метрики по дневным доходностям (доли капитала, без сложного процента) и функции оценки."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def month_keys(days: np.ndarray) -> np.ndarray:
    """yyyymmdd -> yyyymm; хвост сентября 2025 (4 дня) присоединён к октябрю 2025."""
    m = days // 100
    return np.where(m == 202509, 202510, m)


def monthly(ret: np.ndarray, mkeys: np.ndarray, months: np.ndarray) -> np.ndarray:
    """ret: [..., n_days] -> [..., n_months] сумма (простая доходность без реинвестирования)."""
    out = np.zeros(ret.shape[:-1] + (len(months),))
    for j, m in enumerate(months):
        out[..., j] = ret[..., mkeys == m].sum(axis=-1)
    return out


def max_dd(ret: np.ndarray) -> np.ndarray:
    """Максимальная просадка кумулятивной суммы (в долях капитала), по последней оси."""
    cum = np.cumsum(ret, axis=-1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros(cum.shape[:-1] + (1,)), cum], axis=-1), axis=-1)[..., 1:]
    return (peak - cum).max(axis=-1)


def summary_from_daily(ret: np.ndarray, mret: np.ndarray) -> dict:
    """ret: [k, n_days], mret: [k, n_months] -> dict of arrays."""
    mu = ret.mean(axis=-1)
    sd = ret.std(axis=-1)
    neg = np.where(ret < 0, ret, 0.0)
    dsd = np.sqrt((neg ** 2).mean(axis=-1))
    sharpe = np.where(sd > 0, mu / np.where(sd > 0, sd, 1) * np.sqrt(ANN), 0.0)
    sortino = np.where(dsd > 0, mu / np.where(dsd > 0, dsd, 1) * np.sqrt(ANN), 0.0)
    mdd = max_dd(ret)
    tot = ret.sum(axis=-1)
    msd = mret.std(axis=-1)
    return dict(total=tot, sharpe=sharpe, sortino=sortino, mdd=mdd,
                calmar=np.where(mdd > 0, tot / np.where(mdd > 0, mdd, 1), 0.0),
                m_mean=mret.mean(axis=-1), m_median=np.median(mret, axis=-1), m_min=mret.min(axis=-1),
                m_max=mret.max(axis=-1), m_std=msd, pmr=(mret > 0).mean(axis=-1),
                lose_streak=_longest_losing(mret))


def _longest_losing(mret):
    out = np.zeros(mret.shape[0], int)
    for k in range(mret.shape[0]):
        s = best = 0
        for v in mret[k]:
            s = s + 1 if v <= 0 else 0
            best = max(best, s)
        out[k] = best
    return out


def score_stability(mret: np.ndarray, target_mvol: float = 0.05) -> np.ndarray:
    """Композитная оценка на месячных доходностях, нормированных к месячной волатильности target_mvol.

    Score = mean + 0.5*median + 0.5*min − 0.5*std − 1.0*mean(|убыточные месяцы|) − 0.03*(доля убыточных)*... (в долях)
    Нормировка делает оценку независимой от плеча: сравнивается качество, а не размер позиции.
    """
    sd = mret.std(axis=-1, keepdims=True)
    scale = np.where(sd > 0, target_mvol / np.where(sd > 0, sd, 1), 0.0)
    z = mret * scale
    losing = np.where(z < 0, -z, 0.0)
    return (z.mean(-1) + 0.5 * np.median(z, -1) + 0.5 * z.min(-1) - 0.5 * z.std(-1) - 1.0 * losing.mean(-1)
            - 0.02 * (z <= 0).mean(-1) / 0.1)


def trade_stats(t_ret: np.ndarray) -> dict:
    n = len(t_ret)
    if n == 0:
        return dict(trades=0, win=np.nan, pf=np.nan, avg_trade=np.nan)
    g = t_ret[t_ret > 0].sum()
    lo = -t_ret[t_ret < 0].sum()
    return dict(trades=n, win=float((t_ret > 0).mean()), pf=float(g / lo) if lo > 0 else np.inf,
                avg_trade=float(t_ret.mean()))


def monthly_table(dates: np.ndarray, ret: np.ndarray, trade_exit_days=None, trade_rets=None,
                  start_equity: float = 1.0, compound: bool = True) -> pd.DataFrame:
    """Помесячная таблица (год, месяц, капитал на начало/конец, доходность, сделки, win, PF, макс. просадка)."""
    df = pd.DataFrame({"day": dates, "r": ret})
    df["ym"] = month_keys(df["day"].to_numpy())
    rows = []
    eq = start_equity
    for ym, g in df.groupby("ym", sort=True):
        r = g["r"].to_numpy()
        if compound:
            path = eq * np.cumprod(1 + r)
        else:
            path = eq + np.cumsum(r) * start_equity
        pk = np.maximum.accumulate(np.append(eq, path))[1:]
        mdd = float(((pk - path) / pk).max())
        end = float(path[-1])
        row = dict(year=ym // 100, month=ym % 100, start_equity=eq, end_equity=end, monthly_return=end / eq - 1,
                   max_drawdown=mdd)
        if trade_exit_days is not None:
            sel = month_keys(np.asarray(trade_exit_days)) == ym
            tr = np.asarray(trade_rets)[sel]
            st = trade_stats(tr)
            row.update(trades=st["trades"], win_rate=st["win"], profit_factor=st["pf"])
        rows.append(row)
        eq = end
    return pd.DataFrame(rows)
