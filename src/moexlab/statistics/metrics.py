"""Метрики стратегии (задача §20–21, спецификация §76–80)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns_from_equity(daily_equity: pd.Series) -> pd.Series:
    return daily_equity.pct_change().fillna(daily_equity.iloc[0] / daily_equity.iloc[0] - 1)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min()) if len(equity) else 0.0


def longest_losing_streak(x: np.ndarray) -> int:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def monthly_table(daily_ret: pd.Series) -> pd.DataFrame:
    """Таблица год × месяц доходностей (сложный процент внутри месяца)."""
    m = (1 + daily_ret).groupby([daily_ret.index.year, daily_ret.index.month]).prod() - 1
    t = m.unstack(level=1)
    t.index.name, t.columns.name = "year", "month"
    return t


def monthly_series(daily_ret: pd.Series) -> pd.Series:
    m = (1 + daily_ret).resample("ME").prod() - 1
    return m


def trade_metrics(tr: pd.DataFrame) -> dict:
    if tr is None or tr.empty:
        return dict(n_trades=0)
    R = tr["R"].to_numpy(float)
    wins, losses = R[R > 0], R[R <= 0]
    gp, gl = tr.loc[tr.net_pnl > 0, "net_pnl"].sum(), -tr.loc[tr.net_pnl <= 0, "net_pnl"].sum()
    days = pd.to_datetime(pd.Series(tr["trading_day"])).nunique()
    return dict(
        n_trades=int(len(tr)),
        win_rate=float((R > 0).mean()),
        expectancy_R=float(R.mean()),
        median_R=float(np.median(R)),
        expectancy_R_se=float(R.std(ddof=1) / math.sqrt(len(R))) if len(R) > 1 else float("nan"),
        gross_expectancy_R=float(tr["gross_R"].mean()),
        avg_win_R=float(wins.mean()) if len(wins) else 0.0,
        avg_loss_R=float(-losses.mean()) if len(losses) else 0.0,
        profit_factor=float(gp / gl) if gl > 0 else float("inf"),
        net_pnl=float(tr["net_pnl"].sum()),
        gross_pnl=float(tr["gross_pnl"].sum()),
        commission=float(tr["commission"].sum()),
        spread_cost=float(tr["spread_cost"].sum()),
        slippage_cost=float(tr["slippage_cost"].sum()),
        mae_R=float(tr["mae_R"].mean()),
        mfe_R=float(tr["mfe_R"].mean()),
        avg_bars_held=float(tr["bars_held"].mean()),
        longest_losing_streak=int(longest_losing_streak(R)),
        trades_per_day=float(len(tr) / max(days, 1)),
        long_share=float((tr["direction"] == "LONG").mean()),
    )


def equity_metrics(daily_equity: pd.Series, periods: int = TRADING_DAYS) -> dict:
    if daily_equity is None or len(daily_equity) < 2:
        return {}
    r = daily_equity.pct_change().dropna()
    n_years = max(len(r) / periods, 1e-9)
    total = daily_equity.iloc[-1] / daily_equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if total > -1 else -1.0
    sd = r.std(ddof=1)
    dsd = r[r < 0].std(ddof=1)
    mdd = max_drawdown(daily_equity)
    mret = monthly_series(r)
    return dict(
        total_return=float(total), cagr=float(cagr),
        sharpe=float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan"),
        sortino=float(r.mean() / dsd * math.sqrt(periods)) if dsd and dsd > 0 else float("nan"),
        max_drawdown=float(mdd), calmar=float(cagr / -mdd) if mdd < 0 else float("nan"),
        return_over_maxdd=float(total / -mdd) if mdd < 0 else float("nan"),
        monthly_mean=float(mret.mean()), monthly_median=float(mret.median()),
        monthly_best=float(mret.max()), monthly_worst=float(mret.min()),
        months_positive=int((mret > 0).sum()), months_negative=int((mret < 0).sum()),
        longest_losing_months=int(longest_losing_streak(mret.to_numpy())),
        n_days=int(len(r)), skew=float(r.skew()), kurtosis=float(r.kurt() + 3),
    )


def exposure(tr: pd.DataFrame, n_bars_total: int) -> float:
    if tr is None or tr.empty or n_bars_total <= 0:
        return 0.0
    return float((tr["bars_held"] + 1).sum() / n_bars_total)


def remove_extremes(tr: pd.DataFrame) -> pd.DataFrame:
    """Зависимость от экстремальных сделок/дней (задача §24): ожидание R после удаления."""
    rows = []
    R = tr["R"].sort_values(ascending=False)
    day_R = tr.groupby("trading_day")["R"].sum().sort_values(ascending=False)
    for label, s in [("all", tr["R"]), ("-best1", R.iloc[1:]), ("-best5", R.iloc[5:]), ("-best10", R.iloc[10:]),
                     ("-worst1", R.iloc[:-1]), ("-worst5", R.iloc[:-5])]:
        rows.append(dict(case=label, n=len(s), mean_R=float(s.mean()) if len(s) else np.nan, sum_R=float(s.sum())))
    for k in (1, 5, 10):
        keep = ~tr["trading_day"].isin(day_R.index[:k])
        s = tr.loc[keep, "R"]
        rows.append(dict(case=f"-bestday{k}", n=len(s), mean_R=float(s.mean()) if len(s) else np.nan,
                         sum_R=float(s.sum())))
    return pd.DataFrame(rows)
