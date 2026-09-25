"""Синтетические минутные данные — ТОЛЬКО для проверки кода и статистического конвейера.

Никакие выводы о рынке из них не делаются. Используются для:
  * модульных/интеграционных тестов и тестов причинности;
  * «нулевой гипотезы»: случайное блуждание без преимущества — конвейер обязан НЕ найти стратегию;
  * «внедрённого преимущества»: слабая автокорреляция — конвейер обязан его обнаружить.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from .calendar import MSK


def _session_minutes(d: date, evening: bool) -> list[pd.Timestamp]:
    out = []
    t = datetime.combine(d, time(10, 0))
    end = datetime.combine(d, time(18, 50))
    while t < end:
        if not (time(14, 0) <= t.time() < time(14, 5)):  # промежуточный клиринг
            out.append(t)
        t += timedelta(minutes=1)
    if evening:
        t = datetime.combine(d, time(19, 5))
        end = datetime.combine(d, time(23, 50))
        while t < end:
            out.append(t)
            t += timedelta(minutes=1)
    return out


def generate_minute_bars(
    start: str = "2021-01-04",
    end: str = "2021-12-30",
    seed: int = 0,
    s0: float = 3000.0,
    tick: float = 1.0,
    daily_vol: float = 0.015,
    ar1: float = 0.0,
    ar_horizon_min: int = 10,
    garch: bool = True,
    evening: bool = True,
    base_volume: float = 60.0,
    roll_every_quarter: bool = True,
    code: str = "SYN",
    substeps: int = 6,
) -> pd.DataFrame:
    """Минутные свечи (index — UTC начало свечи).

    ar1 > 0 — внедрённое преимущество: доходность каждого ar_horizon_min-минутного блока
    коррелирует с предыдущим блоком (моментум); ar1 = 0 — чистое случайное блуждание.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end).date
    stamps: list[pd.Timestamp] = []
    for d in days:
        stamps.extend(_session_minutes(d, evening))
    n = len(stamps)
    ts = pd.DatetimeIndex(stamps).tz_localize(MSK).tz_convert("UTC")
    per_day = len(_session_minutes(days[0], evening))
    sigma_min = daily_vol / np.sqrt(per_day)
    # U-образный внутридневной профиль волатильности
    msk_t = pd.DatetimeIndex(stamps)
    minute_of_day = (msk_t.hour * 60 + msk_t.minute).to_numpy()
    prof = 1.0 + 0.8 * np.exp(-(minute_of_day - 600) / 30.0).clip(0, 5) * (minute_of_day >= 600)
    prof = np.where(minute_of_day >= 19 * 60, 0.6, prof)
    # кластеризация волатильности (GARCH(1,1) на дневном уровне)
    day_idx = np.searchsorted(np.array([pd.Timestamp(d) for d in days]), msk_t.normalize(), side="right") - 1
    dv = np.ones(len(days))
    if garch:
        h, w, a_, b_ = 1.0, 0.05, 0.10, 0.85
        for k in range(len(days)):
            z = rng.standard_normal()
            dv[k] = np.sqrt(h)
            h = w + a_ * (z * z) * h + b_ * h
            h = min(max(h, 0.2), 6.0)
    sub = substeps  # шаги внутри минуты: OHLC строится из единого ценового пути
    scale = sigma_min * prof * dv[day_idx]
    steps = rng.standard_normal((n, sub)) * (scale / np.sqrt(sub))[:, None]
    eps = steps.sum(axis=1)
    if ar1 != 0.0:
        blk = np.arange(n) // ar_horizon_min
        blk_ret = pd.Series(eps).groupby(blk).sum().to_numpy()
        drift_blk = np.zeros_like(blk_ret)
        drift_blk[1:] = ar1 * blk_ret[:-1]
        steps = steps + (drift_blk[blk] / ar_horizon_min / sub)[:, None]
    # гэп на открытии торгового дня входит в путь цены
    new_day = np.r_[True, day_idx[1:] != day_idx[:-1]]
    gap = rng.standard_normal(n) * sigma_min * 8 * new_day
    gap[0] = 0.0
    path = np.cumsum((steps + np.c_[gap, np.zeros((n, sub - 1))]).ravel()).reshape(n, sub)
    open_log = np.log(s0) + np.r_[0.0, path[:-1, -1]] + gap
    path = np.log(s0) + path
    rnd = lambda x: np.round(x / tick) * tick  # noqa: E731
    o = rnd(np.exp(open_log))
    c = rnd(np.exp(path[:, -1]))
    h = rnd(np.exp(np.maximum(path.max(axis=1), open_log)))
    l = rnd(np.exp(np.minimum(path.min(axis=1), open_log)))
    h = np.maximum.reduce([h, o, c])
    l = np.minimum.reduce([l, o, c])
    vol = rng.poisson(base_volume * prof * dv[day_idx]).astype(float) + 1
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": vol}, index=ts)
    if roll_every_quarter:
        q = pd.DatetimeIndex(stamps).to_period("Q")
        df["secid"] = [f"{code}{str(p)}" for p in q]
    else:
        df["secid"] = code
    df.index.name = "ts"
    return df


def generate_daily_bars(n_days: int = 1500, seed: int = 0, s0: float = 100.0, tick: float = 0.01,
                        daily_vol: float = 0.015, ar1: float = 0.0, trend_persist: float = 0.0,
                        start: str = "2020-01-03", code: str = "SYN") -> pd.DataFrame:
    """Дневные свечи из единого внутридневного пути (78 шагов). ar1 — автокорреляция дневных доходностей;
    trend_persist — медленный скрытый дрейф (AR(1) с коэффициентом 0.99), амплитуда в долях daily_vol.
    Только для проверки конвейера (NULL / внедрённое преимущество)."""
    rng = np.random.default_rng(seed)
    sub = 78
    days = pd.bdate_range(start, periods=n_days)
    h, w, a_, b_ = 1.0, 0.05, 0.10, 0.85
    ret = np.zeros(n_days)
    mu = 0.0
    prev = 0.0
    o = np.empty(n_days); hi = np.empty(n_days); lo = np.empty(n_days); c = np.empty(n_days)
    logp = np.log(s0)
    for t in range(n_days):
        z = rng.standard_normal()
        sig = daily_vol * np.sqrt(h)
        h = min(max(w + a_ * z * z * h + b_ * h, 0.2), 6.0)
        mu = 0.99 * mu + trend_persist * daily_vol * 0.14 * rng.standard_normal()
        drift = ar1 * prev + mu
        steps = rng.standard_normal(sub) * sig / np.sqrt(sub) + drift / sub
        gap = rng.standard_normal() * sig * 0.2
        path = logp + gap + np.cumsum(steps)
        o[t] = np.exp(logp + gap)
        hi[t] = np.exp(max(path.max(), logp + gap))
        lo[t] = np.exp(min(path.min(), logp + gap))
        c[t] = np.exp(path[-1])
        prev = path[-1] - logp
        logp = path[-1]
    r = lambda x: np.round(x / tick) * tick  # noqa: E731
    df = pd.DataFrame({"open": r(o), "high": r(hi), "low": r(lo), "close": r(c),
                       "volume": rng.poisson(10000, n_days).astype(float)},
                      index=pd.DatetimeIndex(days).tz_localize("UTC"))
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df["trading_day"] = df.index.date
    df["secid"] = code
    df.index.name = "ts"
    return df
