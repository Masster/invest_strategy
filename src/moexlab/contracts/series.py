"""Серии фьючерсов и выбор активной серии (спецификация R-Breaker §6–7).

Код серии MOEX: <базовый код><буква месяца><последняя цифра года>, например MXZ4.
Решение о серии на день D принимается только по завершённым дням < D (причинность).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

MONTH_CODES = "FGHJKMNQUVXZ"  # январь..декабрь


def secid(base: str, year: int, month: int) -> str:
    return f"{base}{MONTH_CODES[month - 1]}{year % 10}"


def parse_secid(s: str, around_year: int) -> tuple[str, int, int]:
    """Возвращает (base, year, month). Год восстанавливается в окрестности around_year."""
    base, m, y = s[:-2], s[-2], int(s[-1])
    month = MONTH_CODES.index(m) + 1
    decade = around_year - around_year % 10
    year = decade + y
    if year > around_year + 5:
        year -= 10
    elif year < around_year - 5:
        year += 10
    return base, year, month


@dataclass(frozen=True)
class SeriesInfo:
    secid: str
    base: str
    expiration: date        # последний день торгов
    first_trade: date | None = None


@dataclass(frozen=True)
class RollConfig:
    mandatory_days_before_expiry: int = 2
    early_roll_consecutive_days: int = 2
    early_roll_max_spread_ratio: float = 1.25
    early_roll_min_oi_ratio: float = 0.5
    early_roll_requires_spread: bool = True


def select_active_series(
    series: list[SeriesInfo],
    daily: pd.DataFrame,
    trading_days: list[date],
    cfg: RollConfig = RollConfig(),
) -> pd.Series:
    """Возвращает Series: торговый день -> secid активной серии.

    daily — таблица по завершённым дням с колонками [trading_day, secid, volume] и опционально
    [median_spread, open_interest]. Решение на день D использует только строки с trading_day < D.

    Правила:
      * front = ближайшая неистёкшая серия; next = следующая;
      * обязательная перекладка: если торговых дней до экспирации front (после D-1) <= N,
        с дня D работаем на next;
      * досрочная перекладка: K последних завершённых дней подряд volume(next) > volume(front),
        median_spread(next) <= 1.25 × median_spread(front) (если спред недоступен и
        early_roll_requires_spread=True — правило не применяется), OI(next) >= 0.5 × OI(front) если есть;
      * после перехода на next обратно на front не возвращаемся;
      * внутри дня серия не меняется (одно значение на день).
    """
    ser = sorted(series, key=lambda s: s.expiration)
    tdays = sorted(trading_days)
    tpos = {d: i for i, d in enumerate(tdays)}
    has_spread = "median_spread" in daily.columns and daily["median_spread"].notna().any()
    has_oi = "open_interest" in daily.columns and daily["open_interest"].notna().any()
    piv_v = daily.pivot_table(index="trading_day", columns="secid", values="volume", aggfunc="sum")
    piv_s = daily.pivot_table(index="trading_day", columns="secid", values="median_spread") if has_spread else None
    piv_o = daily.pivot_table(index="trading_day", columns="secid", values="open_interest") if has_oi else None

    out: dict[date, str] = {}
    current: str | None = None
    for d in tdays:
        alive = [s for s in ser if s.expiration >= d]
        if not alive:
            break
        front = alive[0]
        nxt = alive[1] if len(alive) > 1 else None
        if current is None or all(s.secid != current for s in alive):
            current = front.secid
        cur = next(s for s in alive if s.secid == current)
        # переход имеет смысл только с текущей ближайшей серии
        if cur.secid == front.secid and nxt is not None:
            i = tpos[d]
            prev_day = tdays[i - 1] if i > 0 else None
            # торговые дни после prev_day до экспирации включительно
            if prev_day is not None:
                exp_idx = np.searchsorted(np.array(tdays, dtype=object), front.expiration, side="right")
                days_left = exp_idx - i  # включая D
                if front.expiration > tdays[-1]:
                    # экспирация за пределами известного календаря: добавляем рабочие дни после него
                    days_left += int(np.busday_count(tdays[-1] + pd.Timedelta(days=1), front.expiration
                                                     + pd.Timedelta(days=1)))
                if days_left <= cfg.mandatory_days_before_expiry:
                    current = nxt.secid
                elif _early_roll_ok(front.secid, nxt.secid, tdays[:i], piv_v, piv_s, piv_o, cfg, has_spread):
                    current = nxt.secid
        out[d] = current
    return pd.Series(out, name="active_secid")


def _early_roll_ok(front, nxt, past_days, piv_v, piv_s, piv_o, cfg: RollConfig, has_spread: bool) -> bool:
    k = cfg.early_roll_consecutive_days
    if len(past_days) < k:
        return False
    if cfg.early_roll_requires_spread and not has_spread:
        return False
    last = past_days[-k:]
    for d in last:
        try:
            vf, vn = piv_v.at[d, front], piv_v.at[d, nxt]
        except KeyError:
            return False
        if not (np.isfinite(vn) and np.isfinite(vf) and vn > vf):
            return False
        if has_spread:
            try:
                sf, sn = piv_s.at[d, front], piv_s.at[d, nxt]
            except KeyError:
                return False
            if not (np.isfinite(sf) and np.isfinite(sn) and sn <= cfg.early_roll_max_spread_ratio * sf):
                return False
        if piv_o is not None:
            try:
                of, on = piv_o.at[d, front], piv_o.at[d, nxt]
                if np.isfinite(of) and np.isfinite(on) and on < cfg.early_roll_min_oi_ratio * of:
                    return False
            except KeyError:
                pass
    return True
