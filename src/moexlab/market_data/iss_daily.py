"""Загрузка дневных свечей реальных серий фьючерсов MOEX (формат data/raw/iss_daily/<BASE>/<SECID>.csv).

Происхождение данных: MOEX ISS `candles.json?interval=24`, получено через веб-скрейпер Firecrawl
(прямой доступ к iss.moex.com из окружения закрыт) и перенесено в CSV языковой моделью.
Поэтому набор помечается DATA_QUALITY=LLM_TRANSCRIBED и проходит автоматические проверки целостности.
Дневная свеча ISS для FORTS: дата = торговый день (включая вечернюю сессию предыдущего календарного дня).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts.series import MONTH_CODES, RollConfig, SeriesInfo, select_active_series
from .bars import annotate_daily, stitch_active


@dataclass
class DailyUniverseItem:
    base: str
    stream: pd.DataFrame          # поток активной серии (реальные + сигнальные цены)
    series: dict[str, pd.DataFrame]
    active: pd.Series
    issues: list[str]


def third_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month, 1)
    offs = (weekday - d.weekday()) % 7
    return date(year, month, 1 + offs + 14)


def validate_bars(df: pd.DataFrame, name: str) -> list[str]:
    issues = []
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        issues.append(f"{name}: dates not strictly increasing")
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    bad = (h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9) | (v < 0) | (l <= 0)
    if bad.any():
        issues.append(f"{name}: {int(bad.sum())} OHLC violations e.g. {df.index[bad.to_numpy()][0].date()}")
    # подозрительные скачки закрытия (> 25% за день) — сигнал возможной ошибки переноса данных
    r = np.log(c).diff().abs()
    jumps = r > np.log(1.25)
    if jumps.any():
        issues.append(f"{name}: {int(jumps.sum())} close jumps >25% e.g. {df.index[jumps.to_numpy()][0].date()}")
    return issues


def load_series_dir(base_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out, issues = {}, []
    for p in sorted(base_dir.glob("*.csv")):
        df = pd.read_csv(p, parse_dates=["date"])
        if df.empty:
            continue
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        issues += validate_bars(df, p.stem)
        out[p.stem] = df
    return out, issues


def series_info(base: str, secid: str, df: pd.DataFrame, asof: date, monthly: bool) -> SeriesInfo:
    y_digit, m_code = int(secid[-1]), secid[-2]
    month = MONTH_CODES.index(m_code) + 1
    last = df.index[-1].date()
    decade = last.year - last.year % 10
    year = decade + y_digit
    if year < last.year - 1:
        year += 10
    # истёкшая серия: последний торговый день = последняя свеча; действующая — правило календаря
    if (asof - last).days > 7:
        exp = last
    else:
        weekday = 4 if base in ("GD", "BR", "NG") else 3
        exp = third_weekday(year, month, weekday) if not monthly else date(year, month, 1)
        if exp < last:
            exp = last
    return SeriesInfo(secid=secid, base=base, expiration=exp, first_trade=df.index[0].date())


def build_daily_universe(root: str | Path, bases: list[str], asof: date, monthly: set[str] = frozenset({"BR", "NG"}),
                         roll: RollConfig | None = None, min_volume: float = 0.0) -> dict[str, DailyUniverseItem]:
    """Строит поток активной серии по каждому базовому активу.

    Перекладка: обязательная за 2 дня до экспирации; досрочная — по объёму 2 дня подряд
    (спред на истории недоступен => early_roll_requires_spread=False, отклонение A-08 для дневного исследования).
    """
    root = Path(root)
    roll = roll or RollConfig(early_roll_requires_spread=False)
    res = {}
    for b in bases:
        d = root / b
        if not d.exists():
            continue
        ser, issues = load_series_dir(d)
        if not ser:
            continue
        infos = [series_info(b, s, df, asof, b in monthly) for s, df in ser.items()]
        rows = []
        for s, df in ser.items():
            rows.append(pd.DataFrame({"trading_day": df.index.date, "secid": s, "volume": df["volume"].to_numpy()}))
        daily = pd.concat(rows, ignore_index=True)
        tdays = sorted(set(daily["trading_day"]))
        active = select_active_series(infos, daily, tdays, roll)
        by_sec = {}
        for s, df in ser.items():
            x = df.copy()
            x["trading_day"] = x.index.date
            x.index = pd.DatetimeIndex(x.index).tz_localize("UTC")
            by_sec[s] = x
        stream = stitch_active(by_sec, active)
        if stream.empty:
            continue
        stream = annotate_daily(stream)
        stream.index.name = "ts"
        low_vol = (stream["volume"] < min_volume).sum()
        if low_vol:
            issues.append(f"{b}: {int(low_vol)} active days with volume < {min_volume}")
        res[b] = DailyUniverseItem(base=b, stream=stream, series=by_sec, active=active, issues=issues)
    return res
