"""Нормализованный слой данных для глубокого поиска.

raw  : data/db/snapshot (Parquet, неизменяем; читается через DuckDB in-memory — запись невозможна)
norm : data/normalized/<CODE>_1m.parquet — минутные свечи торгуемого потока инструмента
cache: data/cache/<CODE>_<TF>_<ver>.parquet — агрегированные свечи (ключ = версия данных + версия кода)

Правила нормализации (все воспроизводимы, исходник не меняется):
  * время — UTC в хранении, правила сессий — по МСК (UTC+3, без перехода на летнее время);
  * торговый день = календарная дата МСК будней; сессии: 0 = утренняя (<10:00), 1 = основная (10:00–19:00),
    2 = вечерняя (>=19:00). Свечи выходных дней и ночные свечи 00:00–06:49 МСК исключены (торговать в них
    стратегиям запрещено; позиция, перенесённая через выходные, переоценивается по первой свече понедельника);
  * последний день снимка 2026-09-25 неполный (выгрузка в 18:13 МСК) — исключён; LAST_DAY = 2026-09-24;
  * фьючерсы: поток активной серии; выбор серии на день D только по объёмам завершённых дней < D
    (moexlab.contracts.series.select_active_series): обязательная перекладка за 2 торговых дня до последнего
    дня торгов, досрочная — если объём следующей серии больше 2 дня подряд;
  * сигнальные цены a_* — мультипликативная форвард-корректировка (коэффициент известен в момент перекладки,
    каузально); цены исполнения — только реальные цены активной серии;
  * флаг new_series — первая свеча новой серии: движок обязан закрыть позицию на последней свече старой серии.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..contracts.series import RollConfig, SeriesInfo, select_active_series

ROOT = Path(__file__).resolve().parents[3]
SNAP = ROOT / "data/db/snapshot"
NORM = ROOT / "data/normalized"
CACHE = ROOT / "data/cache"
CODE_VERSION = "norm-v3"

FIRST_DAY = date(2025, 9, 25)
LAST_DAY = date(2026, 9, 24)            # 2026-09-25 неполный
DEV_END = date(2026, 6, 30)             # конец разработки (включительно)
HOLDOUT_START = date(2026, 7, 1)        # FINAL HOLDOUT 2026-07-01 … 2026-09-24 (ранее частично видел v1-анализ)

FUT_BASES = ["Si", "CR", "RI", "MX", "BR", "GD", "SR", "GZ", "RN"]
EQUITIES = ["SBER", "GAZP", "LKOH", "ROSN", "GMKN", "NVTK", "TATN", "MGNT", "PLZL", "VTBR", "ALRS", "AFLT"]
ALL_CODES = FUT_BASES + EQUITIES
TIMEFRAMES = [1, 2, 3, 5, 10, 15, 30, 60, 1440]

MSK_OFFSET = pd.Timedelta(hours=3)

# комиссии (research/sources.md §4): Трейдер — фьючерсы 0,04% (< 5 млн ₽/день), акции 0,05%;
# Премиум — 0,025% / 0,04%. Биржевой сбор включён в брокерскую комиссию.
COMMISSION = {"TRADER": {"future": 0.0004, "equity": 0.0005}, "PREMIUM": {"future": 0.00025, "equity": 0.0004}}


def _con():
    c = duckdb.connect()   # in-memory: снимок читается как внешние Parquet-файлы, запись в него невозможна
    c.execute("SET TimeZone='UTC'")
    return c


def data_version() -> str:
    h = hashlib.sha256()
    for p in sorted(SNAP.rglob("*.parquet")):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]


@lru_cache(maxsize=1)
def instruments() -> pd.DataFrame:
    df = _con().execute(f"SELECT * FROM read_parquet('{SNAP / 'instruments.parquet'}')").df()
    meta = df["meta"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    df["last_trade"] = meta.map(lambda m: m.get("last_trade"))
    df["im_buy"] = meta.map(lambda m: m.get("initial_margin_buy"))
    df["im_sell"] = meta.map(lambda m: m.get("initial_margin_sell"))
    df["basic_asset_size"] = meta.map(lambda m: m.get("basic_asset_size"))
    return df.drop(columns=["meta"])


def raw_candles(ticker: str, interval: str = "1m") -> pd.DataFrame:
    p = SNAP / "candles" / interval / f"{ticker}.parquet"
    df = _con().execute(f"SELECT ts, open, high, low, close, volume FROM read_parquet('{p}') ORDER BY ts").df()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _session_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Будние торговые свечи 06:50–23:59 МСК, торговый день = дата МСК, код сессии."""
    msk = df["ts"].dt.tz_convert(None) + MSK_OFFSET
    mins = msk.dt.hour * 60 + msk.dt.minute
    wd = msk.dt.weekday
    keep = (wd < 5) & (mins >= 6 * 60 + 50)
    out = df[keep.to_numpy()].copy()
    m = mins[keep].to_numpy()
    out["day"] = msk[keep].dt.date.to_numpy()
    out["mod"] = m.astype(np.int16)                      # минута дня МСК (начало свечи)
    out["sess"] = np.where(m < 600, 0, np.where(m < 1140, 1, 2)).astype(np.int8)
    out = out[(out["day"] >= FIRST_DAY) & (out["day"] <= LAST_DAY)]
    return out.reset_index(drop=True)


def _main_days(df: pd.DataFrame) -> list[date]:
    return sorted(set(df.loc[df["sess"] == 1, "day"]))


def build_future(base: str, roll: RollConfig | None = None) -> tuple[pd.DataFrame, dict]:
    roll = roll or RollConfig(early_roll_requires_spread=False)
    ins = instruments()
    ins = ins[(ins["kind"] == "future") & (ins["base"] == base)]
    series, infos, rows = {}, [], []
    for _, r in ins.iterrows():
        p = SNAP / "candles/1m" / f"{r['ticker']}.parquet"
        if not p.exists():
            continue
        d = _session_filter(raw_candles(r["ticker"]))
        if d.empty:
            continue
        series[r["ticker"]] = d
        lt = pd.Timestamp(r["last_trade"]) if r["last_trade"] else pd.Timestamp(r["expiration"])
        exp = (lt.tz_convert(None) + MSK_OFFSET).date() if lt.tzinfo else lt.date()
        infos.append(SeriesInfo(secid=r["ticker"], base=base, expiration=exp, first_trade=d["day"].iloc[0]))
        g = d.groupby("day")["volume"].sum()
        rows.append(pd.DataFrame({"trading_day": g.index, "secid": r["ticker"], "volume": g.to_numpy()}))
    daily = pd.concat(rows, ignore_index=True)
    tdays = sorted(set().union(*[set(_main_days(s)) for s in series.values()]))
    active = select_active_series(infos, daily, tdays, roll)
    parts, factor, prev = [], 1.0, None
    log = []
    for d, sec in active.sort_index().items():
        s = series[sec]
        sel = s[s["day"] == d]
        if sel.empty:
            log.append(f"{d}: no bars for active {sec}")
            continue
        if prev is not None and sec != prev:
            # коэффициент по последним закрытиям обеих серий ДО первой свечи дня перекладки (каузально)
            t0 = sel["ts"].iloc[0]
            po = series[prev]
            po = po[po["ts"] < t0]
            pn = s[s["ts"] < t0]
            if len(po) and len(pn):
                factor *= float(po["close"].iloc[-1]) / float(pn["close"].iloc[-1])
                log.append(f"{d}: roll {prev}->{sec} ratio={po['close'].iloc[-1] / pn['close'].iloc[-1]:.5f}")
            else:
                log.append(f"{d}: roll {prev}->{sec} without overlap (factor unchanged)")
        sel = sel.copy()
        sel["secid"] = sec
        sel["adj"] = factor
        parts.append(sel)
        prev = sec
    out = pd.concat(parts, ignore_index=True)
    return out, {"roll_log": log, "active": {str(k): v for k, v in active.items()}}


def build_equity(ticker: str) -> tuple[pd.DataFrame, dict]:
    d = _session_filter(raw_candles(ticker))
    d["secid"] = ticker
    d["adj"] = 1.0
    return d, {"roll_log": []}


def spec(code: str) -> dict:
    """Спецификация для потока: шаг, стоимость шага (текущая), лот, вид, ГО% (текущее ГО / текущий номинал)."""
    ins = instruments()
    if code in EQUITIES:
        r = ins[ins["ticker"] == code].iloc[0]
        return {"code": code, "kind": "equity", "tick": float(r["tick"]), "lot": int(r["lot"]),
                "tick_value": float(r["tick"]) * int(r["lot"]), "margin_pct": None}
    r = ins[(ins["base"] == code) & ins["tick_value"].notna()].sort_values("expiration").iloc[0]
    return {"code": code, "kind": "future", "tick": float(r["tick"]), "lot": 1,
            "tick_value": float(r["tick_value"]), "im": float(r["im_buy"] or np.nan), "ref_series": r["ticker"]}


def normalized(code: str, rebuild: bool = False) -> pd.DataFrame:
    """Минутный поток инструмента (с кэшем в data/normalized)."""
    NORM.mkdir(parents=True, exist_ok=True)
    p = NORM / f"{code}_1m.parquet"
    mp = NORM / f"{code}_1m.json"
    ver = f"{CODE_VERSION}:{data_version()}"
    if p.exists() and mp.exists() and not rebuild:
        if json.loads(mp.read_text()).get("version") == ver:
            return pd.read_parquet(p)
    df, info = build_future(code) if code in FUT_BASES else build_equity(code)
    df["new_series"] = (df["secid"] != df["secid"].shift(1)).to_numpy() & (np.arange(len(df)) > 0)
    df.to_parquet(p, index=False)
    info["version"] = ver
    info["rows"] = len(df)
    mp.write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    return df


def resample(df1: pd.DataFrame, tf: int) -> pd.DataFrame:
    """N-минутные (tf<1440) или дневные (tf=1440) свечи; ведро не пересекает день и смену серии."""
    if tf == 1:
        out = df1.copy()
    else:
        if tf >= 1440:
            key = [df1["day"], df1["secid"]]
        else:
            key = [df1["day"], df1["secid"], (df1["mod"] // tf).rename("bucket")]
        g = df1.groupby(key, sort=False)
        out = g.agg(ts=("ts", "first"), mod=("mod", "first"), sess=("sess", "first"), open=("open", "first"),
                    high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                    adj=("adj", "last"), ts_last=("ts", "last")).reset_index()
        out = out.sort_values("ts").reset_index(drop=True)
        if "bucket" in out:
            out = out.drop(columns=["bucket"])
    out["new_series"] = (out["secid"] != out["secid"].shift(1)).to_numpy() & (np.arange(len(out)) > 0)
    return out


def bars(code: str, tf: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    ver = hashlib.sha1(f"{CODE_VERSION}:{data_version()}".encode()).hexdigest()[:8]
    p = CACHE / f"{code}_{tf}_{ver}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    out = resample(normalized(code), tf)
    out.to_parquet(p, index=False)
    return out


def to_arrays(b: pd.DataFrame) -> dict:
    """Numpy-представление свечей для движка/признаков."""
    day = pd.to_datetime(b["day"]).dt.strftime("%Y%m%d").astype(np.int32).to_numpy()
    last_of_day = np.append(day[1:] != day[:-1], True)
    new_series = b["new_series"].to_numpy(bool)
    last_of_series = np.append(new_series[1:], True)
    adj = b["adj"].to_numpy(float)
    return dict(ts=b["ts"].to_numpy("datetime64[ns]").astype(np.int64), day=day, mod=b["mod"].to_numpy(np.int32),
                sess=b["sess"].to_numpy(np.int8), o=b["open"].to_numpy(float), h=b["high"].to_numpy(float),
                l=b["low"].to_numpy(float), c=b["close"].to_numpy(float), v=b["volume"].to_numpy(float),
                ao=b["open"].to_numpy(float) * adj, ah=b["high"].to_numpy(float) * adj,
                al=b["low"].to_numpy(float) * adj, ac=b["close"].to_numpy(float) * adj,
                new_series=new_series, last_of_day=last_of_day, last_of_series=last_of_series,
                first_of_day=np.append(True, day[1:] != day[:-1]))
