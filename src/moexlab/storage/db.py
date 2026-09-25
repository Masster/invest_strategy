"""Локальная база рыночных данных (DuckDB) — единый источник для всех исследований.

Таблицы:
  instruments(ticker PK, kind, base, uid, figi, class_code, lot, tick, tick_value, expiration, meta JSON)
  candles(ticker, interval, ts, open, high, low, close, volume, PK(ticker, interval, ts))
      interval: '1m' | '1d'; ts — начало свечи, UTC.
  fetch_log(ticker, interval, day, n_rows, source, fetched_at, PK(ticker, interval, day))
      отмечает УЖЕ запрошенные сутки (в т.ч. пустые) — повторная загрузка их не запрашивает.

Контейнер эфемерный, поэтому источник истины для хранения между сессиями — снимок в Parquet
(data/db/snapshot/<table>/...; сжатие zstd), который коммитится в git. `MarketDB.restore()` восстанавливает
базу из снимка без обращения к API; `MarketDB.snapshot()` обновляет снимок.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data/db/market.duckdb"
SNAPSHOT = ROOT / "data/db/snapshot"

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    ticker VARCHAR PRIMARY KEY, kind VARCHAR, base VARCHAR, uid VARCHAR, figi VARCHAR, class_code VARCHAR,
    lot INTEGER, tick DOUBLE, tick_value DOUBLE, expiration DATE, meta JSON);
CREATE TABLE IF NOT EXISTS candles (
    ticker VARCHAR, interval VARCHAR, ts TIMESTAMPTZ, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume DOUBLE, PRIMARY KEY (ticker, interval, ts));
CREATE TABLE IF NOT EXISTS fetch_log (
    ticker VARCHAR, interval VARCHAR, day DATE, n_rows INTEGER, source VARCHAR, fetched_at TIMESTAMPTZ,
    PRIMARY KEY (ticker, interval, day));
"""


class MarketDB:
    def __init__(self, path: str | Path = DB_PATH, read_only: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if read_only and not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found: run MarketDB().restore() or scripts/fetch_tinvest_data.py")
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        self.con.execute("SET TimeZone='UTC'")
        if not read_only:
            self.con.execute(SCHEMA)

    def close(self):
        self.con.close()

    # ------------------------------------------------------------------ запись
    def upsert_instruments(self, meta: dict[str, dict]) -> None:
        rows = []
        for t, m in meta.items():
            exp = m.get("expiration")
            rows.append(dict(ticker=t, kind=m.get("kind"), base=m.get("base"), uid=m.get("uid"), figi=m.get("figi"),
                             class_code=m.get("class_code"), lot=int(m.get("lot") or 1), tick=m.get("tick"),
                             tick_value=m.get("tick_value"), expiration=pd.to_datetime(exp).date() if exp else None,
                             meta=json.dumps(m, ensure_ascii=False, default=str)))
        df = pd.DataFrame(rows)  # noqa: F841 — используется DuckDB по имени
        self.con.execute("INSERT OR REPLACE INTO instruments SELECT * FROM df")

    def upsert_candles(self, ticker: str, interval: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        x = df[["open", "high", "low", "close", "volume"]].copy()
        x.insert(0, "ts", pd.DatetimeIndex(df.index).tz_convert("UTC"))
        x.insert(0, "interval", interval)
        x.insert(0, "ticker", ticker)
        self.con.execute("INSERT OR REPLACE INTO candles SELECT * FROM x")
        return len(x)

    def log_fetch(self, ticker: str, interval: str, days: list[date], counts: dict[date, int], source: str) -> None:
        if not days:
            return
        lg = pd.DataFrame(dict(ticker=ticker, interval=interval, day=days, n_rows=[counts.get(d, 0) for d in days],
                               source=source, fetched_at=pd.Timestamp.now(tz="UTC")))
        self.con.execute("INSERT OR REPLACE INTO fetch_log SELECT * FROM lg")

    # ------------------------------------------------------------------ чтение
    def fetched_days(self, ticker: str, interval: str) -> set[date]:
        r = self.con.execute("SELECT day FROM fetch_log WHERE ticker=? AND interval=?", [ticker, interval]).fetchall()
        return {x[0] for x in r}

    def instruments(self, kind: str | None = None, base: str | None = None) -> pd.DataFrame:
        q, p = "SELECT * FROM instruments WHERE 1=1", []
        if kind:
            q += " AND kind=?"
            p.append(kind)
        if base:
            q += " AND base=?"
            p.append(base)
        return self.con.execute(q + " ORDER BY base, expiration, ticker", p).df()

    def candles(self, ticker: str, interval: str = "1m", start=None, end=None) -> pd.DataFrame:
        q, p = "SELECT ts, open, high, low, close, volume FROM candles WHERE ticker=? AND interval=?", [ticker, interval]
        if start is not None:
            q += " AND ts >= ?"
            p.append(pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start))
        if end is not None:
            q += " AND ts < ?"
            p.append(pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end))
        df = self.con.execute(q + " ORDER BY ts", p).df()
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.set_index("ts")

    def coverage(self) -> pd.DataFrame:
        return self.con.execute("""
            SELECT c.ticker, c.interval, count(*) AS n, min(ts) AS first_ts, max(ts) AS last_ts, i.kind, i.base
            FROM candles c LEFT JOIN instruments i USING (ticker)
            GROUP BY c.ticker, c.interval, i.kind, i.base ORDER BY i.base, c.ticker, c.interval""").df()

    # ------------------------------------------------------------------ снимок для git
    def snapshot(self, root: Path = SNAPSHOT) -> None:
        """Выгрузка в Parquet (zstd): свечи — по файлу на (interval, ticker), чтобы файлы были < 50 МБ."""
        root.mkdir(parents=True, exist_ok=True)
        for tbl in ("instruments", "fetch_log"):
            self.con.execute(f"COPY {tbl} TO '{root / (tbl + '.parquet')}' (FORMAT parquet, COMPRESSION zstd)")
        for t, iv in self.con.execute("SELECT DISTINCT ticker, interval FROM candles").fetchall():
            d = root / "candles" / iv
            d.mkdir(parents=True, exist_ok=True)
            self.con.execute(f"""COPY (SELECT * FROM candles WHERE ticker='{t}' AND interval='{iv}' ORDER BY ts)
                                 TO '{d / (t + '.parquet')}' (FORMAT parquet, COMPRESSION zstd)""")

    def restore(self, root: Path = SNAPSHOT) -> None:
        """Восстановление базы из снимка (без сети)."""
        for tbl in ("instruments", "fetch_log"):
            p = root / f"{tbl}.parquet"
            if p.exists():
                self.con.execute(f"INSERT OR REPLACE INTO {tbl} SELECT * FROM read_parquet('{p}')")
        files = sorted((root / "candles").glob("*/*.parquet"))
        for p in files:
            self.con.execute(f"INSERT OR REPLACE INTO candles SELECT * FROM read_parquet('{p}')")


def open_db(read_only: bool = True) -> MarketDB:
    """База для исследований; если файла нет, но есть снимок — восстанавливает его автоматически."""
    if not DB_PATH.exists() and (SNAPSHOT / "instruments.parquet").exists():
        db = MarketDB()
        db.restore()
        db.close()
    return MarketDB(read_only=read_only)
