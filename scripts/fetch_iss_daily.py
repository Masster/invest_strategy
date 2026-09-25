"""Прямая загрузка дневных свечей из MOEX ISS (history) — реальные серии фьючерсов и акции TQBR.

python3 scripts/fetch_iss_daily.py [--futures] [--equities] [--from 2020-01-01]
Результат:
  data/raw/iss_daily/<BASE>/<SECID>.csv        (date,open,high,low,close,volume,open_interest)
  data/raw/iss_equity_daily/<TICKER>.csv        (date,open,high,low,close,volume)
  data/raw/iss_manifest.json                    (источник, дата загрузки, число строк, sha256)
Источник: https://iss.moex.com/iss/history/engines/... (DATA_QUALITY = ISS_DIRECT).
Дни без сделок (OPEN/CLOSE пусты или объём 0) отбрасываются.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
ISS = "https://iss.moex.com/iss/history/engines"
QUARTERLY = ["Si", "RI", "MX", "GD", "SR", "GZ", "CR"]
MONTHLY = ["BR"]
MONTH_CODES = "FGHJKMNQUVXZ"
EQUITIES = ["SBER", "GAZP", "LKOH", "ROSN", "GMKN", "NVTK", "TATN", "MGNT", "PLZL", "VTBR", "ALRS", "AFLT"]


def fetch_history(path: str, start: str, cols: str, s: requests.Session) -> pd.DataFrame:
    rows, pos = [], 0
    while True:
        r = s.get(f"{ISS}/{path}.json", timeout=60, params={
            "iss.meta": "off", "from": start, "start": pos, "history.columns": cols})
        r.raise_for_status()
        js = r.json()
        h = js["history"]
        rows += h["data"]
        idx, total, size = js["history.cursor"]["data"][0]
        pos = idx + size
        if pos >= total or not h["data"]:
            break
        time.sleep(0.05)
    return pd.DataFrame(rows, columns=cols.split(","))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"TRADEDATE": "date", "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
                            "VOLUME": "volume", "OPENPOSITION": "open_interest"})
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"].fillna(0) > 0]
    # на TQBR бывают дубликаты режимов торгов — оставляем строку с максимальным объёмом
    df = df.sort_values(["date", "volume"]).drop_duplicates("date", keep="last").sort_values("date")
    return df.drop(columns=[c for c in ("SECID", "BOARDID") if c in df.columns]).reset_index(drop=True)


def futures_secids(base: str, y0: int, y1: int) -> list[str]:
    months = MONTH_CODES if base in MONTHLY else "HMUZ"
    return [f"{base}{m}{y % 10}" for y in range(y0, y1 + 1) for m in months]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--futures", action="store_true")
    ap.add_argument("--equities", action="store_true")
    ap.add_argument("--from", dest="start", default="2020-01-01")
    a = ap.parse_args()
    if not (a.futures or a.equities):
        a.futures = a.equities = True
    s = requests.Session()
    man_p = ROOT / "data/raw/iss_manifest.json"
    manifest = json.loads(man_p.read_text()) if man_p.exists() else {}
    y0, y1 = int(a.start[:4]), date.today().year + 1
    if a.futures:
        cols = "TRADEDATE,SECID,OPEN,LOW,HIGH,CLOSE,VOLUME,OPENPOSITION"
        for base in QUARTERLY + MONTHLY:
            for secid in futures_secids(base, y0, y1):
                df = clean(fetch_history(f"futures/markets/forts/securities/{secid}", a.start, cols, s))
                # код серии повторяется каждые 10 лет: оставляем торги не раньше чем за 2 года до экспирации
                if df.empty:
                    continue
                exp_year = y0 + (int(secid[-1]) - y0) % 10
                while exp_year < pd.Timestamp(df["date"].iloc[-1]).year:
                    exp_year += 10
                df = df[pd.to_datetime(df["date"]).dt.year >= exp_year - 2]
                if df.empty:
                    continue
                p = ROOT / "data/raw/iss_daily" / base / f"{secid}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p, index=False)
                manifest[str(p.relative_to(ROOT))] = dict(
                    source=f"{ISS}/futures/markets/forts/securities/{secid}.json", rows=len(df),
                    first=df["date"].iloc[0], last=df["date"].iloc[-1], fetched=str(date.today()),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                print(secid, len(df), df["date"].iloc[0], df["date"].iloc[-1], flush=True)
    if a.equities:
        cols = "TRADEDATE,BOARDID,OPEN,LOW,HIGH,CLOSE,VOLUME"
        for t in EQUITIES:
            df = clean(fetch_history(f"stock/markets/shares/boards/TQBR/securities/{t}", a.start, cols, s))
            p = ROOT / "data/raw/iss_equity_daily" / f"{t}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
            manifest[str(p.relative_to(ROOT))] = dict(
                source=f"{ISS}/stock/markets/shares/boards/TQBR/securities/{t}.json", rows=len(df),
                first=df["date"].iloc[0], last=df["date"].iloc[-1], fetched=str(date.today()),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            print(t, len(df), df["date"].iloc[0], df["date"].iloc[-1], flush=True)
    man_p.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
