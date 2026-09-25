"""Прямая загрузка дневных свечей MOEX ISS (DATA_QUALITY = DIRECT_ISS).

Планы загрузки (воспроизводимые списки серий) пишутся в data/raw/iss_daily/plan_{A,B,C}.json
и копируются в data/manifests/ (коммитятся вместе с SHA-256 каждого файла; сами CSV не коммитятся:
репозиторий публичный, а данные MOEX не подлежат перераспространению):
  A — квартальные фьючерсы Si RI MX GD SR GZ CR (месяцы H M U Z), 2020 → 2026-12;
  B — месячные фьючерсы BR, 2020-01 → 2026-12;
  C — акции TQBR (12 тикеров).
Формат результата: data/raw/iss_daily/<BASE>/<SECID>.csv и data/raw/iss_equity_daily/<TICKER>.csv
с колонками date,open,high,low,close,volume. Дата свечи FORTS в ISS = торговый день.

Запуск: python3 scripts/fetch_iss_daily.py [--plans A,B,C] [--from 2019-06-01] [--till 2026-09-25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FUT_DIR = ROOT / "data/raw/iss_daily"
EQ_DIR = ROOT / "data/raw/iss_equity_daily"
MANIFESTS = ROOT / "data/manifests"
ISS = "https://iss.moex.com/iss"
QUARTERLY = ["Si", "RI", "MX", "GD", "SR", "GZ", "CR"]
MONTHLY = ["BR"]
MONTH_CODES = "FGHJKMNQUVXZ"
EQUITIES = ["SBER", "GAZP", "LKOH", "ROSN", "GMKN", "NVTK", "TATN", "MGNT", "PLZL", "VTBR", "ALRS", "AFLT"]


def build_plans(first_year: int = 2020, last_year: int = 2026) -> dict[str, list[dict]]:
    """Коды серий <BASE><месяц><последняя цифра года>. Цифры 0..6 однозначно = 2020..2026."""
    a, b = [], []
    for y in range(first_year, last_year + 1):
        for base in QUARTERLY:
            for m in "HMUZ":
                a.append(dict(base=base, secid=f"{base}{m}{y % 10}", expiry_year=y, month_code=m))
        for base in MONTHLY:
            for m in MONTH_CODES:
                b.append(dict(base=base, secid=f"{base}{m}{y % 10}", expiry_year=y, month_code=m))
    c = [dict(base=t, secid=t, board="TQBR") for t in EQUITIES]
    return {"A": a, "B": b, "C": c}


def fetch_candles(url: str, frm: str, till: str, session: requests.Session) -> pd.DataFrame:
    rows, start = [], 0
    while True:
        params = {"interval": 24, "from": frm, "till": till, "iss.meta": "off", "start": start}
        for attempt in range(5):
            try:
                r = session.get(url, params=params, timeout=30)
                r.raise_for_status()
                js = r.json()["candles"]
                break
            except Exception:  # сеть/ISS: повтор с паузой
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        data = js["data"]
        rows += [dict(zip(js["columns"], d)) for d in data]
        if len(data) < 500:
            break
        start += len(data)
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["begin"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date").sort_values("date")
    return df


def fetch_item(item: dict, plan: str, frm: str, till: str, session: requests.Session) -> tuple[str, int]:
    if plan == "C":
        url = f"{ISS}/engines/stock/markets/shares/boards/{item['board']}/securities/{item['secid']}/candles.json"
        out = EQ_DIR / f"{item['secid']}.csv"
    else:
        url = f"{ISS}/engines/futures/markets/forts/securities/{item['secid']}/candles.json"
        out = FUT_DIR / item["base"] / f"{item['secid']}.csv"
    df = fetch_candles(url, frm, till, session)
    # серии без сделок (свечи с нулевым объёмом) не нужны: цены там — расчётные, не торговые
    df = df[df["volume"] > 0]
    if df.empty:
        return item["secid"], 0
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return item["secid"], len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="A,B,C")
    ap.add_argument("--from", dest="frm", default="2019-06-01")
    ap.add_argument("--till", default="2026-09-25")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    plans = build_plans()
    FUT_DIR.mkdir(parents=True, exist_ok=True)
    for k, v in plans.items():
        (FUT_DIR / f"plan_{k}.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    session = requests.Session()
    summary = {}
    for k in a.plans.split(","):
        items = plans[k]
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda it: fetch_item(it, k, a.frm, a.till, session), items))
        got = {s: n for s, n in res if n}
        summary[k] = dict(requested=len(items), with_data=len(got), bars=sum(got.values()))
        print(f"plan {k}: {summary[k]}", flush=True)
    summ = json.dumps(dict(source="MOEX ISS direct", frm=a.frm, till=a.till, plans=summary), indent=1)
    (FUT_DIR / "fetch_summary.json").write_text(summ, encoding="utf-8")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "iss_daily_fetch_summary.json").write_text(summ, encoding="utf-8")
    for k, v in plans.items():
        (MANIFESTS / f"iss_daily_plan_{k}.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    rows = []
    for d in (FUT_DIR, EQ_DIR):
        for p in sorted(d.rglob("*.csv")):
            b = p.read_bytes()
            rows.append(dict(path=str(p.relative_to(ROOT)), rows=b.count(b"\n") - 1,
                             sha256=hashlib.sha256(b).hexdigest()))
    pd.DataFrame(rows).to_csv(MANIFESTS / "iss_daily_sha256.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
