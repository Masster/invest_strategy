"""Загрузка минутных свечей реальных серий фьючерсов из архива T-Invest history-data (нужны сеть + TBANK_TOKEN).

python3 scripts/fetch_tinvest_history.py --bases MX RI Si CR GD BR SR GZ --years 2020 2021 2022 2023 2024 2025 2026
Результат: data/raw/tinvest_minute/<BASE>/<TICKER>.parquet + instruments.json (uid, ticker, шаг, стоимость шага, экспирация).
Лимит архива: 30 запросов/мин — пауза 2.1 с между запросами.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moexlab.broker.tinvest import TInvestClient, download_history_year, quotation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["MX", "RI", "Si", "CR", "GD", "BR", "SR", "GZ"])
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2020, 2027)))
    a = ap.parse_args()
    out = ROOT / "data/raw/tinvest_minute"
    out.mkdir(parents=True, exist_ok=True)
    cli = TInvestClient(sandbox=False)
    fut = cli.futures("INSTRUMENT_STATUS_ALL")
    meta = {}
    for _, r in fut.iterrows():
        base = str(r.get("basicAsset", ""))
        tick = str(r.get("ticker", ""))
        if not any(tick.startswith(b) and len(tick) == len(b) + 2 for b in a.bases):
            continue
        meta[tick] = dict(uid=r.get("uid"), figi=r.get("figi"), ticker=tick, basic_asset=base,
                          min_price_increment=quotation(r.get("minPriceIncrement")),
                          expiration=r.get("expirationDate"), last_trade=r.get("lastTradeDate"))
    (out / "instruments.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    for tick, m in sorted(meta.items()):
        b = next(b for b in a.bases if tick.startswith(b))
        for y in a.years:
            p = out / b / f"{tick}_{y}.parquet"
            if p.exists():
                continue
            try:
                df = download_history_year(m["uid"], y)
            except Exception as e:  # noqa: BLE001
                print(tick, y, "ERR", e)
                time.sleep(2.1)
                continue
            if len(df):
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(p)
                print(tick, y, len(df))
            time.sleep(2.1)


if __name__ == "__main__":
    main()
