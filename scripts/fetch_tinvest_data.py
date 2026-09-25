"""Загрузка ВСЕХ исследовательских данных из T-Invest API (песочница, только чтение) за последний год.

TBANK_CA_BUNDLE=/root/.certs/ca-bundle-ru.crt python3 scripts/fetch_tinvest_data.py [--days 365]

Всё пишется в DuckDB (data/db/market.duckdb, см. moexlab.storage.db); уже запрошенные сутки
(fetch_log) повторно не скачиваются. После загрузки обновляется снимок data/db/snapshot (Parquet, в git).

Что загружается (период: [asof - days, asof]):
  * спецификации фьючерсов Si, RI, MX, RN, GD, SR, GZ, CR, BR (шаг цены, стоимость шага, лот, экспирация, ГО)
    и 12 акций TQBR -> data/raw/tinvest/instruments.json;
  * дневные свечи API всех серий, торговавшихся в периоде -> data/raw/tinvest/daily/<BASE>/<TICKER>.csv
    (только для выбора серий и сверки; дневные свечи исследования строятся из минуток по торговым дням);
  * минутные свечи серий, у которых хотя бы в один день периода доля дневного объёма базового актива ≥ 10%
    (активная серия и следующая вокруг перекладки) -> data/raw/tinvest/minute/<BASE>/<TICKER>.parquet;
  * минутные свечи 12 акций TQBR -> data/raw/tinvest/minute/EQ/<TICKER>.parquet.
Минутки запрашиваются по одним суткам UTC (limit=2400 > 1010 минут самой длинной сессии).
Лимит MarketData 600 запросов/мин; клиент держит паузу 0,12 с и повторяет при 429/5xx.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moexlab.broker.tinvest import TInvestClient, quotation  # noqa: E402
from moexlab.storage.db import MarketDB  # noqa: E402

BASES = ["Si", "RI", "MX", "RN", "GD", "SR", "GZ", "CR", "BR"]
EQUITIES = ["SBER", "GAZP", "LKOH", "ROSN", "GMKN", "NVTK", "TATN", "MGNT", "PLZL", "VTBR", "ALRS", "AFLT"]
SOURCE = "tinvest-sandbox GetCandles"


def q(x) -> float:
    return quotation(x) if isinstance(x, dict) else float("nan")


def fetch_minutes(cli: TInvestClient, db: MarketDB, ticker: str, uid: str, start: date, end: date,
                  asof: date) -> int:
    """Качает минутки по суткам UTC, пропуская уже записанные в fetch_log. Незавершённые сутки (asof) не логируются."""
    done = db.fetched_days(ticker, "1m")
    n = 0
    d = start
    while d <= end:
        if d in done:
            d += timedelta(days=1)
            continue
        t0 = pd.Timestamp(d, tz="UTC")
        df = cli.candles(uid, t0, t0 + pd.Timedelta(days=1), "CANDLE_INTERVAL_1_MIN", limit=2400)
        if len(df):
            if len(df) >= 2400:
                raise RuntimeError(f"{ticker} {d}: limit reached, split the request")
            df = df[df["complete"]]
        k = db.upsert_candles(ticker, "1m", df)
        if d < asof:
            db.log_fetch(ticker, "1m", [d], {d: k}, SOURCE)
        n += k
        d += timedelta(days=1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--asof", default=str(date.today()))
    ap.add_argument("--share-threshold", type=float, default=0.10)
    ap.add_argument("--lead-days", type=int, default=40,
                    help="история серии до первого дня с долей объёма >= порога (R-Breaker §7-8: 20 дней серии)")
    ap.add_argument("--skip-minutes", action="store_true")
    a = ap.parse_args()
    asof = date.fromisoformat(a.asof)
    start = asof - timedelta(days=a.days)
    cli = TInvestClient(sandbox=True)
    db = MarketDB()
    meta: dict[str, dict] = {}

    # --- фьючерсы: список серий, пересекающихся с периодом ---
    fut = cli.futures("INSTRUMENT_STATUS_ALL")
    fut = fut[fut["ticker"].str.match(r"^(%s)[FGHJKMNQUVXZ]\d$" % "|".join(BASES))].copy()
    fut["exp"] = pd.to_datetime(fut["expirationDate"]).dt.date
    fut["first"] = pd.to_datetime(fut["firstTradeDate"]).dt.date
    fut = fut[(fut["exp"] >= start) & (fut["first"] <= asof)]
    for _, r in fut.iterrows():
        base = r["ticker"][:-2]
        meta[r["ticker"]] = dict(kind="future", base=base, uid=r["uid"], figi=r["figi"], ticker=r["ticker"],
                                 class_code=r["classCode"], lot=int(r["lot"]),
                                 tick=q(r["minPriceIncrement"]), tick_value=q(r["minPriceIncrementAmount"]),
                                 expiration=str(r["exp"]), last_trade=r["lastTradeDate"],
                                 first_trade=r["firstTradeDate"], basic_asset_size=q(r.get("basicAssetSize")),
                                 initial_margin_buy=q(r.get("initialMarginOnBuy")),
                                 initial_margin_sell=q(r.get("initialMarginOnSell")))
    for t in EQUITIES:
        s = cli.share(t)
        meta[t] = dict(kind="equity", base=t, uid=s["uid"], figi=s["figi"], ticker=t, class_code="TQBR",
                       lot=int(s["lot"]), tick=q(s.get("minPriceIncrement")), tick_value=float("nan"),
                       isin=s.get("isin"))
    db.upsert_instruments(meta)

    # --- дневные свечи API (выбор серий, сверка) ---
    daily = {}
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(asof + timedelta(days=1), tz="UTC")
    for tick, m in meta.items():
        df = cli.candles(m["uid"], t0, t1, "CANDLE_INTERVAL_DAY")
        if df.empty:
            print(tick, "no daily candles", flush=True)
            continue
        df = df[df["complete"]].drop(columns="complete")
        db.upsert_candles(tick, "1d", df)
        daily[tick] = df
        print("daily", tick, len(df), flush=True)

    if a.skip_minutes:
        db.snapshot()
        return
    # --- какие серии качать минутками ---
    need = []
    for base in BASES:
        ticks = [t for t in daily if meta[t]["kind"] == "future" and meta[t]["base"] == base]
        if not ticks:
            continue
        V = pd.DataFrame({t: daily[t]["volume"] for t in ticks}).fillna(0.0)
        share = V.div(V.sum(axis=1).replace(0, float("nan")), axis=0)
        for t in ticks:
            days = share.index[share[t] >= a.share_threshold]
            if len(days):
                need.append((base, t, max(start, days.min().date() - timedelta(days=a.lead_days)),
                             min(asof, days.max().date() + timedelta(days=1))))
    need += [("EQ", t, start, asof) for t in EQUITIES if t in daily]
    for sub, tick, d0, d1 in need:
        n = fetch_minutes(cli, db, tick, meta[tick]["uid"], d0, d1, asof)
        print("minute", sub, tick, d0, d1, "new rows", n, flush=True)
    print(db.coverage().to_string())
    db.snapshot()
    db.close()


if __name__ == "__main__":
    main()
