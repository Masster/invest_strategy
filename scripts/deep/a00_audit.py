"""Аудит данных снимка T-Invest -> research/audit/*.csv (+ цифры для research/DATA_AUDIT.md).

Только чтение: DuckDB in-memory над Parquet. Ничего не пишет в data/db/snapshot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import data as D  # noqa: E402

OUT = D.ROOT / "research/audit"
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET TimeZone='UTC'")
G = f"read_parquet('{D.SNAP}/candles/1m/*.parquet')"
GD = f"read_parquet('{D.SNAP}/candles/1d/*.parquet')"

res = {}
# 1. схема
res["schema_1m"] = con.execute(f"DESCRIBE SELECT * FROM {G}").df()[["column_name", "column_type"]].to_dict("records")
res["schema_instruments"] = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{D.SNAP}/instruments.parquet')").df()[["column_name", "column_type"]].to_dict("records")
res["schema_fetch_log"] = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{D.SNAP}/fetch_log.parquet')").df()[["column_name", "column_type"]].to_dict("records")

# 2. покрытие и целостность по тикерам
cov = con.execute(f"""
SELECT ticker, count(*) n, min(ts) first_ts, max(ts) last_ts,
  count(*) - count(DISTINCT ts) dup_ts,
  sum(CASE WHEN high < greatest(open, close) OR low > least(open, close) OR high < low THEN 1 ELSE 0 END) ohlc_viol,
  sum(CASE WHEN least(open, high, low, close) <= 0 THEN 1 ELSE 0 END) nonpos,
  sum(CASE WHEN volume <= 0 THEN 1 ELSE 0 END) zero_vol,
  sum(CASE WHEN epoch(ts)::BIGINT % 60 <> 0 THEN 1 ELSE 0 END) not_minute_aligned,
  sum(CASE WHEN high = low THEN 1 ELSE 0 END) flat_bars,
  sum(volume) volume,
  sum(CASE WHEN dayofweek(ts + INTERVAL 3 HOUR) IN (0, 6) THEN 1 ELSE 0 END) weekend_bars,
  sum(CASE WHEN dayofweek(ts + INTERVAL 3 HOUR) NOT IN (0, 6) AND extract(hour FROM ts + INTERVAL 3 HOUR) < 7
           AND NOT (extract(hour FROM ts + INTERVAL 3 HOUR) = 6 AND extract(minute FROM ts) >= 50) THEN 1 ELSE 0 END) night_bars
FROM {G} GROUP BY 1 ORDER BY 1""").df()
cov.to_csv(OUT / "coverage_1m_raw.csv", index=False)
dup_rows = con.execute(f"SELECT count(*) FROM (SELECT ticker, ts, count(*) c FROM {G} GROUP BY 1,2 HAVING c > 1)").fetchone()[0]
res["duplicate_ticker_ts"] = int(dup_rows)

# 3. распределение по часам МСК (будни), по сессиям и месяцам
hours = con.execute(f"""
SELECT CASE WHEN length(ticker) <= 4 AND ticker = upper(ticker) AND ticker NOT SIMILAR TO '.*[FGHJKMNQUVXZ][0-9]' THEN ticker
            ELSE regexp_replace(ticker, '[FGHJKMNQUVXZ][0-9]$', '') END code,
  strftime(ts + INTERVAL 3 HOUR, '%Y-%m') ym, extract(hour FROM ts + INTERVAL 3 HOUR) h, count(*) n, sum(volume) v
FROM {G} WHERE dayofweek(ts + INTERVAL 3 HOUR) NOT IN (0, 6) GROUP BY 1,2,3 ORDER BY 1,2,3""").df()
hours.to_csv(OUT / "bars_by_code_month_hour_msk.csv", index=False)

# 4. нормализованный слой: дни, пропуски минут в основной сессии, выбросы, гэпы, перекладки
rows, outl, gaps, days_all = [], [], [], {}
for code in D.ALL_CODES:
    df = D.normalized(code)
    days = sorted(set(df["day"]))
    days_all[code] = set(days)
    main = df[df["sess"] == 1]
    per_day = main.groupby("day").size()
    exp_main = 540  # 10:00–19:00
    c = df["close"].to_numpy(float)
    same = (df["secid"].to_numpy() == np.roll(df["secid"].to_numpy(), 1))
    same[0] = False
    lr = np.full(len(c), np.nan)
    lr[1:] = np.log(c[1:] / c[:-1])
    lr[~same] = np.nan
    first = np.append(True, df["day"].to_numpy()[1:] != df["day"].to_numpy()[:-1])
    intr = lr.copy()
    intr[first] = np.nan               # внутри дня
    med = np.nanmedian(np.abs(intr))
    mad = med / 0.6745 if med > 0 else np.nanstd(intr)
    z = np.abs(intr) / mad
    big = np.where(z > 25)[0]
    for i in big[np.argsort(-z[big])][:10]:
        outl.append(dict(code=code, ts=df["ts"].iloc[i], secid=df["secid"].iloc[i], prev_close=c[i - 1], close=c[i],
                         ret_pct=100 * (np.exp(lr[i]) - 1), robust_z=z[i], volume=df["volume"].iloc[i]))
    ov = lr.copy()
    ov[~first] = np.nan                # межсессионные (овернайт) гэпы
    gi = np.where(np.abs(ov) > 0.03)[0]
    for i in gi:
        gaps.append(dict(code=code, day=df["day"].iloc[i], secid=df["secid"].iloc[i], prev_close=c[i - 1],
                         open=df["open"].iloc[i], gap_pct=100 * (df["open"].iloc[i] / c[i - 1] - 1)))
    info = json.loads((D.NORM / f"{code}_1m.json").read_text())
    rows.append(dict(code=code, bars=len(df), days=len(days), first_day=days[0], last_day=days[-1],
                     series=df["secid"].nunique(), rolls=len([x for x in info["roll_log"] if "roll" in x]),
                     main_bars_median=int(per_day.median()), main_fill_pct=round(100 * per_day.median() / exp_main, 1),
                     days_main_lt50pct=int((per_day < exp_main * 0.5).sum()),
                     sess_morning_pct=round(100 * (df["sess"] == 0).mean(), 1),
                     sess_evening_pct=round(100 * (df["sess"] == 2).mean(), 1),
                     median_1m_absret_bp=round(1e4 * med, 2), outliers_z25=int(len(big)),
                     gaps_gt3pct=int(len(gi)), median_1m_volume=float(df["volume"].median())))
norm = pd.DataFrame(rows)
norm.to_csv(OUT / "normalized_summary.csv", index=False)
pd.DataFrame(outl).to_csv(OUT / "outliers_1m.csv", index=False)
pd.DataFrame(gaps).to_csv(OUT / "overnight_gaps_gt3pct.csv", index=False)

# 5. календарь: дни, отсутствующие у части инструментов
alld = sorted(set().union(*days_all.values()))
miss = []
for d in alld:
    absent = [c for c in D.ALL_CODES if d not in days_all[c]]
    if absent:
        miss.append(dict(day=d, weekday=pd.Timestamp(d).day_name(), absent=" ".join(absent), n_absent=len(absent)))
pd.DataFrame(miss).to_csv(OUT / "calendar_missing_days.csv", index=False)

# 6. сверка дневных свечей API с агрегатом минут (акции: календарный день UTC == МСК-день для 07:00–23:50)
chk = []
for code in D.EQUITIES:
    d1 = con.execute(f"SELECT ts, open, high, low, close, volume FROM {GD} WHERE ticker='{code}'").df()
    d1["day"] = pd.to_datetime(d1["ts"]).dt.date
    m = con.execute(f"""SELECT CAST(ts AS DATE) AS "day", first(open ORDER BY ts) o, max(high) h, min(low) l,
                        last(close ORDER BY ts) c, sum(volume) v FROM {G} WHERE ticker='{code}' GROUP BY 1""").df()
    m["day"] = pd.to_datetime(m["day"]).dt.date
    j = d1.merge(m, on="day")
    chk.append(dict(code=code, days=len(j), close_mismatch=int((np.abs(j["close"] - j["c"]) > 1e-9).sum()),
                    high_mismatch=int((np.abs(j["high"] - j["h"]) > 1e-9).sum()),
                    vol_rel_diff_median=float(np.median(np.abs(j["volume"] - j["v"]) / j["volume"].clip(lower=1)))))
pd.DataFrame(chk).to_csv(OUT / "daily_vs_minute_check.csv", index=False)

# 7. спецификации
ins = D.instruments()
specs = pd.DataFrame([D.spec(c) for c in D.ALL_CODES])
lastp = {c: float(D.normalized(c)["close"].iloc[-1]) for c in D.ALL_CODES}
specs["last_price"] = specs["code"].map(lastp)
specs["notional_rub"] = specs["last_price"] / specs["tick"] * specs["tick_value"]
specs["tick_pct_bp"] = 1e4 * specs["tick"] / specs["last_price"]
specs["margin_pct"] = specs["im"] / specs["notional_rub"]
specs.to_csv(OUT / "specs.csv", index=False)
res["tick_value_missing_series"] = ins.loc[(ins["kind"] == "future") & ins["tick_value"].isna(), "ticker"].tolist()
res["im_present"] = ins.loc[ins["kind"] == "future", "im_buy"].notna().sum()
fl = con.execute(f"SELECT interval, count(*) n, sum(CASE WHEN n_rows=0 THEN 1 ELSE 0 END) empty FROM read_parquet('{D.SNAP}/fetch_log.parquet') GROUP BY 1").df()
res["fetch_log"] = fl.to_dict("records")
(OUT / "audit_facts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(cov.to_string())
print(norm.to_string())
print(pd.DataFrame(chk).to_string())
print(specs.to_string())
print(json.dumps(res, ensure_ascii=False, default=str)[:3000])
print("missing days:", len(miss))
print(pd.DataFrame(miss).to_string())
print(pd.DataFrame(outl).head(40).to_string())
