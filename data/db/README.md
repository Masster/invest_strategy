# База рыночных данных (источник — только T-Invest API)

Переносимый снимок — `snapshot/` (Parquet, zstd). Его можно отдать любой модели, тесту или инструменту
(pandas, polars, DuckDB, Spark, R) без доступа к API и без этого репозитория.

## Файлы
| Путь | Содержимое |
|---|---|
| `snapshot/instruments.parquet` | инструменты: ticker, kind (future/equity), base, uid, figi, class_code, lot, tick, tick_value, expiration, meta (JSON: last_trade, ГО и др.) |
| `snapshot/candles/1m/<TICKER>.parquet` | минутные свечи: ticker, interval, ts (UTC, начало свечи), open, high, low, close, volume (в лотах) |
| `snapshot/candles/1d/<TICKER>.parquet` | дневные свечи API (календарные сутки T-Invest) |
| `snapshot/fetch_log.parquet` | какие сутки уже запрошены у API (для докачки) |
| `market.duckdb` | рабочая база (не в git; восстанавливается из снимка автоматически) |

Свечи — биржевые (`CANDLE_SOURCE_EXCHANGE`), только завершённые (`isComplete`). Время — UTC; биржевое — МСК (UTC+3).
Фьючерсы хранятся по сериям (SiZ5, SiH6, …); склейку и выбор активной серии делает `moexlab.research.intraday`.

## Чтение
```python
import pandas as pd
df = pd.read_parquet("data/db/snapshot/candles/1m/SiZ5.parquet")
```
```sql
-- DuckDB без репозитория
SELECT ticker, count(*), min(ts), max(ts) FROM 'data/db/snapshot/candles/1m/*.parquet' GROUP BY 1;
```
```python
from moexlab.storage.db import open_db       # в репозитории: восстановит market.duckdb из снимка
db = open_db(); db.candles("SBER", "1m", "2026-01-01", "2026-02-01"); db.coverage()
```

## Обновление
`TBANK_CA_BUNDLE=<bundle с корнем Минцифры> python3 scripts/fetch_tinvest_data.py` — докачивает только новые сутки
и обновляет снимок.
