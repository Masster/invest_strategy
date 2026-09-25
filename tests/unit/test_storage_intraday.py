from datetime import date

import numpy as np
import pandas as pd

from moexlab.research.intraday import annotate_futures, resample_stream
from moexlab.storage.db import MarketDB


def _bars(start_msk: str, n: int, step_min: int = 1) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start_msk, tz="Europe/Moscow"), periods=n, freq=f"{step_min}min").tz_convert("UTC")
    px = 100 + np.arange(n) * 0.1
    return pd.DataFrame({"open": px, "high": px + 0.2, "low": px - 0.2, "close": px + 0.1, "volume": 10.0}, index=idx)


def test_db_roundtrip_snapshot_restore(tmp_path):
    db = MarketDB(tmp_path / "m.duckdb")
    db.upsert_instruments({"SiZ5": dict(kind="future", base="Si", uid="u1", lot=1, tick=1.0, tick_value=1.0,
                                        expiration="2025-12-18")})
    df = _bars("2025-10-01 10:00", 30)
    assert db.upsert_candles("SiZ5", "1m", df) == 30
    assert db.upsert_candles("SiZ5", "1m", df) == 30          # повторная запись не дублирует
    db.log_fetch("SiZ5", "1m", [date(2025, 10, 1)], {date(2025, 10, 1): 30}, "test")
    got = db.candles("SiZ5", "1m")
    assert len(got) == 30 and got.index.tz is not None
    assert np.allclose(got["close"].to_numpy(), df["close"].to_numpy())
    assert db.fetched_days("SiZ5", "1m") == {date(2025, 10, 1)}
    snap = tmp_path / "snap"
    db.snapshot(snap)
    db.close()
    db2 = MarketDB(tmp_path / "m2.duckdb")
    db2.restore(snap)
    assert len(db2.candles("SiZ5", "1m")) == 30
    assert db2.instruments(kind="future").iloc[0]["ticker"] == "SiZ5"
    assert db2.fetched_days("SiZ5", "1m") == {date(2025, 10, 1)}


def test_evening_session_trading_day_changes_with_ets():
    # до ЕТС: вечер вторника 10.03.2026 относится к среде 11.03; с ЕТС: вечер 14.04.2026 — к самому 14.04
    parts = [_bars("2026-03-10 10:00", 600), _bars("2026-03-11 10:00", 600),
             _bars("2026-04-14 10:00", 600), _bars("2026-04-15 10:00", 600)]
    a = annotate_futures(pd.concat(parts), 1)
    msk = a.index.tz_convert("Europe/Moscow")
    ev_pre = a[(msk.date == date(2026, 3, 10)) & (msk.hour == 19)]
    ev_ets = a[(msk.date == date(2026, 4, 14)) & (msk.hour == 19)]
    assert set(ev_pre["trading_day"]) == {date(2026, 3, 11)}
    assert set(ev_ets["trading_day"]) == {date(2026, 4, 14)}
    # входы разрешены только в окне основной сессии
    assert not a.loc[ev_ets.index, "entry_ok"].any()


def test_resample_keeps_series_and_day_boundaries():
    df = annotate_futures(_bars("2026-04-14 10:00", 120), 1)
    df["secid"] = np.where(np.arange(len(df)) < 63, "SiM6", "SiU6")
    for c in ("open", "high", "low", "close"):
        df[f"a_{c}"] = df[c]
    df["adj_shift"] = 0.0
    r = resample_stream(df, 5, "future")
    # ведро 11:00–11:05 разрезано сменой серии -> две свечи с одним началом, разными secid
    assert r.groupby(level=0)["secid"].nunique().max() == 2
    assert r["volume"].sum() == df["volume"].sum()
