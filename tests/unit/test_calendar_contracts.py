from datetime import date, time

import pandas as pd

from moexlab.contracts.series import SeriesInfo, parse_secid, secid, select_active_series
from moexlab.market_data.calendar import SessionConfig, annotate_sessions, assign_trading_day


def test_secid_roundtrip():
    assert secid("MX", 2024, 12) == "MXZ4"
    assert parse_secid("MXZ4", 2024) == ("MX", 2024, 12)
    assert parse_secid("SiH0", 2019) == ("Si", 2020, 3)


def test_evening_session_belongs_to_next_trading_day_msk():
    # 2024-03-01 пятница 19:10 МСК = 16:10 UTC -> торговый день понедельник 2024-03-04
    ts = pd.DatetimeIndex(["2024-03-01 07:30", "2024-03-01 16:10", "2024-03-04 07:30"], tz="UTC")
    days = [date(2024, 3, 1), date(2024, 3, 4)]
    out = assign_trading_day(ts, days, time(19, 0))
    assert list(out) == [date(2024, 3, 1), date(2024, 3, 4), date(2024, 3, 4)]


def test_session_flags_respect_cooldown_and_force_close():
    idx = pd.date_range("2024-03-04 07:00", "2024-03-04 15:49", freq="1min", tz="UTC")  # 10:00-18:49 МСК
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    a = annotate_sessions(df, SessionConfig(), 1)
    msk = a.index.tz_convert("Europe/Moscow")
    ok = a[a.entry_ok]
    # вход исполняется на следующей свече => первая свеча решения 10:14, последняя: следующая < 18:30
    assert msk[a.entry_ok][0].time() == time(10, 14)
    assert msk[a.entry_ok][-1].time() == time(18, 28)
    assert msk[a.force_exit][0].time() == time(18, 35)
    assert a.force_exit.sum() == 1 and len(ok) > 0


def test_mandatory_roll_two_days_before_expiry_and_no_lookahead():
    days = list(pd.bdate_range("2024-03-01", "2024-03-29").date)
    ser = [SeriesInfo("MXH4", "MX", date(2024, 3, 21)), SeriesInfo("MXM4", "MX", date(2024, 6, 20))]
    rows = []
    for d in days:
        rows.append(dict(trading_day=d, secid="MXH4", volume=1000.0))
        rows.append(dict(trading_day=d, secid="MXM4", volume=10.0))
    act = select_active_series(ser, pd.DataFrame(rows), days)
    assert act[date(2024, 3, 19)] == "MXH4"
    assert act[date(2024, 3, 20)] == "MXM4"   # 20 и 21 марта — 2 последних дня: работаем на следующей
    assert act[date(2024, 3, 21)] == "MXM4"
    assert act[date(2024, 3, 22)] == "MXM4"


def test_early_roll_requires_spread_data_by_default():
    days = list(pd.bdate_range("2024-03-01", "2024-03-15").date)
    ser = [SeriesInfo("MXH4", "MX", date(2024, 3, 21)), SeriesInfo("MXM4", "MX", date(2024, 6, 20))]
    rows = [dict(trading_day=d, secid=s, volume=v) for d in days for s, v in (("MXH4", 10.0), ("MXM4", 1000.0))]
    act = select_active_series(ser, pd.DataFrame(rows), days)
    assert set(act.values) == {"MXH4"}
    rows = [dict(r, median_spread=1.0) for r in rows]
    act2 = select_active_series(ser, pd.DataFrame(rows), days)
    assert act2[days[1]] == "MXH4" and act2[days[2]] == "MXM4"  # 2 завершённых дня подряд => с 3-го дня
