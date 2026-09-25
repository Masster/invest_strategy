import numpy as np
import pandas as pd
import pytest

from moexlab.strategies.rbreaker import RBreaker, prev_day_hlc


def levels(H, L, C, sk=0.35, rk=0.07, bk=0.25):
    ssetup = H + sk * (C - L)
    bsetup = L - sk * (H - C)
    senter = 0.535 * H + 0.465 * L
    benter = 0.465 * H + 0.535 * L
    w = ssetup - bsetup
    return dict(sbreak=bsetup - bk * w, bsetup=bsetup, benter=benter, senter=senter, ssetup=ssetup, bbreak=ssetup + bk * w)


def _bars_two_days(H, L, C):
    idx = pd.date_range("2024-03-04 07:00", periods=3, freq="1min", tz="UTC").append(
        pd.date_range("2024-03-05 07:00", periods=3, freq="1min", tz="UTC"))
    df = pd.DataFrame({"open": [C] * 6, "high": [H, C, C, C, C, C], "low": [L, C, C, C, C, C],
                       "close": [C] * 6, "volume": [1.0] * 6}, index=idx)
    df["trading_day"] = [pd.Timestamp("2024-03-04").date()] * 3 + [pd.Timestamp("2024-03-05").date()] * 3
    return df


def test_six_levels_match_spec_formulas():
    H, L, C = 110.0, 100.0, 104.0
    df = _bars_two_days(H, L, C)
    f = RBreaker({"min_days": 0, "min_atr_bars": 0, "min_median_volume": 0}).prepare(df)
    exp = levels(H, L, C)
    row = f.iloc[-1]
    for k, v in exp.items():
        assert row[k] == pytest.approx(v)
    # эквивалентные формы §10.3/10.4
    assert row["senter"] == pytest.approx(0.535 * (H + L) - 0.07 * L)
    assert row["benter"] == pytest.approx(0.535 * (H + L) - 0.07 * H)


def test_level_order_holds_for_normal_day():
    lv = levels(110.0, 100.0, 104.0)
    assert lv["sbreak"] < lv["bsetup"] < lv["benter"] < lv["senter"] < lv["ssetup"] < lv["bbreak"]


def test_first_day_has_no_levels():
    df = _bars_two_days(110.0, 100.0, 104.0)
    f = prev_day_hlc(df)
    assert f["prev_H"].iloc[:3].isna().all()
    assert f["prev_H"].iloc[3:].eq(110.0).all()
