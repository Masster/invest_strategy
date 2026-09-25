import pandas as pd

from moexlab.market_data.iss_daily import split_adjust


def test_reverse_split_5000_is_adjusted():
    idx = pd.date_range("2024-07-10", periods=4, freq="B")
    df = pd.DataFrame({"open": [0.021, 0.0212, 106.0, 107.0], "high": [0.022, 0.0215, 108.0, 108.0],
                       "low": [0.020, 0.0209, 105.0, 106.0], "close": [0.0211, 0.0213, 107.0, 107.5],
                       "volume": [5e9, 5e9, 1e6, 1e6]}, index=idx)
    out, notes = split_adjust(df)
    assert "5000" in notes[0]
    assert abs(out["close"].iloc[1] - 0.0213 * 5000) < 1e-6
    assert out["close"].iloc[3] == 107.5


def test_forward_split_100():
    idx = pd.date_range("2024-04-01", periods=3, freq="B")
    df = pd.DataFrame({"open": [15000.0, 150.5, 151], "high": [15100.0, 152, 152], "low": [14900.0, 149, 150],
                       "close": [15020.0, 151, 151.5], "volume": [1e5, 1e7, 1e7]}, index=idx)
    out, notes = split_adjust(df)
    assert abs(out["close"].iloc[0] - 150.2) < 1e-9 and notes
