import numpy as np
import pandas as pd
import pytest

from moexlab.indicators import core


def _df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_normal(n))
    o = np.r_[100, c[:-1]]
    h = np.maximum(o, c) + rng.random(n)
    l = np.minimum(o, c) - rng.random(n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.integers(1, 100, n).astype(float)})


def test_true_range_definition():
    df = pd.DataFrame({"high": [10, 12, 11.0], "low": [9, 10, 7.0], "close": [9.5, 11, 8.0]})
    tr = core.true_range(df)
    assert list(tr) == [1.0, max(2, 2.5, 0.5), max(4, 0, 4.0)]


def test_atr_sma_is_simple_mean_of_last_14_tr():
    df = _df()
    a = core.atr(df, 14, "SMA")
    tr = core.true_range(df)
    assert a.iloc[20] == pytest.approx(tr.iloc[7:21].mean())
    assert a.iloc[:13].isna().all()


def test_donchian_excludes_current_bar():
    df = _df()
    up, lo = core.donchian(df, 5)
    assert up.iloc[10] == pytest.approx(df["high"].iloc[5:10].max())
    assert lo.iloc[10] == pytest.approx(df["low"].iloc[5:10].min())


@pytest.mark.parametrize("fn", [
    lambda d: core.atr(d, 14), lambda d: core.rsi(d["close"], 14), lambda d: core.adx(d, 14),
    lambda d: core.supertrend(d, 10, 3.0), lambda d: core.zscore(d["close"], 20),
    lambda d: core.bollinger(d["close"], 20)[1], lambda d: core.donchian(d, 20)[0],
    lambda d: core.relative_volume(d["volume"], 20), lambda d: core.ema(d["close"], 10),
])
def test_indicators_are_causal(fn):
    df = _df(300)
    full = fn(df)
    for T in (60, 150, 299):
        cut = fn(df.iloc[:T])
        np.testing.assert_allclose(full.iloc[:T].to_numpy(float), cut.to_numpy(float), equal_nan=True)
