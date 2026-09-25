"""Проверки быстрого движка moexlab.deep.engine на ручных сценариях."""
import numpy as np
import pytest

from moexlab.deep import engine as E


def mk(o, h, l, c, days=None, last_series=None):
    n = len(o)
    days = np.zeros(n, np.int64) if days is None else np.asarray(days, np.int64)
    lod = np.append(days[1:] != days[:-1], True)
    los = np.zeros(n, np.bool_) if last_series is None else np.asarray(last_series, np.bool_)
    return dict(o=np.asarray(o, float), h=np.asarray(h, float), l=np.asarray(l, float), c=np.asarray(c, float),
                day_idx=days, n_days=int(days.max()) + 1, last_of_day=lod, last_of_series=los,
                atr=np.full(n, 1.0))


def sig(n, long_at=(), short_at=(), xl=(), xs=()):
    s = dict(long=np.zeros(n, np.bool_), short=np.zeros(n, np.bool_), exit_long=np.zeros(n, np.bool_),
             exit_short=np.zeros(n, np.bool_))
    for k, idx in (("long", long_at), ("short", short_at), ("exit_long", xl), ("exit_short", xs)):
        for i in idx:
            s[k][i] = True
    return s


NOCOST = {"comm": 0.0, "half_spread": 0.0, "slip": 0.0, "stop_extra": 0.0}


def test_market_entry_next_open_and_signal_exit():
    A = mk([100, 101, 103, 106, 104], [101, 102, 104, 107, 105], [99, 100, 102, 105, 103], [100.5, 102, 103.5, 106, 104])
    r = E.run(A, sig(5, long_at=[0], xl=[2]), costs=NOCOST)
    tr = r[5]
    assert len(tr) == 1
    # вход по open[1]=101, выход по open[3]=106
    assert tr[0] == pytest.approx(106 / 101 - 1)
    assert r[0].sum() == pytest.approx(106 / 101 - 1)
    assert r[2][0] == 1 and r[3][0] == 3


def test_costs_applied_both_sides():
    A = mk([100, 100, 100, 100], [100] * 4, [100] * 4, [100] * 4)
    costs = {"comm": 0.001, "half_spread": 0.5, "slip": 1.0, "stop_extra": 0.0}
    r = E.run(A, sig(4, long_at=[0], xl=[1]), costs=costs)
    # покупка 101.5, продажа 98.5; комиссия 0.1% с каждой стороны
    exp = (98.5 - 101.5) / 101.5 - 0.001 - 0.001 * 98.5 / 101.5
    assert r[5][0] == pytest.approx(exp)


def test_stop_gap_fills_at_open_and_stop_beats_tp():
    # вход long по open[1]=100, ATR=1, stop 2 => 98, tp 3 => 103
    A = mk([100, 100, 95, 100], [100, 101, 96, 104], [100, 99.5, 94, 97], [100, 100.5, 95, 101])
    r = E.run(A, sig(4, long_at=[0]), stop_k=2.0, tp_k=3.0, costs=NOCOST)
    assert r[5][0] == pytest.approx(95 / 100 - 1)      # гэп вниз через стоп: исполнение по open 95
    assert r[6][0] == 1
    A2 = mk([100, 100, 100], [100, 104, 100], [100, 97, 100], [100, 100, 100])
    r2 = E.run(A2, sig(3, long_at=[0]), stop_k=2.0, tp_k=3.0, costs=NOCOST)
    assert r2[6][0] == 1 and r2[5][0] == pytest.approx(98 / 100 - 1)   # и стоп и тейк на свече => стоп


def test_eod_and_roll_forced_exit_at_close():
    A = mk([100, 101, 102, 103], [100, 102, 103, 104], [100, 100, 101, 102], [100, 101.5, 102.5, 103],
           days=[0, 0, 1, 1])
    r = E.run(A, sig(4, long_at=[0]), eod_exit=True, costs=NOCOST)
    assert r[6][0] == 6 and r[5][0] == pytest.approx(101.5 / 101 - 1)
    A2 = mk([100, 101, 102, 103], [100, 102, 103, 104], [100, 100, 101, 102], [100, 101.5, 102.5, 103],
            last_series=[False, True, False, False])
    r2 = E.run(A2, sig(4, long_at=[0]), costs=NOCOST)
    assert r2[6][0] == 7


def test_stop_entry_and_limit_entry():
    A = mk([100, 100, 100], [100, 103, 100], [100, 99, 100], [100, 102, 100])
    s = sig(3, long_at=[0])
    s["mode"] = 1
    s["lvl_long"] = np.array([102.0, 102.0, 102.0])
    r = E.run(A, s, costs=NOCOST)
    assert r[5][0] == pytest.approx(100 / 102 - 1)       # вход по уровню 102, выход в конце данных по 100


def test_limit_requires_trade_through():
    A = mk([100, 100, 100], [100, 101, 100], [100, 99.0, 100], [100, 100, 100])
    s = sig(3, long_at=[0])
    s["mode"] = 2
    s["lvl_long"] = np.array([99.0, 99.0, 99.0])
    r = E.run(A, s, costs=NOCOST)
    assert len(r[5]) == 0                                # касание уровня без прохождения — не исполнено
    A2 = mk([100, 100, 100], [100, 101, 100], [100, 98.5, 100], [100, 100, 100])
    r2 = E.run(A2, s, costs=NOCOST)
    assert len(r2[5]) == 1 and r2[5][0] == pytest.approx(100 / 99 - 1)


def test_mtm_sums_to_trade_return():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    o = np.append(100, c[:-1])
    h = np.maximum(o, c) * 1.002
    l = np.minimum(o, c) * 0.998
    days = np.repeat(np.arange(30), 10)
    A = mk(o, h, l, c, days=days)
    s = sig(300, long_at=range(0, 300, 7), short_at=range(3, 300, 11))
    costs = {"comm": 0.0004, "half_spread": 0.01, "slip": 0.01, "stop_extra": 0.01}
    r = E.run(A, s, stop_k=2.0, trail_k=3.0, costs=costs)
    assert r[0].sum() == pytest.approx(r[5].sum(), rel=1e-9, abs=1e-12)
