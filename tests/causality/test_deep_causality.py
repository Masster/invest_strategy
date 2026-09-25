"""Тест причинности для всех семейств moexlab.deep: сигналы и уровни на префиксе данных
не зависят от того, есть ли в массиве будущие свечи; то же для дневных доходностей движка.

decision_timestamp = закрытие свечи i; допустимы только данные с индексом <= i.
"""
import numpy as np
import pytest

from moexlab.deep import engine as E
from moexlab.deep import features as F
from moexlab.deep.families import REGISTRY, Ctx, build


def synth(n=3000, seed=7, bars_per_day=60, tf=5):
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, n) * 0.002
    c = 100 * np.exp(np.cumsum(r))
    o = np.append(100, c[:-1]) * (1 + rng.normal(0, 0.0005, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    v = rng.integers(1, 1000, n).astype(float)
    di = np.arange(n) // bars_per_day
    dd = np.datetime64("2025-01-06") + di.astype("timedelta64[D]")
    day = np.array([int(str(x).replace("-", "")) for x in dd], np.int32)
    first = np.append(True, day[1:] != day[:-1])
    last = np.append(day[1:] != day[:-1], True)
    mod = (420 + (np.arange(n) % bars_per_day) * tf).astype(np.int32)
    adj = np.where(np.arange(n) > n // 2, 1.01, 1.0)
    A = dict(o=o, h=h, l=l, c=c, v=v, ao=o * adj, ah=h * adj, al=l * adj, ac=c * adj, day=day, mod=mod,
             first_of_day=first, last_of_day=last, new_series=np.zeros(n, bool), last_of_series=np.zeros(n, bool),
             sess=np.ones(n, np.int8), day_idx=di.astype(np.int64), n_days=int(di.max()) + 1)
    A["atr"] = F.atr(A["ah"], A["al"], A["ac"], 14) / adj
    return A


def cut(A, T):
    B = {}
    for k, v in A.items():
        B[k] = v[:T] if isinstance(v, np.ndarray) and len(v) == len(A["o"]) else v
    B["last_of_day"] = B["last_of_day"].copy()
    B["last_of_day"][-1] = True
    return B


FAMS = [f for f in REGISTRY.values()]


@pytest.mark.parametrize("fam", FAMS, ids=[f.name for f in FAMS])
def test_family_signals_causal(fam):
    tf = 1 if 1 in fam.tfs else (1440 if 1440 in fam.tfs else fam.tfs[0])
    A = synth(tf=tf if tf < 1440 else 1, bars_per_day=60 if tf < 1440 else 1)
    params = list(fam.params())
    rng = np.random.default_rng(0)
    pick = [params[i] for i in rng.choice(len(params), size=len(params), replace=False)]
    for p in pick:
        full = build(fam, Ctx(A, tf, "TEST"), p)
        if full is None:
            continue
        for T in (900, 1777, 2401):
            part = build(fam, Ctx(cut(A, T), tf, "TEST"), p)
            for k, v in full.items():
                if isinstance(v, np.ndarray):
                    # две последние свечи префикса могут отличаться флагом last_of_day (расписание, а не цена)
                    a, b = v[:T - 2], part[k][:T - 2]
                    same = (a == b) | (np.isnan(a.astype(float)) & np.isnan(b.astype(float)))
                    assert same.all(), f"{fam.name} {p} key={k} T={T} first diff at {np.where(~same)[0][:5]}"


def test_engine_daily_returns_causal():
    A = synth()
    fam = REGISTRY["T_EMA_X"]
    s = build(fam, Ctx(A, 5, "T"), {"f": 10, "s": 50})
    s.pop("state")
    full = E.run(A, s, stop_k=2.0, trail_k=3.0, state_mode=True)[0]
    for T in (1000, 2000):
        B = cut(A, T)
        sp = build(fam, Ctx(B, 5, "T"), {"f": 10, "s": 50})
        sp.pop("state")
        part = E.run(B, sp, stop_k=2.0, trail_k=3.0, state_mode=True)[0]
        d_last = B["day_idx"][-1]
        # все дни до дня обрезки совпадают (в день обрезки позиция закрывается принудительно)
        np.testing.assert_allclose(full[:d_last], part[:d_last], atol=1e-12)
