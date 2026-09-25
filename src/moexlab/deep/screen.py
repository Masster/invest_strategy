"""Stage 1: массовый скрининг (инструмент × таймфрейм × семейство × параметры × выход × направление × перенос).

Скрининг идёт ТОЛЬКО на периоде разработки (<= DEV_END): массивы обрезаются до вызова семейств,
так что holdout физически недоступен. Результат задания (code, tf):
  data/cache/screen/<tag>/<code>_<tf>.parquet — сводка по каждой конфигурации;
  data/cache/screen/<tag>/<code>_<tf>.npy     — дневные доходности (float32, глобальный календарь dev).
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import engine as E
from . import features as F
from . import metrics as M
from .families import REGISTRY, Ctx, build

SCREEN = D.CACHE / "screen"

EXITS = {
    "X0": dict(),                                   # только сигнал/разворот (+ принудительные)
    "X1": dict(stop_k=1.5),
    "X2": dict(stop_k=3.0),
    "X3": dict(stop_k=1.5, tp_k=3.0),
    "X4": dict(trail_k=3.0),
    "X5": dict(time_bars=20),
}

SPREAD_TICKS = {"RN": 3.0, "MGNT": 2.0, "ALRS": 2.0, "AFLT": 2.0}


def cost_model(code: str, A: dict, tariff: str = "TRADER", slip_mult: float = 1.0, comm_mult: float = 1.0) -> dict:
    sp = D.spec(code)
    kind = sp["kind"]
    tick = sp["tick"]
    st = SPREAD_TICKS.get(code, 1.0)
    return {"comm": D.COMMISSION[tariff][kind] * comm_mult, "half_spread": 0.5 * st * tick,
            "slip": 1.0 * tick * slip_mult, "stop_extra": 1.0 * tick * slip_mult}


def calendar(end=D.DEV_END, start=D.FIRST_DAY) -> np.ndarray:
    days = set()
    for c in D.ALL_CODES:
        b = D.bars(c, 1440)
        d = pd.to_datetime(b["day"]).dt.date
        days |= set(d[(d >= start) & (d <= end)])
    return np.array(sorted(int(x.strftime("%Y%m%d")) for x in days), np.int32)


def load_arrays(code: str, tf: int, end=D.DEV_END, start=D.FIRST_DAY, cal: np.ndarray | None = None) -> dict:
    b = D.bars(code, tf)
    d = pd.to_datetime(b["day"]).dt.date
    b = b[((d >= start) & (d <= end)).to_numpy()].reset_index(drop=True)
    A = D.to_arrays(b)
    cal = cal if cal is not None else calendar(end, start)
    A["cal"] = cal
    A["day_idx"] = np.searchsorted(cal, A["day"]).astype(np.int64)
    assert np.all(cal[A["day_idx"]] == A["day"]), "day not in calendar"
    A["n_days"] = len(cal)
    A["atr"] = F.atr(A["ah"], A["al"], A["ac"], 14) / (A["ac"] / A["c"])
    return A


def run_job(code: str, tf: int, families: list[str] | None = None, tag: str = "s1", end=D.DEV_END,
            tariff: str = "TRADER", zero_cost: bool = False) -> dict:
    out_dir = SCREEN / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{code}_{tf}.parquet"
    if fp.exists():
        return {"code": code, "tf": tf, "skipped": True}
    t0 = time.time()
    cal = calendar(end)
    A = load_arrays(code, tf, end=end, cal=cal)
    ctx = Ctx(A, tf, code)
    costs = cost_model(code, A, tariff)
    if zero_cost:
        costs = {k: 0.0 for k in costs}
    is_eq = code in D.EQUITIES
    mkeys = M.month_keys(cal)
    months = np.unique(mkeys)
    rows, rets = [], []
    fams = [REGISTRY[f] for f in (families or REGISTRY)]
    errors = []
    for fam in fams:
        if tf not in fam.tfs:
            continue
        if fam.group == "carry" and is_eq:
            continue
        for p in fam.params():
            try:
                sig = build(fam, ctx, p)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{fam.name} {p}: {e!r}")
                continue
            if sig is None:
                continue
            state = bool(sig.pop("state", fam.state))
            fixed_dir = sig.pop("dir", None)
            fixed_eod = sig.pop("eod", None)
            dirs = [fixed_dir] if fixed_dir is not None else [1, -1, 0]
            if fixed_eod is not None:
                eods = [bool(fixed_eod) or is_eq]
            elif fam.intraday or is_eq:
                eods = [True]
            elif tf == 1440:
                eods = [False, True]
            else:
                eods = [False, True]
            for ex in fam.exits:
                xp = EXITS[ex]
                for eod in eods:
                    for dr in dirs:
                        res = E.run(A, sig, eod_exit=eod, state_mode=state, direction=dr, costs=costs, **xp)
                        day_ret, expo, te, tx, ts, tr, rsn, risk = res
                        n_tr = len(tr)
                        rets.append(day_ret.astype(np.float32))
                        st = M.trade_stats(tr)
                        hold = float(np.mean(tx - te + 1)) if n_tr else 0.0
                        rows.append(dict(code=code, tf=tf, family=fam.name, group=fam.group,
                                         params=json.dumps(p, sort_keys=True), exit=ex, eod=eod, direction=dr,
                                         trades=n_tr, win=st["win"], pf=st["pf"], avg_trade=st["avg_trade"],
                                         hold_bars=hold, expo_days=float((expo > 0).mean()),
                                         long_share=float((ts > 0).mean()) if n_tr else np.nan))
    if not rows:
        return {"code": code, "tf": tf, "n": 0}
    R = np.vstack(rets)
    mret = M.monthly(R.astype(float), mkeys, months)
    s = M.summary_from_daily(R.astype(float), mret)
    df = pd.DataFrame(rows)
    for k, v in s.items():
        df[k] = v
    for j, m in enumerate(months):
        df[f"m{m}"] = mret[:, j]
    df["score"] = M.score_stability(mret)
    df["row"] = np.arange(len(df))
    np.save(out_dir / f"{code}_{tf}.npy", R)
    df.to_parquet(fp, index=False)
    if errors:
        (out_dir / f"{code}_{tf}.errors.txt").write_text("\n".join(errors))
    return {"code": code, "tf": tf, "n": len(df), "sec": round(time.time() - t0, 1), "errors": len(errors)}


def _worker(args):
    try:
        return run_job(*args)
    except Exception:  # noqa: BLE001
        return {"code": args[0], "tf": args[1], "error": traceback.format_exc()}


def run_all(codes=None, tfs=None, families=None, tag="s1", procs=None, end=D.DEV_END, tariff="TRADER",
            zero_cost=False):
    import multiprocessing as mp
    codes = codes or D.ALL_CODES
    tfs = tfs or D.TIMEFRAMES
    jobs = [(c, tf, families, tag, end, tariff, zero_cost) for tf in sorted(tfs, reverse=True) for c in codes]
    # тяжёлые (1m) — первыми, чтобы выровнять загрузку
    jobs.sort(key=lambda j: j[1])
    procs = procs or os.cpu_count()
    with mp.get_context("fork").Pool(procs) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            print(json.dumps(r, default=str)[:500], flush=True)
