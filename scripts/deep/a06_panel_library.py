"""Библиотека дневных портфельных стратегий (панель 21 инструмента) -> data/cache/screen/p1/.

Формат совместим с moexlab.deep.wf (meta parquet + npy дневных доходностей по календарю разработки).
Запуск с аргументом `full` строит ту же библиотеку на всём периоде (для holdout-стадии, только после фиксации
процедур): data/cache/screen/p1full/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.deep import data as D  # noqa: E402
from moexlab.deep import metrics as M  # noqa: E402
from moexlab.deep import panel as PN  # noqa: E402
from moexlab.deep import screen as S  # noqa: E402
from moexlab.deep.panel_lib import library  # noqa: E402


def main(full=False):
    end = D.LAST_DAY if full else D.DEV_END
    tag = "p1full" if full else "p1"
    out = D.CACHE / "screen" / tag
    out.mkdir(parents=True, exist_ok=True)
    P = PN.load_panel(end=end)
    cal = S.calendar(end)
    assert np.array_equal(np.array(P["days"]), cal), "panel days != calendar"
    rows, rets, weights = [], [], {}
    for fam, cn, p, fn in library(P):
        W = fn()
        r, turn, gross, Wx = PN.backtest(P, W)
        r = r.to_numpy()
        rets.append(r.astype(np.float32))
        rows.append(dict(code=cn, tf=1440, family=fam, group="panel", params=json.dumps(p, sort_keys=True), exit="P",
                         eod=False, direction=0, trades=int((turn > 0).sum()), turnover=float(turn.mean()),
                         gross=float(gross.mean())))
    R = np.vstack(rets)
    mk = M.month_keys(cal)
    months = np.unique(mk)
    mret = M.monthly(R.astype(float), mk, months)
    s = M.summary_from_daily(R.astype(float), mret)
    df = pd.DataFrame(rows)
    for k, v in s.items():
        df[k] = v
    for j, m in enumerate(months):
        df[f"m{m}"] = mret[:, j]
    df["score"] = M.score_stability(mret)
    df["row"] = np.arange(len(df))
    df.to_parquet(out / "PANEL_1440.parquet", index=False)
    np.save(out / "PANEL_1440.npy", R)
    if full:   # holdout не просматриваем здесь: только сохранение
        print("saved", len(df), out)
        return
    pd.set_option("display.width", 250)
    print(len(df))
    print(df.sort_values("sharpe", ascending=False).head(30)[["family", "code", "params", "sharpe", "total", "pmr", "m_min", "mdd", "turnover"]].round(3).to_string())
    print(df.groupby(["family", "code"]).sharpe.agg(["count", "median", "max"]).round(2).to_string())


if __name__ == "__main__":
    main(full=len(sys.argv) > 1 and sys.argv[1] == "full")
