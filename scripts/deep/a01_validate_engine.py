"""Сверка быстрого Numba-движка (moexlab.deep.engine) с событийным движком v1 (moexlab.backtest.engine).

Одни и те же сигналы подаются в оба движка через адаптер; издержки — только комиссия 0,04% (спред и
проскальзывание 0, чтобы округление цен к шагу в v1 не вносило различий). Сравниваются сделки:
время входа/выхода, цены и доходность. Результат: research/audit/engine_validation.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moexlab.backtest.engine import Engine, InstrumentInput, RiskConfig  # noqa: E402
from moexlab.deep import data as D  # noqa: E402
from moexlab.deep import engine as E  # noqa: E402
from moexlab.deep import screen as S  # noqa: E402
from moexlab.deep.families import REGISTRY, Ctx, build  # noqa: E402
from moexlab.execution.models import CommissionModel, ConservativeL1ExecutionModel  # noqa: E402
from moexlab.instruments.spec import InstrumentSpec  # noqa: E402
from moexlab.strategies.base import EntryOrder, ExitPolicy, Strategy  # noqa: E402


class Adapter(Strategy):
    name = family = "ADAPTER"

    def __init__(self, sig, atr, direction, exits):
        super().__init__({}, exits)
        self.sig, self.atr_, self.dir = sig, atr, direction

    def prepare(self, df):
        return pd.DataFrame({"atr": self.atr_}, index=df.index)

    def on_bar_close(self, i, ctx):
        if not ctx.flat:
            return []
        if self.dir >= 0 and self.sig["long"][i]:
            return [EntryOrder(1, "market", tag="L")]
        if self.dir <= 0 and self.sig["short"][i]:
            return [EntryOrder(-1, "market", tag="S")]
        return []

    def exit_signal(self, i, pos):
        k = "exit_long" if pos.direction > 0 else "exit_short"
        return bool(self.sig.get(k, np.zeros(1, bool))[i]) if k in self.sig else False


CASES = [("T_EMA_X", {"f": 10, "s": 50}, 1, 2.0), ("T_DONCH", {"n": 20}, -1, 3.0), ("M_TSMOM", {"n": 20, "z": 0.5}, 1, 1.5),
         ("R_BB", {"n": 20, "k": 2.0}, 1, 3.0)]


def main():
    rows = []
    for code in ["Si", "MX", "BR", "SBER", "GAZP"]:
        for tf in [60, 1440, 15]:
            A = S.load_arrays(code, tf)
            b = D.bars(code, tf)
            b = b[pd.to_datetime(b["day"]).dt.date <= D.DEV_END].reset_index(drop=True)
            sp = D.spec(code)
            spec = InstrumentSpec(code, sp["kind"], tick=sp["tick"], tick_value=sp["tick_value"], lot=sp["lot"],
                                  spread_ticks=0.0, slippage_ticks=0.0, stop_extra_ticks=0.0, margin_fraction=0.1)
            idx = pd.DatetimeIndex(pd.to_datetime(A["ts"], utc=True))
            bars = pd.DataFrame({"open": A["o"], "high": A["h"], "low": A["l"], "close": A["c"], "volume": A["v"],
                                 "trading_day": pd.to_datetime(A["day"].astype(str)).date, "entry_ok": True,
                                 "force_exit": False, "last_in_day": A["last_of_day"], "secid": b["secid"].to_numpy()},
                                index=idx)
            for fam_name, p, dr, k in CASES:
                sig = build(REGISTRY[fam_name], Ctx(A, tf, code), p)
                sig.pop("state", None)
                costs = {"comm": 0.0004, "half_spread": 0.0, "slip": 0.0, "stop_extra": 0.0}
                r = E.run(A, sig, stop_k=k, direction=dr, state_mode=False, costs=costs)
                new = pd.DataFrame({"ent": r[2], "ext": r[3], "side": r[4], "ret": r[5], "rsn": r[6]})
                strat = Adapter(sig, A["atr"], dr, ExitPolicy(initial="atr", initial_k=k, trailing="none",
                                                              intraday=False))
                exe = ConservativeL1ExecutionModel(commission=CommissionModel(broker_fraction=0.0004))
                res = Engine([InstrumentInput(code, spec, bars, strat, tf * 60)], execution=exe,
                             risk=RiskConfig.research(), initial_equity=1e7).run()
                old = res.trades
                if len(old):
                    old = old.assign(ent=idx.get_indexer(old["entry_ts"]), ext=idx.get_indexer(old["exit_ts"]))
                    old["ret"] = (np.where(old["direction"] == "LONG", 1, -1) * (old["exit_price"] / old["entry_price"] - 1)
                                  - 0.0004 * (1 + old["exit_price"] / old["entry_price"]))
                m = new.merge(old[["ent", "ext", "ret", "exit_reason"]], on="ent", how="outer", suffixes=("_new", "_old"),
                              indicator=True) if len(old) else new.assign(_merge="left_only")
                both = m[m["_merge"] == "both"]
                same_exit = (both["ext_new"] == both["ext_old"]).mean() if len(both) else np.nan
                rows.append(dict(code=code, tf=tf, family=fam_name, params=str(p), direction=dr, stop_k=k,
                                 trades_new=len(new), trades_old=len(old), matched_entries=len(both),
                                 same_exit_share=round(float(same_exit), 4),
                                 ret_abs_diff_median=float((both["ret_new"] - both["ret_old"]).abs().median()) if len(both) else np.nan,
                                 sum_ret_new=float(new["ret"].sum()), sum_ret_old=float(old["ret"].sum()) if len(old) else 0.0))
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(D.ROOT / "research/audit/engine_validation.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
