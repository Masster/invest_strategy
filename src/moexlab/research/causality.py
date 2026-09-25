"""Автоматический тест отсутствия заглядывания в будущее (R-Breaker §69, задача §16).

Два уровня проверки:
  1. признаки: prepare(full)[:T] == prepare(full[:T]) для случайных T;
  2. решения: события (входы/выходы) до T-1 на полном и обрезанном наборе совпадают.
Любое расхождение => CAUSALITY_TEST_FAILED.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.engine import Engine, InstrumentInput, RiskConfig
from ..execution.models import ExecutionModel
from ..market_data.calendar import SessionConfig, annotate_sessions


@dataclass
class CausalityReport:
    passed: bool
    n_cuts: int
    feature_mismatches: list
    event_mismatches: list

    def __str__(self) -> str:
        return ("CAUSALITY_TEST_PASSED" if self.passed else "CAUSALITY_TEST_FAILED") + \
            f" cuts={self.n_cuts} feature_mismatch={len(self.feature_mismatches)} event_mismatch={len(self.event_mismatches)}"


def _events_before(events: list[dict], cut) -> list[tuple]:
    out = []
    for e in events:
        if e["ts"] < cut:
            out.append(tuple((k, (round(v, 8) if isinstance(v, float) else v)) for k, v in sorted(e.items())))
    return out


def check_causality(raw_bars: pd.DataFrame, make_strategy, spec, session: SessionConfig | None, bar_minutes: int,
                    n_cuts: int = 20, seed: int = 0, execution: ExecutionModel | None = None,
                    risk: RiskConfig | None = None, min_frac: float = 0.3, annotate=None) -> CausalityReport:
    """raw_bars — свечи ДО разметки сессий (разметка тоже должна быть каузальной)."""
    rng = np.random.default_rng(seed)
    ann = annotate or (lambda b: annotate_sessions(b, session, bar_minutes))
    full = ann(raw_bars)
    st_full = make_strategy()
    f_full = st_full.prepare(full)
    risk = risk or RiskConfig.research()
    res_full = Engine([InstrumentInput(spec.code, spec, full, st_full, bar_minutes * 60)], execution=execution,
                      risk=copy.deepcopy(risk), record_events=True).run()
    n = len(raw_bars)
    feat_bad, ev_bad = [], []
    cuts = sorted(rng.integers(int(n * min_frac), n - 2, size=n_cuts))
    for T in cuts:
        cut_raw = raw_bars.iloc[:T]
        cut = ann(cut_raw)
        st = make_strategy()
        f_cut = st.prepare(cut)
        common = f_cut.index
        a = f_full.loc[common].select_dtypes("number")
        b = f_cut.select_dtypes("number")
        diff = ~np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=1e-9, atol=1e-9)
        if diff.any():
            r, c = np.argwhere(diff)[0]
            feat_bad.append(dict(T=int(T), ts=str(common[r]), column=a.columns[c],
                                 full=float(a.iat[r, c]), cut=float(b.iat[r, c])))
        res_cut = Engine([InstrumentInput(spec.code, spec, cut, st, bar_minutes * 60)], execution=execution,
                         risk=copy.deepcopy(risk), record_events=True).run()
        # сравниваем события строго до последней свечи обрезанного набора
        last_ts = cut.index[-1]
        ea, eb = _events_before(res_full.events, last_ts), _events_before(res_cut.events, last_ts)
        if ea != eb:
            k = next((j for j, (x, y) in enumerate(zip(ea, eb)) if x != y), min(len(ea), len(eb)))
            ev_bad.append(dict(T=int(T), first_diff_index=k, full=ea[k] if k < len(ea) else None,
                               cut=eb[k] if k < len(eb) else None))
    return CausalityReport(passed=not feat_bad and not ev_bad, n_cuts=len(cuts),
                           feature_mismatches=feat_bad, event_mismatches=ev_bad)
