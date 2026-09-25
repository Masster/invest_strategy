"""Внутридневное исследование (R-Breaker 2.1 по спецификации + внутридневные семейства) на минутных данных.

Требует data/raw/tinvest_minute (scripts/fetch_tinvest_history.py). Без данных — сухой прогон на синтетике
(--synthetic), который проверяет только работоспособность конвейера, а не рынок.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moexlab.backtest.engine import Engine, InstrumentInput, RiskConfig  # noqa: E402
from moexlab.execution.models import SCENARIOS, CommissionModel, ConservativeL1ExecutionModel  # noqa: E402
from moexlab.instruments.spec import InstrumentSpec  # noqa: E402
from moexlab.market_data.calendar import SessionConfig, annotate_sessions  # noqa: E402
from moexlab.market_data.synthetic import generate_minute_bars  # noqa: E402
from moexlab.research.causality import check_causality  # noqa: E402
from moexlab.statistics.metrics import trade_metrics  # noqa: E402
from moexlab.strategies.rbreaker import RBreaker  # noqa: E402

# С 23.03.2026 FORTS торгуется единой сессией 07:00–23:50 МСК без дневного клиринга (research/sources.md §6);
# R-Breaker торгует основное окно 10:00–19:00 (допущение A-01, конфигурируется).
SESSION_2026 = SessionConfig(windows_msk=((pd.Timestamp("10:00").time(), pd.Timestamp("19:00").time()),))
SESSION_PRE2026 = SessionConfig()


def run_rbreaker(bars: pd.DataFrame, spec: InstrumentSpec, scenario: str = "NORMAL") -> dict:
    exe = ConservativeL1ExecutionModel(CommissionModel(0.00025), SCENARIOS[scenario])
    risk = RiskConfig(risk_fraction=0.005, risk_fraction_by_tag={"BREAKOUT": 0.005, "REVERSAL": 0.0035})
    res = Engine([InstrumentInput(spec.code, spec, bars, RBreaker(), 60)], execution=exe, risk=risk,
                 record_events=True).run()
    return dict(metrics=trade_metrics(res.trades), trades=res.trades, daily=res.daily_equity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        raw = generate_minute_bars("2021-01-04", "2021-12-30", seed=5, daily_vol=0.015, tick=1.0)
        spec = InstrumentSpec("SYN", "future", tick=1.0, tick_value=1.0, group="equity")
        bars = annotate_sessions(raw, SESSION_PRE2026, 1)
        out = run_rbreaker(bars, spec)
        rep = check_causality(raw, lambda: RBreaker({"min_median_volume": 0}), spec, SESSION_PRE2026, 1, n_cuts=5)
        print("SYNTHETIC (not market evidence):", json.dumps(out["metrics"], default=str, indent=1))
        print(rep)
        return
    src = ROOT / "data/raw/tinvest_minute"
    if not src.exists():
        print("No minute data. Open network access to invest-public-api.tbank.ru, set TBANK_TOKEN, "
              "run scripts/fetch_tinvest_history.py")
        return
    raise SystemExit("Real-data intraday stage: implemented after minute data become available (see RESEARCH_PLAN §7)")


if __name__ == "__main__":
    main()
