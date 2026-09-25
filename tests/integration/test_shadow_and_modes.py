import os

import pandas as pd
import pytest

from moexlab.backtest.engine import Engine, InstrumentInput, RiskConfig
from moexlab.execution.models import CommissionModel, ConservativeL1ExecutionModel
from moexlab.instruments.spec import InstrumentSpec
from moexlab.live.modes import LiveTradingDisabled, Mode, assert_mode_allowed
from moexlab.market_data.bars import annotate_daily
from moexlab.market_data.synthetic import generate_daily_bars
from moexlab.shadow.runner import ReplayAdapter, ShadowRunner
from moexlab.strategies.base import ExitPolicy
from moexlab.strategies.families import DonchianBreakout

SPEC = InstrumentSpec("S", "future", tick=0.01, tick_value=0.01)
EXE = ConservativeL1ExecutionModel(CommissionModel(0.00025))


def make():
    return DonchianBreakout({"n": 20}, ExitPolicy("atr", 2.0, "atr", 3.0, intraday=False))


def test_live_is_disabled_by_default():
    with pytest.raises(LiveTradingDisabled):
        assert_mode_allowed(Mode.LIVE)
    os.environ["LIVE_TRADING"] = "true"
    try:
        with pytest.raises(LiveTradingDisabled):
            assert_mode_allowed(Mode.LIVE, {"live": {"enabled": True, "confirmation_phrase": "x"}}, "x")
    finally:
        os.environ["LIVE_TRADING"] = "false"
    assert_mode_allowed(Mode.SHADOW)


def test_shadow_signals_equal_backtest_decisions(tmp_path):
    """Единое торговое ядро: заявки, которые SHADOW выставляет на свечу t+1, совпадают с тем,
    что бэктест исполняет/выставляет на этой свече."""
    df = annotate_daily(generate_daily_bars(300, seed=3, trend_persist=0.05))
    ad = ReplayAdapter({"S": df}, ticks={"S": 0.01})
    sh = ShadowRunner(ad, {"S": SPEC}, make, EXE, tmp_path / "shadow.jsonl")
    shadow_orders = {}
    for n in range(60, 300):
        ad.advance("S", n)
        recs = sh.cycle()
        shadow_orders[str(df.index[n - 1])] = sorted((r.direction, r.kind, round(r.level or 0, 6)) for r in recs)
    res = Engine([InstrumentInput("S", SPEC, df, make(), 86400)], execution=EXE, risk=RiskConfig.research(),
                 record_events=True).run()
    # каждый вход бэктеста на свече j должен был быть выставлен SHADOW на закрытии свечи j-1
    for e in res.events:
        if e["event"] != "ENTRY":
            continue
        j = df.index.get_loc(e["ts"])
        prev = str(df.index[j - 1])
        if prev in shadow_orders:
            assert any(d == e["dir"] for d, _, _ in shadow_orders[prev]), (prev, shadow_orders[prev])
    assert (tmp_path / "shadow.jsonl").exists()
