import numpy as np
import pandas as pd
import pytest

from moexlab.backtest.engine import Engine, InstrumentInput, RiskConfig
from moexlab.execution.models import SCENARIOS, CommissionModel, ConservativeL1ExecutionModel
from moexlab.indicators.core import atr
from moexlab.instruments.spec import InstrumentSpec
from moexlab.market_data.calendar import SessionConfig, annotate_sessions
from moexlab.market_data.synthetic import generate_minute_bars
from moexlab.strategies.base import EntryOrder, ExitPolicy, Strategy

ZERO = ConservativeL1ExecutionModel(CommissionModel(0, 0, 0), SCENARIOS["ZERO"])


class Scripted(Strategy):
    """Выставляет заранее заданные заявки на закрытии указанных свечей."""
    name = "SCRIPTED"

    def __init__(self, script, exits, atr_value=1.0):
        super().__init__({}, exits)
        self.script, self.atr_value = script, atr_value

    def prepare(self, d):
        return pd.DataFrame({"atr": self.atr_value}, index=d.index)

    def on_bar_close(self, i, ctx):
        return self.script.get(i, []) if ctx.flat else []


def _bars(rows):
    idx = pd.date_range("2024-03-04 07:20", periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    df["trading_day"] = idx[0].date()
    df["entry_ok"] = True
    df["force_exit"] = False
    df["last_in_day"] = False
    df.iloc[-1, df.columns.get_loc("last_in_day")] = True
    return df


SPEC = InstrumentSpec("X", "future", tick=1.0, tick_value=1.0, spread_ticks=0, slippage_ticks=0, stop_extra_ticks=0)


def _run(df, strat, exe=ZERO, risk=None):
    return Engine([InstrumentInput("X", SPEC, df, strat, 60)], execution=exe,
                  risk=risk or RiskConfig.research(), record_events=True).run()


def test_stop_entry_fills_at_level_and_gap_fills_at_open():
    df = _bars([(100, 101, 99, 100), (100, 106, 100, 105), (105, 105, 104, 104), (110, 112, 109, 111),
                (111, 111, 110, 110)])
    st = Scripted({0: [EntryOrder(+1, "stop", 103, "T")], 2: [EntryOrder(+1, "stop", 106, "T")]},
                  ExitPolicy("atr", 50, "none", intraday=True))
    tr = _run(df, st).trades
    assert tr.iloc[0]["entry_price"] == 103          # уровень внутри свечи 1
    assert len(tr) == 1                               # позиция держится до конца данных


def test_gap_through_stop_exits_at_open():
    df = _bars([(100, 100, 100, 100), (100, 101, 99, 100), (90, 91, 89, 90), (90, 90, 90, 90)])
    st = Scripted({0: [EntryOrder(+1, "market", tag="T")]}, ExitPolicy("atr", 5, "none", intraday=True))
    tr = _run(df, st).trades
    assert tr.iloc[0]["exit_price"] == 90 and tr.iloc[0]["exit_reason"] == "INITIAL_STOP"


def test_entry_bar_worst_case_adverse_after_entry():
    # вход по стопу 102; в той же свече low=94 ниже стопа 97 => считаем, что low был ПОСЛЕ входа
    df = _bars([(100, 100, 100, 100), (100, 103, 94, 101), (101, 101, 101, 101)])
    st = Scripted({0: [EntryOrder(+1, "stop", 102, "T")]}, ExitPolicy("atr", 5, "none", intraday=True))
    tr = _run(df, st).trades
    assert tr.iloc[0]["exit_price"] == 97 and tr.iloc[0]["R"] == pytest.approx(-1.0)


def test_stop_checked_before_target_in_same_bar():
    df = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 120, 80, 100), (100, 100, 100, 100)])
    st = Scripted({0: [EntryOrder(+1, "market", tag="T")]}, ExitPolicy("atr", 5, "none", target_rr=1.0, intraday=True))
    tr = _run(df, st).trades
    assert tr.iloc[0]["exit_reason"] == "INITIAL_STOP"


def test_opposite_triggers_in_one_bar_are_skipped():
    df = _bars([(100, 100, 100, 100), (100, 110, 90, 100), (100, 100, 100, 100)])
    st = Scripted({0: [EntryOrder(+1, "stop", 105, "A"), EntryOrder(-1, "stop", 95, "B")]},
                  ExitPolicy("atr", 5, "none", intraday=True))
    res = _run(df, st)
    assert res.trades.empty and any(e["event"] == "AMBIGUOUS_SKIP" for e in res.events)


def test_trailing_stop_is_monotonic_and_uses_closed_bars():
    df = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 110, 100, 109), (109, 115, 108, 114),
                (114, 114, 108, 109), (109, 109, 109, 109)])
    st = Scripted({0: [EntryOrder(+1, "market", tag="T")]}, ExitPolicy("atr", 5, "atr", 5, intraday=True))
    tr = _run(df, st).trades
    # после свечи 3 best=115 => стоп 110; свеча 4 low=108 => выход по 110
    assert tr.iloc[0]["exit_price"] == 110 and tr.iloc[0]["exit_reason"] == "TRAILING_STOP"


def test_position_size_uses_risk_budget_and_floor():
    df = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)])
    st = Scripted({0: [EntryOrder(+1, "market", tag="T")]}, ExitPolicy("atr", 7, "none", intraday=True))
    risk = RiskConfig(risk_fraction=0.005, integer_qty=True, max_margin_fraction=100, portfolio_max_open_risk=None,
                      group_max_open_risk=None, max_participation=1e9)
    tr = _run(df, st, risk=risk).trades
    assert tr.iloc[0]["qty"] == 714   # floor(5000 / 7)


def test_zero_cost_random_entries_have_zero_expectancy():
    raw = generate_minute_bars("2021-01-04", "2021-04-30", seed=3, tick=0.01, substeps=60, evening=False)
    df = annotate_sessions(raw, SessionConfig(), 1)

    class Rand(Strategy):
        name = "RAND"

        def prepare(self, d):
            return pd.DataFrame({"atr": atr(d, 30)}, index=d.index)

        def reset(self):
            self.rng = np.random.default_rng(11)

        def on_bar_close(self, i, ctx):
            if ctx.flat and ctx.entry_ok and self.rng.random() < 0.03:
                return [EntryOrder(1 if self.rng.random() < .5 else -1, "market", tag="R")]
            return []

    spec = InstrumentSpec("S", "future", tick=0.01, tick_value=0.01, spread_ticks=0, slippage_ticks=0, stop_extra_ticks=0)
    st = Rand(exits=ExitPolicy("atr", 1.5, "none", target_rr=1.0, intraday=True))
    tr = Engine([InstrumentInput("S", spec, df, st, 60)], execution=ZERO, risk=RiskConfig.research()).run().trades
    se = tr.R.std() / np.sqrt(len(tr))
    assert len(tr) > 500
    assert abs(tr.R.mean()) < 3 * se


def test_costs_make_random_entries_negative():
    raw = generate_minute_bars("2021-01-04", "2021-03-31", seed=4, tick=0.01, substeps=60, evening=False)
    df = annotate_sessions(raw, SessionConfig(), 1)

    class Every(Strategy):
        name = "EVERY"

        def prepare(self, d):
            return pd.DataFrame({"atr": atr(d, 30)}, index=d.index)

        def on_bar_close(self, i, ctx):
            return [EntryOrder(1 if i % 2 else -1, "market", tag="E")] if ctx.flat and ctx.entry_ok and i % 20 == 0 else []

    spec = InstrumentSpec("S", "future", tick=0.01, tick_value=0.01, spread_ticks=5, slippage_ticks=5)
    exe = ConservativeL1ExecutionModel(CommissionModel(0.0004))
    tr = Engine([InstrumentInput("S", spec, df, Every(exits=ExitPolicy("atr", 1.5, "none", target_rr=1.0)), 60)],
                execution=exe, risk=RiskConfig.research()).run().trades
    assert tr.R.mean() < 0
    assert (tr.commission > 0).all() and (tr.spread_cost > 0).all()
