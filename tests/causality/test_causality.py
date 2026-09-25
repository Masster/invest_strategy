from moexlab.instruments.spec import InstrumentSpec
from moexlab.market_data.calendar import SessionConfig
from moexlab.market_data.synthetic import generate_minute_bars
from moexlab.research.causality import check_causality
from moexlab.strategies.rbreaker import RBreaker

SPEC = InstrumentSpec("SYN", "future", tick=1.0, tick_value=1.0, group="equity")


def test_rbreaker_is_causal():
    raw = generate_minute_bars("2021-01-04", "2021-03-31", seed=7, daily_vol=0.02)
    rep = check_causality(raw, lambda: RBreaker({"min_median_volume": 0, "min_days": 5}), SPEC, SessionConfig(), 1,
                          n_cuts=6, seed=1)
    assert rep.passed, rep


def test_causality_check_detects_lookahead():
    """Контроль самого теста: стратегия, использующая следующую свечу, обязана провалиться."""
    import pandas as pd

    from moexlab.indicators.core import atr
    from moexlab.strategies.base import EntryOrder, ExitPolicy, Strategy

    class Leaky(Strategy):
        name = "LEAKY"

        def prepare(self, d):
            return pd.DataFrame({"atr": atr(d, 14), "fut": d["close"].shift(-1) - d["close"]}, index=d.index)

        def on_bar_close(self, i, ctx):
            x = self.f["fut"][i]
            if ctx.flat and ctx.entry_ok and x == x and abs(x) > 0:
                return [EntryOrder(1 if x > 0 else -1, "market", tag="L")]
            return []

    raw = generate_minute_bars("2021-01-04", "2021-02-15", seed=2)
    rep = check_causality(raw, lambda: Leaky(exits=ExitPolicy("atr", 1.0, "none", max_bars=1)), SPEC,
                          SessionConfig(), 1, n_cuts=4, seed=3)
    assert not rep.passed
