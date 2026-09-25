import pytest

from moexlab.execution.models import SCENARIOS, CommissionModel, ConservativeL1ExecutionModel
from moexlab.instruments.spec import InstrumentSpec

SPEC = InstrumentSpec("X", "future", tick=0.5, tick_value=5.0, spread_ticks=2, slippage_ticks=1, stop_extra_ticks=1)


def test_buy_pays_half_spread_plus_slippage_and_rounds_up():
    m = ConservativeL1ExecutionModel(CommissionModel(0.0, 0.0, 0.0))
    f = m.fill(side=+1, ref_price=100.0, qty=2, spec=SPEC, sigma_per_bar=0.0, bar_seconds=60)
    assert f.price == 101.0            # 100 + 0.5*2*0.5 + 1*0.5
    assert f.spread_cost == pytest.approx(0.5 * 2 * 0.5 * 10 * 2)  # half spread * point_value(10) * qty


def test_sell_is_symmetric_and_rounds_down():
    m = ConservativeL1ExecutionModel(CommissionModel(0.0, 0.0, 0.0))
    f = m.fill(side=-1, ref_price=100.2, qty=1, spec=SPEC, sigma_per_bar=0.0, bar_seconds=60)
    assert f.price == 99.0  # 100.2 - 1.0 = 99.2 -> вниз к шагу 0.5


def test_stop_fill_pays_extra_ticks():
    m = ConservativeL1ExecutionModel(CommissionModel(0.0, 0.0, 0.0))
    a = m.fill(side=+1, ref_price=100.0, qty=1, spec=SPEC, sigma_per_bar=0.0, bar_seconds=60)
    b = m.fill(side=+1, ref_price=100.0, qty=1, spec=SPEC, sigma_per_bar=0.0, bar_seconds=60, is_stop=True)
    assert b.price - a.price == pytest.approx(0.5)


def test_stress_scenarios_increase_cost():
    prices = []
    for sc in ("NORMAL", "STRESS_1", "STRESS_2"):
        m = ConservativeL1ExecutionModel(CommissionModel(0.0, 0.0, 0.0), SCENARIOS[sc])
        prices.append(m.fill(side=+1, ref_price=100.0, qty=1, spec=SPEC, sigma_per_bar=0.0, bar_seconds=60).price)
    assert prices[0] < prices[1] < prices[2]


def test_latency_worsens_price_monotonically():
    out = []
    for ms in (0, 100, 250, 500, 1000, 2000):
        m = ConservativeL1ExecutionModel(CommissionModel(0.0, 0.0, 0.0), latency_ms=ms)
        out.append(m.fill(side=+1, ref_price=100.0, qty=1, spec=SPEC, sigma_per_bar=5.0, bar_seconds=60).price)
    assert all(b >= a for a, b in zip(out, out[1:])) and out[-1] > out[0]


def test_commission_fraction_and_per_contract():
    cm = CommissionModel(broker_fraction=0.0004, exchange_fraction=0.0001, per_contract_rub=2.0)
    assert cm.fee(1_000_000.0, 3, SPEC) == pytest.approx(500.0 + 6.0)
