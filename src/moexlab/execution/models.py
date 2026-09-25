"""Модели исполнения и издержек.

ExecutionModel — единый интерфейс (спецификация R-Breaker §71).
  * ConservativeL1ExecutionModel — работает по свечам: восстанавливает bid/ask как
    цена ± половина спреда, добавляет проскальзывание и неблагоприятный сдвиг за время задержки.
  * OrderBookReplayExecutionModel — интерфейс для воспроизведения исторического стакана
    (данные пока недоступны; реализация поднимает NotImplementedError).

Все цены исполнения округляются к шагу цены в худшую для нас сторону.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..instruments.spec import InstrumentSpec


@dataclass(frozen=True)
class CostScenario:
    name: str = "NORMAL"
    spread_mult: float = 1.0
    slippage_mult: float = 1.0
    commission_mult: float = 1.0


SCENARIOS = {
    "ZERO": CostScenario("ZERO", 0.0, 0.0, 0.0),        # только для диагностики «до расходов»
    "NORMAL": CostScenario("NORMAL", 1.0, 1.0, 1.0),
    "STRESS_1": CostScenario("STRESS_1", 1.5, 2.0, 1.0),
    "STRESS_2": CostScenario("STRESS_2", 2.0, 3.0, 1.0),
}


@dataclass(frozen=True)
class CommissionModel:
    """Комиссия брокера (доля оборота) + биржевой сбор (доля оборота и/или RUB за контракт).

    Значения по умолчанию НЕ являются подтверждённым тарифом: см. research/sources.md.
    """
    broker_fraction: float = 0.0004
    exchange_fraction: float = 0.0
    per_contract_rub: float = 0.0

    def fee(self, notional: float, qty: float, spec: InstrumentSpec) -> float:
        return abs(notional) * (self.broker_fraction + self.exchange_fraction) + \
            abs(qty) * (self.per_contract_rub + spec.exchange_fee_per_contract)


@dataclass(frozen=True)
class Fill:
    price: float          # фактическая цена исполнения
    ref_price: float      # цена-ориентир (сделка/уровень) без спреда и проскальзывания
    spread_cost: float    # RUB
    slippage_cost: float  # RUB (включая задержку)
    commission: float     # RUB
    qty: float


class ExecutionModel:
    def fill(self, *, side: int, ref_price: float, qty: float, spec: InstrumentSpec,
             sigma_per_bar: float, bar_seconds: float, is_stop: bool = False) -> Fill:  # pragma: no cover
        raise NotImplementedError


@dataclass
class ConservativeL1ExecutionModel(ExecutionModel):
    """side=+1 покупка (по ask), side=-1 продажа (по bid).

    Цена = ref ± (spread/2 × spread_mult + slippage × slippage_mult) ± latency_adverse,
    latency_adverse = sigma_per_bar × sqrt(latency / bar_seconds) × sqrt(2/π) × latency_aggr —
    ожидаемое абсолютное смещение случайного блуждания за время задержки; при latency_aggr=1
    считаем, что это смещение всегда против нас (консервативно для пробойных входов).
    """
    commission: CommissionModel
    scenario: CostScenario = SCENARIOS["NORMAL"]
    latency_ms: float = 0.0
    latency_aggr: float = 1.0

    def fill(self, *, side: int, ref_price: float, qty: float, spec: InstrumentSpec,
             sigma_per_bar: float, bar_seconds: float, is_stop: bool = False) -> Fill:
        half_spread = 0.5 * spec.spread_ticks * spec.tick * self.scenario.spread_mult
        slip = (spec.slippage_ticks + (spec.stop_extra_ticks if is_stop else 0.0)) * spec.tick * self.scenario.slippage_mult
        lat = 0.0
        if self.latency_ms > 0 and sigma_per_bar > 0 and bar_seconds > 0:
            lat = sigma_per_bar * math.sqrt((self.latency_ms / 1000.0) / bar_seconds) * math.sqrt(2 / math.pi) \
                * self.latency_aggr
        raw = ref_price + side * (half_spread + slip + lat)
        price = spec.round_price(raw, side_up=(side > 0))
        pv = spec.point_value
        q = abs(qty)
        spread_cost = half_spread * pv * q
        slippage_cost = max(0.0, (price - ref_price) * side * pv * q - spread_cost)
        commission = self.commission.fee(price * pv * q, q, spec) * self.scenario.commission_mult
        return Fill(price=price, ref_price=ref_price, spread_cost=spread_cost, slippage_cost=slippage_cost,
                    commission=commission, qty=q)


class OrderBookReplayExecutionModel(ExecutionModel):
    """Воспроизведение исторического стакана: требует данных уровня C (сделки + стакан)."""

    def fill(self, **kw) -> Fill:
        raise NotImplementedError("Historical order book data is not available in this environment")
