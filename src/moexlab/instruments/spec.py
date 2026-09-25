"""Параметры инструмента.

Здесь НЕТ зашитых рыночных фактов (шаг цены, стоимость шага, ГО): они загружаются из метаданных
источника — T-Invest `InstrumentsService` (таблица instruments в data/db/market.duckdb).
Для синтетических данных используются явно помеченные синтетические спецификации.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstrumentSpec:
    code: str                      # базовый код (MX, RN, SBER ...)
    kind: str                      # "future" | "equity"
    tick: float                    # минимальный шаг цены
    tick_value: float              # стоимость шага в RUB на 1 контракт / 1 лот
    lot: int = 1                   # бумаг в лоте (для акций)
    group: str = "other"           # группа риска
    spread_ticks: float = 1.0      # типичный спред, шагов (оценка из L1-данных)
    slippage_ticks: float = 1.0    # базовое проскальзывание, шагов на сторону
    stop_extra_ticks: float = 1.0  # доп. проскальзывание стоп-заявок (проскок цены за уровень, см. docs/ASSUMPTIONS.md A-05)
    margin_fraction: float = 0.15  # ГО как доля номинала (модель для истории, если нет данных)
    exchange_fee_per_contract: float = 0.0  # биржевой сбор RUB на контракт (если известен)
    source: str = "UNVERIFIED"

    @property
    def point_value(self) -> float:
        """Стоимость изменения цены на 1.0 в RUB на контракт (лот)."""
        return self.tick_value / self.tick

    def round_price(self, price: float, side_up: bool) -> float:
        """Консервативное округление к шагу цены: вверх (side_up) или вниз."""
        n = price / self.tick
        n = math.ceil(n - 1e-9) if side_up else math.floor(n + 1e-9)
        return round(n * self.tick, 10)

    def notional(self, price: float) -> float:
        return price * self.point_value


def load_specs(path: str | Path) -> dict[str, InstrumentSpec]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: InstrumentSpec(**v) for k, v in raw.items()}


def save_specs(specs: dict[str, InstrumentSpec], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in specs.items()}, f, ensure_ascii=False, indent=2)
