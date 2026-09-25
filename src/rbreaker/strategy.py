"""Торговое ядро R-Breaker (§14–26, §33–35).

Модуль не знает ни брокера, ни источника данных (§102, §103): на вход получает цены
последних сделок, на выходе даёт кандидата в сигнал или решение о выходе.
Один и тот же код используется историческим тестом, теневой и реальной торговлей.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .indicators import Levels


class State(str, Enum):
    FLAT = "FLAT"
    ENTRY_PENDING = "ENTRY_PENDING"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    HALTED_DAY = "HALTED_DAY"
    HALTED_WEEK = "HALTED_WEEK"
    HALTED_MANUAL = "HALTED_MANUAL"
    DISABLED_FOR_DAY = "DISABLED_FOR_DAY"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class EntryKind(str, Enum):
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"


class ExitReason(str, Enum):
    INITIAL_STOP = "INITIAL_STOP"
    TRAILING_STOP = "TRAILING_STOP"
    FORCE_CLOSE = "FORCE_CLOSE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    WEEKLY_LOSS_LIMIT = "WEEKLY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    DATA_FAILURE = "DATA_FAILURE"
    BROKER_FAILURE = "BROKER_FAILURE"
    PROTECTION_FAILURE = "PROTECTION_FAILURE"
    MANUAL = "MANUAL"


@dataclass(frozen=True)
class SignalCandidate:
    kind: EntryKind
    direction: Direction
    price: float  # цена сделки, на которой сформирован сигнал


@dataclass
class LevelState:
    """Состояние уровней одного инструмента в пределах торгового дня (§15)."""
    upper_setup_seen: bool = False
    lower_setup_seen: bool = False
    long_break_armed: bool = False
    short_break_armed: bool = False
    initialized: bool = False


class SignalEngine:
    """Генератор сигналов по одному инструменту на один торговый день."""

    def __init__(self, levels: Levels):
        self.levels = levels
        self.st = LevelState()
        self.prev_price: float | None = None

    # ------------------------------------------------------------------
    def observe(self, price: float, *, flat: bool, active: bool, rearm_distance: float | None) -> SignalCandidate | None:
        """Обработать одну цену сделки.

        flat   — позиция по инструменту отсутствует (state == FLAT);
        active — время внутри окна новых сделок (после cooldown и до stop_new_entries).
        rearm_distance — 0.25 × ATR14 (None, если ATR ещё не готов).

        Возвращает кандидата; флаги сбрасываются только вызовом `consume`,
        когда сигнал прошёл все внешние ограничения (§18: «после формирования сигнала»).
        """
        lv, st = self.levels, self.st
        prev = self.prev_price
        self.prev_price = price
        if not active:
            return None

        if not st.initialized:
            # §15–16: первая допустимая цена после cooldown определяет взведение пробоев.
            st.initialized = True
            st.long_break_armed = price < lv.bbreak
            st.short_break_armed = price > lv.sbreak
            # Первая цена не может сама быть пересечением: предыдущей допустимой цены нет.
            prev = None

        if not flat:
            return None

        # --- шаг 8 §57: обновить состояния уровней ---
        if rearm_distance is not None:
            if price <= lv.bbreak - rearm_distance:
                st.long_break_armed = True
            if price >= lv.sbreak + rearm_distance:
                st.short_break_armed = True
        if price >= lv.ssetup:
            st.upper_setup_seen = True
        if price <= lv.bsetup:
            st.lower_setup_seen = True
        if prev is not None and prev < lv.bbreak <= price:
            st.upper_setup_seen = False  # §22
        if prev is not None and prev > lv.sbreak >= price:
            st.lower_setup_seen = False  # §25

        if prev is None:
            return None

        # --- BREAKOUT (приоритет, §26) ---
        if st.long_break_armed and prev < lv.bbreak <= price:
            return SignalCandidate(EntryKind.BREAKOUT, Direction.LONG, price)
        if st.short_break_armed and prev > lv.sbreak >= price:
            return SignalCandidate(EntryKind.BREAKOUT, Direction.SHORT, price)
        # --- REVERSAL ---
        if st.upper_setup_seen and prev > lv.senter >= price:
            return SignalCandidate(EntryKind.REVERSAL, Direction.SHORT, price)
        if st.lower_setup_seen and prev < lv.benter <= price:
            return SignalCandidate(EntryKind.REVERSAL, Direction.LONG, price)
        return None

    def consume(self, sig: SignalCandidate) -> None:
        st = self.st
        if sig.kind is EntryKind.BREAKOUT:
            if sig.direction is Direction.LONG:
                st.long_break_armed = False
            else:
                st.short_break_armed = False
        st.upper_setup_seen = False
        st.lower_setup_seen = False


# ----------------------------------------------------------------------
@dataclass
class TrailingStop:
    """Скользящий защитный приказ (§33–35). Уровень монотонен (§105)."""
    direction: Direction
    entry_price: float
    initial_stop: float
    trailing_distance: float
    best_price: float = field(init=False)
    active_stop: float = field(init=False)

    def __post_init__(self) -> None:
        self.best_price = self.entry_price
        self.active_stop = self.initial_stop

    def is_hit(self, executable_price: float) -> bool:
        """Шаг 1: executable_sell (LONG) / executable_buy (SHORT) против активного стопа."""
        if self.direction is Direction.LONG:
            return executable_price <= self.active_stop
        return executable_price >= self.active_stop

    def update(self, trade_price: float) -> None:
        """Шаги 2–4, вызываются только если выхода не было."""
        if self.direction is Direction.LONG:
            self.best_price = max(self.best_price, trade_price)
            cand = self.best_price - self.trailing_distance
            self.active_stop = max(self.active_stop, cand, self.initial_stop)
        else:
            self.best_price = min(self.best_price, trade_price)
            cand = self.best_price + self.trailing_distance
            self.active_stop = min(self.active_stop, cand, self.initial_stop)

    @property
    def is_trailing(self) -> bool:
        if self.direction is Direction.LONG:
            return self.active_stop > self.initial_stop
        return self.active_stop < self.initial_stop
