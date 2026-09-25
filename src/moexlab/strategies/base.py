"""Базовый интерфейс стратегии.

Стратегия ничего не знает о брокере и источнике данных. Движок вызывает:
  prepare(df)         — один раз: каузальные признаки (значение в строке t зависит только от строк <= t);
  on_bar_close(i, ctx) — на закрытии каждой свечи i: вернуть заявки на вход для свечи i+1;
  exit_signal(i, pos)  — на закрытии свечи i: True => выход по рынку на открытии i+1;
  dynamic_target(i, pos) — необязательная динамическая цель (лимитная заявка на свечу i+1).
Один и тот же объект используется в BACKTEST / SHADOW / PAPER / LIVE (разные только адаптеры).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EntryOrder:
    direction: int                 # +1 LONG, -1 SHORT
    kind: str                      # "market" | "stop" | "limit"
    price: float | None = None     # уровень для stop/limit
    tag: str = ""                  # тип сигнала (BREAKOUT, REVERSAL, ...)
    structural_stop: float | None = None  # цена структурного стопа, если используется
    signal_id: str = ""


@dataclass(frozen=True)
class ExitPolicy:
    """Правила выхода. Все дистанции ATR считаются от entry_atr (ATR на свече решения).

    initial:   "atr" | "percent" | "structural"
    trailing:  "none" | "atr" | "chandelier" | "percent" | "structural"
    """
    initial: str = "atr"
    initial_k: float = 1.5          # множитель ATR или доля цены для percent
    trailing: str = "atr"
    trailing_k: float = 1.5
    trailing_lookback: int = 10     # для structural / chandelier
    target_rr: float | None = None  # полная фиксация на R-кратном
    partial_rr: float | None = None  # частичная фиксация на R-кратном
    partial_frac: float = 0.5
    breakeven_after_partial: bool = True
    max_bars: int | None = None
    intraday: bool = True           # закрывать до конца торгового дня


@dataclass
class BarContext:
    flat: bool
    entry_ok: bool
    position_dir: int
    trading_day: Any
    new_day: bool
    entered_tag: str | None = None   # тип сигнала, по которому на этой свече открылась позиция
    entered_dir: int = 0


@dataclass
class PositionView:
    direction: int
    entry_price: float
    entry_atr: float
    initial_stop: float
    active_stop: float
    bars_held: int
    tag: str
    entry_index: int


class Strategy:
    name: str = "base"
    family: str = "base"

    def __init__(self, params: dict | None = None, exits: ExitPolicy | None = None):
        self.params: dict = dict(params or {})
        self.exits: ExitPolicy = exits or ExitPolicy()
        self.f: dict[str, np.ndarray] = {}

    # --- жизненный цикл ---
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает DataFrame признаков (тот же индекс). Обязательная колонка: atr."""
        raise NotImplementedError

    def bind(self, feats: pd.DataFrame) -> None:
        self.f = {c: feats[c].to_numpy(dtype=float) for c in feats.columns
                  if pd.api.types.is_numeric_dtype(feats[c])}

    def set_bars(self, df: pd.DataFrame) -> None:
        """Массивы свечей (строка i доступна стратегии только на закрытии свечи i)."""
        self._op = df["open"].to_numpy(float)
        self._hi = df["high"].to_numpy(float)
        self._lo = df["low"].to_numpy(float)
        self._cl = df["close"].to_numpy(float)
        self._vol = df["volume"].to_numpy(float)

    def reset(self) -> None:
        """Сброс внутреннего состояния перед прогоном."""

    def on_bar_close(self, i: int, ctx: BarContext) -> list[EntryOrder]:
        return []

    def exit_signal(self, i: int, pos: PositionView) -> bool:
        return False

    def dynamic_target(self, i: int, pos: PositionView) -> float | None:
        return None

    def exits_for(self, tag: str) -> ExitPolicy:
        """Правила выхода для типа сигнала (по умолчанию общие)."""
        return self.exits

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family, "params": self.params, "exits": self.exits.__dict__}


def finite(x: float) -> bool:
    return x is not None and np.isfinite(x)


__all__ = ["EntryOrder", "ExitPolicy", "BarContext", "PositionView", "Strategy", "finite", "field"]
