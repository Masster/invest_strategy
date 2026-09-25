"""Индикаторы: уровни R-Breaker (§10–11), ATR14 (§12)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PrevDay:
    """H/L/C предыдущего торгового дня активной серии (§9)."""
    high: float
    low: float
    close: float

    def is_valid(self) -> bool:
        return self.high > self.low and self.low <= self.close <= self.high


@dataclass(frozen=True)
class Levels:
    sbreak: float
    bsetup: float
    benter: float
    senter: float
    ssetup: float
    bbreak: float

    @classmethod
    def from_prev_day(cls, d: PrevDay, setup_k: float = 0.35, reversal_k: float = 0.07,
                      breakout_k: float = 0.25) -> "Levels":
        H, L, C = d.high, d.low, d.close
        ssetup = H + setup_k * (C - L)
        bsetup = L - setup_k * (H - C)
        mid = (0.5 + reversal_k / 2.0)  # 0.535 при reversal_k = 0.07
        senter = mid * (H + L) - reversal_k * L
        benter = mid * (H + L) - reversal_k * H
        width = ssetup - bsetup
        bbreak = ssetup + breakout_k * width
        sbreak = bsetup - breakout_k * width
        return cls(sbreak=sbreak, bsetup=bsetup, benter=benter, senter=senter, ssetup=ssetup, bbreak=bbreak)

    def is_ordered(self) -> bool:
        """Проверка §11: Sbreak < Bsetup < Benter < Senter < Ssetup < Bbreak."""
        return self.sbreak < self.bsetup < self.benter < self.senter < self.ssetup < self.bbreak

    def as_dict(self) -> dict[str, float]:
        return {
            "Sbreak": self.sbreak, "Bsetup": self.bsetup, "Benter": self.benter,
            "Senter": self.senter, "Ssetup": self.ssetup, "Bbreak": self.bbreak,
        }


def true_range(high: float, low: float, prev_close: float | None) -> float:
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


class RollingATR:
    """SMA(TR, period) по завершённым свечам (§12). Незавершённые свечи сюда не передаются."""

    def __init__(self, period: int = 14):
        self.period = period
        self._trs: deque[float] = deque(maxlen=period)
        self._prev_close: float | None = None
        self.bars_seen = 0

    def reset(self) -> None:
        """Вызывается при смене серии: ATR считается только по новой серии (§7)."""
        self._trs.clear()
        self._prev_close = None
        self.bars_seen = 0

    def update(self, high: float, low: float, close: float) -> None:
        self._trs.append(true_range(high, low, self._prev_close))
        self._prev_close = close
        self.bars_seen += 1

    @property
    def ready(self) -> bool:
        return len(self._trs) == self.period

    @property
    def value(self) -> float | None:
        if not self.ready:
            return None
        return sum(self._trs) / self.period
