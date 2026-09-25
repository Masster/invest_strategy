"""Конфигурация стратегии (спецификация §107, §108)."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

import yaml

STRATEGY_VERSION = "2.1"


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class InstrumentCosts:
    spread_ticks: float
    slippage_ticks: float


@dataclass
class Config:
    raw: dict[str, Any]

    # --- R-Breaker ---
    setup_k: float = 0.35
    reversal_k: float = 0.07
    breakout_k: float = 0.25
    # --- ATR ---
    atr_timeframe_minutes: int = 10
    atr_period: int = 14
    # --- session ---
    opening_cooldown_minutes: int = 15
    stop_new_entries_before_close_minutes: int = 20
    force_close_before_close_minutes: int = 15
    trading_windows_msk: list[tuple[time, time]] = field(default_factory=list)
    trading_day_boundary_msk: time = time(19, 0)
    # --- risk ---
    breakout_fraction: float = 0.005
    reversal_fraction: float = 0.0035
    portfolio_max_open_risk: float = 0.0125
    group_max_open_risk: float = 0.0075
    max_margin_fraction: float = 0.25
    # --- stops ---
    breakout_initial_atr: float = 1.5
    breakout_trailing_atr: float = 1.5
    reversal_initial_atr: float = 1.0
    reversal_trailing_atr: float = 1.0
    # --- limits ---
    max_entries_per_instrument_per_day: int = 4
    daily_loss_fraction: float = 0.02
    weekly_loss_fraction: float = 0.04
    reduce_risk_drawdown: float = 0.08
    halt_drawdown: float = 0.12
    # --- liquidity ---
    median_volume_days: int = 20
    minimum_median_volume: float = 10000
    maximum_spread_fraction: float = 0.0005
    maximum_spread_of_stop: float = 0.10
    # --- execution ---
    maximum_slippage_fraction: float = 0.0005
    maximum_slippage_of_stop: float = 0.10
    # --- assumptions ---
    initial_equity: float = 1_000_000.0
    historical_margin_fraction: float = 0.15
    broker_commission_fraction: float = 0.0004
    exchange_commission_fraction: float = 0.0
    default_spread_ticks: float = 1.0
    default_slippage_ticks: float = 1.0
    instrument_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    early_roll_requires_spread_data: bool = True
    # --- min history (§8) ---
    min_daily_history: int = 20
    min_atr_bars: int = 14
    # --- risk multiplier on top of everything (§93) ---
    risk_multiplier: float = 1.0

    instruments: list[str] = field(default_factory=lambda: ["MX", "RN", "CR", "GD"])

    # Консервативные группы риска (§44)
    risk_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"equity": ["MX", "RN"], "fx_commodity": ["CR", "GD"]}
    )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        version = str(raw.get("strategy", {}).get("version"))
        if version != STRATEGY_VERSION:
            raise ValueError(f"config version {version!r} != strategy version {STRATEGY_VERSION}")
        s, rb, atr = raw["session"], raw["rbreaker"], raw["atr"]
        if atr.get("method", "SMA") != "SMA":
            raise ValueError("ATR method must be SMA (spec §12)")
        r, st, lim = raw["risk"], raw["stops"], raw["limits"]
        liq, ex = raw["liquidity"], raw["execution"]
        a = raw.get("assumptions", {})
        cm = a.get("commission_model", {})
        windows = [(_parse_hhmm(o), _parse_hhmm(c)) for o, c in a.get("trading_windows_msk", [["10:00", "18:50"]])]
        return cls(
            raw=copy.deepcopy(raw),
            setup_k=rb["setup_coefficient"],
            reversal_k=rb["reversal_coefficient"],
            breakout_k=rb["breakout_coefficient"],
            atr_timeframe_minutes=atr["timeframe_minutes"],
            atr_period=atr["period"],
            opening_cooldown_minutes=s["opening_cooldown_minutes"],
            stop_new_entries_before_close_minutes=s["stop_new_entries_before_close_minutes"],
            force_close_before_close_minutes=s["force_close_before_close_minutes"],
            trading_windows_msk=windows,
            trading_day_boundary_msk=_parse_hhmm(a.get("trading_day_boundary_msk", "19:00")),
            breakout_fraction=r["breakout_fraction"],
            reversal_fraction=r["reversal_fraction"],
            portfolio_max_open_risk=r["portfolio_max_open_risk"],
            group_max_open_risk=r["group_max_open_risk"],
            max_margin_fraction=r["max_margin_fraction"],
            breakout_initial_atr=st["breakout_initial_atr"],
            breakout_trailing_atr=st["breakout_trailing_atr"],
            reversal_initial_atr=st["reversal_initial_atr"],
            reversal_trailing_atr=st["reversal_trailing_atr"],
            max_entries_per_instrument_per_day=lim["max_entries_per_instrument_per_day"],
            daily_loss_fraction=lim["daily_loss_fraction"],
            weekly_loss_fraction=lim["weekly_loss_fraction"],
            reduce_risk_drawdown=lim["reduce_risk_drawdown"],
            halt_drawdown=lim["halt_drawdown"],
            median_volume_days=liq["median_volume_days"],
            minimum_median_volume=liq["minimum_median_volume"],
            maximum_spread_fraction=liq["maximum_spread_fraction"],
            maximum_spread_of_stop=liq["maximum_spread_of_stop"],
            maximum_slippage_fraction=ex["maximum_slippage_fraction"],
            maximum_slippage_of_stop=ex["maximum_slippage_of_stop"],
            initial_equity=float(a.get("initial_equity", 1_000_000.0)),
            historical_margin_fraction=float(a.get("historical_margin_fraction_of_notional", 0.15)),
            broker_commission_fraction=float(cm.get("broker_fraction_of_notional", 0.0004)),
            exchange_commission_fraction=float(cm.get("exchange_fraction_of_notional", 0.0)),
            default_spread_ticks=float(a.get("default_spread_ticks", 1.0)),
            default_slippage_ticks=float(a.get("default_slippage_ticks", 1.0)),
            instrument_overrides=dict(a.get("instrument_overrides") or {}),
            early_roll_requires_spread_data=bool(a.get("early_roll_requires_spread_data", True)),
            instruments=list(raw.get("instruments", ["MX", "RN", "CR", "GD"])),
        )

    def costs_for(self, code: str) -> InstrumentCosts:
        o = self.instrument_overrides.get(code, {})
        return InstrumentCosts(
            spread_ticks=float(o.get("spread_ticks", self.default_spread_ticks)),
            slippage_ticks=float(o.get("slippage_ticks", self.default_slippage_ticks)),
        )

    def group_of(self, code: str) -> str | None:
        for g, members in self.risk_groups.items():
            if code in members:
                return g
        return None

    def hash(self) -> str:
        """configuration_hash (§100)."""
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
