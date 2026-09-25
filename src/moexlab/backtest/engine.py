"""Событийный бэктест по свечам (уровень качества A — свечи).

Принципы:
  * решение принимается на закрытии свечи i только по данным <= i, исполняется на свече i+1;
  * внутри свечи порядок цен неизвестен => всегда выбирается наихудший допустимый вариант
    (стоп проверяется раньше цели; после входа внутри свечи сначала проверяется неблагоприятный экстремум;
     цель в свече входа не исполняется; трейлинг обновляется только по закрытой свече);
  * лимитная заявка исполняется только при проторговке на шаг цены за уровень;
  * стоп-приказ, перепрыгнутый гэпом, исполняется по цене открытия;
  * все исполнения идут через ExecutionModel (спред, проскальзывание, задержка, комиссия);
  * общий капитал портфеля, лимиты риска по портфелю и группам, дневной/недельный лимиты, просадка.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..execution.models import CommissionModel, ConservativeL1ExecutionModel, ExecutionModel
from ..instruments.spec import InstrumentSpec
from ..strategies.base import BarContext, EntryOrder, PositionView, Strategy


# ----------------------------------------------------------------------------
@dataclass
class RiskConfig:
    risk_fraction: float = 0.005              # риск на сделку (доля капитала)
    risk_fraction_by_tag: dict = field(default_factory=dict)
    compounding: bool = True                  # False => база риска = начальный капитал (для чистой оценки преимущества)
    integer_qty: bool = True                  # False => дробные контракты (исследование сигнала без дискретности)
    max_margin_fraction: float = 0.25
    portfolio_max_open_risk: float | None = 0.0125
    group_max_open_risk: float | None = 0.0075
    daily_loss_fraction: float | None = 0.02
    weekly_loss_fraction: float | None = 0.04
    reduce_risk_drawdown: float | None = 0.08
    halt_drawdown: float | None = 0.12
    max_entries_per_instrument_per_day: int | None = 4
    max_participation: float = 0.10           # доля объёма предыдущей свечи
    risk_multiplier: float = 1.0

    @classmethod
    def research(cls, risk_fraction: float = 0.005) -> "RiskConfig":
        """Режим оценки сигнала: без портфельных остановок, без дискретности, без сложного процента."""
        return cls(risk_fraction=risk_fraction, compounding=False, integer_qty=False, max_margin_fraction=10.0,
                   portfolio_max_open_risk=None, group_max_open_risk=None, daily_loss_fraction=None,
                   weekly_loss_fraction=None, reduce_risk_drawdown=None, halt_drawdown=None,
                   max_entries_per_instrument_per_day=None, max_participation=1e9)


@dataclass
class InstrumentInput:
    code: str
    spec: InstrumentSpec
    bars: pd.DataFrame          # index UTC начала свечи; open high low close volume trading_day entry_ok force_exit last_in_day [secid]
    strategy: Strategy
    bar_seconds: float = 60.0


@dataclass
class _Pos:
    direction: int
    qty: float
    entry_price: float
    entry_ref: float
    entry_ts: Any
    entry_index: int
    entry_atr: float
    initial_stop: float
    active_stop: float
    best: float
    hh: float
    ll: float
    risk_money: float
    tag: str
    signal_id: str
    secid: str
    target: float | None
    partial_price: float | None
    trading_day: Any
    bars_held: int = 0
    qty_open: float = 0.0
    realized_gross: float = 0.0
    costs_commission: float = 0.0
    costs_spread: float = 0.0
    costs_slippage: float = 0.0
    exit_value_ref: float = 0.0   # сумма ref*qty по выходам
    exit_value: float = 0.0       # сумма price*qty по выходам
    exits: Any = None
    partial_done: bool = False
    exit_pending: str | None = None
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series            # по меткам времени
    daily_equity: pd.Series      # по торговым дням
    events: list[dict]
    meta: dict


class Engine:
    def __init__(self, instruments: list[InstrumentInput], execution: ExecutionModel | None = None,
                 risk: RiskConfig | None = None, initial_equity: float = 1_000_000.0,
                 groups: dict[str, str] | None = None, record_events: bool = False):
        self.ins = instruments
        self.exe = execution or ConservativeL1ExecutionModel(commission=CommissionModel())
        self.risk = risk or RiskConfig()
        self.initial_equity = initial_equity
        self.groups = groups or {x.code: x.spec.group for x in instruments}
        self.record_events = record_events

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        R = self.risk
        n_ins = len(self.ins)
        arrs = []
        for k, x in enumerate(self.ins):
            df = x.bars
            feats = x.strategy.prepare(df)
            if "atr" not in feats:
                raise ValueError(f"{x.strategy.name}: prepare() must return 'atr'")
            x.strategy.set_bars(df)
            x.strategy.bind(feats)
            x.strategy.reset()
            close = df["close"].to_numpy(float)
            dclose = np.diff(close, prepend=np.nan)
            if "secid" in df:
                # приращения через границу серий не используем
                sec = df["secid"].to_numpy()
                dclose[1:][sec[1:] != sec[:-1]] = np.nan
            sigma = pd.Series(dclose).rolling(60, min_periods=10).std(ddof=0).to_numpy()
            arrs.append(dict(
                ts=df.index, o=df["open"].to_numpy(float), h=df["high"].to_numpy(float),
                l=df["low"].to_numpy(float), c=close, v=df["volume"].to_numpy(float),
                day=df["trading_day"].to_numpy(), entry_ok=df["entry_ok"].to_numpy(bool),
                force_exit=df["force_exit"].to_numpy(bool), last_in_day=df["last_in_day"].to_numpy(bool),
                secid=(df["secid"].to_numpy() if "secid" in df else np.array([x.code] * len(df), dtype=object)),
                atr=feats["atr"].to_numpy(float), sigma=sigma, n=len(df),
            ))
        # единая шкала времени
        all_ts = sorted(set().union(*[set(a["ts"]) for a in arrs]))
        pos_of = [dict(zip(a["ts"], range(a["n"]))) for a in arrs]

        cash = self.initial_equity
        positions: list[_Pos | None] = [None] * n_ins
        pending: list[list[EntryOrder]] = [[] for _ in range(n_ins)]
        last_close = [np.nan] * n_ins
        entries_today = [0] * n_ins
        cur_day_ins = [None] * n_ins
        trades: list[dict] = []
        events: list[dict] = []
        eq_ts, eq_val, eq_day = [], [], []
        hwm = self.initial_equity
        day_start_eq = self.initial_equity
        week_start_eq = self.initial_equity
        cur_day = None
        cur_week = None
        halted_day = halted_week = halted_manual = False
        risk_mult = R.risk_multiplier
        seq = 0
        entered_tag: list = [None] * n_ins

        def mtm_equity() -> float:
            e = cash
            for k, p in enumerate(positions):
                if p is not None and np.isfinite(last_close[k]):
                    sp = self.ins[k].spec
                    half = 0.5 * sp.spread_ticks * sp.tick
                    px = last_close[k] - p.direction * half  # консервативно по исполнимой стороне
                    e += p.direction * (px - p.entry_price) * sp.point_value * p.qty_open
            return e

        def open_risk(k_filter=None, group=None) -> float:
            tot = 0.0
            for k, p in enumerate(positions):
                if p is None:
                    continue
                if k_filter is not None and k != k_filter:
                    continue
                if group is not None and self.groups.get(self.ins[k].code) != group:
                    continue
                sp = self.ins[k].spec
                ref = last_close[k] if np.isfinite(last_close[k]) else p.entry_price
                loss = p.direction * (ref - p.active_stop) * sp.point_value * p.qty_open
                tot += max(0.0, loss)
            return tot

        def do_exit(k: int, i: int, ref: float, qty: float, reason: str, limit: bool = False, is_stop: bool = False):
            nonlocal cash
            p = positions[k]
            a = arrs[k]
            sp = self.ins[k].spec
            sig = a["sigma"][i - 1] if i > 0 and np.isfinite(a["sigma"][i - 1]) else 0.0
            if limit:
                price = ref
                comm = self.exe.commission.fee(price * sp.point_value * qty, qty, sp) * self.exe.scenario.commission_mult \
                    if hasattr(self.exe, "commission") else 0.0
                fill_spread = fill_slip = 0.0
            else:
                f = self.exe.fill(side=-p.direction, ref_price=ref, qty=qty, spec=sp, sigma_per_bar=sig,
                                  bar_seconds=self.ins[k].bar_seconds, is_stop=is_stop)
                price, comm, fill_spread, fill_slip = f.price, f.commission, f.spread_cost, f.slippage_cost
            pnl = p.direction * (price - p.entry_price) * sp.point_value * qty
            cash += pnl - comm
            p.realized_gross += p.direction * (ref - p.entry_ref) * sp.point_value * qty
            p.costs_commission += comm
            p.costs_spread += fill_spread
            p.costs_slippage += fill_slip
            p.exit_value += price * qty
            p.exit_value_ref += ref * qty
            p.qty_open -= qty
            if self.record_events:
                events.append(dict(ts=a["ts"][i], code=self.ins[k].code, event="EXIT", reason=reason,
                                   ref=ref, price=price, qty=qty, signal_id=p.signal_id))
            if p.qty_open <= 1e-12:
                qty_total = p.qty
                net = p.realized_gross - p.costs_commission - p.costs_spread - p.costs_slippage
                # net через фактические цены (контроль): сумма pnl по фактическим ценам минус комиссии
                net_actual = p.direction * (p.exit_value - p.entry_price * qty_total) * sp.point_value - p.costs_commission
                trades.append(dict(
                    code=self.ins[k].code, secid=p.secid, strategy=self.ins[k].strategy.name, tag=p.tag,
                    direction="LONG" if p.direction > 0 else "SHORT", signal_id=p.signal_id,
                    entry_ts=p.entry_ts, exit_ts=a["ts"][i], trading_day=p.trading_day,
                    entry_ref=p.entry_ref, entry_price=p.entry_price,
                    exit_ref=p.exit_value_ref / qty_total, exit_price=p.exit_value / qty_total,
                    qty=qty_total, entry_atr=p.entry_atr, initial_stop=p.initial_stop, final_stop=p.active_stop,
                    risk_money=p.risk_money, gross_pnl=p.realized_gross, commission=p.costs_commission,
                    spread_cost=p.costs_spread, slippage_cost=p.costs_slippage, net_pnl=net_actual,
                    net_pnl_check=net, R=net_actual / p.risk_money if p.risk_money > 0 else np.nan,
                    gross_R=p.realized_gross / p.risk_money if p.risk_money > 0 else np.nan,
                    exit_reason=reason, bars_held=p.bars_held,
                    mae_R=p.mae * sp.point_value * qty_total / p.risk_money if p.risk_money > 0 else np.nan,
                    mfe_R=p.mfe * sp.point_value * qty_total / p.risk_money if p.risk_money > 0 else np.nan,
                    partial=p.partial_done,
                ))
                positions[k] = None

        def stop_reason(p: _Pos) -> str:
            moved = (p.active_stop > p.initial_stop) if p.direction > 0 else (p.active_stop < p.initial_stop)
            return "TRAILING_STOP" if moved else "INITIAL_STOP"

        def try_entry(k: int, i: int) -> None:
            """Исполнение заявок, выставленных на закрытии свечи i-1."""
            nonlocal cash, seq
            orders = pending[k]
            pending[k] = []
            if not orders or positions[k] is not None:
                return
            a = arrs[k]
            o, h, l = a["o"][i], a["h"][i], a["l"][i]
            sp = self.ins[k].spec
            trig = []
            for od in orders:
                if od.kind == "market":
                    trig.append((od, o))
                elif od.kind == "stop":
                    if od.direction > 0 and h >= od.price:
                        trig.append((od, max(o, od.price)))
                    elif od.direction < 0 and l <= od.price:
                        trig.append((od, min(o, od.price)))
                elif od.kind == "limit":
                    if od.direction > 0:
                        if o <= od.price:
                            trig.append((od, o))
                        elif l <= od.price - sp.tick:
                            trig.append((od, od.price))
                    else:
                        if o >= od.price:
                            trig.append((od, o))
                        elif h >= od.price + sp.tick:
                            trig.append((od, od.price))
            if not trig:
                return
            dirs = {od.direction for od, _ in trig}
            if len(dirs) > 1:
                # Сработали заявки разных направлений: порядок внутри свечи неизвестен.
                # Пропуск сделки = заглядывание в будущее (мы «знаем», что цена потом развернулась).
                # Берём направление с худшим исходом при худшем внутрисвечном пути: после входа цена
                # идёт к противоположному экстремуму свечи, затем к закрытию.
                def worst_case_R(od_ref):
                    od_, ref_ = od_ref
                    adv = l if od_.direction > 0 else h
                    return od_.direction * (adv - ref_)
                trig.sort(key=worst_case_R)
                if self.record_events:
                    events.append(dict(ts=a["ts"][i], code=self.ins[k].code, event="AMBIGUOUS_WORST_CASE"))
            od, ref = trig[0]
            d = od.direction
            is_stop_entry = od.kind == "stop" and ref != o
            ent_atr = a["atr"][i - 1]
            if not np.isfinite(ent_atr) or ent_atr <= 0:
                return
            ex = self.ins[k].strategy.exits_for(od.tag)
            # начальный стоп от ориентира (цены исполнения мы ещё не знаем) — пересчитаем после
            sig = a["sigma"][i - 1] if np.isfinite(a["sigma"][i - 1]) else 0.0
            probe = self.exe.fill(side=d, ref_price=ref, qty=1.0, spec=sp, sigma_per_bar=sig,
                                  bar_seconds=self.ins[k].bar_seconds, is_stop=is_stop_entry)
            fill_px = probe.price
            if ex.initial == "atr":
                dist = ex.initial_k * ent_atr
            elif ex.initial == "percent":
                dist = ex.initial_k * fill_px
            elif ex.initial == "structural":
                if od.structural_stop is None or not np.isfinite(od.structural_stop):
                    return
                dist = d * (fill_px - od.structural_stop)
                if dist <= sp.tick:
                    return
            else:
                raise ValueError(ex.initial)
            stop = sp.round_price(fill_px - d * dist, side_up=(d < 0))  # консервативно: риск не меньше расчётного
            dist = abs(fill_px - stop)
            # --- размер позиции ---
            eq_now = mtm_equity()
            base = eq_now if R.compounding else self.initial_equity
            frac = R.risk_fraction_by_tag.get(od.tag, R.risk_fraction) * risk_mult
            risk_budget = base * frac
            pv = sp.point_value
            est_costs = 2 * (probe.commission + probe.spread_cost + probe.slippage_cost)
            per_contract = dist * pv + est_costs
            qty = risk_budget / per_contract
            margin_pc = sp.margin_fraction * fill_px * pv
            if margin_pc > 0:
                used_margin = 0.0
                for j, pj in enumerate(positions):
                    if pj is None:
                        continue
                    spj = self.ins[j].spec
                    pxj = last_close[j] if np.isfinite(last_close[j]) else pj.entry_price
                    used_margin += spj.margin_fraction * pxj * spj.point_value * pj.qty_open
                qty = min(qty, max(0.0, (R.max_margin_fraction * eq_now - used_margin) / margin_pc))
            if i > 0 and np.isfinite(a["v"][i - 1]):
                qty = min(qty, R.max_participation * a["v"][i - 1])
            if R.integer_qty:
                qty = math.floor(qty + 1e-9)
            if qty <= 0:
                if self.record_events:
                    events.append(dict(ts=a["ts"][i], code=self.ins[k].code, event="SKIP_QTY"))
                return
            new_risk = qty * per_contract
            if R.portfolio_max_open_risk is not None and open_risk() + new_risk > R.portfolio_max_open_risk * eq_now + 1e-9:
                if self.record_events:
                    events.append(dict(ts=a["ts"][i], code=self.ins[k].code, event="SKIP_PORTFOLIO_RISK"))
                return
            g = self.groups.get(self.ins[k].code)
            if R.group_max_open_risk is not None and g is not None and \
                    open_risk(group=g) + new_risk > R.group_max_open_risk * eq_now + 1e-9:
                if self.record_events:
                    events.append(dict(ts=a["ts"][i], code=self.ins[k].code, event="SKIP_GROUP_RISK"))
                return
            f = self.exe.fill(side=d, ref_price=ref, qty=qty, spec=sp, sigma_per_bar=sig,
                              bar_seconds=self.ins[k].bar_seconds, is_stop=is_stop_entry)
            cash -= f.commission
            seq += 1
            rr_unit = dist
            target = fill_px + d * ex.target_rr * rr_unit if ex.target_rr else None
            partial = fill_px + d * ex.partial_rr * rr_unit if ex.partial_rr else None
            p = _Pos(direction=d, qty=qty, entry_price=f.price, entry_ref=ref, entry_ts=a["ts"][i], entry_index=i,
                     entry_atr=ent_atr, initial_stop=stop, active_stop=stop, best=f.price, hh=h, ll=l,
                     risk_money=qty * per_contract, tag=od.tag,
                     signal_id=od.signal_id or f"{a['day'][i]}|{self.ins[k].code}|{a['secid'][i]}|{od.tag}|{d}|{seq}",
                     secid=a["secid"][i], target=target, partial_price=partial, trading_day=a["day"][i],
                     qty_open=qty, costs_commission=f.commission, costs_spread=f.spread_cost,
                     costs_slippage=f.slippage_cost, exits=ex)
            entered_tag[k] = (od.tag, d)
            p.hh, p.ll = f.price, f.price
            positions[k] = p
            entries_today[k] += 1
            if self.record_events:
                events.append(dict(ts=a["ts"][i], code=self.ins[k].code, event="ENTRY", tag=od.tag, dir=d,
                                   ref=ref, price=f.price, qty=qty, stop=stop, signal_id=p.signal_id))
            # худший случай внутри свечи входа: после входа цена идёт к неблагоприятному экстремуму
            adverse = l if d > 0 else h
            if (d > 0 and adverse <= stop) or (d < 0 and adverse >= stop):
                do_exit(k, i, stop, p.qty_open, "INITIAL_STOP", is_stop=True)

        # ------------------------------------------------------------------
        for ts in all_ts:
            active = [k for k in range(n_ins) if ts in pos_of[k]]
            # смена торгового дня / недели (по первому инструменту, у которого есть свеча)
            k0 = active[0]
            day = arrs[k0]["day"][pos_of[k0][ts]]
            if day != cur_day:
                if cur_day is not None:
                    pass
                cur_day = day
                day_start_eq = mtm_equity()
                halted_day = False
                wk = day.isocalendar()[:2] if hasattr(day, "isocalendar") else None
                if wk != cur_week:
                    cur_week = wk
                    week_start_eq = day_start_eq
                    halted_week = False
            for k in active:
                i = pos_of[k][ts]
                a = arrs[k]
                if cur_day_ins[k] != a["day"][i]:
                    cur_day_ins[k] = a["day"][i]
                    entries_today[k] = 0
                sp = self.ins[k].spec
                strat = self.ins[k].strategy
                entered_tag[k] = None
                o, h, l, c = a["o"][i], a["h"][i], a["l"][i], a["c"][i]
                p = positions[k]
                ex = p.exits if p is not None else strat.exits
                opened_before = p is not None
                # --- 1. позиция, открытая до этой свечи ---
                if p is not None:
                    if p.exit_pending is not None or (ex.intraday and a["force_exit"][i]):
                        do_exit(k, i, o, p.qty_open, p.exit_pending or "FORCE_CLOSE")
                    else:
                        st = p.active_stop
                        hit = (o <= st or l <= st) if p.direction > 0 else (o >= st or h >= st)
                        if hit:
                            ref = min(o, st) if p.direction > 0 else max(o, st)
                            do_exit(k, i, ref, p.qty_open, stop_reason(p), is_stop=(ref != o))
                        else:
                            # частичная фиксация, затем цель (лимитные заявки; только при проторговке)
                            if p.partial_price is not None and not p.partial_done:
                                pp = p.partial_price
                                if (p.direction > 0 and (o >= pp or h >= pp + sp.tick)) or \
                                        (p.direction < 0 and (o <= pp or l <= pp - sp.tick)):
                                    px = max(o, pp) if p.direction > 0 else min(o, pp)
                                    q = p.qty * ex.partial_frac
                                    if R.integer_qty:
                                        q = math.floor(q)
                                    if 0 < q < p.qty_open:
                                        do_exit(k, i, px, q, "PARTIAL_TARGET", limit=True)
                                        p.partial_done = True
                                        if ex.breakeven_after_partial:
                                            be = p.entry_price
                                            p.active_stop = max(p.active_stop, be) if p.direction > 0 else min(p.active_stop, be)
                            tgt = p.target
                            dyn = strat.dynamic_target(i - 1, self._view(p)) if i > 0 else None
                            if dyn is not None and np.isfinite(dyn):
                                tgt = dyn if tgt is None else (min(tgt, dyn) if p.direction > 0 else max(tgt, dyn))
                            if positions[k] is not None and tgt is not None:
                                if (p.direction > 0 and (o >= tgt or h >= tgt + sp.tick)) or \
                                        (p.direction < 0 and (o <= tgt or l <= tgt - sp.tick)):
                                    px = max(o, tgt) if p.direction > 0 else min(o, tgt)
                                    do_exit(k, i, px, p.qty_open, "TARGET", limit=True)
                # --- 2. вход по заявкам прошлой свечи ---
                if positions[k] is None and not opened_before:
                    try_entry(k, i)
                elif positions[k] is None:
                    pending[k] = []  # после выхода на этой свече новый вход не открываем (§26)
                else:
                    pending[k] = []
                last_close[k] = c
                # --- 3. закрытие свечи: сопровождение позиции ---
                p = positions[k]
                if p is not None:
                    ex = p.exits
                    if p.entry_index != i:
                        p.bars_held += 1
                    else:
                        p.bars_held = 0
                    if p.direction > 0:
                        p.mae = max(p.mae, p.entry_price - l)
                        p.mfe = max(p.mfe, h - p.entry_price)
                    else:
                        p.mae = max(p.mae, h - p.entry_price)
                        p.mfe = max(p.mfe, p.entry_price - l)
                    in_entry_bar = p.entry_index == i
                    # трейлинг по закрытой свече; в свече входа экстремумы не используем (порядок неизвестен)
                    hi = c if in_entry_bar else h
                    lo = c if in_entry_bar else l
                    p.hh = max(p.hh, hi)
                    p.ll = min(p.ll, lo)
                    cand = None
                    if ex.trailing == "atr":
                        p.best = max(p.best, hi) if p.direction > 0 else min(p.best, lo)
                        cand = p.best - p.direction * ex.trailing_k * p.entry_atr
                    elif ex.trailing == "chandelier":
                        cur_atr = a["atr"][i]
                        if np.isfinite(cur_atr):
                            ext = p.hh if p.direction > 0 else p.ll
                            cand = ext - p.direction * ex.trailing_k * cur_atr
                    elif ex.trailing == "percent":
                        p.best = max(p.best, hi) if p.direction > 0 else min(p.best, lo)
                        cand = p.best * (1 - p.direction * ex.trailing_k)
                    elif ex.trailing == "structural":
                        lb = ex.trailing_lookback
                        j0 = max(p.entry_index + 1, i - lb + 1)
                        if i - j0 + 1 >= 1 and not in_entry_bar:
                            cand = a["l"][j0:i + 1].min() if p.direction > 0 else a["h"][j0:i + 1].max()
                    if cand is not None and np.isfinite(cand):
                        cand = sp.round_price(cand, side_up=(p.direction < 0))
                        p.active_stop = max(p.active_stop, cand, p.initial_stop) if p.direction > 0 \
                            else min(p.active_stop, cand, p.initial_stop)
                    # выходы по закрытию свечи
                    if ex.max_bars is not None and p.bars_held >= ex.max_bars:
                        p.exit_pending = "TIME_EXIT"
                    if strat.exit_signal(i, self._view(p)):
                        p.exit_pending = p.exit_pending or "SIGNAL_EXIT"
                    nxt_sec = a["secid"][i + 1] if i + 1 < a["n"] else None
                    if ex.intraday and a["last_in_day"][i]:
                        do_exit(k, i, c, p.qty_open, "FORCE_CLOSE")
                    elif nxt_sec is not None and nxt_sec != p.secid:
                        do_exit(k, i, c, p.qty_open, "ROLL")
                    elif i + 1 >= a["n"]:
                        do_exit(k, i, c, p.qty_open, "END_OF_DATA")
                # --- 4. решение на закрытии свечи ---
                ctx = BarContext(flat=positions[k] is None, entry_ok=bool(a["entry_ok"][i]),
                                 position_dir=0 if positions[k] is None else positions[k].direction,
                                 trading_day=a["day"][i], new_day=(i == 0 or a["day"][i - 1] != a["day"][i]),
                                 entered_tag=entered_tag[k][0] if entered_tag[k] else None,
                                 entered_dir=entered_tag[k][1] if entered_tag[k] else 0)
                orders = strat.on_bar_close(i, ctx)
                allowed = (positions[k] is None and ctx.entry_ok and not (halted_day or halted_week or halted_manual)
                           and (R.max_entries_per_instrument_per_day is None
                                or entries_today[k] < R.max_entries_per_instrument_per_day)
                           and i + 1 < a["n"] and a["secid"][i + 1] == a["secid"][i])
                pending[k] = list(orders) if allowed else []
            # --- 5. портфельные ограничения на закрытии метки времени ---
            eq = mtm_equity()
            hwm = max(hwm, eq)
            if R.daily_loss_fraction is not None and not halted_day and eq - day_start_eq <= -R.daily_loss_fraction * day_start_eq:
                halted_day = True
                self._flatten(positions, "DAILY_LOSS_LIMIT", pending)
            if R.weekly_loss_fraction is not None and not halted_week and eq - week_start_eq <= -R.weekly_loss_fraction * week_start_eq:
                halted_week = True
                self._flatten(positions, "WEEKLY_LOSS_LIMIT", pending)
            dd = eq / hwm - 1
            if R.halt_drawdown is not None and not halted_manual and dd <= -R.halt_drawdown:
                halted_manual = True
                self._flatten(positions, "MAX_DRAWDOWN", pending)
            elif R.reduce_risk_drawdown is not None:
                risk_mult = R.risk_multiplier * (0.5 if dd <= -R.reduce_risk_drawdown else 1.0)
            eq_ts.append(ts)
            eq_val.append(eq)
            eq_day.append(cur_day)

        tr = pd.DataFrame(trades)
        equity = pd.Series(eq_val, index=pd.DatetimeIndex(eq_ts), name="equity")
        daily = pd.Series(eq_val, index=eq_day).groupby(level=0).last()
        daily.index = pd.to_datetime(daily.index)
        return BacktestResult(trades=tr, equity=equity, daily_equity=daily, events=events,
                              meta=dict(initial_equity=self.initial_equity, halted_manual=halted_manual))

    @staticmethod
    def _flatten(positions, reason, pending):
        for k, p in enumerate(positions):
            if p is not None:
                p.exit_pending = reason
            pending[k] = []

    @staticmethod
    def _view(p: _Pos) -> PositionView:
        return PositionView(direction=p.direction, entry_price=p.entry_price, entry_atr=p.entry_atr,
                            initial_stop=p.initial_stop, active_stop=p.active_stop, bars_held=p.bars_held,
                            tag=p.tag, entry_index=p.entry_index)
