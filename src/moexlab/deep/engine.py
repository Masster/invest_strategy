"""Быстрый бар-движок (Numba) для массового скрининга.

Семантика (консервативная, только OHLC):
  * сигнал формируется на закрытии свечи i (все признаки используют данные <= i);
  * заявка исполняется на свече i+1: рыночная — по open[i+1] ± (½ спреда + проскальзывание);
    стоп-заявка на вход — по уровню (или по open при гэпе через уровень) ± издержки; лимитная — только если
    цена прошла уровень насквозь (или open уже лучше уровня); заявка живёт одну свечу;
  * защитный стоп: по уровню, при гэпе — по open; + доп. проскальзывание; тейк — только при прохождении уровня
    насквозь; если на свече достижимы стоп и тейк — считается стоп; если вход был стоп-заявкой внутри свечи
    и low (для лонга) достиг стопа — считается, что стоп сработал на той же свече;
  * трейлинг (chandelier) пересчитывается по закрытию свечи и действует со следующей;
  * выход по сигналу / по времени — рыночной заявкой по open следующей свечи;
  * выход в конце дня (eod) и на последней свече серии фьючерса (перекладка) — по close этой свечи
    рыночной заявкой (с издержками);
  * размер: 1 единица номинала на капитал 1 (без плеча, без сложного процента) → доходность в долях капитала;
    переоценка по close каждой свечи, агрегирование по торговым дням.
Причины выхода: 1 стоп, 2 тейк, 3 трейлинг, 4 сигнал, 5 время, 6 конец дня, 7 перекладка, 8 разворот.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def simulate(o, h, l, c, day_idx, n_days, last_of_day, last_of_series,
             sig_long, sig_short, exit_long, exit_short,
             entry_mode, lvl_long, lvl_short, atr,
             stop_k, tp_k, trail_k, time_bars, eod_exit, state_mode, direction,
             comm, half_spread, slip, stop_extra):
    n = len(o)
    day_ret = np.zeros(n_days)
    day_expo = np.zeros(n_days)        # доля свечей дня в позиции (для оценки загрузки)
    max_tr = n // 2 + 2
    t_ent = np.empty(max_tr, np.int64)
    t_ext = np.empty(max_tr, np.int64)
    t_side = np.empty(max_tr, np.int8)
    t_ret = np.empty(max_tr)
    t_rsn = np.empty(max_tr, np.int8)
    t_risk = np.empty(max_tr)
    nt = 0

    pos = 0            # +1 / -1 / 0
    units = 0.0
    mark = 0.0
    entry_px = 0.0
    entry_i = 0
    stop = 0.0
    tp = 0.0
    trail = 0.0
    extreme = 0.0
    trade_pnl = 0.0
    risk0 = 0.0
    pend_entry = 0     # направление заявки на вход, выставленной на закрытии i-1
    pend_lvl = 0.0
    pend_exit = 0      # причина рыночного выхода по open
    need_reset_l = False
    need_reset_s = False

    for i in range(n):
        d = day_idx[i]
        # ---------- 1. рыночный выход по open (сигнал/время/разворот)
        if pos != 0 and pend_exit > 0:
            px = o[i] - pos * (half_spread + slip)
            pnl = units * pos * (px - mark) - comm * units * px
            day_ret[d] += pnl
            trade_pnl += pnl
            t_ent[nt] = entry_i; t_ext[nt] = i; t_side[nt] = pos; t_ret[nt] = trade_pnl; t_rsn[nt] = pend_exit
            t_risk[nt] = risk0
            nt += 1
            if pend_exit == 5 or pend_exit == 4:
                if pos > 0:
                    need_reset_l = True
                else:
                    need_reset_s = True
            pos = 0
        pend_exit = 0
        # ---------- 2. вход
        entered_intrabar = False
        if pos == 0 and pend_entry != 0:
            side = pend_entry
            fill = -1.0
            if entry_mode == 0:
                fill = o[i] + side * (half_spread + slip)
            elif entry_mode == 1:   # стоп-заявка (пробой)
                L = pend_lvl
                if side > 0:
                    if o[i] >= L:
                        fill = o[i] + half_spread + slip
                    elif h[i] >= L:
                        fill = L + half_spread + slip
                        entered_intrabar = True
                else:
                    if o[i] <= L:
                        fill = o[i] - half_spread - slip
                    elif l[i] <= L:
                        fill = L - half_spread - slip
                        entered_intrabar = True
            else:                   # лимит (откат)
                L = pend_lvl
                if side > 0:
                    if o[i] < L:
                        fill = o[i]
                    elif l[i] < L:
                        fill = L
                        entered_intrabar = True
                else:
                    if o[i] > L:
                        fill = o[i]
                    elif h[i] > L:
                        fill = L
                        entered_intrabar = True
            if fill > 0:
                pos = side
                entry_px = fill
                units = 1.0 / fill
                mark = fill
                entry_i = i
                a = atr[i - 1] if i > 0 else atr[i]
                if not (a > 0):
                    a = 0.0
                c_in = comm * units * fill
                day_ret[d] -= c_in
                trade_pnl = -c_in
                stop = fill - side * stop_k * a if stop_k > 0 else 0.0
                tp = fill + side * tp_k * a if tp_k > 0 else 0.0
                trail = 0.0
                extreme = fill
                risk0 = stop_k * a / fill if stop_k > 0 else 0.0
        pend_entry = 0
        # ---------- 3. стопы/тейки внутри свечи
        if pos != 0:
            ex = -1.0
            rsn = 0
            eff_stop = stop
            is_trail = False
            if trail > 0:
                if eff_stop <= 0 or (pos > 0 and trail > eff_stop) or (pos < 0 and trail < eff_stop):
                    eff_stop = trail
                    is_trail = True
            if pos > 0:
                if eff_stop > 0:
                    if o[i] <= eff_stop and not (entered_intrabar or i == entry_i):
                        ex = o[i] - half_spread - slip - stop_extra
                        rsn = 3 if is_trail else 1
                    elif l[i] <= eff_stop:
                        ex = min(eff_stop, o[i]) - half_spread - slip - stop_extra if not entered_intrabar else eff_stop - half_spread - slip - stop_extra
                        rsn = 3 if is_trail else 1
                if ex < 0 and tp > 0 and h[i] > tp:
                    ex = tp
                    rsn = 2
            else:
                if eff_stop > 0:
                    if o[i] >= eff_stop and not (entered_intrabar or i == entry_i):
                        ex = o[i] + half_spread + slip + stop_extra
                        rsn = 3 if is_trail else 1
                    elif h[i] >= eff_stop:
                        ex = max(eff_stop, o[i]) + half_spread + slip + stop_extra if not entered_intrabar else eff_stop + half_spread + slip + stop_extra
                        rsn = 3 if is_trail else 1
                if ex < 0 and tp > 0 and l[i] < tp:
                    ex = tp
                    rsn = 2
            if ex > 0:
                pnl = units * pos * (ex - mark) - comm * units * ex
                day_ret[d] += pnl
                trade_pnl += pnl
                t_ent[nt] = entry_i; t_ext[nt] = i; t_side[nt] = pos; t_ret[nt] = trade_pnl; t_rsn[nt] = rsn
                t_risk[nt] = risk0
                nt += 1
                if pos > 0:
                    need_reset_l = True
                else:
                    need_reset_s = True
                pos = 0
        # ---------- 4. переоценка / принудительный выход по close
        if pos != 0:
            day_expo[d] += 1.0
            forced = 0
            if last_of_series[i]:
                forced = 7
            elif eod_exit and last_of_day[i]:
                forced = 6
            if forced > 0:
                px = c[i] - pos * (half_spread + slip)
                pnl = units * pos * (px - mark) - comm * units * px
                day_ret[d] += pnl
                trade_pnl += pnl
                t_ent[nt] = entry_i; t_ext[nt] = i; t_side[nt] = pos; t_ret[nt] = trade_pnl; t_rsn[nt] = forced
                t_risk[nt] = risk0
                nt += 1
                pos = 0
            else:
                pnl = units * pos * (c[i] - mark)
                day_ret[d] += pnl
                trade_pnl += pnl
                mark = c[i]
                # трейлинг по закрытию
                if trail_k > 0 and atr[i] > 0:
                    if pos > 0:
                        if h[i] > extreme:
                            extreme = h[i]
                        nt_ = extreme - trail_k * atr[i]
                        if nt_ > trail:
                            trail = nt_
                    else:
                        if l[i] < extreme:
                            extreme = l[i]
                        nt_ = extreme + trail_k * atr[i]
                        if trail == 0.0 or nt_ < trail:
                            trail = nt_
                # выход по сигналу / времени — на open следующей свечи
                if pos > 0 and exit_long[i]:
                    pend_exit = 4
                elif pos < 0 and exit_short[i]:
                    pend_exit = 4
                elif time_bars > 0 and (i - entry_i + 1) >= time_bars:
                    pend_exit = 5
        # ---------- 5. сброс state-режима
        if not sig_long[i]:
            need_reset_l = False
        if not sig_short[i]:
            need_reset_s = False
        # ---------- 6. новые заявки на вход (исполнение на i+1)
        if i + 1 < n and not last_of_series[i] and not (eod_exit and last_of_day[i]):
            want = 0
            if sig_long[i] and direction >= 0 and not (state_mode and need_reset_l):
                want = 1
            elif sig_short[i] and direction <= 0 and not (state_mode and need_reset_s):
                want = -1
            if want != 0:
                if pos == 0:
                    pend_entry = want
                    pend_lvl = lvl_long[i] if want > 0 else lvl_short[i]
                elif pos == -want and pend_exit == 0:
                    pend_exit = 8           # разворот: выход и вход по одному open
                    pend_entry = want
                    pend_lvl = lvl_long[i] if want > 0 else lvl_short[i]
                elif pos == -want:
                    pend_entry = want
                    pend_lvl = lvl_long[i] if want > 0 else lvl_short[i]
    # позиция на конец данных: закрываем по последнему close
    if pos != 0:
        i = n - 1
        px = c[i] - pos * (half_spread + slip)
        pnl = units * pos * (px - mark) - comm * units * px
        day_ret[day_idx[i]] += pnl
        trade_pnl += pnl
        t_ent[nt] = entry_i; t_ext[nt] = i; t_side[nt] = pos; t_ret[nt] = trade_pnl; t_rsn[nt] = 9
        t_risk[nt] = risk0
        nt += 1
    return day_ret, day_expo, t_ent[:nt], t_ext[:nt], t_side[:nt], t_ret[:nt], t_rsn[:nt], t_risk[:nt]


def run(A: dict, sig: dict, *, stop_k=0.0, tp_k=0.0, trail_k=0.0, time_bars=0, eod_exit=False, state_mode=False,
        direction=0, costs: dict | None = None):
    """Обёртка: A — массивы свечей (+ day_idx, n_days, atr), sig — сигналы семейства."""
    costs = costs or {"comm": 0.0004, "half_spread": 0.0, "slip": 0.0, "stop_extra": 0.0}
    n = len(A["o"])
    z8 = np.zeros(n, np.bool_)
    zf = np.zeros(n)
    return simulate(A["o"], A["h"], A["l"], A["c"], A["day_idx"], A["n_days"], A["last_of_day"], A["last_of_series"],
                    sig.get("long", z8), sig.get("short", z8), sig.get("exit_long", z8), sig.get("exit_short", z8),
                    int(sig.get("mode", 0)), sig.get("lvl_long", zf), sig.get("lvl_short", zf),
                    sig.get("atr", A["atr"]),
                    float(stop_k), float(tp_k), float(trail_k), int(time_bars), bool(eod_exit), bool(state_mode),
                    int(direction), float(costs["comm"]), float(costs["half_spread"]), float(costs["slip"]),
                    float(costs["stop_extra"]))
