"""Финальные кандидаты (заморожены до вскрытия FINAL HOLDOUT, см. research/FROZEN_CANDIDATES.md).

Все кандидаты — дневные портфели панели (решение на закрытии дня t, исполнение на открытии t+1).
Размер позиции: на уровне кандидата — таргетирование волатильности по прошлым 60 дням
(scale_t = target / σ(r_{t-59..t}), ограничение плеча), т.е. только прошлые данные.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from . import panel as PN
from . import wf as W
from .panel_lib import library

TREND = {"P_TSMOM", "P_EMA", "P_BRK"}


def avg_weights(P, fams, codes, lib=None):
    lib = lib or library(P)
    Ws = [fn() for fam, cn, p, fn in lib if fam in fams and cn in codes]
    Wm = sum(Ws) / len(Ws)
    return Wm, len(Ws)


def fixed_candidates(P):
    lib = library(P)
    out = {}
    W1, n1 = avg_weights(P, TREND, {"FUT"}, lib)
    out["C1_TREND_FUT_ENS"] = (W1, f"равновесный ансамбль {n1} трендовых конфигураций (TSMOM/EMA/пробой) на 9 фьючерсах")
    W2, n2 = avg_weights(P, TREND, {"ALL"}, lib)
    out["C2_TREND_ALL_ENS"] = (W2, f"ансамбль {n2} трендовых конфигураций на всех 21 инструменте (шорт акций с платой за перенос)")
    W3, n3 = avg_weights(P, {"P_XS_MOM"}, {"EQ"}, lib)
    out["C3_XSMOM_EQ_ENS"] = (W3, f"ансамбль {n3} конфигураций кросс-секционного моментума акций (рыночно-нейтральный)")
    W4, n4 = avg_weights(P, {"P_CARRY"}, {"FX"}, lib)
    out["C4_CARRY_FX"] = (W4, "шорт Si и CR (контанго валютных фьючерсов)")
    W5, n5 = avg_weights(P, {"P_TSMOM_L", "P_EMA_L", "P_BRK_L"}, {"FUT"}, lib)
    out["C5_TREND_FUT_LONGONLY"] = (W5, f"только длинная сторона тренда на фьючерсах ({n5} конфигураций)")
    return out


def returns(P, Wt):
    r, turn, gross, Wx = PN.backtest(P, Wt)
    return r.to_numpy(), turn.to_numpy(), gross.to_numpy(), Wx


def vol_target(r, target_ann=0.10, lookback=60, max_scale=5.0, min_obs=20):
    """Масштаб по прошлой реализованной волатильности (каузально: scale для дня t+1 — из r[..t])."""
    s = pd.Series(r)
    sd = s.rolling(lookback, min_periods=min_obs).std().shift(1)
    scale = (target_ann / np.sqrt(252) / sd).clip(upper=max_scale).fillna(0.0).to_numpy()
    return r * scale, scale


def wf_candidate(meta, R, cal, proc: W.Procedure, last_month: int):
    oos, log = W.run_wf(meta, R, cal, proc, first_test_month=202601, last_test_month=last_month)
    return oos, log


FROZEN = {
    "C1_TREND_FUT_ENS": "fixed", "C2_TREND_ALL_ENS": "fixed", "C3_XSMOM_EQ_ENS": "fixed", "C4_CARRY_FX": "fixed",
    "C5_TREND_FUT_LONGONLY": "fixed",
    "C6_WF_PANEL_SHARPE_K10_SMOOTH": W.Procedure("sharpe_k10_smooth", criterion="sharpe", k=10, group_smooth=0.5),
    "C7_WF_PANEL_SORTINO_K1": W.Procedure("sortino_k1", criterion="sortino", k=1),
    "C8_WF_PANEL_FLOOR_K5": W.Procedure("floor_k5", criterion="floor", k=5),
    "C9_META_RP": "meta",
}
