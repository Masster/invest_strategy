"""Сетки гипотез для исследования на годе минутных данных T-Invest (план v2, research/RESEARCH_PLAN.md).

Фиксируется ДО просмотра результатов; изменения — новой версией с записью в research/experiments.csv.
"""
from __future__ import annotations

from .runner import expand_grid, make_config
from ..strategies.base import ExitPolicy

GRID_VERSION = "intraday-v1"

# --- H-RB: R-Breaker 2.1 (параметры спецификации — центр сетки) ---
RB_AXES = {"setup_k": [0.25, 0.35, 0.45], "reversal_k": [0.05, 0.07, 0.10], "breakout_k": [0.15, 0.25, 0.35]}
RB_BASELINE = dict(setup_k=0.35, reversal_k=0.07, breakout_k=0.25)

# --- H-ID: внутридневные семейства на 5-минутных свечах; позиция закрывается до конца основной сессии ---
ID_BAR_MIN = 5
ID_EXIT = dict(initial="atr", initial_k=2.0, trailing="atr", trailing_k=2.0, intraday=True)
ID_MR_EXIT = dict(initial="atr", initial_k=2.0, trailing="none", max_bars=24, intraday=True)
ID_AXES = {
    "ID_ORB": {"n_min": [15, 30, 60], "initial_k": [1.0, 2.0]},
    "TF_DONCHIAN": {"n": [12, 24, 48], "trailing_k": [1.5, 2.5]},
    "TF_EMA_CROSS": {"fast": [6, 12], "slow": [36, 72]},
    "MR_VWAP": {"k": [1.5, 2.0, 3.0]},
    "MR_ZSCORE": {"n": [12, 24], "k": [2.0, 2.5]},
    "MR_RSI": {"lo": [5, 10], "trend_n": [0, 72]},
    "VOL_SQUEEZE": {"pct": [0.1, 0.2], "n": [12, 24]},
    "VOLU_BREAKOUT": {"rv": [2.0, 3.0], "n": [12, 24]},
    "CONTROL_RANDOM": {"p": [0.02], "seed": [1, 2, 3]},
}


def intraday_grid(long_only: bool = False):
    g = []
    g += expand_grid("ID_ORB", {"n_min": ID_AXES["ID_ORB"]["n_min"]}, exits_axes={"initial_k": ID_AXES["ID_ORB"]["initial_k"]},
                     exits_fixed=dict(initial="atr", trailing="atr", trailing_k=2.0, intraday=True))
    g += expand_grid("TF_DONCHIAN", {"n": ID_AXES["TF_DONCHIAN"]["n"]},
                     exits_axes={"trailing_k": ID_AXES["TF_DONCHIAN"]["trailing_k"]},
                     exits_fixed={k: v for k, v in ID_EXIT.items() if k != "trailing_k"})
    g += expand_grid("TF_EMA_CROSS", ID_AXES["TF_EMA_CROSS"], exits_fixed=dict(initial="atr", initial_k=2.0,
                                                                               trailing="none", intraday=True))
    g += expand_grid("MR_VWAP", ID_AXES["MR_VWAP"], exits_fixed=ID_MR_EXIT)
    g += expand_grid("MR_ZSCORE", ID_AXES["MR_ZSCORE"], exits_fixed=ID_MR_EXIT)
    g += expand_grid("MR_RSI", ID_AXES["MR_RSI"], fixed={"n": 2}, exits_fixed=ID_MR_EXIT)
    g += expand_grid("VOL_SQUEEZE", ID_AXES["VOL_SQUEEZE"], fixed={"lookback": 120}, exits_fixed=ID_EXIT)
    g += expand_grid("VOLU_BREAKOUT", ID_AXES["VOLU_BREAKOUT"], exits_fixed=ID_EXIT)
    g += expand_grid("CONTROL_RANDOM", ID_AXES["CONTROL_RANDOM"], exits_fixed=ID_MR_EXIT)
    return _side(g, long_only)


# --- H-DAY: дневной свинг (окна <= 40 дней) ---
SWING = dict(initial="atr", initial_k=2.0, trailing="atr", trailing_k=3.0, intraday=False)
DAY_AXES = {
    "TF_DONCHIAN": {"n": [10, 20, 40]},
    "TF_EMA_CROSS": {"fast": [5, 10], "slow": [20, 40]},
    "MOM_TSMOM": {"lookback": [10, 20, 40]},
    "MR_ZSCORE": {"n": [5, 10], "k": [1.5, 2.0]},
    "MR_RSI": {"lo": [5, 10, 20]},
    "VOL_RANGE_EXPANSION": {"k": [0.5, 0.75, 1.0]},
    "CONTROL_RANDOM": {"p": [0.1], "seed": [1, 2, 3]},
}


def day_grid(long_only: bool = False):
    g = []
    g += expand_grid("TF_DONCHIAN", DAY_AXES["TF_DONCHIAN"], exits_fixed=SWING)
    g += expand_grid("TF_EMA_CROSS", DAY_AXES["TF_EMA_CROSS"], exits_fixed=dict(initial="atr", initial_k=3.0,
                                                                                trailing="none", intraday=False))
    g += expand_grid("MOM_TSMOM", DAY_AXES["MOM_TSMOM"], exits_fixed=dict(initial="atr", initial_k=3.0,
                                                                          trailing="none", intraday=False))
    g += expand_grid("MR_ZSCORE", DAY_AXES["MR_ZSCORE"], exits_fixed=dict(initial="atr", initial_k=2.0,
                                                                          trailing="none", max_bars=5, intraday=False))
    g += expand_grid("MR_RSI", DAY_AXES["MR_RSI"], fixed={"n": 2, "trend_n": 0},
                     exits_fixed=dict(initial="atr", initial_k=2.0, trailing="none", max_bars=5, intraday=False))
    g += expand_grid("VOL_RANGE_EXPANSION", DAY_AXES["VOL_RANGE_EXPANSION"], fixed={"nr": 0},
                     exits_fixed=dict(initial="atr", initial_k=1.0, trailing="none", intraday=True))
    g += expand_grid("CONTROL_RANDOM", DAY_AXES["CONTROL_RANDOM"], exits_fixed=dict(initial="atr", initial_k=2.0,
                                                                                    trailing="none", max_bars=5,
                                                                                    intraday=False))
    return _side(g, long_only)


def _side(g, long_only):
    if not long_only:
        return g
    return [make_config(c.family, dict(dict(c.params), side="long"), ExitPolicy(**dict(c.exits))) for c in g]
