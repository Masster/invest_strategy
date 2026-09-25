"""Заранее зафиксированная сетка гипотез для дневного горизонта (фиксируется ДО просмотра результатов).

Правило: сетка меняется только новой версией файла с записью в research/experiments.csv.
Оси сетки используются для поиска плато (соседи по одному шагу).
"""
from __future__ import annotations

from .runner import expand_grid

GRID_VERSION = "daily-v1"

# стандартный выход: ATR-стоп + ATR-трейлинг, без цели; позиция может переноситься через ночь
SWING = dict(initial="atr", initial_k=2.0, trailing="atr", trailing_k=3.0, intraday=False)
DAYTRADE = dict(initial="atr", initial_k=1.0, trailing="none", intraday=True)

AXES = {
    "TF_DONCHIAN": {"n": [10, 20, 40, 60, 100], "trailing_k": [2.0, 3.0, 4.0]},
    "TF_EMA_CROSS": {"fast": [5, 10, 20], "slow": [50, 100, 200]},
    "TF_SUPERTREND": {"mult": [2.0, 3.0, 4.0], "adx_min": [0, 20]},
    "MOM_TSMOM": {"lookback": [20, 60, 120, 250, "20+60+120"]},
    "MR_ZSCORE": {"n": [5, 10, 20], "k": [1.5, 2.0, 2.5]},
    "MR_RSI": {"lo": [5, 10, 20], "trend_n": [0, 200]},
    "VOL_SQUEEZE": {"pct": [0.1, 0.2, 0.3], "n": [10, 20]},
    "VOL_RANGE_EXPANSION": {"k": [0.5, 0.75, 1.0], "nr": [0, 7]},
    "VOLU_BREAKOUT": {"rv": [1.5, 2.0, 3.0], "n": [10, 20]},
    "BASELINE_01_RBREAKER_DAILY": {"initial_k": [0.5, 0.75, 1.0]},
    "CONTROL_RANDOM": {"p": [0.05, 0.1], "seed": [1, 2, 3]},
}


def daily_grid():
    g = []
    g += expand_grid("TF_DONCHIAN", {"n": AXES["TF_DONCHIAN"]["n"]}, exits_axes={"trailing_k": AXES["TF_DONCHIAN"]["trailing_k"]},
                     exits_fixed={k: v for k, v in SWING.items() if k != "trailing_k"})
    g += expand_grid("TF_EMA_CROSS", AXES["TF_EMA_CROSS"], exits_fixed=dict(initial="atr", initial_k=3.0, trailing="none",
                                                                            intraday=False))
    g += expand_grid("TF_SUPERTREND", AXES["TF_SUPERTREND"], fixed={"n": 10},
                     exits_fixed=dict(initial="atr", initial_k=3.0, trailing="none", intraday=False))
    g += expand_grid("MOM_TSMOM", AXES["MOM_TSMOM"], exits_fixed=dict(initial="atr", initial_k=4.0, trailing="none",
                                                                      intraday=False))
    g += expand_grid("MR_ZSCORE", AXES["MR_ZSCORE"], exits_fixed=dict(initial="atr", initial_k=2.0, trailing="none",
                                                                      max_bars=5, intraday=False))
    g += expand_grid("MR_RSI", AXES["MR_RSI"], fixed={"n": 2},
                     exits_fixed=dict(initial="atr", initial_k=2.0, trailing="none", max_bars=5, intraday=False))
    g += expand_grid("VOL_SQUEEZE", AXES["VOL_SQUEEZE"], fixed={"lookback": 120}, exits_fixed=SWING)
    g += expand_grid("VOL_RANGE_EXPANSION", AXES["VOL_RANGE_EXPANSION"], exits_fixed=DAYTRADE)
    g += expand_grid("VOLU_BREAKOUT", AXES["VOLU_BREAKOUT"], exits_fixed=SWING)
    g += expand_grid("BASELINE_01_RBREAKER_DAILY", {}, exits_axes={"initial_k": AXES["BASELINE_01_RBREAKER_DAILY"]["initial_k"]},
                     exits_fixed={k: v for k, v in DAYTRADE.items() if k != "initial_k"})
    g += expand_grid("CONTROL_RANDOM", AXES["CONTROL_RANDOM"], exits_fixed=dict(initial="atr", initial_k=2.0, trailing="none",
                                                                                max_bars=5, intraday=False))
    return g


GRID_AXES = {fam: ax for fam, ax in AXES.items()}
