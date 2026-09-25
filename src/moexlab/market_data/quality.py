"""Проверки целостности свечей и корректировка сплитов акций (данные T-Invest)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_bars(df: pd.DataFrame, name: str) -> list[str]:
    issues = []
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        issues.append(f"{name}: dates not strictly increasing")
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    bad = (h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9) | (v < 0) | (l <= 0)
    if bad.any():
        issues.append(f"{name}: {int(bad.sum())} OHLC violations e.g. {df.index[bad.to_numpy()][0].date()}")
    # подозрительные скачки закрытия (> 25% за день) — сигнал возможной ошибки переноса данных
    r = np.log(c).diff().abs()
    jumps = r > np.log(1.25)
    if jumps.any():
        issues.append(f"{name}: {int(jumps.sum())} close jumps >25% e.g. {df.index[jumps.to_numpy()][0].date()}")
    return issues


def split_adjust(df: pd.DataFrame, jump: float = 2.5) -> tuple[pd.DataFrame, list[str]]:
    """Корректировка сплитов/консолидаций по наблюдаемому разрыву open_t / close_{t-1} (> jump раз).

    Коэффициент округляется до «круглого» (10, 100, 1000, 5000...). Корректируются цены ДО события
    (стандартная практика; на R-метрики и знаки индикаторов это не влияет). Объём — обратно.
    """
    df = df.copy()
    notes = []
    ratio = (df["open"] / df["close"].shift(1)).to_numpy()
    for i in np.where((ratio > jump) | (ratio < 1 / jump))[0]:
        r = ratio[i]
        k = r if r > 1 else 1 / r
        mag = 10 ** np.floor(np.log10(k))
        k_round = round(k / mag * 2) / 2 * mag
        f = k_round if r > 1 else 1 / k_round
        df.iloc[:i, df.columns.get_indexer(["open", "high", "low", "close"])] *= f
        df.iloc[:i, df.columns.get_loc("volume")] /= f
        notes.append(f"split-adjust {df.index[i].date()}: factor {f:g} (observed {r:.4g})")
    return df, notes
