"""Каузальные индикаторы.

Правило: значение в строке t использует только свечи с индексом <= t (свеча t уже закрыта,
решение принимается на её закрытии и исполняется не раньше следующей свечи).
Никаких центрированных окон и отрицательных сдвигов. Проверяется тестами причинности.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int = 14, method: str = "SMA") -> pd.Series:
    tr = true_range(df)
    return sma(tr, n) if method.upper() == "SMA" else wilder(tr, n)


def donchian(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    """Канал по n ПРЕДЫДУЩИМ свечам (без текущей): пробой = цена выходит за экстремум прошлого."""
    return df["high"].rolling(n, min_periods=n).max().shift(1), df["low"].rolling(n, min_periods=n).min().shift(1)


def rolling_high_low_incl(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    """Экстремумы n последних свечей ВКЛЮЧАЯ текущую закрытую (для структурных стопов)."""
    return df["high"].rolling(n, min_periods=n).max(), df["low"].rolling(n, min_periods=n).min()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = sma(close, n)
    s = close.rolling(n, min_periods=n).std(ddof=0)
    return m, m + k * s, m - k * s


def zscore(close: pd.Series, n: int) -> pd.Series:
    m = sma(close, n)
    s = close.rolling(n, min_periods=n).std(ddof=0)
    return (close - m) / s.replace(0, np.nan)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr_s = wilder(true_range(df), n)
    pdi = 100 * wilder(plus_dm, n) / tr_s
    mdi = 100 * wilder(minus_dm, n) / tr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n)


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """Направление SuperTrend (+1/-1) на закрытии свечи; итеративный, но строго каузальный."""
    a = atr(df, n, "WILDER").to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    c = df["close"].to_numpy()
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    dirn = np.zeros(len(c))
    for i in range(len(c)):
        if not np.isfinite(a[i]):
            dirn[i] = np.nan
            continue
        if i == 0 or not np.isfinite(a[i - 1]):
            dirn[i] = 1.0
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        prev = dirn[i - 1] if np.isfinite(dirn[i - 1]) else 1.0
        if prev > 0:
            dirn[i] = -1.0 if c[i] < flb[i] else 1.0
        else:
            dirn[i] = 1.0 if c[i] > fub[i] else -1.0
    return pd.Series(dirn, index=df.index)


def session_vwap(df: pd.DataFrame, day_col: str = "trading_day") -> pd.Series:
    """VWAP с начала торгового дня по закрытым свечам (типичная цена × объём)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).groupby(df[day_col]).cumsum()
    v = df["volume"].groupby(df[day_col]).cumsum()
    return pv / v.replace(0, np.nan)


def realized_vol(close: pd.Series, n: int) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(n, min_periods=n).std(ddof=0)


def relative_volume(volume: pd.Series, n: int) -> pd.Series:
    """Объём свечи относительно медианы n предыдущих свечей."""
    med = volume.rolling(n, min_periods=n).median().shift(1)
    return volume / med.replace(0, np.nan)


def momentum(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1.0
