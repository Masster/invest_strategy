"""Семейства стратегий для независимого поиска преимущества (задача §8).

Все признаки считаются по «сигнальным» ценам (signal_view: форвард-корректировка перекладок),
исполнение — по реальным ценам активной серии. Каждое семейство имеет 1–3 параметра;
сетки параметров задаются в research/grid.py, а не подбираются здесь.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import core as ind
from ..market_data.bars import signal_view
from .base import BarContext, EntryOrder, ExitPolicy, PositionView, Strategy


def _atr(v: pd.DataFrame, n: int = 14) -> pd.Series:
    return ind.atr(v, n, "WILDER")


class _Base(Strategy):
    """Общие удобства: сигнальные цены, сдвиг корректировки, запрет входа без ATR."""
    long_only = False

    def _adj(self, i: int) -> float:
        s = self.f.get("adj_shift")
        return 0.0 if s is None else s[i]

    def _dirs(self) -> tuple[int, ...]:
        mode = self.params.get("side", "both")
        return {"both": (1, -1), "long": (1,), "short": (-1,)}[mode]


# ---------------------------------------------------------------------------
# 8.1 Тренд
class DonchianBreakout(_Base):
    """Пробой канала Дончиана n свечей стоп-заявкой; выход — стоп/трейлинг из ExitPolicy
    или противоположный канал exit_n (потеря сигнала)."""
    name, family = "TF_DONCHIAN", "trend"

    def prepare(self, df):
        v = signal_view(df)
        n, xn = int(self.params["n"]), int(self.params.get("exit_n", 0) or 0)
        up, lo = ind.donchian(v, n)
        out = pd.DataFrame({"atr": _atr(v), "up": up, "lo": lo,
                            "adj_shift": df.get("adj_shift", pd.Series(0.0, index=df.index))}, index=df.index)
        # уровни для стоп-заявки на СЛЕДУЮЩУЮ свечу: экстремум n последних закрытых свечей (включая текущую)
        out["up_next"], out["lo_next"] = ind.rolling_high_low_incl(v, n)
        if xn:
            xh, xl = ind.rolling_high_low_incl(v, xn)
            out["x_hi"], out["x_lo"] = xh, xl
        out["close_s"] = v["close"]
        return out

    def on_bar_close(self, i, ctx):
        if not (ctx.flat and ctx.entry_ok):
            return []
        f = self.f
        a = f["atr"][i]
        if not np.isfinite(a) or not np.isfinite(f["up_next"][i]):
            return []
        sh = self._adj(i)
        orders = []
        if 1 in self._dirs():
            orders.append(EntryOrder(+1, "stop", f["up_next"][i] - sh + 0.0, "BREAKOUT"))
        if -1 in self._dirs():
            orders.append(EntryOrder(-1, "stop", f["lo_next"][i] - sh, "BREAKOUT"))
        return orders

    def exit_signal(self, i, pos):
        if "x_lo" not in self.f:
            return False
        c = self.f["close_s"][i]
        if pos.direction > 0:
            return np.isfinite(self.f["x_lo"][i]) and c <= self.f["x_lo"][i]
        return np.isfinite(self.f["x_hi"][i]) and c >= self.f["x_hi"][i]


class EMACross(_Base):
    """Пересечение EMA(fast)/EMA(slow): вход по рынку на следующей свече, выход при обратном пересечении."""
    name, family = "TF_EMA_CROSS", "trend"

    def prepare(self, df):
        v = signal_view(df)
        fa, sl = ind.ema(v["close"], int(self.params["fast"])), ind.ema(v["close"], int(self.params["slow"]))
        d = np.sign(fa - sl)
        return pd.DataFrame({"atr": _atr(v), "dir": d, "dir_prev": d.shift(1)}, index=df.index)

    def on_bar_close(self, i, ctx):
        f = self.f
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(f["dir"][i]) or not np.isfinite(f["dir_prev"][i]):
            return []
        d = int(f["dir"][i])
        fresh = self.params.get("fresh_cross_only", False)
        if d != 0 and d in self._dirs() and (not fresh or f["dir_prev"][i] != d):
            return [EntryOrder(d, "market", tag="TREND")]
        return []

    def exit_signal(self, i, pos):
        d = self.f["dir"][i]
        return np.isfinite(d) and d != pos.direction


class SuperTrendFollow(_Base):
    """SuperTrend(n, mult) + необязательный фильтр ADX >= adx_min."""
    name, family = "TF_SUPERTREND", "trend"

    def prepare(self, df):
        v = signal_view(df)
        st = ind.supertrend(v, int(self.params.get("n", 10)), float(self.params["mult"]))
        adx = ind.adx(v, 14)
        return pd.DataFrame({"atr": _atr(v), "dir": st, "adx": adx}, index=df.index)

    def on_bar_close(self, i, ctx):
        f = self.f
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(f["dir"][i]):
            return []
        amin = float(self.params.get("adx_min", 0))
        if amin and not (np.isfinite(f["adx"][i]) and f["adx"][i] >= amin):
            return []
        d = int(f["dir"][i])
        return [EntryOrder(d, "market", tag="TREND")] if d in self._dirs() else []

    def exit_signal(self, i, pos):
        d = self.f["dir"][i]
        return np.isfinite(d) and d != pos.direction


# ---------------------------------------------------------------------------
# 8.3 Моментум
class TimeSeriesMomentum(_Base):
    """Знак доходности за lookback свечей (опционально: согласие нескольких горизонтов)."""
    name, family = "MOM_TSMOM", "momentum"

    def prepare(self, df):
        v = signal_view(df)
        looks = [int(x) for x in str(self.params["lookback"]).split("+")]
        sig = sum(np.sign(v["close"] - v["close"].shift(L)) for L in looks) / len(looks)
        thr = float(self.params.get("agree", 1.0))
        d = np.where(sig >= thr, 1.0, np.where(sig <= -thr, -1.0, 0.0))
        d = pd.Series(d, index=df.index).where(sig.notna())
        return pd.DataFrame({"atr": _atr(v), "dir": d}, index=df.index)

    def on_bar_close(self, i, ctx):
        d = self.f["dir"][i]
        if ctx.flat and ctx.entry_ok and np.isfinite(d) and int(d) in self._dirs():
            return [EntryOrder(int(d), "market", tag="MOMENTUM")]
        return []

    def exit_signal(self, i, pos):
        d = self.f["dir"][i]
        return np.isfinite(d) and d != pos.direction


# ---------------------------------------------------------------------------
# 8.2 Возврат к среднему
class ZScoreReversion(_Base):
    """|z(n)| > k: вход против отклонения по рынку; цель — возврат к средней (динамическая цель),
    выход по времени max_bars."""
    name, family = "MR_ZSCORE", "mean_reversion"

    def prepare(self, df):
        v = signal_view(df)
        n = int(self.params["n"])
        z = ind.zscore(v["close"], n)
        m = ind.sma(v["close"], n)
        return pd.DataFrame({"atr": _atr(v), "z": z, "mean": m,
                             "adj_shift": df.get("adj_shift", pd.Series(0.0, index=df.index))}, index=df.index)

    def on_bar_close(self, i, ctx):
        z = self.f["z"][i]
        k = float(self.params["k"])
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(z):
            return []
        if z <= -k and 1 in self._dirs():
            return [EntryOrder(+1, "market", tag="REVERT")]
        if z >= k and -1 in self._dirs():
            return [EntryOrder(-1, "market", tag="REVERT")]
        return []

    def dynamic_target(self, i, pos):
        m = self.f["mean"][i]
        return m - self._adj(i) if np.isfinite(m) else None


class RSIReversion(_Base):
    """RSI(n) ниже lo / выше hi => вход против движения; выход при возврате RSI к 50 или по времени."""
    name, family = "MR_RSI", "mean_reversion"

    def prepare(self, df):
        v = signal_view(df)
        r = ind.rsi(v["close"], int(self.params.get("n", 2)))
        trend = ind.sma(v["close"], int(self.params.get("trend_n", 0) or 1))
        return pd.DataFrame({"atr": _atr(v), "rsi": r, "trend": trend, "close_s": v["close"]}, index=df.index)

    def on_bar_close(self, i, ctx):
        r = self.f["rsi"][i]
        lo = float(self.params["lo"])
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(r):
            return []
        tn = int(self.params.get("trend_n", 0) or 0)
        up_trend = dn_trend = True
        if tn:
            t = self.f["trend"][i]
            if not np.isfinite(t):
                return []
            up_trend, dn_trend = self.f["close_s"][i] > t, self.f["close_s"][i] < t
        if r <= lo and up_trend and 1 in self._dirs():
            return [EntryOrder(+1, "market", tag="REVERT")]
        if r >= 100 - lo and dn_trend and -1 in self._dirs():
            return [EntryOrder(-1, "market", tag="REVERT")]
        return []

    def exit_signal(self, i, pos):
        r = self.f["rsi"][i]
        return np.isfinite(r) and ((pos.direction > 0 and r >= 50) or (pos.direction < 0 and r <= 50))


class VWAPReversion(_Base):
    """Внутридневной возврат к VWAP: (close - vwap)/ATR за порогом k => вход против отклонения, цель VWAP."""
    name, family = "MR_VWAP", "mean_reversion"

    def prepare(self, df):
        v = signal_view(df)
        vw = ind.session_vwap(v)
        a = _atr(v)
        dev = (v["close"] - vw) / a
        return pd.DataFrame({"atr": a, "dev": dev, "vwap": vw,
                             "adj_shift": df.get("adj_shift", pd.Series(0.0, index=df.index))}, index=df.index)

    def on_bar_close(self, i, ctx):
        d = self.f["dev"][i]
        k = float(self.params["k"])
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(d):
            return []
        if d <= -k and 1 in self._dirs():
            return [EntryOrder(+1, "market", tag="REVERT")]
        if d >= k and -1 in self._dirs():
            return [EntryOrder(-1, "market", tag="REVERT")]
        return []

    def dynamic_target(self, i, pos):
        w = self.f["vwap"][i]
        return w - self._adj(i) if np.isfinite(w) else None


# ---------------------------------------------------------------------------
# 8.4 Волатильность
class SqueezeBreakout(_Base):
    """Сжатие: ширина полос Боллинджера в нижнем перцентиле pct за lookback => пробой канала n."""
    name, family = "VOL_SQUEEZE", "volatility"

    def prepare(self, df):
        v = signal_view(df)
        n = int(self.params.get("n", 20))
        m, up, lo = ind.bollinger(v["close"], n, 2.0)
        width = (up - lo) / m
        lb = int(self.params.get("lookback", 120))
        pct = width.rolling(lb, min_periods=lb).rank(pct=True)
        hi_n, lo_n = ind.rolling_high_low_incl(v, n)
        return pd.DataFrame({"atr": _atr(v), "pct": pct, "hi_n": hi_n, "lo_n": lo_n,
                             "adj_shift": df.get("adj_shift", pd.Series(0.0, index=df.index))}, index=df.index)

    def on_bar_close(self, i, ctx):
        f = self.f
        p = f["pct"][i]
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(p) or p > float(self.params["pct"]):
            return []
        sh = self._adj(i)
        out = []
        if 1 in self._dirs():
            out.append(EntryOrder(+1, "stop", f["hi_n"][i] - sh, "BREAKOUT"))
        if -1 in self._dirs():
            out.append(EntryOrder(-1, "stop", f["lo_n"][i] - sh, "BREAKOUT"))
        return out


class RangeExpansion(_Base):
    """Пробой от открытия: стоп-заявки на open ± k × ATR (вход в день расширения диапазона).
    На дневных свечах уровень от открытия неизвестен до открытия => используем close ± k × ATR
    предыдущей свечи (каузально)."""
    name, family = "VOL_RANGE_EXPANSION", "volatility"

    def prepare(self, df):
        v = signal_view(df)
        a = _atr(v)
        nr = int(self.params.get("nr", 0) or 0)
        rng = v["high"] - v["low"]
        narrow = (rng <= rng.rolling(nr, min_periods=nr).min()) if nr else pd.Series(True, index=df.index)
        return pd.DataFrame({"atr": a, "c": v["close"], "narrow": narrow.astype(float),
                             "adj_shift": df.get("adj_shift", pd.Series(0.0, index=df.index))}, index=df.index)

    def on_bar_close(self, i, ctx):
        f = self.f
        a = f["atr"][i]
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(a) or f["narrow"][i] != 1.0:
            return []
        k = float(self.params["k"])
        c = f["c"][i] - self._adj(i)
        out = []
        if 1 in self._dirs():
            out.append(EntryOrder(+1, "stop", c + k * a, "BREAKOUT"))
        if -1 in self._dirs():
            out.append(EntryOrder(-1, "stop", c - k * a, "BREAKOUT"))
        return out


# ---------------------------------------------------------------------------
# 8.5 Объём
class VolumeBreakout(_Base):
    """Пробой канала n, подтверждённый относительным объёмом закрытой свечи >= rv."""
    name, family = "VOLU_BREAKOUT", "volume"

    def prepare(self, df):
        v = signal_view(df)
        n = int(self.params.get("n", 20))
        up, lo = ind.donchian(v, n)
        rv = ind.relative_volume(df["volume"], int(self.params.get("rv_n", 20)))
        return pd.DataFrame({"atr": _atr(v), "up": up, "lo": lo, "rv": rv, "c": v["close"]}, index=df.index)

    def on_bar_close(self, i, ctx):
        f = self.f
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(f["up"][i]) or not np.isfinite(f["rv"][i]):
            return []
        if f["rv"][i] < float(self.params["rv"]):
            return []
        if f["c"][i] > f["up"][i] and 1 in self._dirs():
            return [EntryOrder(+1, "market", tag="BREAKOUT")]
        if f["c"][i] < f["lo"][i] and -1 in self._dirs():
            return [EntryOrder(-1, "market", tag="BREAKOUT")]
        return []


# ---------------------------------------------------------------------------
class RBreakerDailyApprox(_Base):
    """BASELINE_01_RBREAKER_DAILY: только пробойная часть R-Breaker на дневных свечах.

    Уровни Bbreak/Sbreak из H/L/C предыдущего дня (формулы §10), стоп-заявки внутри дня,
    выход не позже закрытия дня. Разворотная часть на дневных свечах НЕ моделируется
    (нужен внутридневной порядок цен). ATR-стоп считается от дневного ATR × stop_k — это НЕ
    параметры спецификации (там ATR 10-минутных свечей), поэтому результат — только ориентир.
    """
    name, family = "BASELINE_01_RBREAKER_DAILY", "rbreaker"

    def prepare(self, df):
        v = signal_view(df)
        H, L, C = v["high"], v["low"], v["close"]
        ss = H + 0.35 * (C - L)
        bs = L - 0.35 * (H - C)
        w = ss - bs
        return pd.DataFrame({"atr": _atr(v), "bb": ss + 0.25 * w, "sb": bs - 0.25 * w,
                             "adj_shift": df.get("adj_shift", pd.Series(0.0, index=df.index))}, index=df.index)

    def on_bar_close(self, i, ctx):
        f = self.f
        if not (ctx.flat and ctx.entry_ok) or not np.isfinite(f["atr"][i]):
            return []
        sh = self._adj(i)
        out = []
        if 1 in self._dirs():
            out.append(EntryOrder(+1, "stop", f["bb"][i] - sh, "BREAKOUT"))
        if -1 in self._dirs():
            out.append(EntryOrder(-1, "stop", f["sb"][i] - sh, "BREAKOUT"))
        return out


class RandomEntry(_Base):
    """Контроль: случайные входы с вероятностью p. Ожидание после издержек обязано быть < 0."""
    name, family = "CONTROL_RANDOM", "control"

    def prepare(self, df):
        return pd.DataFrame({"atr": _atr(signal_view(df))}, index=df.index)

    def reset(self):
        self.rng = np.random.default_rng(int(self.params.get("seed", 0)))

    def on_bar_close(self, i, ctx):
        if ctx.flat and ctx.entry_ok and self.rng.random() < float(self.params.get("p", 0.1)):
            return [EntryOrder(1 if self.rng.random() < 0.5 else -1, "market", tag="RANDOM")]
        return []


# ---------------------------------------------------------------------------
# 8.6 Внутридневные
class OpeningRangeBreakout(_Base):
    """Пробой диапазона открытия: high/low первых n минут основной сессии (10:00 МСК) текущего торгового дня;
    после завершения диапазона — стоп-заявки на его границах, не более одного входа в день на направление.
    Стоп — из ExitPolicy (ATR) либо противоположная граница диапазона (initial="structural")."""
    name, family = "ID_ORB", "intraday_breakout"

    def prepare(self, df):
        v = signal_view(df)
        n = int(self.params["n_min"])
        msk = df.index.tz_convert("Europe/Moscow")
        mins = msk.hour * 60 + msk.minute
        step = (df.index[1] - df.index[0]).seconds // 60 if len(df) > 1 else 1
        step = max(1, min(step, 60))
        in_rng = (mins >= 600) & (mins + step <= 600 + n)
        day = df["trading_day"].astype(str).to_numpy()
        hi = pd.Series(np.where(in_rng, v["high"], np.nan), index=df.index).groupby(day).cummax()
        lo = pd.Series(np.where(in_rng, v["low"], np.nan), index=df.index).groupby(day).cummin()
        done = pd.Series(mins + step >= 600 + n, index=df.index)
        hi = hi.groupby(day).ffill()
        lo = lo.groupby(day).ffill()
        return pd.DataFrame({"atr": _atr(v), "rng_hi": hi.where(done), "rng_lo": lo.where(done),
                             "adj_shift": df.get("adj_shift", pd.Series(0.0, index=df.index))}, index=df.index)

    def reset(self):
        self._day, self._done = None, set()

    def on_bar_close(self, i, ctx):
        if ctx.trading_day != self._day:
            self._day, self._done = ctx.trading_day, set()
        if ctx.entered_dir:
            self._done.add(ctx.entered_dir)
        if not (ctx.flat and ctx.entry_ok):
            return []
        f = self.f
        h, l_ = f["rng_hi"][i], f["rng_lo"][i]
        if not (np.isfinite(h) and np.isfinite(l_) and np.isfinite(f["atr"][i])):
            return []
        sh = self._adj(i)
        c = self._cl[i]
        out = []
        if 1 in self._dirs() and 1 not in self._done and c < h - sh:
            out.append(EntryOrder(+1, "stop", h - sh, "ORB", structural_stop=l_ - sh))
        if -1 in self._dirs() and -1 not in self._done and c > l_ - sh:
            out.append(EntryOrder(-1, "stop", l_ - sh, "ORB", structural_stop=h - sh))
        return out


FAMILIES = {c.name: c for c in [DonchianBreakout, EMACross, SuperTrendFollow, TimeSeriesMomentum, ZScoreReversion,
                                RSIReversion, VWAPReversion, SqueezeBreakout, RangeExpansion, VolumeBreakout,
                                RBreakerDailyApprox, RandomEntry, OpeningRangeBreakout]}


def _register_rbreaker():
    from .rbreaker import RBreakerConfigurable
    FAMILIES[RBreakerConfigurable.name] = RBreakerConfigurable


_register_rbreaker()
