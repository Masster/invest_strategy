"""BASELINE_01_RBREAKER — R-Breaker MOEX 2.1 (docs/rbreaker_moex_v2_1_spec.md) на свечах.

Отличия свечной реализации от потоковой спецификации (уровень качества A, §67) — все консервативны:
  * флаги уровней (setup_seen, armed) обновляются по закрытой свече и действуют со следующей свечи;
    пересечение Senter/Benter/Bbreak/Sbreak исполняется стоп-заявкой на следующей свече;
    «касание и возврат» внутри одной свечи не торгуется (порядок цен неизвестен, §68);
  * если на одной свече срабатывают заявки разных направлений — вход пропускается (AMBIGUOUS_SKIP);
  * фильтр спреда (§46) без исторического стакана не применим => помечается SPREAD_FILTER_UNVERIFIED;
  * ATR14 считается по завершённым 10-минутным свечам своей серии (§12).
Неоднозначность спецификации (зафиксирована в research/hypotheses/H001_rbreaker.md):
  §20 повторно выставляет upper_setup_seen при любой цене >= Ssetup, поэтому инвалидация §22
  действует только до следующей цены выше Ssetup. Реализовано буквально.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.core import true_range
from .base import BarContext, EntryOrder, ExitPolicy, Strategy


def prev_day_hlc(bars: pd.DataFrame) -> pd.DataFrame:
    """H/L/C предыдущего торгового дня той же серии, привязанные к каждой свече текущего дня."""
    key = ["secid", "trading_day"] if "secid" in bars else ["trading_day"]
    g = bars.groupby(key, sort=False)
    daily = pd.DataFrame({"H": g["high"].max(), "L": g["low"].min(), "C": g["close"].last(),
                          "V": g["volume"].sum()})
    if "secid" in bars:
        daily = daily.sort_index()
        prev = daily.groupby(level=0).shift(1)
        n_days = daily.groupby(level=0).cumcount()  # число завершённых дней до текущего
        med_v = daily.groupby(level=0)["V"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).median())
    else:
        daily = daily.sort_index()
        prev = daily.shift(1)
        n_days = pd.Series(np.arange(len(daily)), index=daily.index)
        med_v = daily["V"].shift(1).rolling(20, min_periods=20).median()
    out = prev.rename(columns=lambda c: f"prev_{c}")
    out["n_prev_days"] = n_days
    out["median_volume_20"] = med_v
    idx = pd.MultiIndex.from_frame(bars[key]) if len(key) > 1 else bars["trading_day"]
    res = out.reindex(idx)
    res.index = bars.index
    return res


def atr_completed_buckets(bars: pd.DataFrame, bucket_min: int = 10, n: int = 14) -> tuple[pd.Series, pd.Series]:
    """ATR SMA(n) по завершённым bucket_min-минутным свечам той же серии.

    Для минутной свечи с началом t значение берётся по последнему ведру, конец которого <= t + 1 мин
    (т.е. к моменту закрытия текущей свечи ведро уже завершено).
    """
    step = bars.index[1] - bars.index[0] if len(bars) > 1 else pd.Timedelta(minutes=1)
    bar_len = min(step, pd.Timedelta(minutes=1)) if step > pd.Timedelta(0) else pd.Timedelta(minutes=1)
    bar_len = pd.Timedelta(minutes=1) if bar_len < pd.Timedelta(minutes=1) else bar_len
    sec = bars["secid"] if "secid" in bars else pd.Series("_", index=bars.index)
    out_atr = pd.Series(np.nan, index=bars.index)
    out_cnt = pd.Series(0.0, index=bars.index)
    for s, part in bars.groupby(sec, sort=False):
        b = part.index.floor(f"{bucket_min}min")
        agg = part.groupby(b).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"),
                                  last_ts=("close", lambda x: x.index[-1]))
        agg = agg.sort_index()
        bucket_end = agg.index + pd.Timedelta(minutes=bucket_min)
        tr = true_range(agg[["high", "low", "close"]])
        atr_v = tr.rolling(n, min_periods=n).mean()
        cnt = pd.Series(np.arange(1, len(agg) + 1), index=agg.index, dtype=float)
        # ведро считается завершённым в момент bucket_end; «последняя свеча» ведра в данных может
        # быть раньше bucket_end — тогда ведро завершено по времени, данных больше не будет.
        ends = pd.DatetimeIndex(bucket_end)
        bar_end = part.index + bar_len
        pos = ends.searchsorted(bar_end, side="right") - 1
        valid = pos >= 0
        vals = np.full(len(part), np.nan)
        cnts = np.zeros(len(part))
        vals[valid] = atr_v.to_numpy()[pos[valid]]
        cnts[valid] = cnt.to_numpy()[pos[valid]]
        out_atr.loc[part.index] = vals
        out_cnt.loc[part.index] = cnts
    return out_atr, out_cnt


class RBreaker(Strategy):
    name = "BASELINE_01_RBREAKER"
    family = "rbreaker"

    DEFAULTS = dict(setup_k=0.35, reversal_k=0.07, breakout_k=0.25, atr_bucket_min=10, atr_n=14,
                    rearm_atr=0.25, min_days=20, min_atr_bars=14, min_median_volume=10000,
                    breakout_stop_atr=1.5, breakout_trail_atr=1.5, reversal_stop_atr=1.0, reversal_trail_atr=1.0,
                    enable_breakout=True, enable_reversal=True)

    def __init__(self, params: dict | None = None):
        p = dict(self.DEFAULTS)
        p.update(params or {})
        super().__init__(p, ExitPolicy(initial="atr", initial_k=p["breakout_stop_atr"], trailing="atr",
                                       trailing_k=p["breakout_trail_atr"], intraday=True))
        self._rev_exits = ExitPolicy(initial="atr", initial_k=p["reversal_stop_atr"], trailing="atr",
                                     trailing_k=p["reversal_trail_atr"], intraday=True)

    def exits_for(self, tag: str) -> ExitPolicy:
        return self._rev_exits if tag == "REVERSAL" else self.exits

    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        if "rb_prev_H" in df:
            # признаки посчитаны по полной истории каждой серии (§7: после перекладки — история НОВОЙ серии)
            f = pd.DataFrame({"prev_H": df["rb_prev_H"], "prev_L": df["rb_prev_L"], "prev_C": df["rb_prev_C"],
                              "n_prev_days": df["rb_n_prev_days"], "median_volume_20": df["rb_median_volume_20"]},
                             index=df.index)
        else:
            f = prev_day_hlc(df)
        H, L, C = f["prev_H"], f["prev_L"], f["prev_C"]
        ssetup = H + p["setup_k"] * (C - L)
        bsetup = L - p["setup_k"] * (H - C)
        mid = 0.5 + p["reversal_k"] / 2
        senter = mid * (H + L) - p["reversal_k"] * L
        benter = mid * (H + L) - p["reversal_k"] * H
        w = ssetup - bsetup
        bbreak = ssetup + p["breakout_k"] * w
        sbreak = bsetup - p["breakout_k"] * w
        if "rb_atr" in df and p["atr_bucket_min"] == 10 and p["atr_n"] == 14:
            a, cnt = df["rb_atr"], df["rb_atr_cnt"]
        else:
            a, cnt = atr_completed_buckets(df, p["atr_bucket_min"], p["atr_n"])
        valid_hlc = (H > L) & (L <= C) & (C <= H)
        ordered = (sbreak < bsetup) & (bsetup < benter) & (benter < senter) & (senter < ssetup) & (ssetup < bbreak)
        hist_ok = (f["n_prev_days"] >= p["min_days"]) & (cnt >= p["min_atr_bars"])
        liq_ok = f["median_volume_20"] >= p["min_median_volume"]
        return pd.DataFrame({
            "atr": a, "ssetup": ssetup, "bsetup": bsetup, "senter": senter, "benter": benter,
            "bbreak": bbreak, "sbreak": sbreak,
            "day_ok": (valid_hlc & ordered & hist_ok & liq_ok).astype(float),
            "levels_ordered": ordered.astype(float), "liq_ok": liq_ok.astype(float),
        }, index=df.index)

    def reset(self) -> None:
        self._day = None
        self._init = False
        self.upper = self.lower = self.long_armed = self.short_armed = False

    def on_bar_close(self, i: int, ctx: BarContext) -> list[EntryOrder]:
        f, p = self.f, self.params
        if ctx.trading_day != self._day:
            self._day = ctx.trading_day
            self._init = False
            self.upper = self.lower = self.long_armed = self.short_armed = False
        if ctx.entered_tag is not None:
            # сигнал сформирован и исполнен на этой свече (§18–24): сбросить признаки
            if ctx.entered_tag == "BREAKOUT" and ctx.entered_dir > 0:
                self.long_armed = False
            if ctx.entered_tag == "BREAKOUT" and ctx.entered_dir < 0:
                self.short_armed = False
            self.upper = self.lower = False
            return []
        if f["day_ok"][i] != 1.0 or not ctx.entry_ok:
            return []
        hi, lo, c = self._hi[i], self._lo[i], self._cl[i]
        bb, sb, ss, bs = f["bbreak"][i], f["sbreak"][i], f["ssetup"][i], f["bsetup"][i]
        se, be, a = f["senter"][i], f["benter"][i], f["atr"][i]
        if not self._init:
            # первая допустимая цена после cooldown (§15–16): закрытие первой свечи окна
            self._init = True
            self.long_armed = c < bb
            self.short_armed = c > sb
            return []
        if not ctx.flat:
            return []
        rearm = p["rearm_atr"] * a if np.isfinite(a) else None
        if rearm is not None:
            if lo <= bb - rearm:
                self.long_armed = True
            if hi >= sb + rearm:
                self.short_armed = True
        if hi >= ss:
            self.upper = True
        if lo <= bs:
            self.lower = True
        if hi >= bb:
            self.upper = c >= ss      # §22 + буквальное §20 (см. docstring)
        if lo <= sb:
            self.lower = c <= bs      # §25 + буквальное §23
        orders: list[EntryOrder] = []
        if p["enable_breakout"]:
            if self.long_armed and c < bb:
                orders.append(EntryOrder(+1, "stop", bb, "BREAKOUT"))
            if self.short_armed and c > sb:
                orders.append(EntryOrder(-1, "stop", sb, "BREAKOUT"))
        if p["enable_reversal"]:
            if self.upper and c > se:
                orders.append(EntryOrder(-1, "stop", se, "REVERSAL"))
            if self.lower and c < be:
                orders.append(EntryOrder(+1, "stop", be, "REVERSAL"))
        return orders


class RBreakerConfigurable(RBreaker):
    """Адаптер для сетки исследований: выходы задаются спецификацией (§29–35), ExitPolicy сетки игнорируется."""
    name = "RBREAKER_2_1"

    def __init__(self, params: dict | None = None, exits: ExitPolicy | None = None):
        super().__init__(params)


def series_features(series_df: pd.DataFrame, bucket_min: int = 10, n: int = 14) -> pd.DataFrame:
    """Признаки R-Breaker, не зависящие от коэффициентов, по полной истории ОДНОЙ серии."""
    f = prev_day_hlc(series_df)
    a, cnt = atr_completed_buckets(series_df, bucket_min, n)
    return pd.DataFrame({"rb_prev_H": f["prev_H"], "rb_prev_L": f["prev_L"], "rb_prev_C": f["prev_C"],
                         "rb_n_prev_days": f["n_prev_days"], "rb_median_volume_20": f["median_volume_20"],
                         "rb_atr": a, "rb_atr_cnt": cnt}, index=series_df.index)
