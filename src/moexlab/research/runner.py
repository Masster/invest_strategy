"""Исследовательский конвейер: прогон сетки, матрица доходностей, walk-forward, журнал экспериментов.

Защита от подглядывания в FINAL HOLDOUT: этап разработки получает данные, ОБРЕЗАННЫЕ по dev_end,
до вызова движка. Holdout прогоняется только функцией final_holdout() для заранее выбранных кандидатов.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.engine import Engine, InstrumentInput, RiskConfig
from ..execution.models import SCENARIOS, CommissionModel, ConservativeL1ExecutionModel
from ..instruments.spec import InstrumentSpec
from ..statistics.metrics import equity_metrics, trade_metrics
from ..strategies.base import ExitPolicy
from ..strategies.families import FAMILIES

LEDGER_FIELDS = ["experiment_id", "timestamp", "hypothesis", "strategy", "parameters", "data", "period", "result",
                 "conclusion", "next_step", "git_commit", "config_hash", "seed", "execution_model", "cost_model"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class StrategyConfig:
    family: str
    params: tuple                     # tuple(sorted(dict.items()))
    exits: tuple                      # tuple(sorted(ExitPolicy.__dict__.items()))

    @property
    def id(self) -> str:
        return f"{self.family}|{json.dumps(dict(self.params), sort_keys=True)}|{config_hash(dict(self.exits))}"

    def build(self):
        cls = FAMILIES[self.family]
        return cls(dict(self.params), ExitPolicy(**dict(self.exits)))

    def describe(self) -> dict:
        return dict(family=self.family, params=dict(self.params), exits=dict(self.exits))


def make_config(family: str, params: dict, exits: ExitPolicy) -> StrategyConfig:
    return StrategyConfig(family, tuple(sorted(params.items())), tuple(sorted(exits.__dict__.items())))


@dataclass
class RunSettings:
    scenario: str = "NORMAL"
    latency_ms: float = 0.0
    commission_fraction: float = 0.00025
    risk_fraction: float = 0.005
    bar_seconds: float = 86400.0


def _execution(rs: RunSettings) -> ConservativeL1ExecutionModel:
    return ConservativeL1ExecutionModel(CommissionModel(rs.commission_fraction), SCENARIOS[rs.scenario],
                                        latency_ms=rs.latency_ms)


def run_single(cfg: StrategyConfig, streams: dict[str, pd.DataFrame], specs: dict[str, InstrumentSpec],
               rs: RunSettings) -> dict:
    """Прогон одной конфигурации по всем инструментам (каждый — отдельный счёт с одинаковым риском).

    Возвращает сделки и дневную доходность портфеля (сумма дневных PnL инструментов / капитал),
    что соответствует равному риску на инструмент без сложного процента.
    """
    trades, daily = [], []
    for code, df in streams.items():
        st = cfg.build()
        eng = Engine([InstrumentInput(code, specs[code], df, st, rs.bar_seconds)], execution=_execution(rs),
                     risk=RiskConfig.research(rs.risk_fraction), initial_equity=1_000_000.0)
        res = eng.run()
        if not res.trades.empty:
            trades.append(res.trades)
        pnl = res.daily_equity.diff().fillna(res.daily_equity.iloc[0] - 1_000_000.0) / 1_000_000.0
        daily.append(pnl.rename(code))
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    D = pd.concat(daily, axis=1).fillna(0.0).sort_index() if daily else pd.DataFrame()
    return dict(trades=tr, daily_by_instrument=D, daily=D.sum(axis=1) if len(D) else pd.Series(dtype=float))


def _worker(args):
    cfg, streams, specs, rs = args
    try:
        out = run_single(cfg, streams, specs, rs)
        return cfg, out, None
    except Exception as e:  # фиксируем, но не останавливаем исследование
        return cfg, None, repr(e)


def run_grid(configs: list[StrategyConfig], streams, specs, rs: RunSettings, workers: int | None = None) -> dict:
    workers = workers or max(1, (os.cpu_count() or 2))
    results, errors = {}, {}
    args = [(c, streams, specs, rs) for c in configs]
    if workers == 1:
        it = map(_worker, args)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        it = ex.map(_worker, args, chunksize=1)
    for cfg, out, err in it:
        if err:
            errors[cfg.id] = err
        else:
            results[cfg.id] = (cfg, out)
    if workers != 1:
        ex.shutdown()
    return dict(results=results, errors=errors)


def returns_matrix(grid_results: dict, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    cols = {}
    for cid, (cfg, out) in grid_results.items():
        cols[cid] = out["daily"]
    M = pd.DataFrame(cols)
    if index is not None:
        M = M.reindex(index)
    return M.fillna(0.0).sort_index()


def summarize(cfg: StrategyConfig, out: dict, start=None, end=None) -> dict:
    tr = out["trades"]
    d = out["daily"]
    if start is not None:
        d = d[d.index >= pd.Timestamp(start, tz=d.index.tz)]
        if not tr.empty:
            tr = tr[pd.to_datetime(tr["trading_day"]) >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.index <= pd.Timestamp(end, tz=d.index.tz)]
        if not tr.empty:
            tr = tr[pd.to_datetime(tr["trading_day"]) <= pd.Timestamp(end)]
    eq = 1.0 + d.cumsum()
    m = dict(strategy_id=cfg.id, family=cfg.family, params=json.dumps(dict(cfg.params)),
             exits=json.dumps(dict(cfg.exits)))
    m.update({k: v for k, v in trade_metrics(tr).items()})
    if len(eq) > 2:
        eq0 = pd.Series([1.0], index=[eq.index[0] - pd.Timedelta(days=1)])
        em = equity_metrics(pd.concat([eq0, eq]))
    else:
        em = {}
    m.update({k: v for k, v in em.items()})
    m["daily_mean"] = float(d.mean()) if len(d) else np.nan
    return m


# ---------------------------------------------------------------------------
def neighbors(configs: list[StrategyConfig], grid_axes: dict[str, list]) -> dict[str, list[str]]:
    """Соседи в сетке: конфигурации той же семьи, отличающиеся по одному параметру на один шаг."""
    by_key = {(c.family, c.params, c.exits): c for c in configs}
    out = {}
    for c in configs:
        p = dict(c.params)
        e = dict(c.exits)
        nb = []
        for axis, values in grid_axes.get(c.family, {}).items():
            src = p if axis in p else (e if axis in e else None)
            if src is None or src[axis] not in values:
                continue
            j = values.index(src[axis])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(values):
                    q = dict(src)
                    q[axis] = values[jj]
                    key = (c.family, tuple(sorted((q if src is p else p).items())),
                           tuple(sorted((q if src is e else e).items())))
                    if key in by_key:
                        nb.append(by_key[key].id)
        out[c.id] = nb
    return out


def robust_score(M: pd.DataFrame, nb: dict[str, list[str]], min_active_days: int = 60) -> pd.Series:
    """Сглаженный по соседям годовой Sharpe (поиск плато, а не пика). Конфигурации с малым числом активных
    дней получают -inf."""
    sr = M.mean() / (M.std(ddof=1) + 1e-12) * np.sqrt(252)
    active = (M != 0).sum()
    sr[active < min_active_days] = -np.inf
    sm = {}
    for cid in M.columns:
        vals = [sr[cid]] + [sr[n] for n in nb.get(cid, []) if n in sr.index]
        vals = [v for v in vals if np.isfinite(v)]
        sm[cid] = float(np.mean(vals)) if vals and np.isfinite(sr[cid]) else -np.inf
    return pd.Series(sm)


def walk_forward(M: pd.DataFrame, families: dict[str, str], nb: dict[str, list[str]], test_years: list[int],
                 train_start: str, last_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Расширяющееся окно: обучение [train_start, Y-1], проверка год Y. Выбор — по robust_score внутри семьи.

    Возвращает (таблица выборов, OOS-доходности по семьям).
    """
    rows = []
    oos = {}
    idx = M.index
    for Y in test_years:
        tr = M[(idx >= pd.Timestamp(train_start, tz=idx.tz)) & (idx < pd.Timestamp(f"{Y}-01-01", tz=idx.tz))]
        te = M[(idx >= pd.Timestamp(f"{Y}-01-01", tz=idx.tz)) &
               (idx <= min(pd.Timestamp(f"{Y}-12-31", tz=idx.tz), pd.Timestamp(last_date, tz=idx.tz)))]
        if len(tr) < 120 or len(te) < 20:
            continue
        score = robust_score(tr, nb)
        for fam in sorted(set(families.values())):
            cols = [c for c in M.columns if families[c] == fam]
            s = score[cols]
            if not np.isfinite(s.max()):
                continue
            best = s.idxmax()
            r = te[best]
            rows.append(dict(family=fam, test_year=Y, selected=best, train_score=float(s.max()),
                             test_sharpe=float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252)),
                             test_return=float(r.sum()), test_days=len(r)))
            oos.setdefault(fam, []).append(r)
    oos_df = pd.DataFrame({k: pd.concat(v) for k, v in oos.items()}).fillna(0.0) if oos else pd.DataFrame()
    return pd.DataFrame(rows), oos_df


# ---------------------------------------------------------------------------
class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, LEDGER_FIELDS).writeheader()

    def next_id(self) -> str:
        with open(self.path, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        return f"E{n + 1:04d}"

    def add(self, **kw) -> str:
        eid = self.next_id()
        row = {k: "" for k in LEDGER_FIELDS}
        row.update(kw)
        row["experiment_id"] = eid
        row["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        row.setdefault("git_commit", git_commit())
        if not row["git_commit"]:
            row["git_commit"] = git_commit()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, LEDGER_FIELDS).writerow({k: row[k] for k in LEDGER_FIELDS})
        return eid


def expand_grid(family: str, axes: dict[str, list], fixed: dict | None = None,
                exits_axes: dict[str, list] | None = None, exits_fixed: dict | None = None) -> list[StrategyConfig]:
    fixed = fixed or {}
    exits_axes = exits_axes or {}
    exits_fixed = exits_fixed or {}
    pk, pv = list(axes), [axes[k] for k in axes]
    ek, ev = list(exits_axes), [exits_axes[k] for k in exits_axes]
    out = []
    for combo in itertools.product(*pv) if pv else [()]:
        p = dict(fixed)
        p.update(dict(zip(pk, combo)))
        for ecombo in itertools.product(*ev) if ev else [()]:
            e = dict(exits_fixed)
            e.update(dict(zip(ek, ecombo)))
            out.append(make_config(family, p, ExitPolicy(**e)))
    return out


__all__ = ["StrategyConfig", "make_config", "RunSettings", "run_single", "run_grid", "returns_matrix", "summarize",
           "neighbors", "robust_score", "walk_forward", "Ledger", "expand_grid", "git_commit", "config_hash",
           "asdict", "field", "copy"]
