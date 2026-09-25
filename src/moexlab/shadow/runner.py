"""SHADOW: реальный поток котировок, те же стратегия и движок, виртуальные исполнения. Заявки НЕ отправляются.

Цикл (для дневных стратегий — раз в день после закрытия основной сессии; для внутридневных — на закрытии свечи):
  1. adapter.history(code) — завершённые свечи активной серии (как в бэктесте);
  2. Engine(close_at_end=False).run() — тот же код решений, что и в историческом тесте;
  3. pending_orders на следующую свечу записываются в журнал вместе с bid/ask/spread/latency из adapter.quote();
  4. на следующем цикле для каждой виртуальной заявки фиксируется наблюдаемое исполнение
     (open следующей свечи / пересечение уровня) и расхождение с моделью исполнения.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..backtest.engine import Engine, InstrumentInput, RiskConfig
from ..execution.models import ExecutionModel
from ..instruments.spec import InstrumentSpec
from ..live.modes import Mode, assert_mode_allowed


class MarketDataAdapter(Protocol):
    def history(self, code: str) -> pd.DataFrame: ...
    def quote(self, code: str) -> dict: ...


@dataclass
class ShadowRecord:
    ts_local: str
    code: str
    bar_ts: str
    kind: str
    direction: int
    level: float | None
    tag: str
    bid: float | None
    ask: float | None
    last: float | None
    spread: float | None
    quote_latency_ms: float
    model_fill_price: float | None
    open_position: dict | None


class ReplayAdapter:
    """Воспроизведение истории как «живого» потока (для тестов и сухого прогона): отдаёт данные до курсора."""

    def __init__(self, streams: dict[str, pd.DataFrame], spread_ticks: dict[str, float] | None = None,
                 ticks: dict[str, float] | None = None):
        self.streams, self.cursor = streams, {k: 0 for k in streams}
        self.spread_ticks, self.ticks = spread_ticks or {}, ticks or {}

    def advance(self, code: str, n: int) -> None:
        self.cursor[code] = n

    def history(self, code: str) -> pd.DataFrame:
        return self.streams[code].iloc[: self.cursor[code]]

    def quote(self, code: str) -> dict:
        h = self.history(code)
        last = float(h["close"].iloc[-1])
        half = 0.5 * self.spread_ticks.get(code, 1.0) * self.ticks.get(code, 0.0)
        return dict(bid=last - half, ask=last + half, last=last)


class ShadowRunner:
    def __init__(self, adapter: MarketDataAdapter, specs: dict[str, InstrumentSpec], make_strategy,
                 execution: ExecutionModel, log_path: str | Path, bar_seconds: float = 86400.0,
                 risk: RiskConfig | None = None):
        assert_mode_allowed(Mode.SHADOW)
        self.adapter, self.specs, self.make_strategy = adapter, specs, make_strategy
        self.execution, self.log_path, self.bar_seconds = execution, Path(log_path), bar_seconds
        self.risk = risk or RiskConfig.research()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def cycle(self) -> list[ShadowRecord]:
        out = []
        for code, spec in self.specs.items():
            hist = self.adapter.history(code)
            if len(hist) < 3:
                continue
            eng = Engine([InstrumentInput(code, spec, hist, self.make_strategy(), self.bar_seconds)],
                         execution=self.execution, risk=self.risk, close_at_end=False)
            res = eng.run()
            t0 = time.perf_counter()
            q = self.adapter.quote(code)
            lat = (time.perf_counter() - t0) * 1000
            pos = res.meta["open_positions"].get(code)
            for od in res.meta["pending_orders"].get(code, []):
                ref = od.price if od.price is not None else (q["ask"] if od.direction > 0 else q["bid"])
                f = self.execution.fill(side=od.direction, ref_price=ref, qty=1, spec=spec, sigma_per_bar=0.0,
                                        bar_seconds=self.bar_seconds, is_stop=(od.kind == "stop"))
                out.append(ShadowRecord(
                    ts_local=pd.Timestamp.now(tz="UTC").isoformat(), code=code, bar_ts=str(hist.index[-1]),
                    kind=od.kind, direction=od.direction, level=od.price, tag=od.tag, bid=q.get("bid"),
                    ask=q.get("ask"), last=q.get("last"),
                    spread=(q["ask"] - q["bid"]) if q.get("ask") is not None and q.get("bid") is not None else None,
                    quote_latency_ms=lat, model_fill_price=f.price, open_position=asdict(pos) if pos else None))
        with open(self.log_path, "a", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(asdict(r), ensure_ascii=False, default=str) + "\n")
        return out
