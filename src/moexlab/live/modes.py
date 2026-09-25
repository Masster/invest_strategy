"""Режимы запуска (задача §33–34). LIVE по умолчанию запрещён.

Реальная заявка возможна только если ОДНОВРЕМЕННО:
  * режим LIVE выбран явно;
  * в конфигурации live.enabled: true;
  * переменная окружения LIVE_TRADING=true;
  * передан подтверждающий ключ, совпадающий с live.confirmation_phrase.
На текущем этапе адаптер реальных заявок не реализован вовсе: LiveBroker.post_order поднимает исключение.
"""
from __future__ import annotations

import os
from enum import Enum


class Mode(str, Enum):
    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    SHADOW = "SHADOW"    # реальный поток, виртуальные исполнения, заявки не отправляются
    PAPER = "PAPER"      # песочница T-Invest
    LIVE = "LIVE"        # реальные деньги — ЗАПРЕЩЕНО на текущем этапе


class LiveTradingDisabled(RuntimeError):
    pass


def assert_mode_allowed(mode: Mode, config: dict | None = None, confirmation: str | None = None) -> None:
    if mode is not Mode.LIVE:
        return
    cfg = (config or {}).get("live", {})
    env_ok = os.environ.get("LIVE_TRADING", "false").lower() == "true"
    if not (cfg.get("enabled") is True and env_ok and confirmation and confirmation == cfg.get("confirmation_phrase")):
        raise LiveTradingDisabled("LIVE trading is disabled (LIVE_TRADING=false by default; owner decision required)")
    raise LiveTradingDisabled("LIVE broker adapter is intentionally not implemented at this stage")


class LiveBroker:
    def post_order(self, *a, **kw):  # pragma: no cover - намеренно недоступно
        raise LiveTradingDisabled("Real orders are forbidden at this stage")
