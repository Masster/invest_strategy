"""Адаптер T-Invest API (REST-шлюз). Только чтение данных и песочница; реальные заявки запрещены.

Документация: https://developer.tbank.ru/invest (см. research/sources.md §1–3).
Токен берётся ТОЛЬКО из переменной окружения TBANK_TOKEN; он не пишется в журнал, отчёты и исключения.
TLS: сертификат *.tbank.ru выпущен НУЦ Минцифры (Russian Trusted Root CA) — его нет в стандартных
хранилищах. Путь к бандлу с этим корнем задаётся TBANK_CA_BUNDLE (иначе — REQUESTS_CA_BUNDLE / системный).
Проверено на живом API 2026-09-25 (песочница: инструменты, GetCandles).
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

PROD = "https://invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1."
SANDBOX = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1."
HISTORY = "https://invest-public-api.tbank.ru/history-data"


class TokenMissing(RuntimeError):
    pass


def _token() -> str:
    t = os.environ.get("TBANK_TOKEN", "")
    if not t:
        raise TokenMissing("TBANK_TOKEN is not set")
    return t


def quotation(q: dict | None) -> float:
    if not q:
        return float("nan")
    return int(q.get("units", 0)) + int(q.get("nano", 0)) / 1e9


@dataclass
class TInvestClient:
    sandbox: bool = True
    timeout: float = 30.0
    min_interval_s: float = 0.12   # ≤ ~8 запросов/с (лимит MarketData 600/мин)

    def __post_init__(self):
        self._last = 0.0
        self._s = requests.Session()
        # verify передаётся в каждый запрос: session.verify перекрывается REQUESTS_CA_BUNDLE из окружения
        self._verify = os.environ.get("TBANK_CA_BUNDLE") or True

    def _post(self, service_method: str, body: dict) -> dict:
        base = SANDBOX if self.sandbox else PROD
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = None
        for attempt in range(6):
            self._last = time.monotonic()
            try:
                r = self._s.post(base + service_method, json=body, timeout=self.timeout, verify=self._verify,
                                 headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"})
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                reset = r.headers.get("x-ratelimit-reset", "")
                time.sleep(float(reset) + 1 if reset.isdigit() else 2 ** attempt)
                continue
            break
        if r is None:
            raise RuntimeError(f"T-Invest {service_method}: network error after retries")
        if r.status_code != 200:
            # тело ответа может содержать tracking id, но не токен; токен в исключение не попадает
            raise RuntimeError(f"T-Invest {service_method} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    # --- инструменты ---
    def futures(self, status: str = "INSTRUMENT_STATUS_ALL") -> pd.DataFrame:
        js = self._post("InstrumentsService/Futures", {"instrumentStatus": status})
        return pd.DataFrame(js.get("instruments", []))

    def share(self, ticker: str, class_code: str = "TQBR") -> dict:
        js = self._post("InstrumentsService/ShareBy", {"idType": "INSTRUMENT_ID_TYPE_TICKER", "classCode": class_code,
                                                       "id": ticker})
        return js["instrument"]

    def find(self, query: str) -> pd.DataFrame:
        js = self._post("InstrumentsService/FindInstrument", {"query": query})
        return pd.DataFrame(js.get("instruments", []))

    def futures_margin(self, instrument_id: str) -> dict:
        return self._post("InstrumentsService/GetFuturesMargin", {"instrumentId": instrument_id})

    # --- рыночные данные ---
    def candles(self, instrument_id: str, start: pd.Timestamp, end: pd.Timestamp,
                interval: str = "CANDLE_INTERVAL_1_MIN", limit: int | None = None) -> pd.DataFrame:
        body = {"instrumentId": instrument_id, "from": start.isoformat(), "to": end.isoformat(), "interval": interval}
        if limit:
            body["limit"] = limit
        js = self._post("MarketDataService/GetCandles", body)
        rows = [dict(ts=pd.Timestamp(c["time"]), open=quotation(c["open"]), high=quotation(c["high"]),
                     low=quotation(c["low"]), close=quotation(c["close"]), volume=float(c.get("volume", 0)),
                     complete=bool(c.get("isComplete", True))) for c in js.get("candles", [])]
        return pd.DataFrame(rows).set_index("ts") if rows else pd.DataFrame()

    def order_book(self, instrument_id: str, depth: int = 10) -> dict:
        js = self._post("MarketDataService/GetOrderBook", {"instrumentId": instrument_id, "depth": depth})
        bids = [(quotation(b["price"]), float(b["quantity"])) for b in js.get("bids", [])]
        asks = [(quotation(a["price"]), float(a["quantity"])) for a in js.get("asks", [])]
        return dict(bids=bids, asks=asks, ts=js.get("orderbookTs"), last=quotation(js.get("lastPrice")))

    def last_prices(self, instrument_ids: list[str]) -> dict[str, float]:
        js = self._post("MarketDataService/GetLastPrices", {"instrumentId": instrument_ids})
        return {p["instrumentUid"]: quotation(p["price"]) for p in js.get("lastPrices", [])}

    # --- песочница: заявки разрешены ТОЛЬКО здесь ---
    def sandbox_open_account(self) -> str:
        assert self.sandbox, "orders are allowed only in sandbox"
        return self._post("SandboxService/OpenSandboxAccount", {})["accountId"]

    def sandbox_post_order(self, account_id: str, instrument_id: str, qty: int, direction: str,
                           order_type: str = "ORDER_TYPE_MARKET", price: float | None = None,
                           order_id: str | None = None) -> dict:
        assert self.sandbox, "orders are allowed only in sandbox"
        body: dict[str, Any] = {"accountId": account_id, "instrumentId": instrument_id, "quantity": str(qty),
                                "direction": direction, "orderType": order_type,
                                "orderId": order_id or str(time.time_ns())}
        if price is not None:
            units = int(price)
            body["price"] = {"units": str(units), "nano": int(round((price - units) * 1e9))}
        return self._post("SandboxService/PostSandboxOrder", body)


def download_history_year(instrument_id: str, year: int, timeout: float = 120.0) -> pd.DataFrame:
    """Архив минутных свечей за год (ZIP с CSV без заголовка, ';', порядок: uid;time;open;close;high;low;volume)."""
    r = requests.get(HISTORY, params={"instrument_id": instrument_id, "year": year}, timeout=timeout,
                     verify=os.environ.get("TBANK_CA_BUNDLE") or True,
                     headers={"Authorization": f"Bearer {_token()}"})
    if r.status_code == 429:
        raise RuntimeError("history-data rate limit (30/min)")
    if r.status_code != 200:
        raise RuntimeError(f"history-data HTTP {r.status_code}")
    frames = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            df = pd.read_csv(z.open(name), sep=";", header=None,
                             names=["uid", "ts", "open", "close", "high", "low", "volume", "_"], usecols=range(7))
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()[["open", "high", "low", "close", "volume"]]
