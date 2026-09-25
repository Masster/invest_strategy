# moexlab — исследование торговых стратегий для Московской биржи (исполнение через Т-Инвестиции)

Цель — найти воспроизводимое преимущество после издержек или честно показать, что его нет.
**Реальные заявки запрещены** (`LIVE_TRADING=false`; адаптер реальных заявок намеренно не реализован).

## Быстрый старт
```bash
pip install -r requirements.txt
pytest                                   # модульные, интеграционные тесты и тесты причинности
python3 scripts/validate_pipeline.py     # проверка статистики на синтетике (NULL / PLANTED)
python3 scripts/run_daily_research.py    # дневное исследование на реальных сериях (data/raw/iss_daily)
python3 scripts/run_intraday_research.py --synthetic   # сухой прогон R-Breaker 2.1
```

## Главное
* Итоговый отчёт: `reports/FINAL_RESEARCH_REPORT.md`
* План и заранее зафиксированные правила отбора: `research/RESEARCH_PLAN.md`
* Журнал всех экспериментов (включая неудачные): `research/experiments.csv`
* Источники фактов (тарифы, API, биржа): `research/sources.md`
* Допущения: `docs/ASSUMPTIONS.md`; исходная спецификация: `docs/rbreaker_moex_v2_1_spec.md`

## Архитектура
```
src/moexlab/
  market_data/   календарь FORTS, агрегация, поток активной серии, загрузчик ISS, синтетика (только тесты)
  contracts/     коды серий, выбор активной серии (перекладка)
  instruments/   спецификация инструмента (без зашитых рыночных фактов)
  indicators/    каузальные индикаторы
  strategies/    единый интерфейс; R-Breaker 2.1; семейства (тренд, моментум, возврат, волатильность, объём)
  backtest/      событийный движок по свечам (худший внутрисвечный путь)
  execution/     модели исполнения/издержек: NORMAL / STRESS_1 / STRESS_2, задержка
  risk/          размер позиции, анализ цели доходности (Келли)
  statistics/    метрики, бутстреп, FDR, DSR, PBO, White RC, Hansen SPA, Монте-Карло
  research/      сетка гипотез, walk-forward, журнал экспериментов, тест причинности
  reporting/     графики
  broker/        T-Invest REST (чтение + песочница)
  shadow/        SHADOW: тот же движок на живом потоке, заявки не отправляются
  live/          режимы RESEARCH / BACKTEST / SHADOW / PAPER / LIVE (LIVE запрещён)
```
Исторический тест, SHADOW и будущая торговля используют один и тот же код стратегии и движка;
различаются только адаптеры данных/исполнения (`tests/integration/test_shadow_and_modes.py`).

## Секреты
Токен T-Invest — только переменная окружения `TBANK_TOKEN`. Он не пишется в код, git, журналы и отчёты.
