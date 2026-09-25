# Источники и факты: T-Invest API, тарифы Т-Инвестиций, MOEX FORTS

Дата сбора: 2026-09-25. Формат строки: факт — источник — дата доступа — уверенность.

Уровни уверенности:
- **VERIFIED** — прочитано в первоисточнике (официальная документация T-Invest / MOEX / тарифный PDF / данные ISS).
- **SECONDARY** — вторичный источник (СМИ, пост в Пульсе, пересказ) или вывод/расчёт на основе первичных данных.
- **UNVERIFIED** — не удалось подтвердить; использовать только как гипотезу.

Методика доступа: tbank.ru, moex.com, iss.moex.com, invest-public-api.tbank.ru заблокированы для curl в контейнере; страницы получены через Firecrawl (scrape) и через клон GitHub-репозитория `RussianInvestments/investAPI` (последний коммит `3eaf23a`, 2025-11-07). **Важно:** актуальная документация теперь живёт на `developer.tbank.ru/invest/...`, и местами она **новее и отличается** от GitHub-версии (лимиты, песочница). Где есть расхождение — приоритет у developer.tbank.ru.

---

## 1. Т-Инвестиции API

### 1.1 Адреса, протоколы
- Prod-контур: `invest-public-api.tbank.ru:443` (ранее `invest-public-api.tinkoff.ru:443`); песочница: `sandbox-invest-public-api.tbank.ru:443`. gRPC + REST. — https://developer.tbank.ru/invest/intro/developer/sandbox/ ; https://github.com/RussianInvestments/investAPI/blob/main/src/docs/url_difference.md — 2026-09-25 — VERIFIED
- Актуальная документация: https://developer.tbank.ru/invest/ (Docusaurus). GitHub-зеркало src/docs отстаёт (последний коммит 2025-11-07). — https://github.com/RussianInvestments/investAPI — 2026-09-25 — VERIFIED

### 1.2 Лимиты запросов (актуальная версия developer.tbank.ru)
- Общая рекомендация: суммарно не более 50 запросов/сек по всем счетам и токенам с одного адреса; бана не будет, если с IP ≤ 1000 запросов/мин. — https://developer.tbank.ru/invest/intro/intro/limits — 2026-09-25 — VERIFIED
- Unary-лимиты в минуту (на сервис, сумма по методам сервиса): инструменты 200; методы Bonds/Shares/Options/Futures/Etfs/GetAssets — 15 (индивидуальный, введён в июне 2026); счета 100; операции 200; формирование отчётов 5; **котировки (MarketDataService, в т.ч. GetCandles) 600**; стоп-ордера 50; песочница 200; ордера 100; сигналы 100; автоследование 100; **getHistory (архив) 30**; getOrders 200; **postOrder 15/сек (900/мин)**; cancelOrder 300; replaceOrder 300; postOrderAsync 600; getStopOrders 60. — https://developer.tbank.ru/invest/intro/intro/limits ; https://developer.tbank.ru/invest/release/News_June_2026 — 2026-09-25 — VERIFIED
- Стримы: активных stream-соединений MarketData — 32; сервис ордеров — 16 на каждый тип стрима; сервис операций — 11 на каждый тип стрима. В одном стриме котировок ≤ 300 подписок суммарно (свечи + стаканы + сделки); подписка Info не ограничена; ≤ 100 запросов подписки в минуту. Счётчик стрим-соединений обновляется раз в 2 минуты. Лимита на число активных заявок нет. — https://developer.tbank.ru/invest/intro/intro/limits — 2026-09-25 — VERIFIED
- Троттлинг доставки в стриме: Orderbook и Candles — не чаще 1 сообщения на подписку за 100 мс; Trades, LastPrice, Info — без ограничений. — там же — 2026-09-25 — VERIFIED
- В старой (GitHub, 2025-11) версии были «грейды» 1–5 (16/64 стримов котировок, postOrder 300/мин, cancelOrder 100/мин); на developer.tbank.ru система грейдов больше не описана. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/limits.md — 2026-09-25 — VERIFIED (устаревшее)

### 1.3 MarketDataService: GetCandles
- Максимальный период одного запроса и максимальный `limit` (актуальная таблица CandleInterval): 5 сек — до 200 мин (limit 2500); 10 сек — до 200 мин (1250); 30 сек — до 20 ч (2500); 1 мин — до 1 дня (2400); 2 мин — до 1 дня (1200); 3 мин — до 1 дня (750); 5 мин — до недели (2400); 10 мин — до недели (1200); 15 мин — до 3 недель (2400); 30 мин — до 3 недель (1200); 1 час — до 3 месяцев (2400); 2 часа — до 3 месяцев (2400); 4 часа — до 3 месяцев (700); 1 день — до 6 лет (2400); 1 неделя — до 5 лет (300); 1 месяц — до 10 лет (120). «Максимальное значение интервала ориентировочное и может отличаться в большую сторону». — https://developer.tbank.ru/invest/services/quotes/marketdata (раздел CandleInterval) ; https://developer.tbank.ru/invest/intro/intro/load_history — 2026-09-25 — VERIFIED
- Секундные интервалы (5/10/30 сек) есть в актуальной версии, отсутствуют в GitHub-версии 2025-11. — там же — 2026-09-25 — VERIFIED
- Страница описания сервиса дополнительно пишет «максимально допустимый период получения свечей за один запрос — один календарный год» (противоречит таблице; для минуток действует таблица: 1 день). — https://developer.tbank.ru/invest/services/quotes/head-marketdata — 2026-09-25 — VERIFIED (текст), трактовка SECONDARY
- Дата начала истории различается по интервалам и инструментам; в сервисе инструментов есть поля `first_1min_candle_date` и `first_1day_candle_date`. — https://developer.tbank.ru/invest/intro/intro/load_history — 2026-09-25 — VERIFIED
- Источник свечей `candle_source_type`: CANDLE_SOURCE_EXCHANGE (биржевые), CANDLE_SOURCE_INCLUDE_WEEKEND (с учётом торгов выходного дня/дилерских), UNSPECIFIED (все). Для бэктеста по бирже использовать EXCHANGE. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/marketdata.md — 2026-09-25 — VERIFIED
- Дневные свечи по фьючерсам в API отличаются от биржевых: брокер считает день по календарному дню, биржа — от открытия предыдущей вечерней сессии (после перехода MOEX на ЕТС в 2026 разница, вероятно, уменьшилась — не проверено). — https://developer.tbank.ru/invest/services/quotes/head-marketdata — 2026-09-25 — VERIFIED (текст) / UNVERIFIED (эффект ЕТС)
- Unary GetCandles не возвращает текущую незакрытую минуту (обсуждение, есть поле `is_complete`). — https://github.com/RussianInvestments/investAPI/issues/61 — 2026-09-25 — SECONDARY
- Цены фьючерсов в API — в пунктах; стоимость = price / min_price_increment × min_price_increment_amount. Объёмы стакана/сделок — в лотах. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/faq_marketdata.md — 2026-09-25 — VERIFIED

### 1.4 GetLastTrades, стакан, стримы
- GetLastTrades: «Метод гарантирует получение информации за последний час» (обезличенные сделки; параметр trade_source: EXCHANGE/DEALER/ALL). Глубже часа — не гарантировано; для истории сделок использовать архив history-trades (см. §3). — https://developer.tbank.ru/invest/services/quotes/marketdata — 2026-09-25 — VERIFIED
- Глубина стакана (подписка и GetOrderBook): допустимые значения 1, 10, 20, 30, 40, 50 (ошибка SUBSCRIPTION_STATUS_DEPTH_IS_INVALID для прочих). — https://developer.tbank.ru/invest/services/quotes/marketdata — 2026-09-25 — VERIFIED
- MarketDataStream (bidirectional) подписки: стаканы, свечи, обезличенные сделки, торговые статусы (Info), последние цены; есть server-side stream для gRPC-web. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/head-marketdata.md — 2026-09-25 — VERIFIED
- Есть GetTechAnalysis (технические индикаторы на стороне брокера) и GetMarketValues. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/marketdata.md — 2026-09-25 — VERIFIED

### 1.5 Заявки (OrdersService)
- Типы: ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, ORDER_TYPE_BESTPRICE. Есть PostOrder, PostOrderAsync, ReplaceOrder, CancelOrder, GetOrders, GetOrderState, GetOrderPrice, GetMaxLots, стрим OrderStateStream/TradesStream. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/orders.md — 2026-09-25 — VERIFIED

### 1.6 Стоп-заявки (StopOrdersService)
- Типы: STOP_ORDER_TYPE_TAKE_PROFIT, STOP_ORDER_TYPE_STOP_LOSS, STOP_ORDER_TYPE_STOP_LIMIT. Дочерняя биржевая заявка: EXCHANGE_ORDER_TYPE_MARKET или _LIMIT. Экспирация: GoodTillCancel / GoodTillDate. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/stoporders.md — 2026-09-25 — VERIFIED
- **Трейлинг-стоп поддерживается** как подтип take-profit: `take_profit_type = TAKE_PROFIT_TYPE_TRAILING` + `trailing_data` (indent/indent_type и защитный spread/spread_type; величина абсолютная в единицах цены или относительная в %), статусы TRAILING_STOP_ACTIVE/ACTIVATED. — там же — 2026-09-25 — VERIFIED
- Доступность трейлинг-стопа именно для фьючерсов FORTS — в документации явно не оговорена. — — 2026-09-25 — UNVERIFIED (проверить эмпирически в песочнице/на малом объёме)
- Стоп-заявка хранится у брокера и после активации превращается в биржевую (появляется `exchange_order_id`); GetStopOrders возвращает только ещё не сработавшие (или по фильтру статуса). — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/faq_stoporders.md ; head-stoporders.md — 2026-09-25 — VERIFIED (текст), «хранится у брокера» — SECONDARY (вывод)

### 1.7 Инструменты, фьючерсы, ГО
- GetFuturesMargin(instrument_id) → initial_margin_on_buy, initial_margin_on_sell, min_price_increment, min_price_increment_amount. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/instruments.md — 2026-09-25 — VERIFIED
- InstrumentStatus: INSTRUMENT_STATUS_BASE (по умолчанию, торгуемые через API) и INSTRUMENT_STATUS_ALL (все). — там же — 2026-09-25 — VERIFIED
- Можно ли получить GetCandles по **экспирированным** фьючерсам через API — в документации прямо не сказано. Косвенно: в каталоге архивов минутных свечей (history-md) по фьючерсам 2874 записи, в т.ч. уже истёкшие контракты (например 92K6_SPBFUT — май 2026), а пример скрипта history-trades использует CRZ5_SPBFUT (истёк в 12.2025). — https://developer.tbank.ru/invest/services/history-md/candles/futures ; https://developer.tbank.ru/invest/services/history-md/head-history-md — 2026-09-25 — SECONDARY
- Исторические данные инструмента, существовавшего до корпоративного действия, становятся недоступны. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/faq_instruments.md — 2026-09-25 — VERIFIED

### 1.8 MCP-сервер Т-Инвестиций
- Запущен в июне 2026. Возможности: поиск бумаг, котировки/объёмы/стаканы, новости с сентиментом, баланс/портфель/ГО, доходность/риски, ребалансировка, **выставление рыночных и лимитных заявок, стоп-заявок, отмена ордеров**, анализ операций/налогов, прогнозы инвестдомов. Авторизация — токеном API (read-only / полный доступ / «переводы»). Совместим с Claude Code/Desktop, Cursor и др. Брокер не несёт ответственности за действия ИИ-агентов. Точный URL эндпойнта (`https://invest-public-api.tbank.ru/mcp`) на странице обзора не показан — он на подстраницах подключения. — https://developer.tbank.ru/invest/mcp ; https://developer.tbank.ru/invest/release/News_June_2026 — 2026-09-25 — VERIFIED (описание), UNVERIFIED (URL и список tools)

---

## 2. Песочница

- Все методы вызываются по адресу `sandbox-invest-public-api.tbank.ru:443` с теми же контрактами, что и в prod. — https://developer.tbank.ru/invest/intro/developer/sandbox/ — 2026-09-25 — VERIFIED
- Сервисы в песочнице (актуальная таблица): котировки — да; **стоп-заявки — да** (в GitHub-версии 2025-11 было «Нет» и фраза «В песочнице нет стоп-заявок»); торговые поручения — да; getBrokerReport/getDividendsForeignIssuer — пустые ответы. — https://developer.tbank.ru/invest/intro/developer/sandbox/url_difference ; https://github.com/RussianInvestments/investAPI/blob/main/src/docs/head-sandbox.md — 2026-09-25 — VERIFIED
- Рыночные данные в песочнице — реальные: песочница «получает информацию от торговых площадок о последних сделках по всем инструментам и исполняет активные заявки». — https://developer.tbank.ru/invest/intro/developer/sandbox/ — 2026-09-25 — VERIFIED
- **Модель исполнения (нереалистичная):** market-ордер исполняется по last price; заявки не влияют на рынок (1 лот и 10 000 лотов — по одной цене); лимитная заявка при наличии встречки хотя бы на 1 лот исполняется **полностью** по лучшей встречной цене; неисполненная лимитка «стоит в рынке», не участвует в стакане и исполняется, когда last price достигает её цены (по цене заявки); все неисполненные заявки снимаются после окончания торговой сессии. — там же — 2026-09-25 — VERIFIED
- Комиссия в песочнице — 0,05% от объёма сделки для любого инструмента (не равна реальной). — там же — 2026-09-25 — VERIFIED
- Маржинальная торговля и короткие позиции открыты, плечо упрощённо = 2 в обе стороны; ГО, ставки риска, ликвидность портфеля не рассчитываются; купоны/дивиденды/налоги не начисляются. (В GitHub-версии 2025-11: шорт по фьючерсам был недоступен, при покупке фьючерса списывалась полная стоимость, вариационная маржа не считалась, поставки при экспирации нет.) — https://developer.tbank.ru/invest/intro/developer/sandbox/ ; https://github.com/RussianInvestments/investAPI/blob/main/src/docs/head-sandbox.md — 2026-09-25 — VERIFIED (текущая формулировка); поведение ВМ/экспирации в текущей версии — UNVERIFIED
- Пополнение только в рублях, лимит 30 000 000 ₽; счета хранятся 3 месяца с последнего использования. — https://developer.tbank.ru/invest/intro/developer/sandbox/ — 2026-09-25 — VERIFIED
- Лимит песочницы — 200 unary-запросов/мин. — https://developer.tbank.ru/invest/intro/intro/limits — 2026-09-25 — VERIFIED

Вывод: песочница пригодна для проверки интеграции (API-вызовы, стоп-заявки, стримы), **но не для оценки проскальзывания, частичных исполнений и очереди** — исполнение оптимистично.

---

## 3. Исторические данные

### 3.1 T-Invest: архив минутных свечей (history-data)
- GET `https://invest-public-api.tbank.ru/history-data?instrument_id=<figi|uid>&year=<YYYY>` с заголовком `Authorization: Bearer <token>` → ZIP-архив минутных свечей за год. Лимит — 30 запросов/мин. — https://developer.tbank.ru/invest/services/quotes/get_history ; https://developer.tbank.ru/invest/intro/intro/limits — 2026-09-25 — VERIFIED
- Формат: в ZIP — CSV-файлы по дням, **без заголовков**, разделитель `;`: INSTRUMENT_UID; UTC начала свечи; open; close; high; low; volume (в лотах). Порядок полей: open, close, high, low (не OHLC!). — https://developer.tbank.ru/invest/services/history-md/head-history-md — 2026-09-25 — VERIFIED
- Данные обновляются раз в сутки (ночью), текущего дня в архиве нет. — https://github.com/RussianInvestments/investAPI/blob/main/src/docs/get_history.md — 2026-09-25 — VERIFIED
- Есть UI-каталог архивов (поиск по тикеру `ticker_classcode`, прямые ссылки, генерация скрипта): https://developer.tbank.ru/invest/services/history-md/candles/futures (фьючерсы, 2874 позиции на 2026-09-25) и старый https://russianinvestments.github.io/invest-history-index-ui/ ; скрипт download_md.sh + figi.txt в репозитории. — https://github.com/RussianInvestments/investAPI/tree/main/src/marketdata — 2026-09-25 — VERIFIED

### 3.2 T-Invest: архив обезличенных сделок (history-trades) — новое
- Сервис history-md отдаёт: архив **всех сделок за день по всем инструментам**; архив **всех сделок за день по инструменту**; архив минутных свечей за год. URL из примера: `https://invest-public-api.tbank.ru/history-trades/<YYYY-MM-DD>?instrumentId=<TICKER_CLASSCODE>` (например `SiZ6_SPBFUT`). Лимит — **30 файлов/мин с одного IP** (429 при превышении). — https://developer.tbank.ru/invest/services/history-md/head-history-md — 2026-09-25 — VERIFIED
- Формат сделок (CSV с заголовком): TRADE_TS (UTC), TICKER_CC, DIRECTION (BUY/SELL — агрессор), PRICE, QUANTITY (лоты), TRADE_SOURCE (EXCHANGE/DEALER), INSTRUMENT_UID. — там же — 2026-09-25 — VERIFIED
- Пример скрипта качает с 2020-01-01; «исторические данные доступны не по всем инструментам и не с начала истории»; в примере curl **без** токена — нужна ли авторизация, не проверено. — там же — 2026-09-25 — VERIFIED (текст) / UNVERIFIED (авторизация, реальная глубина по FORTS)

### 3.3 MOEX ISS
- Свечи: `https://iss.moex.com/iss/engines/futures/markets/forts/securities/<SECID>/candles.json?interval=<1|10|60|24|7|31|4>&from=YYYY-MM-DD&till=YYYY-MM-DD`. **Не более 500 свечей за ответ, курсора нет** — пагинация параметром `start=` (проверено: часовые свечи SiU6 за 01.06–16.09.2026 оборвались ровно на 500-й свече). — запрос через Firecrawl к iss.moex.com — 2026-09-25 — VERIFIED
- История по дням: `/iss/history/engines/futures/markets/forts/securities/<SECID>.json?from=&till=` (TRADEDATE, VOLUME, OPENPOSITION…), страница 100 строк, есть блок `history.cursor` (INDEX/TOTAL/PAGESIZE). — там же — 2026-09-25 — VERIFIED
- **Экспирированные контракты доступны** в ISS (история и свечи SiU6, MXU6 и др. после их экспирации 17–18.09.2026 получены). — там же — 2026-09-25 — VERIFIED
- Спецификации текущих контрактов: `/iss/engines/futures/markets/forts/securities.json` (MINSTEP, STEPPRICE, LOTVOLUME, LASTTRADEDATE, INITIALMARGIN и др.). — там же — 2026-09-25 — VERIFIED
- Анонимный ISS отдаёт текущие данные с задержкой ~15 мин (для истории несущественно). — — UNVERIFIED (не проверялось в этот раз)
- Сырые тиковые сделки и стакан через публичный ISS за прошлые дни недоступны (ISS trades — только текущий день). — — UNVERIFIED

### 3.4 MOEX ALGOPACK / Datashop
- ALGOPACK (data.moex.com): тарифы «Стартовый» (15-мин задержка свечей и сделок, FUTOI с лагом T-15 дней) и «Promo» — **610 ₽/мес** (онлайн стаканы/свечи/сделки, Super Candles, Mega Alerts, HI2, FUTOI, API, Python-библиотека `moexalgo`). Модули: Super Candles (tradestats/orderstats/obstats; 5 мин; с 2020 г.; акции, фьючерсы, валюта), FUTOI (5 мин, с 2020), HI2 (дневной, с 2020), Mega Alerts (1 мин, с 2024), Real-time market data (10 сек, без истории). API: `https://apim.moex.com/iss/datashop/algopack/...` с Bearer APIKEY. — https://data.moex.com/products/algopack ; https://moexalgo.github.io/ — 2026-09-25 — VERIFIED (цена Promo — со страницы через Firecrawl) 
- Полные исторические тиковые сделки/ордерлог FORTS в ALGOPACK не заявлены (только агрегаты 5-мин Super Candles); ордерлог/тики — отдельные платные продукты Datashop, цены не найдены. — — UNVERIFIED
- Сервис «Открытые позиции по фьючерсам в реальном времени» (часть Algopack) платный — 600 ₽/мес «с 3 марта» (год не указан в сниппете). — https://www.moex.com/ru/derivatives/open-positions-online.aspx?code=NM — 2026-09-25 — SECONDARY

---

## 4. Тарифы и комиссии Т-Инвестиций

Общее: «биржевая комиссия уже включена в брокерскую» (п. «Биржевая комиссия — БЕСПЛАТНО» во всех тарифах; комиссия брокера по фьючерсам «не может быть менее комиссии организатора торговли»). Комиссия по фьючерсам считается в % от рублёвой стоимости контракта (по шагу цены на момент сделки), берётся за каждую сделку (вход и выход), и за исполнение расчётного фьючерса. Оборот для ступеней — дневной, по фьючерсам на MOEX. — https://cdn.tbank.ru/static/documents/invest-tariff-trader.pdf ; https://www.tbank.ru/invest/tariffs/ — 2026-09-25 — VERIFIED

| Тариф | Абон. плата | Акции (MOEX, базовый список) | Фьючерсы (кроме «доп. списка») |
|---|---|---|---|
| Инвестор | 0 ₽ | 0,3% | 0,1% |
| Трейдер | 390 ₽/мес (0 ₽ если активы ≥1,5 млн ₽, или оборот ЦБ/валюта ≥5 млн ₽ за период, или нет сделок) | 0,05% (айсберг 0,06%) | 0,04% при обороте до 5 млн ₽/день; 0,03% — 5–10 млн; 0,025% — свыше 10 млн. Фьючерсы из доп. списка — 0,08% |
| Премиум | 2 990 ₽/мес или 0 ₽ при активах в Т-Банке > 3 млн ₽ | 0,04% (по лендингу «от 0,04%») | 0,025% до 12 млн ₽/день; 0,02% — 12–17 млн; 0,015% — свыше 17 млн. Доп. список — 0,06% |
| Private | через сервис Private | не извлечено из PDF | 2.1.2: 0,02% (12–17 млн ₽/день); 2.1.3: 0,015% (>17 млн); **2.1.1 (до 12 млн) не извлечено** |

- Инвестор/Трейдер/Премиум по фьючерсам и доп. список — https://www.tbank.ru/invest/help/brokerage/account/forts/trade-futures/ — 2026-09-25 — VERIFIED
- Трейдер (PDF Т-ТРЕЙД-26 0522): ступени 0,040/0,03/0,025%; акции 0,05%; обслуживание 390 ₽ и условия бесплатности — https://cdn.tbank.ru/static/documents/invest-tariff-trader.pdf — 2026-09-25 — VERIFIED
- Премиум (PDF Т-ПРЕМ-260603): 2.1.2 = 0,02%, 2.1.3 = 0,015%; 2.1.1 в PDF-парсере потерян, но 0,025% подтверждено справкой и официальным постом от 19.09.2025 — https://cdn.tbank.ru/static/documents/invest-tariff-premium.pdf ; https://www.tbank.ru/invest/social/profile/T-Investments/c7697557-dbbe-46ef-988c-38346b93c66b/ — 2026-09-25 — VERIFIED
- Премиум — стоимость 2 990 ₽/мес или 0 ₽ при >3 млн ₽; акции «от 0,04%» — https://www.tbank.ru/invest/tariffs/ — 2026-09-25 — VERIFIED
- **Private:** новость Т-Банка от 16.12.2024: для клиентов T-Private брокерская комиссия по фьючерсам **0%**, «обязательной остаётся лишь минимальная комиссия, взимаемая организатором торгов» (т.е. фактически платится биржевой сбор). Однако текущий PDF тарифа «Private» содержит ступени 0,02%/0,015% для оборота >12 млн ₽/день; значение для первой ступени (до 12 млн) не извлечено — **нужно проверить вручную** (скачать PDF в браузере). — https://www.tbank.ru/about/news/16122024-t-investments-eliminates-futures-commissions-for-t-private-clients/ ; https://acdn.t-static.ru/static/documents/invest-tariff-private.pdf — 2026-09-25 — SECONDARY / UNVERIFIED (актуальная ставка Private до 12 млн)
- Условия сервиса Private (активы и т.п.) в PDF условий сервиса не содержат комиссий по фьючерсам; сервис бесплатен при подключении (п.1.5), критерии подключения (порог активов) в документе не найдены. — https://acdn.t-bank-app.ru/static/documents/docs-terms-of-service-private.pdf — 2026-09-25 — VERIFIED (текст) / UNVERIFIED (порог)
- Перенос непокрытой позиции (если на конец дня не хватает рублей на ГО): Трейдер — 40 ₽/день до 50 тыс. ₽, 75 ₽ до 100 тыс., … (таблица в справке); Премиум — 35 ₽/70 ₽…; Private — бесплатно до 25 000 ₽, далее от 170 ₽/день. — https://www.tbank.ru/invest/help/brokerage/account/forts/trade-futures/ ; invest-tariff-private.pdf — 2026-09-25 — VERIFIED
- Иллюстрация издержек (расчёт, SECONDARY): MX ≈ 219 500 ₽ за контракт (закрытие MXU6 31.08.2026) → Трейдер 0,04% ≈ 88 ₽ за сторону (≈176 ₽ круг); Премиум 0,025% ≈ 55 ₽/сторона. Si ≈ 86 160 ₽ (закрытие SiU6 31.08.2026) → Трейдер ≈ 34 ₽/сторона. Для сравнения тик MX = 25 ₽, тик Si = 1 ₽ → комиссия Трейдер ≈ 3,5 тика MX и ≈ 34 тика Si за сторону. — расчёт по ISS-ценам 08.2026 — 2026-09-25 — SECONDARY

---

## 5. Биржевые сборы MOEX

### 5.1 Срочный рынок (FORTS)
- Модель «мейкер–тейкер»: **мейкер платит 0**, комиссию (биржевая + клиринговая) платит только тейкер. — https://www.moex.com/s93 — 2026-09-25 — VERIFIED
- Ставки для тейкера, безадресные заявки, фьючерсы, % от стоимости контракта: валютные 0,00462; процентные 0,01650; фондовые (на акции) 0,01980; индексные 0,00660; товарные 0,01320; вечные фьючерсы на акции 0,03000; цифровые фондовые 0,04000. Адресные заявки: 0,00154 / 0,00550 / 0,00660 / 0,00220 / 0,00440 / 0,01500 / 0,02000. Минимум 1 коп. Пример на сайте: MX стоимостью 280 000 ₽ → 18,48 ₽. — https://www.moex.com/s93 — 2026-09-25 — VERIFIED
- Клиринговая комиссия за исполнение (экспирацию): валютные 0,00154%; процентные 0,00550%; фондовые 0,00660%; индексные 0,00220%; товарные 0,00440%. Исполнение вечного фьючерса в квартальный по поручению стороны — 1% от стоимости контракта. Есть сборы за ошибочные транзакции и Flood Control. — https://www.moex.com/s93 ; https://fs.moex.com/files/18033 — 2026-09-25 — VERIFIED
- Для клиента Т-Инвестиций биржевой сбор включён в брокерскую комиссию (п.10/12/13 «Биржевая комиссия — БЕСПЛАТНО») → в бэктесте моделировать только брокерский % (для Private — возможно, только биржевой сбор тейкера). — тарифные PDF — 2026-09-25 — VERIFIED (для Трейдер/Премиум), UNVERIFIED (Private)

### 5.2 Фондовый рынок (акции)
- Режим основных торгов T+ («Стакан Т+1»), асимметричная модель: мейкер 0,00%, тейкер 0,03%; симметричная модель: 0,015%/0,015%. — https://www.moex.com/s1197 — 2026-09-25 — VERIFIED (через Firecrawl query)

---

## 6. Расписание FORTS

- **С 23.03.2026** срочный рынок MOEX перешёл на **Единую торговую сессию (ЕТС)**: промежуточный клиринг (14:00) отменён; клиринговая сессия mark-to-market 23:50–00:30; расчётная клиринговая сессия в 20:00 следующего дня (T+1); вечерняя сессия относится к текущему торговому дню; экспирация «в торгах» в 14:00 или 19:00 без остановки торгов (поставочные — в клиринге); маржинальные требования выставляются в 10:00 и исполняются до 17:30. — https://www.moex.com/ru/derivatives/unified-trading-session — 2026-09-25 — VERIFIED
- **С 14.07.2026** расширенное расписание будней: 06:50–07:00 аукцион открытия; 07:00–10:00 утренняя сессия; 10:00–19:00 основная; 19:00–23:50 вечерняя (итого 17 ч). В утреннюю сессию торгуются все инструменты, кроме опционов на валютные пары; для валютных фьючерсов в утро — ценовые границы ±3% от цены на 23:50 предыдущего дня. (Страница ЕТС показывает 07:00–09:00 утро и 09:00–19:00 основную — возможное расхождение в границе основной сессии 09:00 vs 10:00.) — https://www.rbc.ru/quote/10/07/2026/6a50de6f9a79471c7e1b0461 (ссылается на https://www.moex.com/n101980) ; https://www.moex.com/ru/derivatives/unified-trading-session — 2026-09-25 — SECONDARY (РБК) / VERIFIED (страница ЕТС); граница 09:00/10:00 — UNVERIFIED
- Подтверждение по данным: часовые свечи ISS по SiU6 до 13.07.2026 начинаются в 08:00 (аукцион 08:50), с 14.07.2026 — в 06:00 (аукцион 06:50). — ISS candles SiU6 — 2026-09-25 — VERIFIED
- Выходные: дополнительная торговая сессия выходного дня 09:50–10:00 аукцион, 10:00–19:00 торги; с 18.07.2026 в неё добавлены валютные фьючерсы (границы ±3%). Состав инструментов выходного дня (индексные/товарные/акционные) — не проверен. — https://www.moex.com/ru/derivatives/unified-trading-session ; RBC — 2026-09-25 — VERIFIED (время) / UNVERIFIED (полный состав)
- Праздничные дни: торги проводятся в формате доп. сессии выходного дня 09:50–19:00. Торговый календарь: https://www.moex.com/ru/tradingcalendar (2026: 337 торговых дней, 84 дня торгов в выходные, 10 — в праздники, 28 без торгов — по сниппету). — https://www.moex.com/n96571 ; https://www.moex.com/ru/tradingcalendar — 2026-09-25 — SECONDARY
- Т-Инвестиции для экспирации: торги по фьючерсу останавливаются в 19:00 последнего дня, ВМ начисляется после 23:50. — https://www.tbank.ru/invest/help/brokerage/account/forts/trade-futures/ — 2026-09-25 — VERIFIED

---

## 7. Спецификации контрактов (данные ISS на 2026-09-25)

Источник для всей таблицы: `https://iss.moex.com/iss/engines/futures/markets/forts/securities.json?iss.only=securities&securities.columns=SECID,ASSETCODE,MINSTEP,STEPPRICE,LOTVOLUME,LASTTRADEDATE,INITIALMARGIN` — 2026-09-25 — VERIFIED. STEPPRICE для долларовых активов зависит от курса USD/RUB (8,49057 ₽ за 0,01 BR ⇒ курс ≈ 84,9) и меняется ежедневно. ГО — биржевое (у брокера может быть выше; смотреть GetFuturesMargin).

| Код | Базовый актив | Шаг цены | Стоимость шага, ₽ | Лот (ед. базового) | Ближайший контракт / посл. день | ГО биржи, ₽ | Цикл |
|---|---|---|---|---|---|---|---|
| MX (MIX) | Индекс МосБиржи | 25 | 25,00 | 1 | MXZ6 / 2026-12-17 | 27 973,53 | квартальный (H/M/U/Z), 3-й четверг |
| MM (MXI) | Индекс МосБиржи мини | 0,05 | 0,50 | 1 | MMZ6 / 2026-12-17 | 2 797,30 | квартальный |
| RI (RTS) | Индекс РТС | 10 | 16,98114 | 1 | RIZ6 / 2026-12-17 | 23 387,06 | квартальный |
| Si | USD/RUB | 1 | 1,00 | 1000 USD | SiZ6 / 2026-12-17 | 13 571,23 | квартальный |
| CR | CNY/RUB | 0,001 | 1,00 | 1000 CNY | CRZ6 / 2026-12-17 | 1 278,12 | квартальный |
| GD (GOLD) | золото, USD/унц | 0,1 | 8,49057 | 1 унция | GDZ6 / 2026-12-18 | 33 334,04 | квартальный (3-я пятница) |
| BR | нефть Brent | 0,01 | 8,49057 | 10 барр. | BRV6 / 2026-10-01; далее BRX6 11-02, BRZ6 12-01 | 19 135,62 (BRV6) | **ежемесячный** |
| NG | природный газ | 0,001 | 8,49057 | 100 MMBtu | NGU6 / 2026-09-28; NGV6 10-28; NGX6 11-25 | 7 645,17 (NGU6) | **ежемесячный** |
| SR | акции Сбербанка | 1 | 1,00 | 100 акций | SRZ6 / 2026-12-17 | 5 351,53 | квартальный |
| GZ | акции Газпрома | 1 | 1,00 | 100 акций | GZZ6 / 2026-12-17 | 1 944,04 | квартальный |
| LK | акции Лукойла | 1 | 1,00 | 10 акций | LKZ6 / 2026-12-17 | 10 287,18 | квартальный |
| RN | акции Роснефти | 1 | 1,00 | 100 акций | RNZ6 / 2026-12-17 | 8 213,71 | квартальный |
| IMOEXF | вечный на индекс МосБиржи | 0,5 | 5,00 | 10 | бессрочный (2100-01-01) | 2 326,99 | вечный |
| CNYRUBF | вечный CNY/RUB TOM | 0,001 | 1,00 | 1000 CNY | бессрочный | 1 019,76 | вечный |
| USDRUBF | вечный USD/RUB TOM | 0,01 | 10,00 | 1000 USD | бессрочный | 12 784,03 | вечный |
| GLDRUBF | вечный золото/RUB TOM | 0,1 | 0,10 | 1 (г) | бессрочный | 1 289,45 | вечный |

- Для акционных/индексных/валютных квартальных контрактов последний день торгов — 3-й четверг месяца экспирации (17.12.2026 для Z6); GD/товарные на зарубежные активы — 3-я пятница (18.12.2026). Также в листинге есть контракты H7/M7/U7 и дальние MX/MM/Si/CR до 2028. — ISS securities — 2026-09-25 — VERIFIED (даты), правило «3-й четверг» — SECONDARY (вывод из дат)
- Цена MX в ISS котируется в «пунктах ×100» (например MXU6 закрылся 31.08.2026 на 219 500 при MM 2 194,35), т.е. MX ≈ 10× стоимости MM по деньгам (25 ₽ за 25 пунктов vs 0,5 ₽ за 0,05). — ISS candles MXU6/MMU6 — 2026-09-25 — VERIFIED (данные) / SECONDARY (интерпретация)

---

## 8. Ликвидность (средний дневной объём, контрактов)

Метод: месячные свечи ISS (`interval=31`) по фронтальному контракту; объём месяца / число будних торговых дней (июль 2026 — 23, август 2026 — 21; сессии выходного дня в знаменателе не учтены). Проверка: сумма дневных VOLUME по SiU6 за июль (history) = 34 645 100 = объёму месячной свечи. Источник: `https://iss.moex.com/iss/engines/futures/markets/forts/securities/<SECID>/candles.json?interval=31&from=2026-07-01&till=2026-08-31` — 2026-09-25 — SECONDARY (расчёт по первичным данным).

| Контракт (фронт) | ADV июль 2026 | ADV август 2026 | Среднее Jul–Aug |
|---|---|---|---|
| SiU6 | 1 506 309 | 1 471 097 | ≈ 1,49 млн |
| CRU6 | 10 091 346 | 10 428 391 | ≈ 10,3 млн |
| CNYRUBF | 4 191 591 | 4 174 373 | ≈ 4,18 млн |
| IMOEXF | 2 011 947 | 1 816 785 | ≈ 1,92 млн |
| MMU6 | 627 946 | 568 250 | ≈ 0,60 млн |
| MXU6 | 464 172 | 442 916 | ≈ 0,45 млн |
| GZU6 | 411 685 | 355 019 | ≈ 0,38 млн |
| USDRUBF | 328 937 | 344 765 | ≈ 0,34 млн |
| GLDRUBF | 514 553 | 631 666 | ≈ 0,57 млн |
| SRU6 | 242 933 | 137 061 | ≈ 0,19 млн |
| GDU6 | 141 291 | 193 615 | ≈ 0,17 млн |
| RIU6 | 122 653 | 98 006 | ≈ 0,11 млн |
| LKU6 | 33 326 | 25 617 | ≈ 0,03 млн |
| RNU6 | 34 386 | 21 544 | ≈ 0,03 млн |
| BRU6 (фронт в августе) | — | 602 420 | ≈ 0,6 млн (август) |
| NGQ6 (фронт в августе, до ~26.08) | — | 442 462 | ≈ 0,44–0,5 млн |

- Однодневный срез 24.09.2026 (history): BRV6 1 382 746 контрактов (OI 391 560); BRX6 163 587; всего 861 инструмент/серия на FORTS в выгрузке дня. — https://iss.moex.com/iss/history/engines/futures/markets/forts/securities.json?date=2026-09-24 — 2026-09-25 — VERIFIED
- Открытый интерес SiU6 в середине квартала ≈ 9–12 млн контрактов (июль–август 2026). — ISS history SiU6 — 2026-09-25 — VERIFIED
- Объём в контрактах несопоставим между инструментами: CR (лот 1000 CNY ≈ 12,8 тыс. ₽) vs MX (≈ 220 тыс. ₽). Для оценки ёмкости стратегии использовать VALUE/оборот в ₽. — вывод — SECONDARY

---

## 9. Открытые вопросы / UNVERIFIED

1. **Тариф Private, фьючерсы до 12 млн ₽/день** — 0% (новость 12.2024) или иная ставка в текущем PDF (ступени 0,02%/0,015% выше 12 млн присутствуют). Скачать https://acdn.t-static.ru/static/documents/invest-tariff-private.pdf вручную. Также ставка Private по акциям (п.1.1) не извлечена.
2. **Трейлинг-стоп на фьючерсах FORTS** — API поддерживает TAKE_PROFIT_TYPE_TRAILING, но применимость к фьючерсам не задокументирована; проверить в песочнице (стоп-заявки теперь там доступны) и на реальном счёте 1 контрактом.
3. **GetCandles по экспирированным фьючерсам** через API (нужен instrument_uid истёкшего контракта; искать через FindInstrument / Futures с INSTRUMENT_STATUS_ALL) — не проверено; архивы history-data по истёкшим контрактам, судя по каталогу, есть (SECONDARY).
4. **history-trades**: нужна ли авторизация Bearer; реальная глубина истории для FORTS (пример скрипта — с 2020 года); включены ли сделки вечерней/утренней сессий и выходного дня.
5. Песочница после обновления документации: начисляется ли вариационная маржа и списывается ли ГО по фьючерсам (старая версия: списывалась полная стоимость, ВМ не считалась); как исполняются стоп-заявки в песочнице.
6. Граница утренней/основной сессии после 14.07.2026: 10:00 (РБК, новость MOEX n101980) vs 09:00 (схема на странице ЕТС). Проверить на https://www.moex.com/s1167 или в спецификациях.
7. Полный состав инструментов сессии выходного дня (индексные MX/MM/IMOEXF, товарные BR/NG/GD, акционные) — не проверен.
8. Точный URL и список tools MCP-сервера (`https://invest-public-api.tbank.ru/mcp` — не подтверждено в прочитанных страницах; см. https://developer.tbank.ru/invest/mcp/agents-connect/coding-agents).
9. Задержка анонимного ISS (≈15 мин) и отсутствие исторических тиков в публичном ISS — не проверялись в этом сеансе.
10. Стоимость Datashop-продуктов с полным ордерлогом/тиками FORTS — не найдена.
11. Лендинг tbank.ru/invest/tariffs указывает Premium «от 0,025% для фьючерсов», тогда как тариф содержит ступени до 0,015% — лендинг отстаёт; опираться на PDF/справку.
