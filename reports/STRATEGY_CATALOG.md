# Каталог стратегий исследования v2 (со ссылками на первоисточники)

Все стратегии, реализованные и протестированные в исследовании. Колонка «код» ведёт на реализацию в репозитории, «источник» — на описание идеи. Цифры — период разработки 2025-09-25…2026-06-30 после издержек (Трейдер), если не указано иное; финальные кандидаты — весь год + holdout. Ссылки с пометкой «проверена» открыты через Firecrawl 25.09.2026; остальные — устойчивые адреса Wikipedia/DOI, вживую не проверялись (сервис поиска ограничил запросы).

Итог: из 47 семейств только дневной трендовый ансамбль на фьючерсах (C1) пережил walk-forward и holdout, и то без статистической значимости.

### 1. T_DONCH — Пробой канала Дончиана
- Группа: тренд. Правило: close > max(high, n) — лонг, < min(low, n) — шорт; выход по каналу n/2
- Результат: Stage 1: 24300 конф., с плюсом 7%, медианный Sharpe -4.6, сделка -12.1 б.п.. **Вердикт: отвергнута**
- Источник: [StockCharts ChartSchool: Price (Donchian) Channels](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/price-channels) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 2. T_DONCH_STOP — Пробой Дончиана стоп-заявкой
- Группа: тренд. Правило: стоп-заявка на уровне экстремума n свечей
- Результат: Stage 1: 24300 конф., с плюсом 7%, медианный Sharpe -5.5, сделка -11.1 б.п.. **Вердикт: отвергнута**
- Источник: [StockCharts: Donchian Trading Guidelines](https://chartschool.stockcharts.com/table-of-contents/overview/donchian-trading-guidelines) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 3. T_EMA_X — Пересечение EMA
- Группа: тренд. Правило: EMA(f) > EMA(s) — лонг, иначе шорт
- Результат: Stage 1: 82620 конф., с плюсом 10%, медианный Sharpe -3.0, сделка -11.7 б.п.. **Вердикт: отвергнута внутри дня; в дневном ансамбле C1 — положительна**
- Источник: [Wikipedia: Moving average](https://en.wikipedia.org/wiki/Moving_average) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 4. T_PRICE_MA — Цена относительно MA
- Группа: тренд. Правило: close выше EMA(n) + полоса — лонг
- Результат: Stage 1: 48600 конф., с плюсом 4%, медианный Sharpe -6.9, сделка -11.6 б.п.. **Вердикт: отвергнута**
- Источник: [Faber (2007), A Quantitative Approach to Tactical Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 5. T_HMA — Наклон Hull MA
- Группа: тренд. Правило: наклон HMA(n) > 0 — лонг
- Результат: Stage 1: 24300 конф., с плюсом 9%, медианный Sharpe -4.6, сделка -11.4 б.п.. **Вердикт: отвергнута**
- Источник: [Alan Hull: The Hull Moving Average](https://alanhull.com/the-hull-moving-average/) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 6. T_SUPERTREND — SuperTrend
- Группа: тренд. Правило: направление SuperTrend(ATR n, множитель m)
- Результат: Stage 1: 43740 конф., с плюсом 10%, медианный Sharpe -3.9, сделка -12.0 б.п.. **Вердикт: отвергнута**
- Источник: [TradingView: Supertrend (O. Seban)](https://www.tradingview.com/support/solutions/43000634738-supertrend/) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 7. T_ADX — ADX / направленное движение
- Группа: тренд. Правило: ADX > порог и +DI > −DI — лонг
- Результат: Stage 1: 29160 конф., с плюсом 6%, медианный Sharpe -4.3, сделка -12.3 б.п.. **Вердикт: отвергнута внутри дня; дневной ансамбль — 6/6 мес. в плюсе в WF**
- Источник: [Wikipedia: Average directional movement index (Wilder)](https://en.wikipedia.org/wiki/Average_directional_movement_index) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 8. T_MTF — Тренд нескольких масштабов
- Группа: тренд. Правило: знаки ROC(n), ROC(nk), ROC(nk²) совпадают
- Результат: Stage 1: 38880 конф., с плюсом 7%, медианный Sharpe -5.3, сделка -11.3 б.п.. **Вердикт: отвергнута**
- Источник: [Hurst, Ooi, Pedersen (2017), A Century of Evidence on Trend-Following Investing](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 9. T_KELTNER — Пробой канала Кельтнера
- Группа: тренд. Правило: close > EMA(n) + k·ATR — лонг
- Результат: Stage 1: 43740 конф., с плюсом 7%, медианный Sharpe -5.0, сделка -11.6 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: Keltner channel](https://en.wikipedia.org/wiki/Keltner_channel) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 10. M_TSMOM — Временной моментум (TSMOM)
- Группа: моментум. Правило: ROC(n) > z·σ — лонг, < −z·σ — шорт
- Результат: Stage 1: 87480 конф., с плюсом 9%, медианный Sharpe -3.5, сделка -11.7 б.п.. **Вердикт: отвергнута внутри дня; дневная версия — часть C1**
- Источник: [Moskowitz, Ooi, Pedersen (2012), Time series momentum, JFE 104(2)](https://ideas.repec.org/a/eee/jfinec/v104y2012i2p228-250.html) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 11. M_ACCEL — Ускорение моментума
- Группа: моментум. Правило: ROC(n) > 0 и растёт
- Результат: Stage 1: 19440 конф., с плюсом 3%, медианный Sharpe -10.8, сделка -11.2 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: Momentum (technical analysis)](https://en.wikipedia.org/wiki/Momentum_(technical_analysis)) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 12. M_ROC_VOTE — Голосование ROC нескольких горизонтов
- Группа: моментум. Правило: сумма знаков ROC(b·1,2,4,8) ≥ порог
- Результат: Stage 1: 38880 конф., с плюсом 5%, медианный Sharpe -6.9, сделка -11.4 б.п.. **Вердикт: отвергнута**
- Источник: [Hurst, Ooi, Pedersen (2017)](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 13. R_BB — Возврат от полос Боллинджера
- Группа: возврат к среднему. Правило: close < SMA − k·σ — лонг, выход на SMA
- Результат: Stage 1: 77760 конф., с плюсом 3%, медианный Sharpe -5.4, сделка -11.5 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: Bollinger Bands](https://en.wikipedia.org/wiki/Bollinger_Bands) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 14. R_BB_LIMIT — Боллинджер лимитными заявками
- Группа: возврат к среднему. Правило: лимит на полосе (вход откатом, без спреда)
- Результат: Stage 1: 43740 конф., с плюсом 2%, медианный Sharpe -8.2, сделка -10.4 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: Bollinger Bands](https://en.wikipedia.org/wiki/Bollinger_Bands) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 15. R_RSI — RSI-возврат (в т.ч. RSI(2))
- Группа: возврат к среднему. Правило: RSI(n) < lo — лонг, выход при RSI > 50
- Результат: Stage 1: 77760 конф., с плюсом 4%, медианный Sharpe -8.4, сделка -10.6 б.п.; без издержек +0,8 б.п./сделка. **Вердикт: отвергнута (валовое преимущество < издержек)**
- Источник: [StockCharts ChartSchool: RSI(2) (L. Connors)](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 16. R_MA_DEV — Отклонение от MA в ATR
- Группа: возврат к среднему. Правило: close < EMA(n) − k·ATR — лонг
- Результат: Stage 1: 43740 конф., с плюсом 4%, медианный Sharpe -5.6, сделка -11.0 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: Average true range](https://en.wikipedia.org/wiki/Average_true_range) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 17. R_VWAP — Отклонение от сессионного VWAP
- Группа: возврат к среднему. Правило: (close − VWAP)/σ < −k — лонг до VWAP
- Результат: Stage 1: 15876 конф., с плюсом 4%, медианный Sharpe -3.8, сделка -13.5 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: Volume-weighted average price](https://en.wikipedia.org/wiki/Volume-weighted_average_price) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 18. R_EXTREME — Разворот после экстремального движения
- Группа: возврат к среднему. Правило: ROC(m) < −z·σ — лонг
- Результат: Stage 1: 58320 конф., с плюсом 8%, медианный Sharpe -3.7, сделка -11.4 б.п.. **Вердикт: отвергнута**
- Источник: [Jegadeesh (1990), Evidence of Predictable Behavior of Security Returns, JF](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 19. R_GAP_FADE — Против гэпа открытия
- Группа: возврат к среднему. Правило: гэп > z·σ дня — встречная позиция до закрытия гэпа/конца дня
- Результат: Stage 1: 4536 конф., с плюсом 37%, медианный Sharpe -0.4, сделка -8.2 б.п.; без издержек +5,6 б.п.. **Вердикт: отвергнута**
- Источник: [Lou, Polk, Skouras (2019), A tug of war: overnight vs intraday returns, JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 20. R_VOLSPIKE_FADE — Против всплеска объёма
- Группа: возврат к среднему. Правило: объём > k·среднего и большой бар — встречная позиция
- Результат: Stage 1: 17280 конф., с плюсом 6%, медианный Sharpe -4.2, сделка -11.5 б.п.. **Вердикт: отвергнута**
- Источник: [Gervais, Kaniel, Mingelgrin (2001), The High-Volume Return Premium, JF](https://doi.org/10.1111/0022-1082.00349) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 21. V_SQUEEZE — Пробой после сжатия (squeeze)
- Группа: волатильность. Правило: ширина Боллинджера в нижнем перцентиле + пробой канала
- Результат: Stage 1: 38880 конф., с плюсом 15%, медианный Sharpe -4.4, сделка -11.3 б.п.. **Вердикт: отвергнута**
- Источник: [StockCharts ChartSchool: Bollinger Band Squeeze](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/bollinger-band-squeeze) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 22. V_NR_BREAK — Пробой узкого диапазона (NR4/NR7)
- Группа: волатильность. Правило: бар с минимальным диапазоном за n — стоп-заявки на его high/low
- Результат: Stage 1: 14580 конф., с плюсом 7%, медианный Sharpe -5.4, сделка -10.4 б.п.. **Вердикт: отвергнута**
- Источник: [StockCharts ChartSchool: Narrow Range Day NR7 (T. Crabel)](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/narrow-range-day-nr7) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 23. V_RANGE_EXP — Расширение диапазона
- Группа: волатильность. Правило: диапазон бара > k·ATR и закрытие у края — по направлению (или против)
- Результат: Stage 1: 29160 конф., с плюсом 10%, медианный Sharpe -3.2, сделка -11.3 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: Average true range (Wilder)](https://en.wikipedia.org/wiki/Average_true_range) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 24. V_REGIME_TREND — Тренд в режиме высокой/низкой волатильности
- Группа: режимы. Правило: EMA-тренд только при vol выше/ниже медианы
- Результат: Stage 1: 29160 конф., с плюсом 13%, медианный Sharpe -2.8, сделка -10.9 б.п.. **Вердикт: отвергнута**
- Источник: [Moreira, Muir (2017), Volatility-Managed Portfolios, JF](https://doi.org/10.1111/jofi.12513) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 25. U_VOL_BREAK — Пробой на повышенном объёме
- Группа: объём. Правило: пробой канала при объёме > k·среднего
- Результат: Stage 1: 19440 конф., с плюсом 8%, медианный Sharpe -4.7, сделка -11.8 б.п.. **Вердикт: отвергнута**
- Источник: [Gervais, Kaniel, Mingelgrin (2001)](https://doi.org/10.1111/0022-1082.00349) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 26. U_SIGNED_VOL — Знаковый объём (прокси order flow)
- Группа: объём / order flow. Правило: доля объёма в барах вверх/вниз за n > порог
- Результат: Stage 1: 58320 конф., с плюсом 5%, медианный Sharpe -5.7, сделка -11.4 б.п.. **Вердикт: отвергнута (стакана и сделок в данных нет)**
- Источник: [Easley, López de Prado, O'Hara (2012), Flow Toxicity and Liquidity (bulk volume), RFS](https://doi.org/10.1093/rfs/hhs053) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 27. U_OBV — On-Balance Volume
- Группа: объём. Правило: OBV > EMA(OBV) и цена > EMA
- Результат: Stage 1: 9720 конф., с плюсом 8%, медианный Sharpe -4.4, сделка -11.6 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: On-balance volume (J. Granville)](https://en.wikipedia.org/wiki/On-balance_volume) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 28. U_BAR_PRESSURE — Давление бара (Close Location Value)
- Группа: объём / order flow. Правило: средневзвешенная по объёму позиция close в диапазоне
- Результат: Stage 1: 19440 конф., с плюсом 5%, медианный Sharpe -7.9, сделка -12.4 б.п.. **Вердикт: отвергнута**
- Источник: [Wikipedia: Accumulation/distribution index (Chaikin)](https://en.wikipedia.org/wiki/Accumulation/distribution_index) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 29. I_ORB — Пробой диапазона открытия (ORB)
- Группа: внутридневная. Правило: стоп-заявки на high/low первых 15–60 мин сессии, выход в конце дня
- Результат: Stage 1: 9072 конф., с плюсом 7%, медианный Sharpe -3.6, сделка -10.8 б.п.; без издержек +1,9 б.п.. **Вердикт: отвергнута**
- Источник: [Opening Range Breakout (T. Crabel)](https://tradersmastermind.com/trading-strategy-opening-range-breakout/) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 30. I_PDAY_BRK — Пробой high/low вчерашнего дня
- Группа: внутридневная. Правило: стоп-заявки на уровнях прошлого дня
- Результат: Stage 1: 2016 конф., с плюсом 12%, медианный Sharpe -2.9, сделка -16.1 б.п.. **Вердикт: отвергнута**
- Источник: [StockCharts: Price Channels](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/price-channels) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 31. I_TOD — Сезонность времени дня
- Группа: сезонность. Правило: вход в заданную минуту МСК, удержание 30–240 мин
- Результат: Stage 1: 5544 конф., с плюсом 2%, медианный Sharpe -4.5, сделка -11.9 б.п.. **Вердикт: отвергнута**
- Источник: [Gao, Han, Li, Zhou (2018), Market Intraday Momentum, JFE](https://ideas.repec.org/a/eee/jfinec/v129y2018i2p394-414.html) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 32. I_OVERNIGHT — Ночная позиция
- Группа: сезонность. Правило: покупка в конце вечерней сессии, продажа утром
- Результат: Stage 1: 240 конф., с плюсом 2%, медианный Sharpe -26.1, сделка -11.3 б.п.. **Вердикт: отвергнута (валово +8…18 б.п., меньше издержек)**
- Источник: [Lou, Polk, Skouras (2019), A tug of war, JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 33. H_ON — Ночной дрейф акций (точное исполнение)
- Группа: сезонность. Правило: по минуткам: вход по close последней свечи, выход по open утра
- Результат: SBER гросс +8,5 б.п./ночь (t=5,9), после издержек −0,7 б.п.. **Вердикт: отвергнута**
- Источник: [Lou, Polk, Skouras (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/overnight.py

### 34. I_DOW — Эффект дня недели
- Группа: сезонность. Правило: удержание в заданный день недели
- Результат: Stage 1: 420 конф., с плюсом 29%, медианный Sharpe -0.6, сделка -10.0 б.п.. **Вердикт: отвергнута**
- Источник: [French (1980), Stock Returns and the Weekend Effect, JFE](https://doi.org/10.1016/0304-405X(80)90021-5) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 35. C_CARRY — Carry: шорт контанго (1 инструмент)
- Группа: carry. Правило: постоянный шорт фьючерса (с фильтром EMA)
- Результат: Stage 1: 162 конф., с плюсом 0%, медианный Sharpe -0.9, сделка -233.0 б.п.. **Вердикт: отвергнута**
- Источник: [Koijen, Moskowitz, Pedersen, Vrugt (2018), Carry, JFE](https://doi.org/10.1016/j.jfineco.2017.11.002) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/families.py

### 36. X_LEADLAG — Lead-lag (A → B)
- Группа: межинструментальная. Правило: движение лидера (MX, RI, BR, Si, GD, SR, SBER) → сделка по отстающему
- Результат: 43 тыс. вместе с парами: 5% с плюсом, медианный Sharpe −4,3. **Вердикт: отвергнута**
- Источник: [Lo, MacKinlay (1990), When Are Contrarian Profits Due to Overreaction?, RFS](https://doi.org/10.1093/rfs/3.2.175) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/xasset.py

### 37. X_PAIR — Парный трейдинг (спред z-score)
- Группа: межинструментальная. Правило: z-score спреда rB − β·rA; две ноги
- Результат: 18 пар: 8% с плюсом, медианный Sharpe −2,2. **Вердикт: отвергнута**
- Источник: [Gatev, Goetzmann, Rouwenhorst (2006), Pairs Trading, RFS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/xasset.py

### 38. P_TSMOM — Портфельный TSMOM с таргетом волатильности
- Группа: портфель / тренд. Правило: вес = sign(ROC n)·σ*/σ_i
- Результат: дневной, фьючерсы: медианный Sharpe 0,7 (dev). **Вердикт: вошла в C1 (подтверждена слабо)**
- Источник: [Moskowitz, Ooi, Pedersen (2012)](https://ideas.repec.org/a/eee/jfinec/v104y2012i2p228-250.html) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/panel.py

### 39. P_EMA — Портфельный EMA-тренд
- Группа: портфель / тренд. Правило: вес = sign(EMA f − EMA s)·σ*/σ_i
- Результат: дневной, фьючерсы: медианный Sharpe 0,5 (dev). **Вердикт: вошла в C1**
- Источник: [Hurst, Ooi, Pedersen (2017)](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/panel.py

### 40. P_BRK — Портфельный пробой (Дончиан)
- Группа: портфель / тренд. Правило: вес по направлению последнего пробоя канала n
- Результат: дневной, фьючерсы: медианный Sharpe 0,7 (dev). **Вердикт: вошла в C1**
- Источник: [StockCharts: Donchian Channels](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/price-channels) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/panel.py

### 41. P_XS_MOM — Кросс-секционный моментум
- Группа: портфель. Правило: лонг top-k / шорт bottom-k по доходности за n дней
- Результат: акции: dev до Sharpe 2,3 до платы за шорт; C3 — год −11,9%. **Вердикт: отвергнута (C3)**
- Источник: [Jegadeesh, Titman (1993), Returns to Buying Winners and Selling Losers, JF](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/panel.py

### 42. P_XS_REV — Кросс-секционный краткосрочный разворот
- Группа: портфель. Правило: обратный ранжир (лузеры в лонг)
- Результат: медианный Sharpe −1,7 (акции). **Вердикт: отвергнута**
- Источник: [Jegadeesh (1990), JF](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/panel.py

### 43. P_CARRY — Carry-портфель Si/CR
- Группа: портфель / carry. Правило: шорт Si и CR с таргетом волатильности
- Результат: C4: год +0,8%, holdout −3,7%. **Вердикт: отвергнута**
- Источник: [Koijen et al. (2018), Carry, JFE](https://doi.org/10.1016/j.jfineco.2017.11.002) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/panel.py

### 44. ML_DAILY — LightGBM, дневной, панель 21 инструмента
- Группа: ML. Правило: регрессия доходности следующего дня, ежемесячное переобучение
- Результат: вне выборки IC ≈ 0, Sharpe −1,3…−2,5. **Вердикт: отвергнута**
- Источник: [LightGBM (Ke et al., 2017), официальный репозиторий](https://github.com/microsoft/LightGBM) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/scripts/deep/a07_ml_panel.py

### 45. ML_HOURLY — LightGBM, часовой классификатор (C10)
- Группа: ML. Правило: направление на 4–16 свечей, вход при уверенности > порога
- Результат: IC вне выборки 0,05–0,10; holdout −7,3% (0/3 мес.). **Вердикт: отвергнута**
- Источник: [LightGBM, официальный репозиторий](https://github.com/microsoft/LightGBM) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/scripts/deep/a07b_ml_hourly.py

### 46. C1_TREND_FUT_ENS — Ансамбль дневных трендовых правил на 9 фьючерсах
- Группа: финальный кандидат. Правило: среднее весов 30 правил (TSMOM/EMA/пробой), таргет 10% годовых
- Результат: год +8,8%, медиана +0,8%/мес, худший −3,3%, DD −8,1%, WF +13,3%, holdout +0,7%. **Вердикт: ЛУЧШИЙ, но статистически не доказан (p≈0,22)**
- Источник: [Hurst, Ooi, Pedersen (2017); Moskowitz, Ooi, Pedersen (2012)](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/candidates.py

### 47. C5_TREND_FUT_LONGONLY — Только лонг-сторона тренда на фьючерсах
- Группа: финальный кандидат. Правило: как C1, но веса ≥ 0
- Результат: год +6,3%, holdout +5,4% (3/3), на разработке ≈ 0. **Вердикт: режимно-зависим, не рекомендован**
- Источник: [Moskowitz, Ooi, Pedersen (2012)](https://ideas.repec.org/a/eee/jfinec/v104y2012i2p228-250.html) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/candidates.py

### 48. C6-C8_WF — Процедуры walk-forward отбора (Sharpe/Sortino/floor)
- Группа: финальный кандидат. Правило: ежемесячный выбор лучших из 423 портфелей по прошлому
- Результат: WF Sharpe до 2,2, holdout −0,7…−0,9%. **Вердикт: отвергнуты holdout'ом**
- Источник: [Bailey, López de Prado (2014), The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) — не проверена вживую (устойчивый адрес/DOI)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/src/moexlab/deep/wf.py

### 49. C9_META_RP — Мета-стратегия риск-паритета (C1+C3+C4)
- Группа: портфель. Правило: веса обратно прошлой волатильности, лаг 1 день
- Результат: WF Sharpe 1,36, holdout −2,1%. **Вердикт: отвергнута**
- Источник: [Hurst, Ooi, Pedersen (2017)](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing) — проверена (Firecrawl, 25.09.2026)
- Код: https://github.com/Masster/invest_strategy/blob/claude/deep-strategy-search/scripts/deep/a08_final_eval.py
