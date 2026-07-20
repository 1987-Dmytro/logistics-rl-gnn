# /close — терминальное закрытие сессии (operator-only)

Вызывается ТОЛЬКО оператором в конце рабочего дня. Порядок:

1. **Daily log — полный REWRITE** `knowledge/daily_logs/<today>.md`: сводка (конкретика, не
   «работал над X») · решения (→ `decisions/dec-*.md` при значимых) · next · блокеры · `[[wikilinks]]`.
2. **Память:** durable-уроки дня → native-память (topic-файлы + однострочники в MEMORY.md);
   MEMORY.md держать ≤200L/25KB.
3. **hot.md curated** — переписать под завтрашний старт: What's Hot / Next / Blockers +
   `**Last update:** <today>`. Маркер `AUTO-GEN END` НЕ трогать.
4. Если установлен валидатор: `python3 scripts/check-wikilinks.py` → 0 битых в сегодняшнем логе.
5. Итог оператору: сделано / next / блокеры / обновлённые файлы. Коммит предложить, НЕ выполнять.
