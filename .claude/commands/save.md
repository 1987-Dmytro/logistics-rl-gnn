# /save — mid-session checkpoint (НЕ закрывает сессию)

Только по явному запросу оператора.

1. **Daily log — APPEND** (не rewrite): в `knowledge/daily_logs/<today>.md` дописать секцию
   `## Checkpoint HH:MM` — что сделано с прошлого чекпойнта, связи `[[wikilinks]]`.
2. **hot.md curated** — обновить What's Hot / Next / Blockers ниже маркера `AUTO-GEN END`
   (сам маркер НЕ трогать) + строку `**Last update:**`.
3. `python3 scripts/refresh-hot-cache.py` — освежить AUTO-секцию.
4. Если установлен валидатор: `python3 scripts/check-wikilinks.py` — битые линки в вывод (не чинить молча).
5. Durable-урок появился? → записать в native-память (topic-файл + строка в MEMORY.md).

Коммиты — отдельно, по запросу оператора.
