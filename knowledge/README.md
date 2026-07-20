# knowledge/ — vault проекта logistics-rl-gnn

Второй мозг проекта: Obsidian-vault + источник Tier-2 чтений для Claude Code.

## Конституция
1. Файлы создаются только по запросу; папки не реструктурируются без запроса.
2. Один дом на факт: volatile-state → `hot.md` (curated-блок) · durable-уроки → native-память Claude ·
   решения → `decisions/dec-*.md` · процедуры → `runbooks/`.
3. `[[wikilinks]]` связывают ноты; валидатор — `scripts/check-wikilinks.py` (если установлен M4).
4. Frontmatter на нотах единообразный (`type` / `date` / `tags`) — vault остаётся queryable.

## Obsidian
Открывать как vault ПАПКУ `knowledge/` (не корень проекта — скан большого репозитория вешает
Electron). `.obsidian/` — в `.gitignore`.

## Структура
`hot.md` live-state (AUTO+CURATED) · `index.md` авто-индекс (Stop-hook) · `daily_logs/` ·
`decisions/` · `architecture/` · `runbooks/` · `templates/`.
