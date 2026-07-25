# /save — mid-session checkpoint (does NOT close the session)

Only on the operator's explicit request.

1. **Daily log — APPEND** (not a rewrite): in `knowledge/daily_logs/<today>.md` append a section
   `## Checkpoint HH:MM` — what was done since the previous checkpoint, links via `[[wikilinks]]`.
2. **hot.md curated** — update What's Hot / Next / Blockers below the `AUTO-GEN END` marker
   (do NOT touch the marker itself) plus the `**Last update:**` line.
3. `python3 scripts/refresh-hot-cache.py` — refresh the AUTO section.
4. If the validator is installed: `python3 scripts/check-wikilinks.py` — report broken links (do not
   fix them silently).
5. A durable lesson showed up? → write it to native memory (topic file + a line in MEMORY.md).

Commits are separate, on the operator's request.
