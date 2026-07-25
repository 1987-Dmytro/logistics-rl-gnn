# /close — terminal session close (operator-only)

Invoked ONLY by the operator at the end of the workday. Order:

1. **Daily log — full REWRITE** of `knowledge/daily_logs/<today>.md`: summary (specifics, not "worked
   on X") · decisions (→ `decisions/dec-*.md` for significant ones) · next · blockers · `[[wikilinks]]`.
2. **Memory:** the day's durable lessons → native memory (topic files + one-liners in MEMORY.md);
   keep MEMORY.md ≤200L/25KB.
3. **hot.md curated** — rewrite it for tomorrow's start: What's Hot / Next / Blockers +
   `**Last update:** <today>`. Do NOT touch the `AUTO-GEN END` marker.
4. If the validator is installed: `python3 scripts/check-wikilinks.py` → 0 broken links in today's log.
5. Report to the operator: done / next / blockers / updated files. Offer a commit, do NOT run it.
