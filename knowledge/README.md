# knowledge/ — vault of the logistics-rl-gnn project

The project's second brain: an Obsidian vault plus the source of Tier-2 reads for Claude Code.

## Constitution
1. Files are created only on request; folders are not restructured without a request.
2. One home per fact: volatile state → `hot.md` (the curated block) · durable lessons → Claude's
   native memory · decisions → `decisions/dec-*.md` · procedures → `runbooks/`.
3. `[[wikilinks]]` connect the notes; the validator is `scripts/check-wikilinks.py` (if M4 is installed).
4. Note frontmatter is uniform (`type` / `date` / `tags`) — the vault stays queryable.

## Obsidian
Open the FOLDER `knowledge/` as the vault (not the project root — scanning a large repository hangs
Electron). `.obsidian/` belongs in `.gitignore`.

## Structure
`hot.md` live state (AUTO+CURATED) · `index.md` the auto-index (Stop hook) · `daily_logs/` ·
`decisions/` · `architecture/` · `runbooks/` · `templates/`.
