# CLAUDE.md — logistics-rl-gnn

> Dynamic vehicle routing (CVRPTW) via Graph Neural Networks + Reinforcement Learning
> This file = rules + pointers (NOT a code map). Target ≤200 lines; every line must pass the
> removal test: "drop it — will Claude get it wrong?". A new rule is added on the SECOND repeat
> of a mistake.

## ⚠️ Prohibitions (do NOT violate)

1. **NEVER commit heavy/raw artefacts** (OSM dumps, datasets, `*.pt` checkpoints, wandb logs).
   Data and weights live outside the repo (DVC/links); git holds code and configs only. Check:
   before `git add` — no binaries/`*.pt`/large `data/` in the diff.
2. **NEVER hardcode CUDA / `.cuda()`** — the code is device-agnostic via `torch.device`. The target
   hardware is CPU-only. Check: `grep -rn "\.cuda()\|device=.cuda" src` = empty.
3. **NEVER show a before/after without an honest baseline** (OR-Tools + the greedy heuristic) on an
   IDENTICAL instance with a fixed seed. A comparison without a shared baseline is void.
4. **NEVER publish a metric without a pinned seed and a saved run config** — reproducibility is the
   goal of this project. A number without seed+config does not go out.
5. **NEVER mix real and synthetic data without an explicit flag/tag** — the case rests on "the data
   is real". Every instance is tagged `real`/`synthetic`.
6. **NEVER push to `main` without a green `pytest`.** Red or unrun tests do not reach main.

## Language policy

- Conversation: mirror the operator (ru).
- ALL artifacts are English: code, comments, docstrings, commit messages, docs, vault entries
  (decisions, runbooks, daily logs, hot.md curated), README.
- Never let the conversation language leak into artifacts. If an artifact arrives in the wrong
  language, translate it in the same change.

## Session start — where the truth lives

- **Live state** (what is active right now / next / blockers) → `knowledge/hot.md` (injected
  automatically at SessionStart). The curated block is updated by hand or via `/close`.
- **Memory** (durable lessons) — native auto-memory; volatile state — hot.md only. One home per fact.

## Vault (second brain)

- `knowledge/` = an Obsidian vault. In Obsidian open ONLY `knowledge/`, not the project root.
- Do NOT create vault files unless asked; do NOT restructure folders. `[[wikilinks]]` are pointers
  for the reader (the harness does not resolve them).
- Layout: `daily_logs/` journals · `decisions/dec-*.md` decisions · `architecture/` standards ·
  `runbooks/` procedures · `templates/` templates.

## Commands (dev)

- Install the dev environment: `pip install -e ".[dev]"` (runtime deps stay empty until the
  hardware is fixed).
- Tests: `pytest` · one test: `pytest tests/test_x.py::test_name` · with coverage: `pytest --cov=src`.
- Lint: `ruff check src tests` · autofix: `ruff check --fix` · format: `ruff format`.

## Path-scoped rules

- Bulky instructions for specific paths → `.claude/rules/*.md` with `paths:` frontmatter
  (0 tokens at startup; sample: `.claude/rules/_TEMPLATE.md`). Keep only always-needed rules here.

<!-- ─── /init fold-in: code map / stack — filled in by native /init; afterwards apply the
     removal test, total file ≤200 lines ─── -->
