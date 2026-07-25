<!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->

# Hot Cache

**Auto-refreshed:** 2026-07-25 14:52:53 (every SessionStart)
**Branch:** `main`

## 🔀 Recent commits (top 5)

```
5e19af1 docs(i18n): regenerate generated docs (route_sheet, final_metrics, demo assets)
185dc21 docs(i18n): translate decision records 0001-0013, runbooks, vault prose
26195b3 chore(i18n): translate CLAUDE.md + vault scripts (+ language policy section)
82078ae chore(i18n): translate code comments and script output strings to English
afe319f docs(readme): оговорка о train-распределении weekday в Custom scenarios
```

## 📋 Recent decisions

- `0013-time-matched-benchmark.md` — 0013 — Time-matched: OR-Tools anytime vs the system (task #15, the final benchmark)
- `0012-pathB-residual-verdict.md` — 0012 — Path B: the verdict on the pre-registration (Phase 6b)
- `0011-pathB-residual-curriculum-prereg.md` — 0011 — Path B: residual curriculum, PRE-REGISTRATION (Phase 6b)

## 📅 Recent daily logs

- `2026-07-25.md`
- `2026-07-23.md`
- `2026-07-22.md`

<!-- AUTO-GEN END (everything below preserved across refreshes) -->

# Hot Cache — curated

**Last update:** 2026-07-25 11:34 (/save; edited by hand / `/close`; the section above is auto — do NOT touch the marker)

## 🔥 What's Hot
- **The demo became PROVABLE (Phase 9 acceptance, commit `1d9bc5f`):** a provenance banner (ckpt + sha256 +
  the training date + [[0007-phase6b-congestion-training]]; the sha is checked against `system_metrics`, a
  mismatch / no weights → `SystemExit`, there is NO silent fallback) · the portfolio candidate table in [2/5]
  and [4/5] · `--no-model` (a portfolio without RL candidates) + the line "with the model vs without".
  **The lesson:** a counterfactual must NOT be extracted from the winning system — greedy+polish out of the
  portfolio with the model gave Δ ≤ 0 identically; after the fix (a separate portfolio, the full budget) the
  default prints a day plan of −52.7 € (−8.2 %), and a re-plan of **+5.8 € AGAINST the model** — the durable
  verdict [[0012-pathB-residual-verdict]] reproduced in one run. `pytest 156` · adversarial-verified (27 agents, 8 findings).
- **Custom scenarios:** `scenarios/*.yaml` + `config/scenario.py` (weekday → the real
  `opening_hours` + `c(dow,h)`, pharmacies by name/id/all, demand overrides, the fleet K/Q, a chain of
  events; validated by the existing guards + `max(demand) ≤ Q`). `demo.py --scenario` — the same
  pipeline and rendering; **the default Tuesday is byte for byte as before** (587.9 € / 578.7 €). Ready:
  `friday_south` (14 pharmacies, K=2 → 201.3 €) and `monday_rush` (62 + 2 incidents → inaction ∞).
  K/Q live in `Instance.meta` (`im.fleet_of`): the env/polish/OR-Tools defaults bind at `def` time.
- **The Phase 8 human-readable layer is DONE (3 commits `fbf28d9`+`1e349cd`+`a1e3b0b`):** `route_sheet.py`
  (a human-readable plan from the eval pipeline, the parity guard cost==system_metrics per_seed, the real
  pharmacy names 62/62 via `enrich_names.py`) · `demo.py` (a 5-step narrative demonstration) ·
  **`compare.html`** — the screencast frame: A/B/C (do-nothing/OR-Tools/the system in ONE residual world)
  + two iframes B|C, every hop along real streets (`nx.shortest_path` over graph.graphml). All outputs →
  `demo_out/` (outside git). The header numbers == the demo output, statics/residual separated, OR@2s honest
  (not converged; at ~30s it wins). **Adversarial-verified** (7 agents: 2 confirmed+fixed — the map price is
  tied to the drawn plan; the vacuous test strengthened). `pytest 136` · `ruff` clean.
- **The project is in publishable form (Phase 9 done):** README (a full EN rewrite), LICENSE (MIT),
  pyproject metadata, `.github/workflows/ci.yml`, `docs/linkedin_post_draft.md` (2 variants × EN/RU),
  the honest table `docs/final_metrics.md`. The numbers come ONLY from durable artefacts, statics/dynamics
  quarantined. Assembled+**adversarial-verified** by a workflow (the verify caught a red CI and an overclaim in
  LinkedIn — both fixed). Commits `36b8a72`+`219fd7b`. **Before publication: fill in
  `<user>`/`<repo-url>`** (the README bibtex, pyproject urls, 4 LinkedIn blocks — `TODO(publish)`).
- **The research outcome — honest, the "RL=quality" thesis is CLOSED ([[honest-asymmetric-verdicts]]):**
  Phase 6b (0007→0008→0009→0010→[[0012-pathB-residual-verdict]]) + time-matched
  ([[0013-time-matched-benchmark]]). On QUALITY RL does not beat the classics, and **there is no static latency
  edge** (OR-Tools time-matched: parity at <1s 6/10, at 30s it wins 8/10, median −18.6€/seed). The only
  durable contribution of GNN+RL is **the dynamic reaction** (a re-plan in 689ms vs OR's 2001ms). Quality comes
  from polish, not from the policy (after polish RL is even +1.0% worse than greedy).
- **Key durable numbers:** greedy **825.4€** · OR-Tools@30s **611.1€** · the system **631.6€**
  (**−23.5%** vs greedy, **+3.4%** vs OR; vehicle-hours **−39.6%** = window idling, NOT distance ~flat).
  Reaction: the system **689ms** vs OR **2001ms** (×2.9). One scorer `env/scoring.py:evaluate_solution`
  ([[0002-baselines]], `results/*.json` outside git).
- **Do not bring back (lessons):** the decoder WITHOUT `C·tanh` + a normalised advantage (else the Phase 6
  collapse); polish feasibility STRICTLY as the env (a per-customer return ≤ T_max — else env-infeasible under asymmetric OSM).

## ⏭️ Next
- **Before publication:** fill in `<user>`/`<repo-url>` in README/pyproject/LinkedIn; after the push to
  GitHub CI activates (`.[dev,data,env]`, the heavy tests are skipped — green on a bare runner).
- **Push:** `main` is ahead of `origin` by **10** commits (a614509, 67a75d6, 0d739dd, 086d75c, 36b8a72,
  219fd7b, fbf28d9, 1e349cd, a1e3b0b, **1d9bc5f**) — not pushed (committing was requested).
- Optional: record a screencast over `compare.html` (the frame was updated: row C names the REAL
  portfolio winner, "reaction" is marked as a durable median); a LinkedIn post (after the author's
  review); optionally a deployment layer (the RL start as an anytime candidate, quality carried by multistart+polish).
- Optional: your own `scenarios/*.yaml` for a specific conversation (Friday/Saturday/a narrow cluster/an urgent
  order) — the schema and the loader are ready, README §Custom scenarios.
- **The research arc is CLOSED** — do not open a new RL-"quality" branch without a new lever.

## 🚧 Blockers
- none
