# Final metrics — before/after (Augsburg, seeds 0–9, full-62)

> greedy (before) vs the polished portfolio (after) on IDENTICAL instances, one scorer; OR-Tools is the upper bar. Numbers from durable artifacts (parity to decision 0002/0009).

| Metric | greedy (before) | OR-Tools | System (after) | Δ vs greedy | Δ vs OR |
|---|---:|---:|---:|---:|---:|
| Costs, € | 825.4 | 611.1 | **631.6** | **-23.5%** | +3.4% |
| Distance, km (fuel proxy) | 158.8 | 144.7 | **157.3** | **-1.0%** | — |
| Vehicle-hours on duty* | 19.3 | 11.2 | **11.6** | **-39.6%** | — |
| Vehicles used | 7.2 | 6.3 | **6.4** | -11.1% | — |
| On-time, % | 100 | 100 | 100 | — | — |
| Unserved | 0.0 | 0.0 | 0.0 | — | — |
| Re-plan latency | 7 ms | 2001 ms | 689 ms | — | **×2.9 faster** |

\*Vehicle-hours = travel + idle waiting for windows + service. The gain **-39.6%** comes almost entirely from **less idling** (windows): distance is ~flat (-1.0%), service is identical (the same 62 pharmacies). This saves DUTY HOURS (labor), not kilometers.

**Bottom line:** costs **-23.5%** and duty hours **-39.6%** vs greedy (window planning, not distance — it is ~flat), within **+3.4%** of OR-Tools. Event reaction: a neural start ~15ms (the speed ceiling), the deployed system (portfolio+polish) 689ms = **×2.9** vs OR-Tools AT the same quality (+3.4%). Guaranteed ≥ greedy by construction (0008).

<sub>Provenance: baselines.json (0002) · system_metrics.json (parity 0009 631.6€) · polish_summary.json (0009, re-plan latency 5×6 events; the neural floor 14–19ms — ablation 0010, quality-inferior). Statics — seeds 0–9 full-62. Outside git (#1).</sub>
## Time-matched — anytime OR-Tools vs the system (task #15)

> We give OR-Tools THE SAME wall-clock, quality measured on IDENTICAL instances (full-62, seeds 0–9, one scorer). It answers: is the system's latency edge honest in STATICS.

| OR-Tools budget | cost, € (±std) | wins/seed vs system | median Δ/seed, € |
|---|---:|---:|---:|
| 0.7s | 629.7 ± 44 | 6/10 | -4.9 |
| 2.0s | 626.1 ± 42 | 7/10 | -10.5 |
| 5.0s | 625.0 ± 42 | 7/10 | -11.2 |
| 30.0s | 610.5 ± 39 | 8/10 | -18.6 |

**System (statics):** 631.6€ at wall-clock **≥30s** (polish 30000ms/candidate ×≤3 + decode → ≥30s; candidates run in sequence, wall-clock is higher).

**Verdict (paired, same instances — the difficulty σ cancels):** by **30s** OR-Tools beats the system on **8/10** seeds, median **-18.6€/seed**; already at **0.7s** — 6/10 (median -4.9€), i.e. parity below 1s. The system has NO static advantage in quality or in latency; its edge is DYNAMICS only (re-plan on a residual, 689ms/827€), not statics.

<sub>PAIRED: median-Δ/wins on the shared seeds 0–9 (the instance σ cancels) — discipline 0010, not unpaired σ. CONFLATION QUARANTINE (Phase 8): 631.6€ is statics (≥30s polish), 689ms is the dynamic re-plan latency on a residual (cost 827€), not one point '631.6€ @ 689ms'. Provenance: timematch.json (parity 30s=611.1€/0002) + system_metrics.json.</sub>
