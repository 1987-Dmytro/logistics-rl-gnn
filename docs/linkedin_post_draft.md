# LinkedIn post drafts — Dynamic Pharmacy Delivery Routing (GNN + RL)

> Drafts only — final review by the author. Every number is from a durable seeded artifact.
> Honesty rules enforced: **no static/dynamic conflation**, **no "we beat OR-Tools"** (static quality
> is *within 3.4 % of* OR-Tools; the edge vs OR-Tools is *reaction latency*, a different setting).

---

## Variant A — business / "before → after" (EN)

**Before → after on real-world delivery routing.**

I built a dynamic route planner for same-day medication delivery to 62 pharmacies in Augsburg — on
the real OpenStreetMap road network, with real opening-hours time windows, not a toy grid.

On the static daily plan, vs the status-quo greedy dispatcher:
• **−23.5 % operating cost**
• **−39.6 % vehicle-hours on duty** (mostly cutting idle waiting at closed time windows — same
  kilometers, far less waiting)
• Lands **within 3.4 %** of a strong static OR-Tools reference

And in the dynamic setting — a mid-day disruption (traffic, breakdown, urgent order), replanning the
residual problem — it **re-plans in under 0.7 s**, about **2.9× faster** than an OR-Tools re-solve,
so a dispatcher gets an answer in the time it takes to read the alert.

Stack: Graph Neural Network + Reinforcement Learning (POMO) for construction, a candidate portfolio,
and classical local-search polish — every number seeded and reproducible.

**Honest caveat:** the static quality here comes from the classical local-search polish, not the
neural net — given the same wall-clock, OR-Tools actually edges out the system on static cost. Where
GNN+RL genuinely earns its place is **reaction speed**: an instant re-plan when the day changes.

Write-up, code, and the honest trade-offs: https://github.com/1987-Dmytro/logistics-rl-gnn

#OperationsResearch #ReinforcementLearning #Logistics #GraphNeuralNetworks #VehicleRouting

---

## Variant B — technical / research honesty (EN)

**An honest negative result is still a result.**

I spent this project asking one question on real Augsburg delivery data (62 pharmacies, real OSM
network): **does a GNN + RL policy actually beat classical routing on quality?**

Short answer: **no — and I made that hard to fake.**
• The final "RL-by-quality" test was a **pre-registered gate committed to git before the training
  code** (one run, no retry). It failed.
• Comparisons were **paired** on identical seeds; a byte-identical greedy candidate made "never worse
  than greedy" true *by construction*, not by trust.
• A **time-matched** benchmark removed the usual "you under-budgeted OR-Tools" objection: given equal
  wall-clock, OR-Tools matches the system's static quality in under a second.

So where does the neural machinery genuinely help? **Reaction latency and candidate diversity.** In
the dynamic setting it re-plans in ~0.7 s vs ~2 s for an OR-Tools re-solve — an edge in *speed of
reaction*, not in solution cost.

Meanwhile the deployed system still delivers **−23.5 % cost vs the status-quo heuristic**, within
3.4 % of a strong static OR-Tools reference (a 30 s budget — a reference, not a proven optimum). The
quality lever turned out to be classical local search, not the policy — and saying so plainly is the
point.

Full decision log, methodology, and reproducible numbers: https://github.com/1987-Dmytro/logistics-rl-gnn

#ReinforcementLearning #OperationsResearch #MachineLearning #ResearchIntegrity #VehicleRouting

---

