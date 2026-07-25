---
type: decision
id: dec-2026-07-20-mdp-spec
date: 2026-07-20
status: accepted
tags: [decision, mdp, cvrptw, spec]
---

# 0001 — Dynamic CVRPTW: delivery to the pharmacies of Augsburg

## 1. Problem statement + mission
A GNN+RL agent for dynamic routing of medicine delivery to every pharmacy in
Augsburg on the real OSM network, with online re-planning on jams/breakdowns/
urgent orders. The goal is a reproducible case with reduced time and distance
against OR-Tools.

## 2. Instance input
- Road graph [REAL]: OSM Augsburg drive; free-flow from maxspeed + the German
  fallback (primary/secondary 50, residential/Tempo-30 → 30, living_street 7).
- Depot [REAL]: PHOENIX VZ, Benzstraße 10, 86391 Stadtbergen → geocode → snap
  to the nearest OSM node. One depot. A dispatch window.
- Customers [REAL locations]: pharmacies amenity=pharmacy within the Augsburg boundary.
  - demand d_i [ASSUMED]: 3–12 boxes, from a distribution.
  - service s_i [ASSUMED]: ~4 min/stop.
  - window [e_i,l_i]: from opening_hours [REAL where present], else synthetic [ASSUMED, tagged].
- Fleet [ASSUMED, in config]: homogeneous vans; K=6; Q=60 boxes;
  v_eff(e)=the road limit; tour T_max=4h; cost c_f (per vehicle) + c_d (€/km)
  + c_t (€/h); start/finish at the depot.
- Clock [REAL clock]: the episode's start datetime → (weekday, hour).

## 3. MDP
- State: statics (node features, depot) + dynamics (vehicle position, remaining
  capacity, current time, the visited/unvisited/feasible sets, partial
  routes, active events) → a GNN embedding over the current graph.
- Action: pick the next feasible node (masking) or return to the depot;
  autoregressively; vehicle routes are built sequentially.
- Reward: −(c_f·vehicles_used + c_d·distance + c_t·time
  + lateness_penalty + unserved_penalty); sparse (end of episode).
- Transition: move the vehicle; time += travel_time(e,τ)·congestion; service
  (capacity↓, waiting when arriving before e_i); mark visited;
  stochastic events may fire.
- Constraints: hard via masking — capacity, windows, return to the depot;
  soft via penalties — lateness, overtime (T_max).
- Objective: minimise the expected cost over a distribution of instances
  (the real Augsburg + perturbations).

## 4. Congestion & speed model
- t0(e) = length(e)/v(e); v from OSM maxspeed + the fallback (OSMnx add_edge_speeds
  + a German override → add_edge_travel_times).
- t(e,τ) = t0(e)·c(class(e),dow(τ),h(τ))·(1 + Σ_k I_k(e,τ)).
- c(class,dow,h) [ASSUMED, calibrated]: a deterministic weekly-hourly
  profile per road class; calibrated against the TomTom Augsburg curve. Tagged simulated-on-real.
- I_k [ASSUMED]: stochastic incidents; the spawn rate is ~Poisson and depends
  on (dow,h); magnitude δ_k (∞=closure→the edge is removed); duration/decay.
  An incident is a re-plan TRIGGER.
- Implementation: edge-level; cache the free-flow matrix τ0 and the paths P_ij; on an event
  recompute ONLY the OD pairs whose path is affected.

## 5. Eval plan
- Metrics: total time, distance (a fuel proxy), % on-time, number of vehicles,
  unserved customers.
- Baselines: OR-Tools + greedy nearest-feasible on identical instances/seed.
- Criteria: (static) RL within the target gap to OR-Tools [X% — calibrated
  in Phase 4]; (dynamic) re-plan latency ≪ an OR-Tools re-solve at comparable
  cost. Everything reproducible: seed + config + data snapshot.
- Test slices: tiny with a known optimum → a small synthetic one → the full Augsburg.

## 6. Assumptions and failure modes
- FIFO simplification: the multiplier is taken per step by departure time, strict FIFO
  is not guaranteed (an explicit TDVRP assumption).
- maxspeed coverage in OSM < 100% → a fallback; log the coverage % during data collection,
  the threshold is a Phase 2 verify gate.
- demand/fleet/service are parameterised [ASSUMED] and live in the config.
- T_max=4h = the cap on one tour. Rationale: (a) multi-tour pharma distribution operations
  in the 08–18 window (2–4 tours/day ≈ 3–4h); (b) inside EU 561/2006 (≤4.5h of continuous
  driving). It caps the FULL route time (driving+waiting+service) — a simplification against
  the pure EU driving time. Currently weakly binding (tours ~2h at K=8), it becomes active
  under congestion (Phase 7).

## 7. Backlog (not in v1)
Cold chain (2–8°C, a sub-fleet/constraint); multi-depot; a heterogeneous fleet;
a time-series forecast of c(·); unsupervised segmentation.
