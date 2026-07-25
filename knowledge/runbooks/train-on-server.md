---
type: runbook
id: run-train-on-server
date: 2026-07-21
tags: [runbook, training, server, pomo, phase6b]
---

# Training on the Ryzen server (NJ)

**Purpose:** a reproducible full training run (POMO / dynamic) on the NJ server
(Ryzen 9 9900X, 12C/24T, 96 GB, CPU-only). The code travels from the Mac via git, the data via rsync,
the run happens in tmux, and the weights and metrics come back. The Mac is for smoke runs only; full runs
happen here.

> **Environment variables (set them for yourself):**
> `SERVER` — the server's SSH host · `REPO=~/logistics-rl-gnn` — the repo path on the server ·
> `SNAP=augsburg_20260720` — the active data snapshot · `TRAIN=scripts/train_pomo.py` —
> the phase's current train script (for Step 2 replace it with the dynamic runner).

## Preconditions

- `ssh $SERVER` works.
- The current commit is **pushed** from the Mac (the server pulls a version from history, not the working tree).
- The snapshot `data/snapshots/$SNAP/` is built on the Mac (otherwise — Phase 2 `scripts/build_snapshot.py`).

## Steps

### 1. Synchronise the code (git)

```bash
# on the Mac
git push
# on the server
ssh $SERVER
cd $REPO && git pull
```

### 2. Synchronise the data (rsync — the snapshot is outside git)

The snapshot is gitignored (prohibition #1), so we transfer it explicitly. This guarantees the data are
**identical**: rebuilding would hit OSM again and could diverge → breaking the comparison with baselines.

```bash
# on the Mac
rsync -av data/snapshots/$SNAP/  "$SERVER:$REPO/data/snapshots/$SNAP/"
```

### 3. The environment (once)

```bash
# on the server, in $REPO
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheels (AMD)
pip install -e ".[data,env,baselines,model]"
```

### 4. Environment parity

```bash
pytest -q     # the same green as on the Mac — it confirms the version and environment agree
```

### 5. Launch in tmux

```bash
tmux new -s train
# threads = physical cores (12); compare 12 vs 24 (SMT) on your own load
OMP_NUM_THREADS=12 python -u $TRAIN 2>&1 | tee results/pomo_$(date +%Y%m%d).log
# Ctrl-b d — detach (the run keeps going); tmux attach -t train — come back
```

The server never sleeps — `caffeinate` is not needed (unlike on the MacBook Air).

### 6. Monitoring

```bash
tail -f results/pomo_*.log  # ep N|train(g±%)|val(g±%[OR±%])|mem±%|H||g||std|es N/patience|fr HASH
```

Healthy signs: `train g↓` · `|g|>0` · entropy neither 0 nor maximal · `std>0` (the shared baseline is not
degenerate) · `fr` (the instance hash) changes every epoch (the RNG is fresh, the draws do not repeat) ·
`mem` does NOT grow monotonically (growth = memorisation: train↓ with a frozen val). **The gate for the
first ~15 epochs:** if `val` is flat (within the noise) from about epoch 3 and `es` only accumulates without
improving → the selection signal is weak (val saturated at the heuristic, as in the previous run) → raise
`val_n_range=(62,62)` (the lever in config/pomo.py — val/test at the deployment size) and restart.
`es N/patience` → early-stop.

### 7. Collect the result

```bash
# on the Mac (the refit writes policy_pomo_refit.pt — the glob takes it and the old best 770.4€)
rsync -av "$SERVER:$REPO/results/policy_pomo_*.pt"  results/
rsync -av "$SERVER:$REPO/results/pomo_*.log"        results/
```

The weights stay outside git (prohibition #1) → only metrics/the summary + a decision with the numbers are
committed. The refit writes into `policy_pomo_refit.pt` — the old `policy_pomo_best.pt` (770.4€) is NOT
overwritten (it is needed for comparison).

## Reproducibility (mandatory)

Log next to the weights: `seed`, the config (N_starts / epochs / lr / clip), the versions
(`torch`, `torch-geometric`, `ortools`) — following the pattern of `results/baselines.json`.
Without that the run cannot be repeated and the "after" cannot be defended.

## Troubleshooting

- **torch pulls CUDA/a heavy build** → install strictly with `--index-url …/whl/cpu` **before** `pip install -e`.
- **torch-geometric does not build** → torch first (CPU), then PyG; check the version compatibility.
- **Slow / throttling** → tune `OMP_NUM_THREADS` (12 physical cores usually beat 24 with SMT); use `htop` to
  make sure it is not swapping.
- **`pytest` red on the server, green on the Mac** → the versions diverged: compare `pip freeze` for the key
  packages, rebuild `.venv`.

## Links

- Training: [[0006-pomo-static]] · [[0003-phase6-training]]
- Data/snapshot: [[0001-mdp-spec]] (Phase 2)
