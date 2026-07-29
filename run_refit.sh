#!/bin/bash
cd /root/logistics-rl-gnn
source .venv/bin/activate
LOG="results/pomo_refit_$(date +%Y%m%d_%H%M).log"
echo "$LOG" > results/.last_refit_log
OMP_NUM_THREADS=12 python -u scripts/train_pomo.py 2>&1 | tee "$LOG"
echo "TRAIN_DONE=${PIPESTATUS[0]}" | tee -a "$LOG"
