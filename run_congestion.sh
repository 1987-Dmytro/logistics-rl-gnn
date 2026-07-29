#!/bin/bash
cd /root/logistics-rl-gnn
source .venv/bin/activate
LOG="results/pomo_congestion_$(date +%Y%m%d_%H%M).log"
echo "$LOG" > results/.last_cong_log
OMP_NUM_THREADS=12 python -u scripts/train_pomo.py --congestion 2>&1 | tee "$LOG"
echo "TRAIN_DONE=${PIPESTATUS[0]}" | tee -a "$LOG"
