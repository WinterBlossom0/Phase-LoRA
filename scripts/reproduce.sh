#!/usr/bin/env bash
# The paper's whole experiment, end to end, on one 16 GB GPU (RTX 5080 Laptop GPU): about 28.5 GPU-hours, measured
# (training 22.5 h, test scoring 4.2 h, base models 0.8 h, forgetting about 1 h at 13 s per run). Then the analysis on the CPU (under 1 min).
# Environment: `uv sync --extra dev` builds .venv from uv.lock with the exact versions of the runs; run this script
# with that environment active, or set PYTHON. Network is needed only for the data and the two base models.
# Every step skips the work whose result file exists, so a rerun after a crash resumes.
# run_sweep.py takes the GPU lock per training run; score_test.py takes it itself, so it is never wrapped in flock
# (that deadlocks). PHASE_LORA_GPU_LOCK names the lock file (default ~/.gpu.lock). Stops at the first failing step.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
LOG=runs/reproduce.log
PY="${PYTHON:-python3}"
export PHASE_LORA_GPU_LOCK="${PHASE_LORA_GPU_LOCK:-$HOME/.gpu.lock}"
step() { echo "=== $(date '+%F %T %Z') $*" | tee -a "$LOG"; }

step "data (pinned; checked against data_manifest.json) and base models (pinned; weights checked by sha256)"
if ! "$PY" scripts/prepare_data.py --check; then
    "$PY" scripts/prepare_data.py 2>&1 | tee -a "$LOG"
fi
"$PY" scripts/prepare_models.py 2>&1 | tee -a "$LOG"

step "base models: greedy, repetition-blocked translation, test loss"
"$PY" scripts/score_test.py --base --modes greedy norepeat loss 2>&1 | tee -a "$LOG"

step "grid, selection, main runs"
"$PY" scripts/run_sweep.py --phase all 2>&1 | tee -a "$LOG"

step "lists of the main runs to score"
"$PY" scripts/run_sweep.py --phase list > runs/score_all.txt
"$PY" scripts/run_sweep.py --phase list --datasets svamp arc > runs/score_greedy.txt
"$PY" scripts/run_sweep.py --phase list --datasets mafand_tsn mafand_lug --seeds 1 >> runs/score_greedy.txt
mapfile -t ALL < runs/score_all.txt
mapfile -t GREEDY < runs/score_greedy.txt
echo "${#ALL[@]} main runs; ${#GREEDY[@]} greedy scores" | tee -a "$LOG"
if [ "${#ALL[@]}" -eq 0 ] || [ "${#GREEDY[@]}" -eq 0 ]; then
    echo "empty run list" | tee -a "$LOG"
    exit 1
fi

step "repetition-blocked translation scores and test loss of every main run"
"$PY" scripts/score_test.py --modes norepeat loss --runs "${ALL[@]}" 2>&1 | tee -a "$LOG"

step "greedy scores: SVAMP and ARC every seed, translation seed 1 (runaway share)"
"$PY" scripts/score_test.py --modes greedy --runs "${GREEDY[@]}" 2>&1 | tee -a "$LOG"

step "forgetting: WikiText-103 test loss of every main run and both base models"
flock "$PHASE_LORA_GPU_LOCK" "$PY" scripts/forgetting.py 2>&1 | tee -a "$LOG"

step "analysis (CPU): paper/generated/"
CUDA_VISIBLE_DEVICES="" "$PY" scripts/paper_analysis.py 2>&1 | tee -a "$LOG"

step "done"
