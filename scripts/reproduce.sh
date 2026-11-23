#!/usr/bin/env bash
# The paper's whole experiment, end to end, on one 16 GB GPU (RTX 5080 Laptop GPU): about 43 GPU-hours, measured
# (Qwen3: training 21 h, test scoring 4.2 h, base models 0.8 h, forgetting about 1 h; LFM3: training 13 h, scoring 2 h,
# forgetting about 0.5 h). Then the analysis on the CPU (a few minutes).
# Environments: `uv sync --extra dev` builds .venv from uv.lock with the exact versions of the Qwen3 runs; run this
# script with it active, or set PYTHON. LFM3 needs .venv-lfm3 (see requirements-lfm3.txt; set PYTHON_LFM to move it);
# run_sweep.py starts LFM3 runs with it. Network is needed only for the data and the base models.
# Every step skips the work whose result file exists, so a rerun after a crash resumes.
# run_sweep.py takes the GPU lock per training run; score_test.py takes it itself, so it is never wrapped in flock
# (that deadlocks). PHASE_LORA_GPU_LOCK names the lock file (default ~/.gpu.lock). Stops at the first failing step.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
LOG=runs/reproduce.log
PY="${PYTHON:-python3}"
PY_LFM="${PYTHON_LFM:-.venv-lfm3/bin/python}"
LFM=lfm3-1.7b-base
export PHASE_LORA_GPU_LOCK="${PHASE_LORA_GPU_LOCK:-$HOME/.gpu.lock}"
step() { echo "=== $(date '+%F %T %Z') $*" | tee -a "$LOG"; }

step "data (pinned; checked against data_manifest.json) and base models (pinned; weights checked by sha256)"
if ! "$PY" scripts/prepare_data.py --check; then
    "$PY" scripts/prepare_data.py 2>&1 | tee -a "$LOG"
fi
"$PY" scripts/prepare_models.py 2>&1 | tee -a "$LOG"

step "base models: greedy, repetition-blocked translation, test loss"
"$PY" scripts/score_test.py --base --models qwen3-0.6b-base qwen3-1.7b-base --modes greedy norepeat loss 2>&1 | tee -a "$LOG"
"$PY_LFM" scripts/score_test.py --base --models "$LFM" --modes greedy norepeat loss 2>&1 | tee -a "$LOG"

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

# each model is scored in its own environment
of_model() { local want="$1"; shift; for run in "$@"; do [[ "$run" == *"_${want}_"* ]] && echo "$run"; done; }
not_of_model() { local skip="$1"; shift; for run in "$@"; do [[ "$run" != *"_${skip}_"* ]] && echo "$run"; done; }
mapfile -t ALL_QWEN < <(not_of_model "$LFM" "${ALL[@]}")
mapfile -t ALL_LFM < <(of_model "$LFM" "${ALL[@]}")
mapfile -t GREEDY_QWEN < <(not_of_model "$LFM" "${GREEDY[@]}")
mapfile -t GREEDY_LFM < <(of_model "$LFM" "${GREEDY[@]}")

step "repetition-blocked translation scores and test loss of every main run"
"$PY" scripts/score_test.py --modes norepeat loss --runs "${ALL_QWEN[@]}" 2>&1 | tee -a "$LOG"
"$PY_LFM" scripts/score_test.py --modes norepeat loss --runs "${ALL_LFM[@]}" 2>&1 | tee -a "$LOG"

step "greedy scores: SVAMP and ARC every seed, translation seed 1 (runaway share)"
"$PY" scripts/score_test.py --modes greedy --runs "${GREEDY_QWEN[@]}" 2>&1 | tee -a "$LOG"
"$PY_LFM" scripts/score_test.py --modes greedy --runs "${GREEDY_LFM[@]}" 2>&1 | tee -a "$LOG"

step "forgetting: WikiText-103 test loss of every main run and the base models"
flock "$PHASE_LORA_GPU_LOCK" "$PY" scripts/forgetting.py --runs "${ALL_QWEN[@]}" 2>&1 | tee -a "$LOG"
flock "$PHASE_LORA_GPU_LOCK" "$PY_LFM" scripts/forgetting.py --runs "${ALL_LFM[@]}" 2>&1 | tee -a "$LOG"

step "analysis (CPU): paper/generated/"
CUDA_VISIBLE_DEVICES="" "$PY" scripts/paper_analysis.py 2>&1 | tee -a "$LOG"

step "done"
