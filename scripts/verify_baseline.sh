#!/usr/bin/env bash
# One-command check of a stored result: retrains one LoRA baseline run (rank 64, SVAMP, Qwen3-0.6B-Base, seed 1;
# about 30 s of training on one 16 GB GPU), scores it on the SVAMP test set and compares with the stored files.
#   bash scripts/verify_baseline.sh
# Uses the pinned data and model (downloaded on first use). Writes only to verify_runs/ (PHASE_LORA_RUNS_DIR).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
LOCK="${PHASE_LORA_GPU_LOCK:-$HOME/.gpu.lock}"
TAG=additive_r64_lr0.001_e5_svamp_qwen3-0.6b-base_s1_c5cb296d81
export PHASE_LORA_RUNS_DIR="$PWD/verify_runs"
mkdir -p "$PHASE_LORA_RUNS_DIR"

"$PY" scripts/prepare_data.py --check || "$PY" scripts/prepare_data.py
"$PY" scripts/prepare_models.py

# the exact arguments of the stored run, read from its result file
ARGV=$("$PY" -c '
import json, sys
args = json.load(open(sys.argv[1]))["args"]
print(" ".join(f"--{key.replace(chr(95), chr(45))} {value}" for key, value in args.items() if value is not None))
' "runs/$TAG.json")
echo "training: $ARGV"
# shellcheck disable=SC2086  # one word per argument
flock "$LOCK" timeout 1800 "$PY" scripts/train_compare.py $ARGV
"$PY" scripts/score_test.py --modes greedy --runs "$PHASE_LORA_RUNS_DIR/$TAG.json"

"$PY" - "$TAG" <<'PY'
import json, sys
tag = sys.argv[1]
stored, new = (json.load(open(f"{root}/test/{tag}.json")) for root in ("runs", "verify_runs"))
same = sum(a == b for a, b in zip(stored["items"], new["items"]))
print(f"SVAMP accuracy: stored {stored['score']:.2f}, retrained {new['score']:.2f}; "
      f"{same}/{len(stored['items'])} test items with the same answer")
PY
