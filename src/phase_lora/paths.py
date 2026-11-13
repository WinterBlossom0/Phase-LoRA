"""Project paths, so scripts and tests work from any working directory."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"     # base checkpoints live here
# result JSONs and best-eval adapter weights; the override lets a rerun check write elsewhere.
# It must be absolute: run_sweep starts training runs in ROOT, so a relative one would name
# a different folder in the sweep and in its runs.
_override = os.environ.get("PHASE_LORA_RUNS_DIR", "")
if _override and not Path(_override).is_absolute():
    raise ValueError(f"PHASE_LORA_RUNS_DIR must be an absolute path, got {_override!r}")
RUNS_DIR = Path(_override) if _override else ROOT / "runs"
DATA_DIR = ROOT / "data"        # pinned dataset splits, written by scripts/prepare_data.py
