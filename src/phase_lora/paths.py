"""Project paths, so scripts and tests work from any working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"      # one checkpoint. "complex" is a property of the
RUNS_DIR = ROOT / "runs"        # forward pass, not of the weights on disk.
