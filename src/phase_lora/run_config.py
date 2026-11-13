"""One run's settings and file name, shared by train_compare.py and run_sweep.py."""

import argparse
import hashlib
import json

from .paths import ROOT
from .train_utils import EVAL_SPLIT

# The files whose content decides what a training run computes, and what a score is.
TRAIN_CODE = sorted((ROOT / "src" / "phase_lora").glob("*.py")) + [ROOT / "scripts" / "train_compare.py"]
SCORE_CODE = TRAIN_CODE + [ROOT / "scripts" / "score_test.py"]


def code_version(files):
    """Hash of the given source files. Stored in every result, so a sweep resumed after
    a code change cannot silently mix results from two versions of the code."""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:12]


# Older code versions whose results stay valid for some kinds or datasets, with the reason.
# Kept outside the hashed files, so recording an entry does not change the hash it refers to.
LEGACY_VERSIONS = json.loads((ROOT / "code_versions.json").read_text())


def train_version_ok(version, kind):
    """The run was trained by the current code, or by a recorded older version valid for its kind."""
    return version == code_version(TRAIN_CODE) or kind in LEGACY_VERSIONS["train"].get(version, {}).get("kinds", [])


KINDS = ["additive", "polar", "dora", "phase_dora", "stiefel", "phase_stiefel"]
LAM_KINDS = ("stiefel", "phase_stiefel")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=KINDS, required=True)
    # No defaults. Every value that shapes a run is named on the command line, so
    # no arm can silently inherit a different setting than the arm it is compared to.
    p.add_argument("--model", required=True, help="folder name under models/, e.g. qwen3-0.6b-base")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--train", type=int, required=True, help="train examples, clamped to the split size")
    p.add_argument("--eval", type=int, required=True, help="eval examples")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--eval-every", type=int, required=True)
    p.add_argument("--max-tokens", type=int, required=True,
                   help="tokens per batch, padding included; the row count per batch varies "
                        "with row length and no row is ever cut")
    p.add_argument("--grad-ckpt", choices=["on", "off"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset", choices=list(EVAL_SPLIT), required=True)
    p.add_argument("--clip", type=float, required=True, help="max gradient norm")
    p.add_argument("--patience", type=int, required=True,
                   help="stop after this many evals with no new best (0 disables)")
    p.add_argument("--lam", type=float, default=None,
                   help="PoLAR landing penalty; required for stiefel kinds, meaningless for the rest")
    return p


def parse(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.kind in LAM_KINDS and args.lam is None:
        p.error(f"--lam is required for --kind {args.kind}")
    return args


def run_tag(settings):
    """File stem for one run: readable fields, then a hash of every setting, so two
    runs that differ in any setting can never share, and overwrite, one file."""
    lam = f"_lam{settings['lam']}" if settings["kind"] in LAM_KINDS else ""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    return (f"{settings['kind']}_r{settings['rank']}_lr{settings['lr']}_e{settings['epochs']}{lam}"
            f"_{settings['dataset']}_{settings['model']}_s{settings['seed']}_{digest}")
