"""The whole experiment: learning-rate grid, selection, then the main runs. Resumable.

  python scripts/run_sweep.py --phase grid     # 1-epoch grid, seed 1
  python scripts/run_sweep.py --phase main     # 5 epochs at the selected settings
  python scripts/run_sweep.py --phase all
  python scripts/run_sweep.py --phase all --dry-run   # list runs, start nothing
  python scripts/run_sweep.py --phase list [--datasets ...] [--seeds ...]   # result files of the main runs

The design has two blocks. Main: three adapter families, each with its phase twin, at rank 64 on both
models, all datasets, 5 seeds. Ablation: the LoRA family at rank 8 on the 0.6B model, 3 seeds.

Every run is one train_compare.py process under the shared GPU lock. A run whose
result file exists is skipped, so after a crash the same command resumes. A failed
run is logged to runs/sweep_failures.log and the sweep moves on; at the end of the phase
each failed run gets one retry, and a run that fails twice makes the sweep exit non-zero. Main runs go seed by seed, so a sweep stopped early still holds
complete sets of every method for the seeds it finished.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # run without installing

import torch
from transformers import AutoTokenizer

from phase_lora import MODEL_DIR, RUNS_DIR
from phase_lora.run_config import LAM_KINDS, TRAIN_CODE, code_version, parse, run_tag, train_version_ok
from phase_lora.train_utils import build_data, token_batches

# The protocol. Each value is a decision of the owner.
# The whole design runs end to end in about 28 GPU-hours on one 16 GB laptop GPU (see scripts/reproduce.sh).
MODELS = ["qwen3-0.6b-base", "qwen3-1.7b-base"]
DATASETS = ["mafand_tsn", "mafand_lug", "svamp", "arc"]
FAMILIES = {"LoRA": ("additive", "polar"), "DoRA": ("dora", "phase_dora"), "PoLAR": ("stiefel", "phase_stiefel")}
KINDS = [kind for pair in FAMILIES.values() for kind in pair]
SEEDS = [1, 2, 3, 4, 5]
BLOCKS = [{"models": MODELS, "kinds": KINDS, "ranks": [64], "seeds": SEEDS},                  # main
          {"models": ["qwen3-0.6b-base"], "kinds": list(FAMILIES["LoRA"]), "ranks": [8], "seeds": [1, 2, 3]}]   # rank ablation
EPOCHS = 5
# 1e-2 because PoLAR's paper tunes it at 4e-3 with a 4x larger update scale (alpha 32, r 8)
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2]
LAM_GRID = [1e-3, 5e-3, 0.1, 0.5]   # PoLAR landing penalty, in the units of the official PoLAR code
GRID_EPOCHS, GRID_SEED = 1, 1
EVALS_PER_EPOCH = 2
FIXED = {"max-tokens": 3072, "clip": 1.0, "patience": 0, "grad-ckpt": "on",
         "train": 1_000_000, "eval": 1_000_000}   # train/eval above every split size: all rows

GPU_LOCK = os.environ.get("PHASE_LORA_GPU_LOCK", str(Path.home() / ".gpu.lock"))   # one GPU job at a time
MIN_FREE_RAM_KB = 1_572_864            # 1.5 GB: below this WSL can crash, so the sweep stops
SELECTION = RUNS_DIR / "selected_hparams.json"
FAILURES = RUNS_DIR / "sweep_failures.log"
# Set by SIGTERM, SIGHUP or Ctrl-C. A handler that raised could fire between Popen
# and the try that stops the child, and leave the run alive with no owner.
STOP = threading.Event()


def eval_every(model):
    """Steps between dev evals per dataset, so each epoch gets EVALS_PER_EPOCH evals."""
    tok = AutoTokenizer.from_pretrained(MODEL_DIR / model)
    out = {}
    for dataset in DATASETS:
        lengths = [len(x) for x, _ in build_data(tok, "train", FIXED["train"], dataset)]
        steps = len(token_batches(lengths, FIXED["max-tokens"], torch.Generator().manual_seed(0)))
        out[dataset] = math.ceil(steps / EVALS_PER_EPOCH)
    return out


def argv_for(model, dataset, kind, rank, lr, lam, seed, epochs, every):
    values = {**FIXED, "kind": kind, "model": model, "dataset": dataset, "rank": rank, "alpha": rank,
              "lr": lr, "seed": seed, "epochs": epochs, "eval-every": every}
    if kind in LAM_KINDS:
        values["lam"] = lam
    return [item for key, value in values.items() for item in (f"--{key}", str(value))]


def block_arms(block):
    """(model, dataset, rank, kind) of every arm of a design block."""
    for model in block["models"]:
        for dataset in DATASETS:
            for rank in block["ranks"]:
                for kind in block["kinds"]:
                    yield model, dataset, rank, kind


def grid_runs(every):
    for block in BLOCKS:
        for model, dataset, rank, kind in block_arms(block):
            for lr in LR_GRID:
                for lam in (LAM_GRID if kind in LAM_KINDS else [None]):
                    yield argv_for(model, dataset, kind, rank, lr, lam, GRID_SEED, GRID_EPOCHS, every[model][dataset])


def select(every):
    """Lowest best dev loss in the grid, per model, dataset, kind and rank."""
    check_code_version(list(grid_runs(every)))
    chosen, missing = {}, []
    for argv in grid_runs(every):
        settings = vars(parse(argv))
        path = RUNS_DIR / f"{run_tag(settings)}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        result = json.loads(path.read_text())
        score = math.inf if result["diverged_at"] else result["best_eval"]
        key = "|".join(str(settings[k]) for k in ("model", "dataset", "kind", "rank"))
        candidates = chosen.setdefault(key, {"candidates": []})["candidates"]
        candidates.append({"lr": settings["lr"], "lam": settings["lam"], "best_eval": score})
    if missing:
        sys.exit(f"grid incomplete, {len(missing)} runs missing (first: {missing[0]}); run --phase grid")
    all_diverged = [key for key, entry in chosen.items()
                    if all(c["best_eval"] == math.inf for c in entry["candidates"])]
    if all_diverged:
        sys.exit(f"every grid setting diverged for {all_diverged}; no setting to select")
    for entry in chosen.values():
        entry["selected"] = min(entry["candidates"], key=lambda c: c["best_eval"])
    partial = SELECTION.with_name(SELECTION.name + ".tmp")     # a kill mid-write must not leave half a file
    partial.write_text(json.dumps(chosen, indent=1))
    os.replace(partial, SELECTION)
    return chosen


def main_runs(every, chosen):
    for seed in SEEDS:
        for block in BLOCKS:
            if seed not in block["seeds"]:
                continue
            for model, dataset, rank, kind in block_arms(block):
                pick = chosen[f"{model}|{dataset}|{kind}|{rank}"]["selected"]
                yield argv_for(model, dataset, kind, rank, pick["lr"], pick["lam"], seed, EPOCHS, every[model][dataset])


def free_ram_kb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def run_one(argv, log_path):
    """One run under the GPU lock. Returns its exit code, or None if RAM ran low."""
    command = ["flock", GPU_LOCK, sys.executable, str(ROOT / "scripts" / "train_compare.py"), *argv]
    with open(log_path, "a") as log:          # a retry keeps the first attempt's traceback
        # own process group, so the watchdog stops flock, the launcher and python together
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        try:
            while True:
                if STOP.is_set():
                    sys.exit("stopped by a signal")
                if free_ram_kb() < MIN_FREE_RAM_KB:
                    return None
                try:
                    # returns when the run ends, without a polling delay
                    return child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # the group does not receive the terminal's Ctrl-C, so stop it here whenever
            # this function leaves with the run still alive (low RAM, a stop signal, any error)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()


def check_code_version(runs):
    """Stop if a finished run was trained by code that is neither current nor recorded as valid for its kind."""
    current = code_version(TRAIN_CODE)
    stale = []
    for argv in runs:
        settings = vars(parse(argv))
        path = RUNS_DIR / f"{run_tag(settings)}.json"
        if path.exists() and not train_version_ok(json.loads(path.read_text()).get("code_version"), settings["kind"]):
            stale.append(path.name)
    if stale:
        sys.exit(f"{len(stale)} finished runs were trained by other code (current {current}), "
                 f"e.g. {stale[0]}. Delete them to retrain, or restore that code; results are never mixed.")


def execute(runs, label, dry_run):
    check_code_version(runs)
    todo = [argv for argv in runs if not (RUNS_DIR / f"{run_tag(vars(parse(argv)))}.json").exists()]
    print(f"{label}: {len(runs)} runs, {len(runs) - len(todo)} already done, {len(todo)} to run")
    if dry_run:
        for argv in todo[:5]:
            print("   ", " ".join(argv))
        return
    (RUNS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    failed = run_all(todo, label)
    if failed:
        # one retry recovers a transient fault; a run that fails twice needs a person
        print(f"{label}: retrying {len(failed)} failed runs once", flush=True)
        failed = run_all(failed, f"{label} retry")
    check_code_version(runs)    # again: catches a code edit made while this phase ran
    if failed:
        sys.exit(f"{label}: {len(failed)} runs failed twice (see {FAILURES}); later steps need every run")
    print(f"{label}: finished, 0 failed")


def run_all(todo, label):
    """Run each argv in turn; return the ones that failed."""
    failed = []
    for number, argv in enumerate(todo, 1):
        if STOP.is_set():
            sys.exit("stopped by a signal")
        tag = run_tag(vars(parse(argv)))
        start = time.time()
        code = run_one(argv, RUNS_DIR / "logs" / f"{tag}.log")
        if code is None:
            sys.exit(f"stopped: available RAM fell below {MIN_FREE_RAM_KB // 1024} MB during {tag}")
        if code != 0:
            failed.append(argv)
            with open(FAILURES, "a") as log:
                log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  exit {code}  {tag}\n")
        print(f"  {label} [{number}/{len(todo)}] {'FAILED' if code else 'ok'}  {time.time() - start:5.0f}s  {tag}",
              flush=True)
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["grid", "main", "all", "list"], required=True)
    p.add_argument("--dry-run", action="store_true", help="list the runs left to do, start nothing")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS, help="--phase list: only these")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="--phase list: only these")
    a = p.parse_args()
    if a.phase == "list":
        every = {model: eval_every(model) for model in MODELS}
        chosen = json.loads(SELECTION.read_text())
        for argv in main_runs(every, chosen):
            settings = vars(parse(argv))
            if settings["dataset"] in a.datasets and settings["seed"] in a.seeds:
                print(RUNS_DIR / f"{run_tag(settings)}.json")
        return
    # kill and a closed terminal would end the process without unwinding; the flag makes
    # run_one leave through its finally, which stops the training run in progress
    for stop_signal in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        # a signal that nohup set to ignore stays ignored, so a closed terminal cannot stop a nohup batch
        if signal.getsignal(stop_signal) != signal.SIG_IGN:
            signal.signal(stop_signal, lambda _number, _frame: STOP.set())
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    every = {model: eval_every(model) for model in MODELS}
    print(f"evals every {every} steps")
    if a.phase in ("grid", "all"):
        execute(list(grid_runs(every)), "grid", a.dry_run)
    if a.phase in ("main", "all"):
        if a.dry_run and not SELECTION.exists():
            print("main: needs the finished grid to pick each method's settings")
            return
        execute(list(main_runs(every, select(every))), "main", a.dry_run)


if __name__ == "__main__":
    main()
