"""Forgetting of every finished main run: the language-modelling loss of the fine-tuned model on the WikiText-103
test set, against the loss of the untrained base model on the same text.

  python scripts/forgetting.py                         # every main run, then the two base models
  python scripts/forgetting.py --runs runs/<tag>.json  # chosen runs
  python scripts/forgetting.py --check                 # adapter swap equals a fresh load (CPU-cheap, GPU-short)

Protocol (the usual WikiText perplexity setup, as in GPTQ): the 4,358 rows of the test split are joined with
"\\n\\n", tokenized without special tokens and cut into non-overlapping 2,048-token windows; the remainder after the
last full window is dropped. Every window is scored on its own: the loss is the mean negative log-likelihood of
tokens 2..2048 given the tokens before them, averaged over all scored tokens of all windows. Forgetting is the
fine-tuned loss minus the base model's loss.

Speed without a change to the result: each base model is loaded once per adapter kind, and the runs of that kind
are swapped in by loading their saved weights into the same adapter modules. --check confirms on real runs that
a swapped-in adapter gives the same loss, bit for bit, as score_test.load_run's fresh load.
Results go to <runs>/forgetting/<tag>.json; a run whose result exists is skipped, so a restart resumes.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # run without installing
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_sweep
import score_test
from phase_lora import MODEL_DIR, RUNS_DIR
from phase_lora.run_config import TRAIN_CODE, code_version, train_version_ok
from phase_lora.train_utils import attach

WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-103-raw-v1/test-00000-of-00001.parquet")
TEXT_SHA256 = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
WINDOW = 2048
# PHASE_LORA_FORGETTING_OUT sends smoke tests elsewhere, so their results never mix with the real ones
OUT_DIR = Path(os.environ.get("PHASE_LORA_FORGETTING_OUT") or RUNS_DIR / "forgetting")


def wikitext_windows(tok):
    """(windows, n_tokens): the test text as a (n, WINDOW) LongTensor of token ids."""
    repository, revision, filename = WIKITEXT
    path = hf_hub_download(repository, filename, repo_type="dataset", revision=revision)
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    found = hashlib.sha256(text.encode()).hexdigest()
    if found != TEXT_SHA256:
        sys.exit(f"WikiText test text sha256 {found}, expected {TEXT_SHA256}")
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
    count = len(ids) // WINDOW
    return ids[:count * WINDOW].view(count, WINDOW), len(ids)


@torch.inference_mode()
def mean_nll(model, windows, batch_size, limit=None):
    """Mean next-token negative log-likelihood (nats) over every scored token of the windows."""
    windows = windows[:limit]
    total, count = 0.0, 0
    for start in range(0, len(windows), batch_size):
        ids = windows[start:start + batch_size].cuda()
        logits = model(input_ids=ids).logits
        # one window at a time in float32: a whole batch of float32 logits would not fit (vocabulary 151,936)
        for row in range(ids.shape[0]):
            total += F.cross_entropy(logits[row, :-1].float(), ids[row, 1:], reduction="sum").item()
            count += ids.shape[1] - 1
    return total / count, count


def write_atomic(path, data):
    """Write then rename: a preempted container never leaves a partial file that a restart would skip."""
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=1))
    os.replace(partial, path)


def fresh_base(model_name):
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR / model_name, dtype=torch.bfloat16).cuda()
    model.requires_grad_(False)
    return model


def swap_in(model, result):
    """Load the run's best weights into the adapter modules; every trainable tensor must come from the file."""
    weights = torch.load(RUNS_DIR / result["weights"], map_location="cuda")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(weights) != trainable:
        sys.exit(f"{result['tag']}: saved weights {len(weights)} tensors, adapter has {len(trainable)} trainable tensors")
    model.load_state_dict(weights, strict=False)
    model.eval()        # train(False) also clears the Phase DoRA factor cache (polar.PhaseDoRALinear.train)
    return model


def group_runs(paths):
    """(model, kind, rank) -> run results, so one base load serves every run of an adapter kind."""
    groups = defaultdict(list)
    for path in paths:
        result = json.loads(Path(path).read_text())
        if not train_version_ok(result.get("code_version"), result["args"]["kind"]):
            sys.exit(f"{result['tag']}: trained by code {result.get('code_version')}, not a valid version")
        if result["weights"] is None:
            sys.exit(f"{result['tag']}: no saved weights")
        args = result["args"]
        groups[(args["model"], args["kind"], args["rank"])].append(result)
    for key, results in groups.items():
        if len({result["args"]["alpha"] for result in results}) != 1:
            sys.exit(f"{key}: runs of one adapter kind differ in alpha, so one adapter build cannot serve them all")
    return groups


def record(tag, args, nll, tokens, n_text_tokens, base_nll):
    return {"tag": tag, "args": args, "metric": "WikiText-103 test mean NLL (nats/token)", "nll": nll,
            "perplexity": float(torch.tensor(nll).exp()), "base_nll": base_nll,
            "forgetting": None if base_nll is None else nll - base_nll,
            "scored_tokens": tokens, "text_tokens": n_text_tokens, "window": WINDOW,
            "wikitext": {"repository": WIKITEXT[0], "revision": WIKITEXT[1], "file": WIKITEXT[2], "sha256": TEXT_SHA256},
            "code_version": code_version(TRAIN_CODE + [Path(__file__).resolve()]),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version()}


def measure(paths, batch_size):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = group_runs(paths)
    for model_name in sorted({key[0] for key in groups}):
        tok = AutoTokenizer.from_pretrained(MODEL_DIR / model_name)
        windows, n_text_tokens = wikitext_windows(tok)
        base_path = OUT_DIR / f"base_{model_name}.json"
        if base_path.exists():
            base_nll = json.loads(base_path.read_text())["nll"]
        else:
            started = time.time()
            base = fresh_base(model_name)
            base_nll, tokens = mean_nll(base, windows, batch_size)
            del base
            torch.cuda.empty_cache()
            write_atomic(base_path, record(f"base_{model_name}", {"model": model_name}, base_nll, tokens,
                                          n_text_tokens, None))
            print(f"base {model_name}: nll {base_nll:.4f} ({time.time() - started:.0f}s)", flush=True)
        for (name, kind, rank), results in sorted(groups.items()):
            if name != model_name:
                continue
            todo = [result for result in results if not (OUT_DIR / f"{result['tag']}.json").exists()]
            if not todo:
                continue
            first = todo[0]["args"]
            # lam only shapes training (the landing step); the forward pass does not read it
            model = attach(fresh_base(model_name), kind, rank, first["alpha"], lam=first["lam"])
            for result in todo:
                started = time.time()
                nll, tokens = mean_nll(swap_in(model, result), windows, batch_size)
                write_atomic(OUT_DIR / f"{result['tag']}.json",
                             record(result["tag"], result["args"], nll, tokens, n_text_tokens, base_nll))
                print(f"{result['tag']}: nll {nll:.4f}, forgetting {nll - base_nll:+.4f} "
                      f"({time.time() - started:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()


def check(paths, batch_size, limit):
    """For each adapter kind: swap two runs into one model; the second must score exactly as a fresh load_run."""
    groups = group_runs(paths)
    failures = 0
    for (model_name, kind, rank), results in sorted(groups.items()):
        if len(results) < 2:
            continue
        first, second = results[0], results[1]
        tok = AutoTokenizer.from_pretrained(MODEL_DIR / model_name)
        windows, _ = wikitext_windows(tok)
        model = attach(fresh_base(model_name), kind, rank, first["args"]["alpha"], lam=first["args"]["lam"])
        swap_in(model, first)
        mean_nll(model, windows, batch_size, limit)
        swapped, _ = mean_nll(swap_in(model, second), windows, batch_size, limit)
        del model
        _, fresh_model = score_test.load_run(second)
        fresh, _ = mean_nll(fresh_model, windows, batch_size, limit)
        del fresh_model
        torch.cuda.empty_cache()
        same = swapped == fresh
        failures += not same
        print(f"{model_name} {kind} r{rank}: swapped {swapped:.10f} fresh {fresh:.10f} {'EQUAL' if same else 'DIFFERENT'}",
              flush=True)
    if failures:
        sys.exit(f"{failures} adapter kinds differ between swap and fresh load")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", help="result files; default: every main run of the design")
    parser.add_argument("--batch-size", type=int, default=4, help="windows per forward pass")
    parser.add_argument("--check", action="store_true", help="swap-versus-fresh equality on the first windows only")
    parser.add_argument("--check-windows", type=int, default=4)
    a = parser.parse_args()
    paths = a.runs or [line for line in run_sweep_list()]
    if a.check:
        check(paths, a.batch_size, a.check_windows)
    else:
        measure(paths, a.batch_size)


def run_sweep_list():
    """The result files of every main run of the design, as run_sweep --phase list prints them."""
    every = {model: run_sweep.eval_every(model) for model in run_sweep.MODELS}
    chosen = json.loads(run_sweep.SELECTION.read_text())
    return [str(RUNS_DIR / f"{run_sweep.run_tag(vars(run_sweep.parse(argv)))}.json")
            for argv in run_sweep.main_runs(every, chosen)]


if __name__ == "__main__":
    main()
