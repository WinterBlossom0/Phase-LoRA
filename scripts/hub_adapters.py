"""The trained adapters on the Hugging Face Hub: export, check, upload, and load.

  python scripts/hub_adapters.py export <stage dir>          # every main run -> <stage>/phase-lora/
  python scripts/hub_adapters.py check <stage dir> [--runs N] # staged files rebuild the trained models exactly
  python scripts/hub_adapters.py upload <stage dir> <user>     # one public repo: <user>/phase-lora

Load one adapter (LFM3 needs the transformers of requirements-lfm3.txt):

  from hub_adapters import load
  tok, model = load("WinterBlossom/phase-lora", "qwen3-1.7b-base", "svamp", "phase-lora", seed=1)

Layout: <model>/<task>/<method>/seed-<n>/adapter.safetensors and config.json. The adapters are the trainable
tensors only, under the names that phase_lora.train_utils.attach gives them; config.json holds what attach needs.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # run without installing
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_lora import RUNS_DIR
from phase_lora.train_utils import attach

GITHUB = "https://github.com/WinterBlossom0/Phase-LoRA"
PAPER = "https://doi.org/10.5281/zenodo.24393045"
# Hub repository and pinned revision of each base model, as scripts/prepare_models.py downloads them
BASES = {"qwen3-0.6b-base": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"),
         "qwen3-1.7b-base": ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
         "lfm3-1.7b-base": ("LiquidAI/LFM3-1.7B-Base", "7bca5038823c0d0f14efe06439109354a10e014c")}
TASKS = {"mafand_tsn": "en-setswana", "mafand_lug": "en-luganda", "svamp": "svamp", "arc": "arc-challenge"}
TASK_TITLES = {"mafand_tsn": "English→Setswana (chrF++)", "mafand_lug": "English→Luganda (chrF++)",
               "svamp": "SVAMP (accuracy %)", "arc": "ARC-Challenge (accuracy %)"}
# code name of each adapter kind -> public method name
METHODS = {"additive": "lora", "polar": "phase-lora", "dora": "dora", "phase_dora": "phase-dora",
           "stiefel": "polar", "phase_stiefel": "phase-polar"}
METHOD_TITLES = {"additive": "LoRA", "polar": "Phase LoRA", "dora": "DoRA", "phase_dora": "Phase DoRA",
                 "stiefel": "PoLAR", "phase_stiefel": "Phase PoLAR"}
# The score each task reports: repetition-blocked decoding for translation, greedy decoding for the rest
SCORE_DIR = {"mafand_tsn": "test_norepeat", "mafand_lug": "test_norepeat", "svamp": "test", "arc": "test"}
LICENSES = {"qwen3-0.6b-base": "Apache-2.0 (Qwen3)", "qwen3-1.7b-base": "Apache-2.0 (Qwen3)",
            "lfm3-1.7b-base": "LFM Open License v1.0 (LFM3)"}


def main_runs():
    """Every main run of the design, from runs/score_all.txt (run_sweep --phase list)."""
    return [json.loads((ROOT / line).read_text()) for line in (RUNS_DIR / "score_all.txt").read_text().split()]


REPO = "phase-lora"


def folder(result):
    args = result["args"]
    return Path(args["model"]) / TASKS[args["dataset"]] / METHODS[args["kind"]] / f"seed-{args['seed']}"


def run_scores(result):
    """(test score, forgetting) of one run; forgetting is None if not measured."""
    tag, dataset = result["tag"], result["args"]["dataset"]
    test = json.loads((RUNS_DIR / SCORE_DIR[dataset] / f"{tag}.json").read_text())["score"]
    forgetting_path = RUNS_DIR / "forgetting" / f"{tag}.json"
    forgetting = json.loads(forgetting_path.read_text())["forgetting"] if forgetting_path.exists() else None
    return test, forgetting


def export(stage):
    by_model = {}
    for result in main_runs():
        args = result["args"]
        model = args["model"]
        target = stage / REPO / folder(result)
        target.mkdir(parents=True, exist_ok=True)
        weights = torch.load(RUNS_DIR / result["weights"], map_location="cpu")
        save_file({name: tensor.contiguous() for name, tensor in weights.items()}, target / "adapter.safetensors")
        test, forgetting = run_scores(result)
        repository, revision = BASES[model]
        config = {"method": METHOD_TITLES[args["kind"]], "kind": args["kind"], "base_model": repository,
                  "base_revision": revision, "task": TASKS[args["dataset"]], "rank": args["rank"],
                  "alpha": args["alpha"], "lam": args["lam"], "learning_rate": args["lr"], "epochs": args["epochs"],
                  "seed": args["seed"], "test_score": test, "test_metric": TASK_TITLES[args["dataset"]],
                  "forgetting_nats_per_token": forgetting, "run_tag": result["tag"],
                  "training_code_version": result["code_version"], "code": GITHUB}
        (target / "config.json").write_text(json.dumps(config, indent=1) + "\n")
        by_model.setdefault(model, []).append((args["dataset"], args["kind"], test, forgetting))
    (stage / REPO / "README.md").write_text(model_card(by_model))
    print(f"{REPO}: " + ", ".join(f"{model} {len(rows)}" for model, rows in by_model.items()) + " adapters")


def score_table(rows):
    mean = {}
    for dataset, kind, test, forgetting in rows:
        mean.setdefault((dataset, kind), []).append(test)
    lines = ["| Method | " + " | ".join(TASK_TITLES[d] for d in TASKS) + " |", "|---" * (len(TASKS) + 1) + "|"]
    for kind in METHODS:
        cells = [f"{statistics.mean(mean[(dataset, kind)]):.1f}" for dataset in TASKS]
        lines.append(f"| {METHOD_TITLES[kind]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def model_card(by_model):
    total = sum(len(rows) for rows in by_model.values())
    bases = "\n".join(f"- {BASES[model][0]}" for model in by_model)
    tables = "\n\n".join(f"**{BASES[model][0]}** (`{model}/`)\n\n{score_table(rows)}" for model, rows in by_model.items())
    licenses = "; ".join(sorted(set(LICENSES[model] for model in by_model)))
    return f"""---
license: other
license_name: mixed
base_model:
{bases}
tags:
- lora
- phase-lora
- adapter
---

# Phase LoRA: every trained adapter

All {total} adapters of the paper [Phase LoRA: Low-Rank Adapters in Polar Coordinates]({PAPER}):
{len(by_model)} base models × 4 tasks × 6 methods × 5 seeds, rank 64 on `q_proj` and `v_proj`.
Code: [{GITHUB}]({GITHUB}).

Mean test score over 5 seeds:

{tables}

## Layout

`<model>/<task>/<method>/seed-<n>/` holds `adapter.safetensors` and `config.json` (settings, test score, forgetting).
Models: {", ".join(f"`{model}`" for model in by_model)}. Tasks: `en-setswana`, `en-luganda`, `svamp`,
`arc-challenge`. Methods: `lora`, `phase-lora`, `dora`, `phase-dora`, `polar`, `phase-polar`.

## Use

```bash
git clone {GITHUB} && cd Phase-LoRA && uv sync
```
```python
import sys; sys.path[:0] = ["scripts", "src"]
from hub_adapters import load
tok, model = load("WinterBlossom/{REPO}", "qwen3-1.7b-base", "svamp", "phase-lora", seed=1)
```

LFM3 loads with transformers 5.18.0 or newer (`requirements-lfm3.txt` in the code repository). Its 13 convolution
layers have no `q_proj` or `v_proj`, so its adapters sit on its 11 attention layers.

## License

Base models: {licenses}. Training data: MAFAND-MT (CC BY-NC 4.0, so the translation adapters are for
non-commercial use), SVAMP (MIT), ARC (CC BY-SA 4.0).

## Citation

```bibtex
@misc{{ovalekar2026phaselora,
  title     = {{Phase {{LoRA}}: Low-Rank Adapters in Polar Coordinates}},
  author    = {{Ovalekar, Vihaan}},
  year      = {{2026}},
  publisher = {{Zenodo}},
  doi       = {{10.5281/zenodo.24393045}},
  url       = {{{PAPER}}}
}}
```
"""


def build(config, weights, device="cuda"):
    """Base model at its pinned revision plus the adapter, as phase_lora.train_utils.attach builds it."""
    tok = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["base_revision"],
                                                 dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.requires_grad_(False)
    torch.manual_seed(config["seed"])          # as training; every tensor is then replaced by the saved weights
    model = attach(model, config["kind"], config["rank"], config["alpha"], lam=config["lam"])
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(weights) != trainable:
        raise ValueError(f"adapter holds {len(weights)} tensors, the rebuilt model has {len(trainable)} trainable")
    model.load_state_dict({name: tensor.to(device) for name, tensor in weights.items()}, strict=False)
    model.eval()
    return tok, model


def load(repository, model, task, method, seed=1, device="cuda"):
    """Download one adapter from the Hub and return (tokenizer, model)."""
    from huggingface_hub import hf_hub_download
    prefix = f"{model}/{task}/{method}/seed-{seed}"
    config = json.loads(Path(hf_hub_download(repository, f"{prefix}/config.json")).read_text())
    weights = load_file(hf_hub_download(repository, f"{prefix}/adapter.safetensors"))
    return build(config, weights, device)


def check(stage, count):
    """Staged tensors equal the trained ones bit for bit, for every run; for `count` runs of each kind and model,
    the model built from the staged files gives the same logits as score_test.load_run, bit for bit."""
    import score_test
    results = main_runs()
    for result in results:
        staged = load_file(stage / REPO / folder(result) / "adapter.safetensors")
        trained = torch.load(RUNS_DIR / result["weights"], map_location="cpu")
        if staged.keys() != trained.keys() or not all(torch.equal(staged[k], trained[k]) for k in trained):
            sys.exit(f"{result['tag']}: staged tensors differ from the trained ones")
    print(f"{len(results)} staged adapters equal the trained tensors")
    chosen = {}
    for result in results:
        key = (result["args"]["model"], result["args"]["kind"])
        if len(chosen.setdefault(key, [])) < count:
            chosen[key].append(result)
    prompt = "Question: A farmer has 12 cows and buys 7 more. How many cows does the farmer have?\nAnswer:"
    for (model_name, kind), picked in sorted(chosen.items()):
        for result in picked:
            target = stage / REPO / folder(result)
            config = json.loads((target / "config.json").read_text())
            tok, hub_model = build(config, load_file(target / "adapter.safetensors"))
            ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
            with torch.no_grad():
                hub_logits = hub_model(input_ids=ids).logits
            del hub_model
            _, trained_model = score_test.load_run(result)
            with torch.no_grad():
                trained_logits = trained_model(input_ids=ids).logits
            del trained_model
            score_test.release_gpu()
            same = torch.equal(hub_logits, trained_logits)
            print(f"{model_name} {kind} seed {result['args']['seed']}: {'EQUAL' if same else 'DIFFERENT'}", flush=True)
            if not same:
                sys.exit("the staged adapter does not rebuild the trained model")


def upload(stage, user):
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = f"{user}/{REPO}"
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    # upload_large_folder resumes an interrupted upload; huggingface_hub 2.x removed it, so upload_folder there
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=repo_id, folder_path=stage / REPO, repo_type="model")
    else:
        api.upload_folder(repo_id=repo_id, folder_path=stage / REPO, repo_type="model",
                          commit_message="Trained adapters, model card")
    print(f"uploaded https://huggingface.co/{repo_id}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["export", "check", "upload"])
    parser.add_argument("stage", type=Path)
    parser.add_argument("user", nargs="?")
    parser.add_argument("--runs", type=int, default=1, help="check: runs per method and model to rebuild on the GPU")
    a = parser.parse_args()
    if a.action == "export":
        export(a.stage)
    elif a.action == "check":
        check(a.stage, a.runs)
    else:
        if not a.user:
            sys.exit("upload needs the Hub user name")
        upload(a.stage, a.user)


if __name__ == "__main__":
    main()
