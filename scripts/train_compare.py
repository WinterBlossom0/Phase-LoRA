"""One training run: one method, rank, seed, dataset and model, same budget for every arm.

  python3 scripts/train_compare.py --kind polar --model qwen3-0.6b-base --rank 8 \
      --alpha 8 --lr 1e-3 --train 1000000 --eval 1000000 --epochs 5 \
      --eval-every 35 --max-tokens 3072 --grad-ckpt on --seed 1 --dataset mafand_tsn \
      --clip 1.0 --patience 0

The result file runs/<tag>.json is written last and marks the run complete; a run
whose result file exists is skipped, so a crashed sweep resumes where it stopped.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing

# Batch shapes change every step. With the default CUDA cache this reserved 30 GB for 6.6 GB of tensors
# (measured), and WSL moved the excess to host RAM without an error. Expandable segments stop the fragmentation
# (PyTorch CUDA notes); the 0.9 cap turns any spill into an out-of-memory error. Set before torch loads.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,per_process_memory_fraction:0.9"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_lora import MODEL_DIR, RUNS_DIR
from phase_lora.run_config import TRAIN_CODE, code_version, parse, run_tag
from phase_lora.train_utils import (EVAL_SPLIT, attach, build_data, collate, landing_modules, make_optimizer,
                                   token_batches)


@torch.no_grad()
def evaluate(model, rows, pad, batches):
    model.eval()
    tot = n = 0
    for batch in batches:
        ids, lab, att = (t.cuda() for t in collate([rows[j] for j in batch], pad))
        k = (lab != -100).sum().item()
        tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.item() * k
        n += k
    model.train()
    return tot / n


def write_atomic(path, write):
    """Write to a sibling temp file, then rename: a crash never leaves a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def main():
    # read at start: a stamp read at the end would credit an edit made mid-run
    version = code_version(TRAIN_CODE)
    a = parse()
    settings = vars(a)
    tag = run_tag(settings)
    result_path, ckpt, partial = RUNS_DIR / f"{tag}.json", RUNS_DIR / f"{tag}.pt", RUNS_DIR / f"{tag}.pt.partial"
    if result_path.exists():
        print(f"done already, skipping: runs/{tag}.json")
        return
    # weights without a result file come from an unfinished attempt at this exact run;
    # left in place, a rerun that never beats the base would report them as its own
    for leftover in (ckpt, partial):
        leftover.unlink(missing_ok=True)

    torch.manual_seed(a.seed)
    model_dir = MODEL_DIR / a.model
    if not model_dir.is_dir():
        sys.exit(f"no such model: {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tr = build_data(tok, "train", a.train, a.dataset)
    ev = build_data(tok, EVAL_SPLIT[a.dataset], a.eval, a.dataset)
    g = torch.Generator().manual_seed(a.seed)
    # every epoch's batches up front: the cosine schedule needs the exact step count
    plan = [token_batches([len(x) for x, _ in tr], a.max_tokens, g) for _ in range(a.epochs)]
    ev_batches = token_batches([len(x) for x, _ in ev], a.max_tokens, torch.Generator().manual_seed(0))
    steps = sum(len(epoch) for epoch in plan)
    rows_per_step = [len(batch) for epoch in plan for batch in epoch]

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda()
    model.requires_grad_(False)
    model = attach(model, a.kind, a.rank, a.alpha, lam=a.lam)
    if a.grad_ckpt == "on":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()

    params = [q for q in model.parameters() if q.requires_grad]
    # Run the batch with the most padded tokens once, so a budget too big for the GPU
    # fails here, not hours in. No optimizer step, so the weights do not change.
    heaviest = max((batch for epoch in plan for batch in epoch),
                   key=lambda batch: len(batch) * max(len(tr[j][0]) for j in batch))
    ids, lab, att = (t.cuda() for t in collate([tr[j] for j in heaviest], pad))
    model(input_ids=ids, attention_mask=att, labels=lab).loss.backward()
    for q in params:
        q.grad = None
    opt = make_optimizer(params, a.lr, a.clip, landing=landing_modules(model))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    print(f"{a.kind}: {sum(q.numel() for q in params)/1e3:.1f}K trainable, "
          f"{len(tr)} examples, {a.max_tokens} tokens/batch, "
          f"{min(rows_per_step)}-{max(rows_per_step)} rows/step, {steps} steps over {a.epochs} epochs")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    save_seconds = []

    def save_best():
        """Trainable params only. Written on every new best, so the checkpoint is the
        model the reported eval number describes. It stays .partial until the run
        completes, so a crashed run never leaves weights that look finished. Timed
        apart from training: arms differ in how often they set a new best."""
        save_start = time.time()
        weights = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
        write_atomic(partial, lambda path: torch.save(weights, path))
        save_seconds.append(time.time() - save_start)

    base_loss = evaluate(model, ev, pad, ev_batches)
    print(f"  step    0  ep 0.00  eval {base_loss:.4f}  (before training)")

    hist, t0, run, eval_seconds = [(0, base_loss)], time.time(), 0.0, []
    best, bad, stopped, diverged = base_loss, 0, None, None
    schedule = [(ep, i, batch) for ep, epoch in enumerate(plan) for i, batch in enumerate(epoch)]
    for s, (ep, i, batch) in enumerate(schedule):
        ids, lab, att = (t.cuda() for t in collate([tr[j] for j in batch], pad))
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        if not torch.isfinite(loss):
            diverged = s + 1
            print(f"  step {s+1:5d}  DIVERGED"); break
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run = loss.item() if s == 0 else 0.98 * run + 0.02 * loss.item()
        if (s + 1) % a.eval_every == 0 or s + 1 == steps:
            eval_start = time.time()
            e = evaluate(model, ev, pad, ev_batches)
            eval_seconds.append(time.time() - eval_start)
            hist.append((s + 1, e))
            print(f"  step {s+1:5d}  ep {ep + (i+1)/len(plan[ep]):4.2f}  eval {e:.4f}  "
                  f"train {run:.4f}  {time.time()-t0:.0f}s")
            # a new best must be lower by more than 1e-4
            if e < best - 1e-4:
                best, bad = e, 0
                save_best()
            else:
                bad += 1
                if a.patience and bad >= a.patience:
                    stopped = s + 1
                    print(f"  early stop at {stopped}: no new best in {a.patience} evals")
                    break

    # wall seconds; train excludes evals and checkpoint saves; loss.item() syncs the GPU every step
    train_seconds = time.time() - t0 - sum(eval_seconds) - sum(save_seconds)
    if partial.exists():
        os.replace(partial, ckpt)
    result = {"args": settings, "tag": tag, "code_version": version, "hist": hist,
              "best_eval": best, "stopped_at": stopped, "diverged_at": diverged,
              "weights": ckpt.name if ckpt.exists() else None, "rows_per_step": rows_per_step,
              "train_seconds": train_seconds, "eval_seconds": eval_seconds, "save_seconds": save_seconds,
              "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
    # written last: its existence is what marks the run complete
    write_atomic(result_path, lambda path: path.write_text(json.dumps(result)))
    print(f"  best eval {best:.4f}   -> runs/{tag}.json")
    print(f"  weights -> runs/{ckpt.name}" if ckpt.exists() else "  weights: never improved on base, nothing saved")
    print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GB used, "
          f"{torch.cuda.max_memory_reserved()/2**30:.2f} GB reserved")


if __name__ == "__main__":
    main()
