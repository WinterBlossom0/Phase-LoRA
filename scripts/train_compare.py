"""Vanilla LoRA vs Phase LoRA (polar), same base, same data, same budget.

  torch-python scripts/train_compare.py --kind additive
  torch-python scripts/train_compare.py --kind polar --epochs 5 --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_lora import MODEL_DIR, RUNS_DIR
from phase_lora.stiefel import install_phase_stella, install_stella
from phase_lora.train_utils import attach, build_data, collate

@torch.no_grad()
def evaluate(model, rows, pad, bs=4):
    model.eval()
    tot = n = 0
    for i in range(0, len(rows), bs):
        ids, lab, att = (t.cuda() for t in collate(rows[i : i + bs], pad))
        k = (lab != -100).sum().item()
        tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.item() * k
        n += k
    model.train()
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["additive", "unitary", "orthogonal", "polar", "stiefel", "phase_stiefel", "stella", "phase_stella"], required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--b-init", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train", type=int, default=100_000)   # clamped to the split size
    p.add_argument("--eval", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--bs", type=int, default=5)          # peak throughput, both adapters
    p.add_argument("--no-ckpt", action="store_true", help="disable gradient checkpointing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", choices=["no_robots", "yoruba"], default="no_robots")
    p.add_argument("--lam", type=float, default=None,
                   help="PoLAR landing penalty; required for stiefel kinds, taken from the grid")
    p.add_argument("--patience", type=int, default=5,
                   help="stop after this many evals with no new best (0 disables)")
    a = p.parse_args()
    if a.kind in ("stiefel", "phase_stiefel") and a.lam is None:
        p.error(f"--lam is required for --kind {a.kind}")

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    pad = tok.pad_token_id or tok.eos_token_id
    tr = build_data(tok, "train", a.train, a.max_len, a.dataset)
    ev = build_data(tok, "test", a.eval, a.max_len, a.dataset)

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda()
    model.requires_grad_(False)
    model = attach(model, a.kind, a.rank, a.alpha, a.eps, a.b_init, lam=a.lam)
    if not a.no_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    if a.kind == "stella":
        install_stella(model, opt)          # Alg. 1 lines 6, 8-9 as optimizer hooks
    if a.kind == "phase_stella":
        install_phase_stella(model, opt)    # ...the same, through the chart
    per_epoch = len(tr) // a.bs
    steps = per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    print(f"{a.kind}: {sum(q.numel() for q in params)/1e3:.1f}K trainable, "
          f"{len(tr)} examples, {per_epoch} steps/epoch x {a.epochs} = {steps} steps")

    base_loss = evaluate(model, ev, pad)
    print(f"  step    0  ep 0.00  eval {base_loss:.4f}  (before training)")

    hist, t0, run = [(0, base_loss)], time.time(), 0.0
    best, bad, stopped = base_loss, 0, None
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(a.epochs):
        order = torch.randperm(len(tr), generator=g).tolist()     # reshuffle each epoch
        for i in range(per_epoch):
            s = ep * per_epoch + i
            batch = [tr[j] for j in order[i * a.bs : (i + 1) * a.bs]]
            ids, lab, att = (t.cuda() for t in collate(batch, pad))
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                print(f"  step {s+1:5d}  DIVERGED"); break
            run = loss.item() if s == 0 else 0.98 * run + 0.02 * loss.item()
            if (s + 1) % a.eval_every == 0 or s + 1 == steps:
                e = evaluate(model, ev, pad)
                hist.append((s + 1, e))
                print(f"  step {s+1:5d}  ep {(s+1)/per_epoch:4.2f}  eval {e:.4f}  "
                      f"train {run:.4f}  {time.time()-t0:.0f}s")
                # 1e-4 counts as a real improvement; the eval noise band is ~1e-2, so
                # patience must absorb several non-improving evals. Patience 3 would
                # have cut the 3-epoch polar run at step 3000, one eval before its
                # actual best at 3500.
                if e < best - 1e-4:
                    best, bad = e, 0
                else:
                    bad += 1
                    if a.patience and bad >= a.patience:
                        stopped = s + 1
                        print(f"  early stop at {stopped}: no new best in {a.patience} evals")
                        break
        else:
            continue
        break

    lam_tag = f"_lam{a.lam}" if a.kind in ("stiefel", "phase_stiefel") else ""
    # rank AND dataset in the tag, or a yoruba run at the same hyperparameters
    # silently overwrites the completed no_robots baseline
    ds_tag = "" if a.dataset == "no_robots" else f"_{a.dataset}"
    tag = (f"{a.kind}_r{a.rank}_lr{a.lr}_e{a.epochs}{lam_tag}" if a.kind in ("unitary", "orthogonal", "polar", "stiefel", "phase_stiefel", "stella", "phase_stella")
           else f"{a.kind}_r{a.rank}_a{a.alpha}_lr{a.lr}_e{a.epochs}"
                + (f"_b{a.b_init}" if a.kind == "phase" else "")) + ds_tag + f"_s{a.seed}"
    json.dump({"args": vars(a), "hist": hist, "stopped_at": stopped}, open(RUNS_DIR / f"{tag}.json", "w"))
    print(f"  best eval {min(e for _, e in hist):.4f}   -> runs/{tag}.json")
    print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GB")


if __name__ == "__main__":
    main()
