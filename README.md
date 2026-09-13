# Phase LoRA

**One question:** if you write LoRA's two factor matrices in polar form — a
magnitude and an angle per entry, instead of a single signed number — does the
model train any better?

Nothing else changes. Same model, same data, same number of trainable
parameters, same learning rate, same everything. Only the coordinates the
optimizer moves in.

The short answer, from 72 training runs: **it depends on the dataset, and the
effect is small either way.** It helps on English instruction data, it hurts on
English→Yoruba translation. There is no consistent win.

## Why it might have worked

Standard LoRA learns `ΔW = B·A`. If some component of that update needs to flip
sign during training, the only way there is *through zero* — and the gradient of
each factor is proportional to the other, so both shrink together and the flip
stalls right at the crossing.

In polar coordinates a sign flip is just a rotation by π at constant magnitude.
The angle's gradient is largest exactly where the magnitude's gradient vanishes,
so the two take turns instead of dying together.

That's the hypothesis. The experiment below is what happened when it was tested
against real baselines instead of assumed.

## What's being compared

Six methods, each in a plain ("real") version and a phase version:

| family | real version | phase version | what it constrains |
|---|---|---|---|
| LoRA | `additive` | `polar` | nothing — plain low-rank update |
| PoLAR | `stiefel` | `phase_stiefel` | factors kept near-orthonormal by a penalty |
| StelLA | `stella` | `phase_stella` | factors kept exactly orthonormal by a retraction |

PoLAR ([Lion et al., NeurIPS 2025](https://arxiv.org/abs/2506.03133)) and StelLA
are published methods, reimplemented here from their papers. The LoRA baseline is
stock [PEFT](https://github.com/huggingface/peft), not a reimplementation, so the
comparison isn't against a strawman.

**Parameter counts are matched, not approximately.** A complex number costs two
real numbers, so every phase variant is built at half the requested rank
internally: `--rank 64` means complex rank 32, which spans real rank 64 and costs
exactly the same bytes as real LoRA at rank 64. Scaling follows the same rule
(`α/2r` against real LoRA's `α/r`). At rank 64 that's 9.2M trainable parameters
for every arm, within 3%.

## Setup

- **Model:** Qwen3-0.6B-Base, frozen, bf16. Adapters on `q_proj` and `v_proj` in
  all 28 layers.
- **Datasets:** [no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
  (English instruction following, 5 epochs) and
  [OPUS-100 en→yo](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
  (English→Yoruba translation, 9773 examples, 10 epochs, last 600 rows held out).
- **Optimizer:** AdamW, lr 1e-3, cosine decay, batch 5, max length 384, gradient
  checkpointing, gradient clip 1.0.
- **Loss is masked to assistant replies only** — the model is never scored on the
  prompt it was given.
- **Early stopping:** eval every 500 steps, stop after 5 evals with no new best.
- 3 seeds per cell. Hardware: one RTX 5080, ~4.7 GB peak, ~70 min per Yoruba run.

## Results

Numbers are **eval loss — lower is better.** Every cell is 3 seeds, mean ± sample
standard deviation.

### English→Yoruba (10 epochs, untrained model scores 5.76)

| method | rank 8 | rank 64 |
|---|---|---|
| LoRA | **0.8566 ± 0.0021** | 0.8609 ± 0.0025 |
| Phase LoRA | 0.8719 ± 0.0052 | 0.8546 ± 0.0015 |
| StelLA | 0.8683 ± 0.0026 | **0.8501 ± 0.0013** |
| Phase StelLA | 0.8793 ± 0.0018 | 0.8557 ± 0.0015 |
| PoLAR | 0.8919 ± 0.0014 | 0.8529 ± 0.0010 |
| Phase PoLAR | 0.9149 ± 0.0044 | 0.8571 ± 0.0043 |

At rank 8 the phase version is **worse in all three families** — by 0.015, 0.011
and 0.023 nats, all far outside seed noise. At rank 64 it's a wash: Phase LoRA
beats LoRA by 0.006, the other two lose by about 0.005.

Also worth noting: plain LoRA is the *only* method that gets worse going from rank
8 to rank 64. The orthonormality-constrained methods all improve substantially
with rank, which is the effect PoLAR's paper is about.

### English instruction following (5 epochs)

| method | rank 8 | rank 64 |
|---|---|---|
| LoRA | 2.4550 ± 0.0004 | 2.5748 ± 0.0054 |
| Phase LoRA | 2.4390 ± 0.0006 | 2.4897 ± 0.0033 |
| StelLA | 2.4355 ± 0.0002 | 2.4742 ± 0.0022 |
| Phase StelLA | 2.4335 ± 0.0003 | 2.4651 ± 0.0025 |
| PoLAR | **2.4291 ± 0.0011** | 2.4478 ± 0.0015 |
| Phase PoLAR | 2.4298 ± 0.0005 | **2.4419 ± 0.0008** |

Here the phase version wins or ties in 5 of 6 pairs, and the rank-64 gaps are
large — Phase LoRA beats LoRA by 0.085 nats. Opposite sign to Yoruba.

**Read those two tables together and the honest conclusion is that the polar
chart is not a free win.** It changes the optimization path enough to matter, but
which direction it moves depends on the task.

## Known caveats

Reported because they'd be reasonable things for a reader to object to.

1. **The gradient clip is not applied identically to every arm.** Training clips
   gradients to norm 1.0, then calls the optimizer. For StelLA and Phase StelLA
   an optimizer hook replaces the gradient with its projected version *after* the
   clip, so those two arms effectively train with a looser clip than the other
   four. This matters most for the Yoruba rank-64 column, where StelLA is the best
   result. Whether it changed any number depends on how often the clip actually
   bound, which isn't recorded in the logs.

2. **The landing penalty λ is not matched across the no_robots runs.** PoLAR used
   λ=0.005 there and Phase PoLAR used λ=0.001, so that one pair isn't a clean
   comparison on that dataset. The Yoruba runs use λ=0.1 for both, picked by a
   grid search over {1e-3, 5e-3, 0.1} at 2 epochs on seed 0.

3. **One model, two datasets, three seeds.** Enough to say the effect is real and
   direction-dependent; not enough to say why.

## Running it

Requires a CUDA GPU and a local copy of the base model in `model/`.

```bash
pip install -e ".[dev]"

# plain LoRA baseline
python scripts/train_compare.py --kind additive --rank 64 --alpha 64 \
  --lr 1e-3 --dataset yoruba --epochs 10 --seed 0

# the phase version of it
python scripts/train_compare.py --kind polar --rank 64 --alpha 64 \
  --lr 1e-3 --dataset yoruba --epochs 10 --seed 0

# PoLAR needs a landing penalty; there is no default on purpose
python scripts/train_compare.py --kind stiefel --rank 64 --alpha 64 --lam 0.1 \
  --lr 1e-3 --dataset yoruba --epochs 10 --seed 0
```

Results land in `runs/<tag>.json` as `{args, hist, stopped_at}`, where `hist` is a
list of `(step, eval_loss)`. The `runs/` directory is not tracked in git.

Convention throughout: `--alpha` is set equal to `--rank`, so the scaling factor
is 1.

Self-checks for the adapter math:

```bash
pytest                                      # polar and unitary demos
python src/phase_lora/stiefel.py            # PoLAR / StelLA demos, needs a GPU
```

## Layout

```
src/phase_lora/
  polar.py        Phase LoRA — the polar reparametrisation of B·A
  stiefel.py      PoLAR and StelLA, real and phase versions
  unitary.py      block-diagonal orthogonal/unitary adapters (an earlier dead end)
  train_utils.py  data prep, chat masking, adapter attachment
  paths.py
scripts/train_compare.py   one run, one method, one seed
tests/test_adapters.py
model/                     Qwen3-0.6B-Base checkpoint (not in git)
runs/                      result JSONs (not in git)
```

`unitary.py` is kept because it's the experiment that motivated the rest: forcing
the factors onto the unitary group *shrinks* the reachable set and lost to its own
matched control (2.5445 vs 2.5004). That's why the phase variants here constrain
the stacked real factors instead of imposing a complex constraint — the goal is to
change the chart, not the hypothesis class.

## License

MIT — see [LICENSE](LICENSE).
