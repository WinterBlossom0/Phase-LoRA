<div align="center">

# Phase LoRA

### Can complex numbers make a LoRA adapter better?

Same model. Same data. Same parameter count. The only difference is whether the
adapter's factors are **real numbers** or **complex numbers written as magnitude + angle**.

Three families of adapter, each run head-to-head against its own complex twin.
**72 training runs. 12 matchups. 3 seeds each.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-ee4c2c)
![Model](https://img.shields.io/badge/model-Qwen3--0.6B-7b3fe4)
![Runs](https://img.shields.io/badge/runs-72-success)

</div>

---

## 🏁 The verdict

> **Complex does not reliably beat real.**
>
> Across 12 head-to-head matchups the complex version won 6, lost 4, and tied 2 —
> and **five of those six wins came from the same dataset.**
>
> On English instruction following it wins or ties all six matchups, once by
> **0.085 nats — 28× the seed noise.** On English→Yoruba translation it reverses,
> hardest at rank 8, where the complex version **loses all three matchups.**
>
> So the answer isn't "complex is better" or "complex is worse." It's that the
> coordinates you write the factors in change the optimization path enough to
> matter, and which way they move depends on the task.

---

## 🤔 The idea

A standard LoRA adapter learns `ΔW = B·A`, where `B` and `A` hold plain signed
numbers. Suppose some part of that update needs to **flip sign** during training.
The only route is *through zero* — and because each factor's gradient is
proportional to the other, both shrink as they approach the crossing. The flip
stalls exactly where it needs to happen.

Write the same factors as a **magnitude and an angle** instead, and a sign flip is
just a rotation by π while the magnitude stays put. The angle's gradient is
largest precisely where the magnitude's gradient dies, so the two relay instead of
collapsing together.

That's the hypothesis. Everything below is what happened when it was tested
against real baselines instead of assumed.

---

## 🥊 The matchups

Three adapter families. Each one exists in a **real** version and a **complex
(phase)** version, and they fight only each other.

| | real version | complex version | what the factors are constrained to |
|:--|:--|:--|:--|
| **LoRA** | `additive` | `polar` | nothing — plain low-rank update |
| **PoLAR** | `stiefel` | `phase_stiefel` | pushed toward orthonormal by a penalty |
| **StelLA** | `stella` | `phase_stella` | held exactly orthonormal by a retraction |

**PoLAR** and **StelLA** are published methods, reimplemented here from their
papers — PoLAR from its Alg. 2 (the landing field), StelLA from its Alg. 1 (the
Riemannian gradient and polar retraction). Full citations are
[at the bottom](#-references). The LoRA baseline is stock
[PEFT](https://github.com/huggingface/peft), not a reimplementation, so the
comparison isn't against a strawman.

### ⚖️ The fight is weight-matched

A complex number costs two real numbers. So every complex variant is built at
**half the requested rank** internally: `--rank 64` means complex rank 32, which
spans real rank 64 and occupies exactly the same memory as real LoRA at rank 64.
Scaling follows the same rule (`α/2r` against real LoRA's `α/r`).

At rank 64 every arm trains **9.2M parameters**, all six within 3% of each other.
No arm gets to win by being bigger.

---

## 📊 Scoreboard: complex vs. real, 12 matchups

Loss, lower is better. **Δ** is complex minus real, so **negative means complex
won**. Every Δ is a paired difference across the same 3 seeds, ± its own standard
deviation. A matchup is called a **tie** when the gap is under 2σ of that paired
difference — i.e. when it can't be separated from seed noise.

### English → Yoruba translation

| matchup | rank | real | complex | Δ | winner |
|:--|:--:|--:|--:|--:|:--|
| LoRA vs Phase LoRA | 8 | **0.8566** | 0.8719 | `+0.0152 ± 0.0068` | ❌ real |
| PoLAR vs Phase PoLAR | 8 | **0.8919** | 0.9149 | `+0.0231 ± 0.0049` | ❌ real |
| StelLA vs Phase StelLA | 8 | **0.8683** | 0.8793 | `+0.0110 ± 0.0019` | ❌ real |
| LoRA vs Phase LoRA | 64 | 0.8609 | **0.8546** | `−0.0063 ± 0.0027` | ✅ **complex** |
| PoLAR vs Phase PoLAR | 64 | 0.8529 | 0.8571 | `+0.0043 ± 0.0044` | ➖ tie (1.0σ) |
| StelLA vs Phase StelLA | 64 | **0.8501** | 0.8557 | `+0.0056 ± 0.0027` | ❌ real |

### English instruction following

| matchup | rank | real | complex | Δ | winner |
|:--|:--:|--:|--:|--:|:--|
| LoRA vs Phase LoRA | 8 | 2.4550 | **2.4390** | `−0.0160 ± 0.0006` | ✅ **complex** |
| PoLAR vs Phase PoLAR | 8 | 2.4291 | 2.4298 | `+0.0007 ± 0.0006` | ➖ tie (1.2σ) |
| StelLA vs Phase StelLA | 8 | 2.4355 | **2.4335** | `−0.0020 ± 0.0003` | ✅ **complex** |
| LoRA vs Phase LoRA | 64 | 2.5748 | **2.4897** | `−0.0851 ± 0.0031` | 🏆 **complex, by a lot** |
| PoLAR vs Phase PoLAR | 64 | 2.4478 | **2.4419** | `−0.0059 ± 0.0015` | ✅ **complex** |
| StelLA vs Phase StelLA | 64 | 2.4742 | **2.4651** | `−0.0091 ± 0.0015` | ✅ **complex** |

<div align="center">

### ✅ complex 6 &nbsp;&nbsp;•&nbsp;&nbsp; ❌ real 4 &nbsp;&nbsp;•&nbsp;&nbsp; ➖ tie 2

*five of the six complex wins are on instruction following — where it never once loses*

</div>

**The pattern is the dataset, not the method.** On instruction following, complex
wins or ties every single matchup and never once loses.

Translation reverses it, and rank 8 is where the reversal is sharpest: **complex
loses all three matchups outright there** — by 0.015, 0.023 and 0.011 nats against
LoRA, PoLAR and StelLA respectively, every one of them outside seed noise. Give
the adapters rank 64 and the damage mostly stops: one win, one tie, one loss.

And the looser the constraint on the factors, the bigger the swing. Plain LoRA —
the family with no constraint at all — produces the largest effect in **both**
directions (−0.085 and +0.015). The orthonormality-constrained families barely
move either way, which suggests the constraint, not the coordinate system, is
doing most of the work there.

---

## 🥇 Which method is actually best?

Forget the complex-vs-real axis for a moment — here is every method ranked
against every other, in each condition. **Longer bar = lower loss = better.** Bars
are scaled within each panel, so they show the spread of that panel, not absolute
quality.

<table>
<tr><td valign="top">

**English → Yoruba, rank 8**

```
LoRA          0.8566  ██████████████
StelLA        0.8683  ███████████
Phase LoRA    0.8719  ███████████
Phase StelLA  0.8793  █████████
PoLAR         0.8919  ██████
Phase PoLAR   0.9149  █
```

</td><td valign="top">

**English → Yoruba, rank 64**

```
StelLA        0.8501  ██████████████
PoLAR         0.8529  ███████████
Phase LoRA    0.8546  █████████
Phase StelLA  0.8557  ███████
Phase PoLAR   0.8571  ██████
LoRA          0.8609  █
```

</td></tr>
<tr><td valign="top">

**Instruction following, rank 8**

```
PoLAR         2.4291  ██████████████
Phase PoLAR   2.4298  ██████████████
Phase StelLA  2.4335  ████████████
StelLA        2.4355  ███████████
Phase LoRA    2.4390  █████████
LoRA          2.4550  █
```

</td><td valign="top">

**Instruction following, rank 64**

```
Phase PoLAR   2.4419  ██████████████
PoLAR         2.4478  █████████████
Phase StelLA  2.4651  ████████████
StelLA        2.4742  ███████████
Phase LoRA    2.4897  █████████
LoRA          2.5748  █
```

</td></tr>
</table>

**No method wins everywhere.** PoLAR and Phase PoLAR own instruction following at
both ranks. StelLA takes translation at rank 64. Plain LoRA takes translation at
rank 8 — and finishes dead last in the other three panels. It is by far the most
task-sensitive adapter of the six, which is exactly why it also swings hardest
between its real and complex versions.

---

<details>
<summary><b>🔬 Full experimental setup</b></summary>

<br>

- **Model** — Qwen3-0.6B-Base, frozen, bf16. Adapters on `q_proj` and `v_proj` in
  all 28 layers.
- **Datasets** —
  [no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (English
  instruction following, 5 epochs) and
  [OPUS-100 en→yo](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
  (English→Yoruba translation, 9773 examples, 10 epochs, last 600 rows held out
  and never trained on).
- **Optimizer** — AdamW, lr 1e-3, cosine decay, batch 5, max length 384, gradient
  checkpointing, gradient clip 1.0.
- **Loss masking** — scored on assistant replies only; the model is never graded
  on the prompt it was handed.
- **Early stopping** — eval every 500 steps, stop after 5 evals with no new best.
- **Seeds** — 0, 1, 2 for every cell. The seed controls both init and shuffle
  order, so the Δ columns above are genuine paired differences.
- **Hardware** — one RTX 5080. ~4.7 GB peak, ~70 minutes per translation run.
- **Reference point** — the untrained model scores 5.76 on the translation eval,
  so every adapter here is doing real work.

</details>

<details>
<summary><b>▶️ Running it</b></summary>

<br>

Needs a CUDA GPU and a local copy of the base model in `model/`.

```bash
pip install -e ".[dev]"
```

One run = one method, one rank, one seed:

```bash
# real LoRA
python scripts/train_compare.py --kind additive --rank 64 --alpha 64 \
  --lr 1e-3 --dataset yoruba --epochs 10 --seed 0

# its complex twin
python scripts/train_compare.py --kind polar --rank 64 --alpha 64 \
  --lr 1e-3 --dataset yoruba --epochs 10 --seed 0

# PoLAR needs a landing penalty — there is deliberately no default
python scripts/train_compare.py --kind stiefel --rank 64 --alpha 64 --lam 0.1 \
  --lr 1e-3 --dataset yoruba --epochs 10 --seed 0
```

`--alpha` is always set equal to `--rank`, so the scaling factor is 1 everywhere.

Results land in `runs/<tag>.json` as `{args, hist, stopped_at}`, where `hist` is a
list of `(step, eval_loss)`. `runs/` is not tracked in git.

Self-checks for the adapter math:

```bash
pytest                              # polar and unitary demos
python src/phase_lora/stiefel.py    # PoLAR / StelLA demos, needs a GPU
```

</details>

<details>
<summary><b>📁 Repository layout</b></summary>

<br>

```
src/phase_lora/
  polar.py        Phase LoRA — the complex reparametrisation of B·A
  stiefel.py      PoLAR and StelLA, real and complex versions
  unitary.py      block-diagonal orthogonal/unitary adapters (an earlier dead end)
  train_utils.py  data prep, chat masking, adapter attachment
  paths.py
scripts/train_compare.py   one run: one method, one rank, one seed
tests/test_adapters.py
model/                     Qwen3-0.6B-Base checkpoint (not in git)
runs/                      result JSONs (not in git)
```

`unitary.py` is kept because it's the dead end that shaped everything else.
Forcing the factors onto the unitary group *shrinks* the set of updates the
adapter can reach, and it lost to its own matched control (2.5445 vs 2.5004).
That's why the complex variants here constrain the stacked real factors rather
than imposing a genuine complex constraint — the goal is to change the
coordinates, not the hypothesis class.

</details>

---

## 📚 References

The two published methods benchmarked here, and the geometry they rely on:

> **PoLAR: Polar-Decomposed Low-Rank Adapter Representation** <br>
> Kai Lion, Liang Zhang, Bingcong Li, Niao He — ETH Zurich, NeurIPS 2025 <br>
> 📄 [arXiv:2506.03133](https://arxiv.org/abs/2506.03133) <br>
> *`--kind stiefel` implements its Alg. 2 — Stiefel direction factors trained by the landing field.*

> **StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold** <br>
> Zhizhong Li, Sina Sajadmanesh, Jingtao Li, Lingjuan Lyu — Sony AI <br>
> 📄 [arXiv:2510.01938](https://arxiv.org/abs/2510.01938) <br>
> *`--kind stella` implements its Alg. 1 — Riemannian gradient plus an exact polar retraction.*

> **Fast and accurate optimization on the orthogonal manifold without retraction** <br>
> Pierre Ablin, Gabriel Peyré <br>
> 📄 [arXiv:2102.07432](https://arxiv.org/abs/2102.07432) <br>
> *The landing field itself, which PoLAR builds on.*

And the baseline everything here is measured against:

> **LoRA: Low-Rank Adaptation of Large Language Models** <br>
> Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen — Microsoft, ICLR 2022 <br>
> 📄 [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) <br>
> *`--kind additive`, run through stock PEFT.*

---

<div align="center">

**MIT licensed** — see [LICENSE](LICENSE)

</div>
