# Phase LoRA

**Low-rank adapters in polar coordinates: every weight of the adapter becomes a magnitude and an angle. Same parameters, same expressiveness, better fine-tuning.**

## Highlights

- **Better in 20 of 24 head-to-head comparisons** against LoRA, DoRA and PoLAR, with exactly the same number of parameters.
- **Up to +3.35 chrF++** in translation and **+2.2 points** on ARC-Challenge; the gains for LoRA and DoRA are statistically significant (p < 10⁻⁴).
- **Forgets 1.5–3.5× less than LoRA:** after fine-tuning, the model stays much closer to the original on ordinary Wikipedia text, in all 8 settings.
- **Steadier:** Phase LoRA varies less between random seeds than LoRA in all 8 settings.
- **Drop-in:** no extra parameters and no change to what the adapter can express. It also matched LoRA's best accuracy on Hugging Face PEFT's own benchmark and forgot less there too.

## How it works

Phase LoRA writes each entry of the low-rank factors as a complex number in polar form, $`\rho\,e^{i\varphi}`$, and uses the real part of the complex product as the weight update:

```math
\Delta W = \frac{\alpha}{r}\,\mathrm{Re}\!\left(A B^{H}\right), \qquad A = \rho_A e^{i\varphi_A},\quad B = \rho_B e^{i\varphi_B}
```

At complex rank r/2 this reaches exactly the updates that LoRA reaches at rank r, with exactly as many parameters. What changes is how the optimizer moves through them: it adjusts a magnitude and an angle for every entry instead of a raw value. The same idea gives **Phase DoRA** and **Phase PoLAR**.

## Results

Qwen3-0.6B-Base and Qwen3-1.7B-Base, rank 64, mean of 5 seeds. Translation (En→Tsn = English→Setswana, En→Lug = English→Luganda) is scored in chrF++; SVAMP and ARC-Challenge in accuracy (%). **Bold** marks the better adapter of each pair.

| Adapter | Params 0.6B / 1.7B | En→Tsn 0.6B | En→Tsn 1.7B | En→Lug 0.6B | En→Lug 1.7B | SVAMP 0.6B | SVAMP 1.7B | ARC 0.6B | ARC 1.7B |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Base model* | – | *8.8* | *11.2* | *9.5* | *12.6* | *0.3* | *1.0* | *62.6* | *80.1* |
| LoRA | 9.18M / 12.85M | 30.1 | 32.6 | 29.9 | 31.5 | 69.5 | **76.5** | 68.0 | 79.3 |
| **Phase LoRA** | 9.18M / 12.85M | **33.5** | **33.8** | **31.2** | **31.7** | **71.9** | 76.2 | **68.7** | **81.5** |
| DoRA | 9.26M / 12.93M | 30.6 | 32.6 | 30.5 | 31.1 | 71.2 | **78.1** | 67.9 | 79.1 |
| **Phase DoRA** | 9.26M / 12.93M | **33.6** | **33.6** | **31.1** | **31.8** | **72.1** | 75.8 | **68.9** | **81.3** |
| PoLAR | 9.40M / 13.07M | 32.4 | **32.2** | 30.8 | 31.3 | 68.9 | **77.9** | 67.4 | 80.5 |
| **Phase PoLAR** | 9.40M / 13.07M | **32.5** | 32.1 | **31.2** | **31.3** | **70.1** | 77.0 | **67.7** | **80.8** |

| Phase version vs original | Effect | 95 % CI | p |
|---|:---:|:---:|:---:|
| Phase LoRA vs LoRA | **+1.51** | [+0.94, +2.08] | 4 × 10⁻⁶ |
| Phase DoRA vs DoRA | **+1.35** | [+0.76, +1.94] | 4 × 10⁻⁵ |
| Phase PoLAR vs PoLAR | +0.28 | [−0.20, +0.75] | 0.25 |
| All three pairs | **+1.04** | [+0.73, +1.36] | 8 × 10⁻¹⁰ |

The effect is how far the phase version is ahead, in units of the normal run-to-run variation: 1 means ahead by about as much as one run typically differs from its method's average. Phase LoRA and Phase DoRA also reach a lower test loss (+1.77 and +1.84 in the same units).

### Forgetting

How much each fine-tuned model loses of its general language ability: how much worse it predicts ordinary Wikipedia text (the WikiText-103 test set) than the untouched base model, as the rise in loss per token, mean of 5 seeds. **Lower is better; bold marks the lowest of each column (before rounding).**

| Model | Adapter | Setswana | Luganda | SVAMP | ARC |
|---|---|:---:|:---:|:---:|:---:|
| 0.6B | LoRA | +3.57 | +4.20 | +0.40 | +0.045 |
| | Phase LoRA | +1.68 | +1.97 | **+0.19** | **+0.031** |
| | DoRA | +3.52 | +3.75 | +0.43 | +0.048 |
| | Phase DoRA | **+1.66** | +1.97 | +0.20 | +0.034 |
| | PoLAR | +1.69 | +1.61 | +0.19 | +0.059 |
| | Phase PoLAR | +1.74 | **+1.50** | +0.30 | +0.071 |
| 1.7B | LoRA | +1.79 | +2.50 | +0.24 | +0.074 |
| | Phase LoRA | **+0.61** | +0.80 | **+0.07** | **+0.021** |
| | DoRA | +1.79 | +2.43 | +0.24 | +0.074 |
| | Phase DoRA | +0.62 | +0.74 | +0.33 | +0.021 |
| | PoLAR | +0.81 | **+0.72** | +0.13 | +0.039 |
| | Phase PoLAR | +0.77 | +0.74 | +0.19 | +0.045 |

Phase LoRA forgets the least overall: lowest in 5 of the 8 columns, average place 1.9 of 6. LoRA and DoRA forget about 2.4× as much on average. The base models score 2.539 (0.6B) and 2.240 (1.7B).

---

## Use it

```python
from transformers import AutoModelForCausalLM
from phase_lora.train_utils import attach

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base").requires_grad_(False)
model = attach(model, kind="polar", r=64, alpha=64)   # Phase LoRA on q_proj and v_proj
```

| `kind` | Adapter |
|---|---|
| `additive` · `polar` | LoRA · Phase LoRA |
| `dora` · `phase_dora` | DoRA · Phase DoRA |
| `stiefel` · `phase_stiefel` | PoLAR · Phase PoLAR (also pass `lam`, the landing weight; the runs here pick it from {1e-3, 5e-3, 0.1, 0.5}) |

PoLAR's landing step runs inside the optimizer: train PoLAR and Phase PoLAR with `make_optimizer(params, lr, clip, landing=landing_modules(model))` from `phase_lora.train_utils`, where `params` is a list of the trainable parameters.

## Reproduce

```bash
uv sync --extra dev                  # exact library versions from uv.lock
source .venv/bin/activate
python -m pytest                     # CPU tests, under a minute
bash scripts/verify_baseline.sh      # retrains one stored LoRA run and compares, ~2 min on one GPU
bash scripts/reproduce.sh            # the full experiment, ~29 GPU-hours on one 16 GB GPU
```

`verify_baseline.sh` downloads the pinned data and both base models on its first run, retrains the stored LoRA run on SVAMP (0.6B, seed 1) and prints the stored and the new accuracy side by side: both 70.33, with 300/300 identical answers on an RTX 5080 Laptop GPU. `reproduce.sh` runs everything else (grid, main runs, scoring, forgetting, analysis) and resumes after a crash. The forgetting numbers above were measured on one NVIDIA H100 with `scripts/modal_forgetting.py`; `scripts/forgetting.py` measures them on any GPU, about 1 hour on a 16 GB laptop GPU. Every run's training record, per-item test scores and log are in `runs/`; the trained weights are not in git.

---

## PEFT benchmark

I also ran Phase LoRA and PoLAR on Hugging Face PEFT's own [MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA): Llama-3.2-3B trained on MetaMathQA and tested on all 1319 GSM8K problems, on an NVIDIA L40S, the GPU of PEFT's official results. **Phase LoRA matched LoRA's best accuracy (51.7 vs 51.6) and forgot less than LoRA in all five settings.**

**Test accuracy** (higher is better):

| Setting | LoRA | Phase LoRA | PoLAR |
|---|:---:|:---:|:---:|
| Benchmark defaults (lr 1e-4, cosine, 5k steps, weight decay 0.1) | **46.2** | 45.3 | 38.1 |
| Defaults, weight decay 0 | **46.6** | 44.9 | 39.3 |
| lr 1e-3, plateau schedule, 10k steps, weight decay 0 | 44.1 | **50.3** | 48.1 |
| lr 1e-3, plateau schedule, 10k steps, weight decay 0.1 | 46.7 | **51.7** | 50.7 |
| lr 1e-3, plateau schedule, 10k steps, weight decay 0.5 | **51.6** | 48.7 | 44.3 |

**Forgetting**, the rise of the loss on Wikipedia text after training (lower is better):

| Setting | LoRA | Phase LoRA | PoLAR |
|---|:---:|:---:|:---:|
| Benchmark defaults | 0.051 | **0.044** | 0.057 |
| Defaults, weight decay 0 | 0.050 | **0.045** | 0.058 |
| lr 1e-3, plateau, 10k, weight decay 0 | 0.489 | 0.284 | **0.077** |
| lr 1e-3, plateau, 10k, weight decay 0.1 | 0.335 | 0.176 | **0.066** |
| lr 1e-3, plateau, 10k, weight decay 0.5 | 0.113 | 0.073 | **0.040** |

PEFT's official LoRA result at the defaults is 47.8 accuracy and 0.050 forgetting. Each cell is one run with one seed. The first two rows use the benchmark's own settings. **Plateau schedule:** a scheduler I added to the benchmark: a 10 % linear warmup, then the learning rate halves after two 250-step checks without a lower training loss; the same for every method. The settings after the first two rows were chosen while test results were visible.

```bash
git clone https://github.com/huggingface/peft && cd peft
git checkout 66a7c305bac9d7f8183b1f158280b77be971a3d4
git apply /path/to/Phase-LoRA/peft_benchmark/peft_methods.patch
pip install -e .
cd method_comparison/MetaMathQA
python run.py -v experiments/phase_lora/llama-3.2-3B-rank32
```

The patch adds `PhaseLoraConfig` and `PolarConfig` as regular PEFT methods (with tests), the plateau schedule and the experiment configs; PoLAR follows its [official code](https://github.com/kcc-lion/polar). Llama-3.2-3B is gated: accept its license on Hugging Face first. `modal_run.py` runs the same experiments on Modal L40S GPUs and expects a Modal volume `peft-bench-hf` holding the Llama-3.2-3B Hugging Face cache. Each folder in `peft_benchmark/results/` holds the result JSON, training log and exact-code record of one run.

---

## Experiment details

| | |
|---|---|
| **Tasks** | English→Setswana and English→Luganda (MAFAND-MT), SVAMP, ARC-Challenge |
| **Adapters** | rank 64 on `q_proj` and `v_proj`; both adapters of a pair have the same parameter count |
| **Learning rate** | chosen per adapter, task and model from {3e-4, 1e-3, 3e-3, 1e-2} by the dev loss of a one-epoch run; LoRA and Phase LoRA picked the same rate in all 8 settings |
| **Training** | Adam, cosine decay, 5 epochs, best-dev-loss checkpoint, 5 seeds |
| **Translation decoding** | greedy; a 4-gram may appear at most twice per reply |
| **PoLAR landing step** | as in the official PoLAR code: clip the gradients, then replace the gradient G of each factor X by (G Xᵀ − X Gᵀ) X + λ X(XᵀX − I); λ chosen with the learning rate |
| **Forgetting** | WikiText-103 test split (pinned revision, checksum), rows joined with blank lines, non-overlapping 2,048-token windows; mean next-token loss after fine-tuning minus the base model's |
| **Statistics** | per setup, scores standardized by the seed-to-seed SD; OLS with one effect per setup, HC3 errors, Holm correction across the pairs |

Data and model revisions are pinned and checked by checksum (`data_manifest.json`, `scripts/prepare_models.py`).

**Scope.** The gains are clearest for LoRA and DoRA; the PoLAR pair is close (Phase PoLAR ahead in 6 of 8 settings, not significant). At rank 8 (0.6B, three seeds) Phase LoRA is not better than LoRA. On the PEFT benchmark's default settings LoRA scores slightly higher (46.2 vs 45.3).

## Repository layout

```
src/phase_lora/       the adapters: polar.py (Phase LoRA, Phase DoRA), stiefel.py (PoLAR, Phase PoLAR)
scripts/              data and model preparation, training, sweep, scoring, analysis, verify_baseline.sh
tests/                CPU tests
runs/                 every run's training record, test scores and log
peft_benchmark/       PEFT patch and MetaMathQA results
```

## License

MIT
