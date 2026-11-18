# Phase LoRA

**Store every LoRA weight as a magnitude and an angle. Same parameter count, same expressiveness, better fine-tuning.**

Phase LoRA writes each entry of the low-rank factors as $`\rho\,e^{i\varphi}`$ and uses the real part of the complex product as the weight update:

```math
\Delta W = \frac{\alpha}{r}\,\mathrm{Re}\!\left(A B^{H}\right), \qquad A = \rho_A e^{i\varphi_A},\quad B = \rho_B e^{i\varphi_B}
```

At complex rank r/2 it can express exactly the updates that LoRA can at rank r, with exactly as many parameters. The only difference is the coordinates the optimizer moves in: magnitudes and angles instead of raw values. The same change applies to DoRA and PoLAR.

Across 2 models and 4 tasks, the phase version beats the original in **20 of 24 head-to-head settings**. The gain reaches **+3.4 chrF++** on English→Setswana translation and **+2.2 points** on ARC-Challenge. Pooled over all three pairs, the phase versions are ahead by **1.04 seed-to-seed standard deviations (p = 8 × 10⁻¹⁰)**.

## Results

Qwen3-0.6B-Base and Qwen3-1.7B-Base, rank 64, mean of 5 seeds. Translation (En→Tsn = English→Setswana, En→Lug = English→Luganda) is scored in chrF++; SVAMP and ARC-Challenge in accuracy (%). **Bold** marks the better adapter of each real/phase pair.

| Adapter | Params 0.6B / 1.7B | En→Tsn 0.6B | En→Tsn 1.7B | En→Lug 0.6B | En→Lug 1.7B | SVAMP 0.6B | SVAMP 1.7B | ARC 0.6B | ARC 1.7B |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Base model* | – | *8.8* | *11.2* | *9.5* | *12.6* | *0.3* | *1.0* | *62.6* | *80.1* |
| LoRA | 9.18M / 12.85M | 30.1 | 32.6 | 29.9 | 31.5 | 69.5 | **76.5** | 68.0 | 79.3 |
| Phase LoRA | 9.18M / 12.85M | **33.5** | **33.8** | **31.2** | **31.7** | **71.9** | 76.2 | **68.7** | **81.5** |
| DoRA | 9.26M / 12.93M | 30.6 | 32.6 | 30.5 | 31.1 | 71.2 | **78.1** | 67.9 | 79.1 |
| Phase DoRA | 9.26M / 12.93M | **33.6** | **33.6** | **31.1** | **31.8** | **72.1** | 75.8 | **68.9** | **81.3** |
| PoLAR | 9.40M / 13.07M | 32.4 | **32.2** | 30.8 | 31.3 | 68.9 | **77.9** | 67.4 | 80.5 |
| Phase PoLAR | 9.40M / 13.07M | **32.5** | 32.1 | **31.2** | **31.3** | **70.1** | 77.0 | **67.7** | **80.8** |

| Phase version vs original | Effect (seed SDs) | 95 % CI | p (Holm) |
|---|:---:|:---:|:---:|
| Phase LoRA vs LoRA | **+1.51** | [+0.94, +2.08] | 4 × 10⁻⁶ |
| Phase DoRA vs DoRA | **+1.35** | [+0.76, +1.94] | 4 × 10⁻⁵ |
| Phase PoLAR vs PoLAR | +0.28 | [−0.20, +0.75] | 0.25 |
| All three pairs | **+1.04** | [+0.73, +1.36] | 8 × 10⁻¹⁰ |

Phase LoRA and Phase DoRA also reach a lower test loss than LoRA and DoRA (+1.77 and +1.84 seed SDs).

### Where it does not win

- **SVAMP on 1.7B:** Phase LoRA 76.2 vs 76.5, Phase DoRA 75.8 vs 78.1.
- **PoLAR:** the closest pair. Phase PoLAR is ahead in 6 of 8 settings, but the pair's test is not significant (p = 0.25), and its test loss is higher than PoLAR's on both translation tasks (1.09 seed SDs worse, p = 0.001).
- **Rank 8:** the gain shrinks. Setswana +0.8, Luganda −0.1, SVAMP +0.4, ARC −2.1, against +3.3, +1.3, +2.3, +0.6 at rank 64; none of the rank-8 differences is significant.
- **PEFT benchmark:** Phase LoRA is behind LoRA in 3 of 5 settings (−0.9, −1.7, −2.9) and ahead in 2 by larger margins (+6.2, +5.0). It forgets less than LoRA in all 5 (see below).

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
| `stiefel` · `phase_stiefel` | PoLAR · Phase PoLAR (also pass `lam`, the landing weight; our runs pick it from {1e-3, 5e-3, 0.1, 0.5}) |

PoLAR's landing step runs inside the optimizer. Train PoLAR and Phase PoLAR with `make_optimizer(params, lr, clip, landing=landing_modules(model))` from `phase_lora.train_utils`, where `params` is a list of the trainable parameters. With any other optimizer they train without the landing step.

## Reproduce

```bash
uv sync --extra dev                  # exact library versions from uv.lock
source .venv/bin/activate
python -m pytest                     # CPU tests, under a minute
bash scripts/verify_baseline.sh      # retrains one stored LoRA run and compares, ~2 min on one GPU
bash scripts/reproduce.sh            # the full experiment, ~28 GPU-hours on one 16 GB GPU
```

`verify_baseline.sh` downloads the pinned data and both base models on its first run. It then retrains the stored LoRA run on SVAMP (0.6B, seed 1), scores it, and prints the stored and the new accuracy side by side: both 70.33, with 300/300 identical answers on an RTX 5080 Laptop GPU. `reproduce.sh` runs everything else (grid, main runs, scoring, analysis) and resumes after a crash.

Every run's training record, per-item test scores and log are in `runs/`. The trained weights are not in git.

---

## Experiment details

| | |
|---|---|
| **Tasks** | English→Setswana and English→Luganda (MAFAND-MT), SVAMP, ARC-Challenge |
| **Adapters** | rank 64 on `q_proj` and `v_proj`; both adapters of a pair have the same parameter count |
| **Learning rate** | chosen per adapter, task and model from {3e-4, 1e-3, 3e-3, 1e-2} by the dev loss of a one-epoch run |
| **Training** | 5 epochs, best-dev-loss checkpoint, 5 seeds |
| **Translation decoding** | greedy; a 4-gram may appear at most twice per reply |
| **PoLAR landing step** | as in the [official PoLAR code](https://github.com/kcc-lion/polar): clip the gradients, then replace the gradient G of each factor X by (G Xᵀ − X Gᵀ) X + λ X(XᵀX − I); λ is chosen with the learning rate from {1e-3, 5e-3, 0.1, 0.5} |
| **Ablation** | LoRA vs Phase LoRA at rank 8, 0.6B, 3 seeds |

On both translation tasks, PoLAR and Phase PoLAR picked the largest learning rate (1e-2), and all 40 of those runs were still improving when training stopped.

**Statistics.** An effect is how far the phase version is ahead, in units of the seed-to-seed standard deviation (seed SD); for test loss, a positive effect means a lower loss. Each test regresses the run scores on a phase indicator with one effect per setup (OLS, HC3 errors, Holm correction across the three pairs). Runs are not matched by seed. The setups differ a lot (I² = 59 %); a random-effects estimate across the 24 setups gives +0.86 (p = 1 × 10⁻⁴; Hartung–Knapp p = 0.002).

Data and model revisions are pinned and checked by checksum (`data_manifest.json`, `scripts/prepare_models.py`).

---

## PEFT benchmark

We also ran Phase LoRA and PoLAR on Hugging Face PEFT's own [MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA): Llama-3.2-3B trained on MetaMathQA and tested on all 1319 GSM8K problems, on an NVIDIA L40S (the GPU of PEFT's official results). **Phase LoRA reached the best accuracy of the benchmark (51.7) and forgot less than LoRA in every setting.**

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

PEFT's official LoRA result at the defaults is 47.8 accuracy and 0.050 forgetting. Each cell is one run with one seed (a GSM8K score varies by about ±1.4 points). The benchmark tests the final model and only logs its validation set. The settings after the first two rows were chosen while test results were visible, so they form a sweep, not a tuned comparison. **Plateau schedule:** a 10 % linear warmup, then the learning rate halves after two 250-step checks without a lower training loss; the same for every method.

```bash
git clone https://github.com/huggingface/peft && cd peft
git checkout 66a7c305bac9d7f8183b1f158280b77be971a3d4
git apply /path/to/Phase-LoRA/peft_benchmark/peft_methods.patch
pip install -e .
cd method_comparison/MetaMathQA
python run.py -v experiments/phase_lora/llama-3.2-3B-rank32
```

The patch adds `PhaseLoraConfig` and `PolarConfig` as regular PEFT methods (with tests), the plateau schedule and the experiment configs. PoLAR follows its official code, landing step included. Llama-3.2-3B is gated: accept its license on Hugging Face first. `modal_run.py` runs the same experiments on Modal L40S GPUs and expects a Modal volume `peft-bench-hf` that holds the Llama-3.2-3B Hugging Face cache. Each folder in `peft_benchmark/results/` holds the result JSON, training log and exact-code record of one run.

---

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
