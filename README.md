# Phase LoRA

**Low-rank adapters with factors in polar coordinates, at the same parameter count as the original.**

## Results at a glance

Qwen3-0.6B-Base and Qwen3-1.7B-Base, rank 64, mean over 5 seeds. Translation: chrF++; SVAMP and ARC-Challenge: accuracy (%). **Bold** marks the better adapter of each real/phase pair.

| Adapter | Params 0.6B / 1.7B | En→Tsn 0.6B | En→Tsn 1.7B | En→Lug 0.6B | En→Lug 1.7B | SVAMP 0.6B | SVAMP 1.7B | ARC 0.6B | ARC 1.7B |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Base model* | – | *8.8* | *11.2* | *9.5* | *12.6* | *0.3* | *1.0* | *62.6* | *80.1* |
| LoRA | 9.18M / 12.85M | 30.1 | 32.6 | 29.9 | 31.5 | 69.5 | **76.5** | 68.0 | 79.3 |
| **Phase LoRA** | 9.18M / 12.85M | **33.5** | **33.8** | **31.2** | **31.7** | **71.9** | 76.2 | **68.7** | **81.5** |
| DoRA | 9.26M / 12.93M | 30.6 | 32.6 | 30.5 | 31.1 | 71.2 | **78.1** | 67.9 | 79.1 |
| **Phase DoRA** | 9.26M / 12.93M | **33.6** | **33.6** | **31.1** | **31.8** | **72.1** | 75.8 | **68.9** | **81.3** |
| PoLAR | 9.40M / 13.07M | **29.4** | 28.5 | **29.5** | 29.3 | 71.5 | 77.8 | **67.5** | **81.7** |
| **Phase PoLAR** | 9.40M / 13.07M | 28.7 | **28.7** | 22.4¹ | **31.2** | **71.7** | **78.0** | 67.1 | 80.2 |

¹ Two of the five seeds reached 13.0 and 11.2; the other three 27.6–30.6.

### Where the phase versions fall behind

- **SVAMP on 1.7B:** Phase LoRA 76.2 vs 76.5, Phase DoRA 75.8 vs 78.1.
- **The PoLAR pair:** Phase PoLAR is below PoLAR on Setswana 0.6B, Luganda 0.6B and both ARC settings, and its family-level test is not significant (p = 0.075).
- **Rank 8:** Phase LoRA is level with LoRA (Luganda −0.1) or below it (ARC −2.1); none of the rank-8 differences is significant.
- **PEFT benchmark at its default settings:** Phase LoRA scores 45.3 vs LoRA's 46.2 (see below).

---

## The method

Every entry of a LoRA factor is stored as a magnitude and an angle, $\rho\,e^{i\varphi}$, and the weight update is the real part of the complex product:

$$
\Delta W = \frac{\alpha}{r}\,\mathrm{Re}\!\left(A B^{H}\right), \qquad A = \rho_A e^{i\varphi_A},\quad B = \rho_B e^{i\varphi_B}
$$

At complex rank $r/2$ this has **the same parameter count and the same function class as LoRA at rank $r$**; only the coordinates that the optimizer moves differ. Phase DoRA and Phase PoLAR apply the same change to DoRA and PoLAR.

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
| `stiefel` · `phase_stiefel` | PoLAR · Phase PoLAR (also pass `lam=5e-3`) |

## Reproduce

```bash
uv sync --extra dev                  # exact library versions from uv.lock
python -m pytest                     # CPU tests, under a minute
bash scripts/verify_baseline.sh      # retrains one stored LoRA run and compares, ~2 min on one GPU
bash scripts/reproduce.sh            # the full experiment, ~25 GPU-hours on one 16 GB GPU
```

`verify_baseline.sh` downloads the pinned data and model, retrains the stored LoRA run on SVAMP (0.6B, seed 1), scores it and prints the stored and the new accuracy side by side (both 70.33, 300/300 answers identical on an RTX 5080). `reproduce.sh` runs everything else: grid, main runs, scoring and analysis, and resumes after a crash.

Every run's training record, per-item test scores and log are in `runs/`. The trained weights are not in git.

---

## Experiment details

| | |
|---|---|
| **Tasks** | English→Setswana and English→Luganda (MAFAND-MT), SVAMP, ARC-Challenge |
| **Adapters** | rank 64 on `q_proj` and `v_proj`; each pair has the same parameter count |
| **Learning rate** | chosen per adapter from {3e-4, 1e-3, 3e-3, 1e-2} by dev loss |
| **Training** | 5 epochs, best-dev-loss checkpoint, 5 seeds |
| **Translation decoding** | greedy, a 4-gram may appear at most twice per reply |
| **PoLAR landing field** | ½(G Xᵀ − X Gᵀ) X + 4λ X(XᵀX − I), λ = 5·10⁻³, applied before gradient clipping (both PoLAR arms) |
| **Ablation** | LoRA vs Phase LoRA at rank 8, 0.6B, 3 seeds |

Data and model revisions are pinned and checked by checksum (`data_manifest.json`, `scripts/prepare_models.py`).

**Tests**, phase minus real, in units of the seed-to-seed standard deviation. Each test is a regression of the run scores on a phase indicator with one effect per setup (OLS with HC3 errors, Holm-corrected); runs are not matched by seed:

| Comparison | Effect | 95 % CI | p |
|---|:---:|:---:|:---:|
| Phase LoRA vs LoRA | **+1.51** | [+0.94, +2.08] | 4 × 10⁻⁶ |
| Phase DoRA vs DoRA | **+1.35** | [+0.76, +1.94] | 4 × 10⁻⁵ |
| Phase PoLAR vs PoLAR | +1.08 | [−0.11, +2.27] | 0.075 |
| All pairs, all tasks | **+1.31** | [+0.84, +1.78] | 1 × 10⁻⁷ |

The setups differ a lot from one another (I² = 75 %). A random-effects estimate across the 24 setups gives **+0.80** (p = 0.007; Hartung–Knapp p = 0.048).

**Rank 8 vs rank 64** (LoRA → Phase LoRA, 0.6B): Setswana +0.8 vs +3.3, Luganda −0.1 vs +1.3, SVAMP +0.4 vs +2.3, ARC −2.1 vs +0.6.

---

## PEFT benchmark

Phase LoRA and PoLAR were also run on Hugging Face PEFT's own [MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA): Llama-3.2-3B, trained on MetaMathQA, tested on all 1319 GSM8K problems, on an NVIDIA L40S (the GPU of PEFT's official results).

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

PEFT's official LoRA result in the default setting is 47.8 accuracy and 0.050 forgetting.

> **Read with care.** Each cell is a single run with one seed; one GSM8K score carries about ±1.4 points of noise. The benchmark trains for a fixed number of steps and tests the final model; its 50-problem validation set is only logged, never used to choose a setting or a checkpoint. The settings after the first two rows were chosen while test results were visible, so they are a sweep, not a tuned comparison.

**Plateau schedule:** a 10 % linear warmup, then the learning rate halves after two checks (250 steps each) without a lower training loss; the same for every method.

```bash
git clone https://github.com/huggingface/peft && cd peft
git checkout 66a7c305bac9d7f8183b1f158280b77be971a3d4
git apply /path/to/Phase-LoRA/peft_benchmark/peft_methods.patch
pip install -e .
cd method_comparison/MetaMathQA
python run.py -v experiments/phase_lora/llama-3.2-3B-rank32
```

The patch adds `PhaseLoraConfig` and `PolarConfig` as regular PEFT methods (with tests), the plateau schedule and the experiment configs. PoLAR follows its [official code](https://github.com/kcc-lion/polar), landing field included. Llama-3.2-3B is gated: accept its license on Hugging Face first. `modal_run.py` runs the same experiments on Modal GPUs. Each folder in `peft_benchmark/results/` holds the result JSON, training log and exact-code record of one run.

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
