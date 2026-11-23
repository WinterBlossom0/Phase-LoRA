"""Every number, table and figure of the paper, from the saved runs. Reads only.

  python scripts/paper_analysis.py

Inputs:  the design of scripts/run_sweep.py and runs/selected_hparams.json (which runs count),
         runs/<tag>.json (training curves), runs/test_norepeat/ (repetition-blocked translation, primary),
         runs/test/ (greedy: SVAMP, ARC, and translation seed 1), runs/test_loss/ (teacher-forced test loss),
         runs/test*/base_*.json (untrained models).
Outputs: paper/generated/numbers.json, tab_*.tex, fig_*.pdf.
The tests and sensitivity analyses were fixed before the Setswana, SVAMP and ARC runs and apply unchanged to Luganda,
which was added after those results. The "post_hoc" entry of numbers.json holds analyses added after the results were
seen; the paper reports them as such.
"""

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # run without installing
sys.path.insert(0, str(ROOT / "scripts"))

from transformers import AutoTokenizer

import run_sweep
import score_test
from phase_lora import RUNS_DIR as RUNS
from phase_lora.run_config import SCORE_CODE, code_version, parse, run_tag, train_version_ok
from phase_lora.train_utils import TRANSLATION

OUT = ROOT / "paper" / "generated"
MODELS, DATASETS, FAMILIES = run_sweep.MODELS, run_sweep.DATASETS, run_sweep.FAMILIES
(MAIN,) = run_sweep.BLOCKS
MAIN_RANK = MAIN["ranks"][0]
MODEL_NAMES = {"qwen3-0.6b-base": "0.6B", "qwen3-1.7b-base": "1.7B", "lfm3-1.7b-base": "LFM3"}
TASK_NAMES = {"mafand_tsn": "En$\\to$Tsn", "mafand_lug": "En$\\to$Lug", "svamp": "SVAMP", "arc": "ARC-C"}
PLAIN_TASK_NAMES = {"mafand_tsn": "En→Tsn", "mafand_lug": "En→Lug", "svamp": "SVAMP", "arc": "ARC-C"}
NAMES = {"additive": "LoRA", "polar": "Phase LoRA", "dora": "DoRA", "phase_dora": "Phase DoRA",
         "stiefel": "PoLAR", "phase_stiefel": "Phase PoLAR"}
ALPHA = 0.05
BOOTSTRAP_DRAWS = 2000
# Adapted layers and (in, out) of q_proj and v_proj, from models/*/config.json. Qwen3: every one of 28 layers.
# LFM3: only its 11 attention layers have q_proj and v_proj; its 13 convolution layers get no adapter.
ADAPTED = {"qwen3-0.6b-base": (28, [(1024, 2048), (1024, 1024)]),
           "qwen3-1.7b-base": (28, [(2048, 2048), (2048, 1024)]),
           "lfm3-1.7b-base": (11, [(2048, 2048), (2048, 512)])}
# A greedy reply counts as runaway when it fills the generation cap: no end token, no newline.
# Re-tokenising the decoded text can merge a few tokens, hence 90 % of the cap.
RUNAWAY_SHARE = 0.9
BLOCK_RULES = set()     # the repetition-block rule of each blocked score; the paper needs exactly one
SCORE_DIRS = {"greedy": "test", "norepeat": "test_norepeat", "loss": "test_loss"}


def runaway_percent(items, tok, cap):
    return 100 * statistics.mean(len(tok(item["hyp"], add_special_tokens=False)["input_ids"]) >= RUNAWAY_SHARE * cap
                                 for item in items)


def read_score(tag, train_version, dataset, mode):
    """One score file of a mode: greedy or norepeat decoding, or the test loss (decoder None).
    The oldest recorded scorers decoded greedily (their files hold no decoder)."""
    path = RUNS / SCORE_DIRS[mode] / f"{tag}.json"
    if not path.exists():
        sys.exit(f"missing score {path}")
    score = json.loads(path.read_text())
    score_version, scored_train = score["versions"]
    expected = None if mode == "loss" else mode
    if score.get("decoder", "greedy") != expected:
        sys.exit(f"{path.name}: decoder {score.get('decoder')}, expected {expected}")
    if mode == "norepeat":
        BLOCK_RULES.add(json.dumps(score["block"], sort_keys=True))
    if scored_train != train_version or not score_test.score_valid(score_version, code_version(SCORE_CODE), dataset, mode):
        sys.exit(f"{path.name}: scored by {score['versions']}, not a valid scorer for training {train_version}")
    return score


def item_values(score, dataset):
    """Per test item: sentence chrF++ for translation, 0/100 correctness for the accuracy tasks."""
    if dataset in TRANSLATION:
        return np.array([item["chrf++"] for item in score["items"]])
    return np.array([100.0 * item["correct"] for item in score["items"]])


def load():
    """(block, dataset, model, rank, kind) -> seed -> run record, for exactly the runs of the design."""
    selected = json.loads((RUNS / "selected_hparams.json").read_text())
    every = {model: run_sweep.eval_every(model) for model in MODELS}
    toks = {model: AutoTokenizer.from_pretrained(ROOT / "models" / model, trust_remote_code=True) for model in MODELS}
    caps = {(model, dataset): score_test.generation_cap(tok, dataset) for model, tok in toks.items() for dataset in TRANSLATION}
    runs = defaultdict(dict)
    for seed in MAIN["seeds"]:
        for model, dataset, rank, kind in run_sweep.block_arms(MAIN):
            pick = selected[f"{model}|{dataset}|{kind}|{rank}"]["selected"]
            argv = run_sweep.argv_for(model, dataset, kind, rank, pick["lr"], pick["lam"], seed, run_sweep.EPOCHS,
                                      every[model][dataset])
            tag = run_tag(vars(parse(argv)))
            path = RUNS / f"{tag}.json"
            if not path.exists():
                sys.exit(f"missing main run {path.name}")
            run = json.loads(path.read_text())
            if not train_version_ok(run["code_version"], kind):
                sys.exit(f"{path.name}: trained by code {run['code_version']}, not valid for {kind}")
            version = run["code_version"]
            record = {"steps": [step for step, _ in run["hist"]], "losses": [loss for _, loss in run["hist"]],
                      "diverged_at": run["diverged_at"], "lr": pick["lr"],
                      "test_loss": read_score(tag, version, dataset, "loss")["score"]}
            if dataset in TRANSLATION:
                blocked = read_score(tag, version, dataset, "norepeat")
                record.update(test=blocked["score"], items=item_values(blocked, dataset),
                              hyps=[item["hyp"] for item in blocked["items"]],
                              refs=[item["ref"] for item in blocked["items"]],
                              runaway_blocked=runaway_percent(blocked["items"], toks[model], caps[(model, dataset)]),
                              length_ratio=(sum(len(item["hyp"]) for item in blocked["items"])
                                            / sum(len(item["ref"]) for item in blocked["items"])))
                if seed == 1:
                    greedy = read_score(tag, version, dataset, "greedy")
                    record.update(greedy=greedy["score"],
                                  runaway=runaway_percent(greedy["items"], toks[model], caps[(model, dataset)]))
            else:
                greedy = read_score(tag, version, dataset, "greedy")
                record.update(test=greedy["score"], items=item_values(greedy, dataset))
            runs[("main", dataset, model, rank, kind)][seed] = record
    return runs, toks, caps


def load_base(toks, caps):
    """(dataset, model) -> score of the untrained model with the reported decoder, its loss, and its greedy
    translation score and runaway share."""
    base = {}
    for model in MODELS:
        for dataset in DATASETS:
            tag = f"base_{model}_{dataset}"
            decoder = "norepeat" if dataset in TRANSLATION else "greedy"
            base[(dataset, model)] = read_score(tag, None, dataset, decoder)["score"]
            base[(f"test_loss|{dataset}", model)] = read_score(tag, None, dataset, "loss")["score"]
            if dataset in TRANSLATION:
                greedy = read_score(tag, None, dataset, "greedy")
                base[(f"greedy|{dataset}", model)] = greedy["score"]
                base[(f"runaway|{dataset}", model)] = runaway_percent(greedy["items"], toks[model], caps[(model, dataset)])
    return base


# ---------------------------------------------------------------- statistics (the same for every task)

def ols_phase(y, phase, groups):
    """OLS y ~ C(groups) + phase with HC3 errors and a t reference: (effect, ci_low, ci_high, p)."""
    levels = sorted(set(groups))
    columns = [np.ones(len(y)), np.asarray(phase, dtype=float)]
    columns += [np.array([group == level for group in groups], dtype=float) for level in levels[1:]]
    fit = sm.OLS(np.asarray(y, dtype=float), np.column_stack(columns)).fit(cov_type="HC3", use_t=True)
    low, high = fit.conf_int(ALPHA)[1]
    return float(fit.params[1]), float(low), float(high), float(fit.pvalues[1])


def setup_rows(runs, family, dataset, model, field, sign, seeds=None):
    """(score, phase flag) of every run of one setup; sign = -1 turns a loss into a higher-is-better score."""
    real, phase = FAMILIES[family]
    rows = []
    for kind, flag in ((real, 0), (phase, 1)):
        arm = runs[("main", dataset, model, MAIN_RANK, kind)]
        rows += [(sign * record[field], flag) for seed, record in arm.items() if seeds is None or seed in seeds]
    return rows


def standardised(rows):
    """Subtract the setup mean and divide by the pooled within-arm standard deviation."""
    real = [score for score, flag in rows if flag == 0]
    phase = [score for score, flag in rows if flag == 1]
    pooled = math.sqrt((statistics.variance(real) + statistics.variance(phase)) / 2)
    if pooled == 0:
        sys.exit("a setup has no spread between runs; standardisation is undefined")
    centre = statistics.mean(real + phase)
    return [((score - centre) / pooled, flag) for score, flag in rows]


def z_rows_of(runs, field, sign, families, datasets, seeds=None, drop_setup=None):
    """(z, phase flag, setup, family) of every run of the chosen setups."""
    rows = []
    for family in families:
        for dataset in datasets:
            for model in MODELS:
                setup = f"{family}|{dataset}|{model}"
                if setup != drop_setup:
                    rows += [(z, flag, setup, family)
                             for z, flag in standardised(setup_rows(runs, family, dataset, model, field, sign, seeds))]
    return rows


def ols_of(rows):
    return ols_phase([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])


def family_tests(runs, field, sign, seeds=None):
    """P0 (all families), P1 (each family, Holm over the families) and P2 (family x task, Holm over all of them), with the
    gatekeeping: P1 is tested only if P0 rejects, P2 of a family only if that family's P1 rejects."""
    z_rows = z_rows_of(runs, field, sign, FAMILIES, DATASETS, seeds)
    effect, low, high, p = ols_of(z_rows)
    p0 = {"effect_sd": effect, "ci": [low, high], "p": p, "reject": p < ALPHA}
    p1 = []
    for family in FAMILIES:
        effect, low, high, p = ols_of([r for r in z_rows if r[3] == family])
        p1.append({"family": family, "effect_sd": effect, "ci": [low, high], "p": p})
    p2 = []
    for family in FAMILIES:
        for dataset in DATASETS:
            raw, per_model = [], {}
            for model in MODELS:
                rows = setup_rows(runs, family, dataset, model, field, sign, seeds)
                raw += [(score, flag, model) for score, flag in rows]
                per_model[model] = (statistics.mean(s for s, f in rows if f == 1) - statistics.mean(s for s, f in rows if f == 0))
            # oriented so that a positive effect is better (for the loss: a lower loss)
            effect, low, high, p = ols_of(raw)
            p2.append({"family": family, "dataset": dataset, "effect": effect, "ci": [low, high], "p": p,
                       "per_model": per_model})
    for tests in (p1, p2):
        adjusted = multipletests([t["p"] for t in tests], alpha=ALPHA, method="holm")[1]
        for test, p_holm in zip(tests, adjusted):
            test["p_holm"] = float(p_holm)
    for test in p1:
        test["tested"] = p0["reject"]
        test["reject"] = test["tested"] and test["p_holm"] < ALPHA
    for test in p2:
        test["tested"] = next(t["reject"] for t in p1 if t["family"] == test["family"])
        test["reject"] = test["tested"] and test["p_holm"] < ALPHA
    return {"P0": p0, "P1": p1, "P2": p2}


def permutation_p(rows, rng, draws=10000):
    """Two-sided p of the phase-minus-real mean z when the phase label is shuffled within each setup."""
    def statistic(flags):
        return (np.mean([z for (z, _, _, _), f in zip(rows, flags) if f == 1])
                - np.mean([z for (z, _, _, _), f in zip(rows, flags) if f == 0]))
    by_setup = defaultdict(list)
    for index, row in enumerate(rows):
        by_setup[row[2]].append(index)
    flags = np.array([row[1] for row in rows])
    observed = statistic(flags)
    extreme = 0
    for _ in range(draws):
        shuffled = flags.copy()
        for indices in by_setup.values():
            shuffled[indices] = rng.permutation(flags[indices])
        extreme += abs(statistic(shuffled)) >= abs(observed) - 1e-12
    return float((extreme + 1) / (draws + 1))


def pooled_sensitivity(runs, field, sign):
    """The sensitivity analyses of P0 and P1."""
    rng = np.random.default_rng(0)
    out = {}
    effect, low, high, p = ols_of(z_rows_of(runs, field, sign, ["DoRA", "PoLAR"], DATASETS))
    out["P0_new_arms_only"] = {"effect_sd": effect, "ci": [low, high], "p": p}
    out["P0_leave_one_task_out"] = {}
    for dataset in DATASETS:
        effect, low, high, p = ols_of(z_rows_of(runs, field, sign, FAMILIES, [d for d in DATASETS if d != dataset]))
        out["P0_leave_one_task_out"][dataset] = {"effect_sd": effect, "ci": [low, high], "p": p}
    setups = [f"{f}|{d}|{m}" for f in FAMILIES for d in DATASETS for m in MODELS]
    left_out = [ols_of(z_rows_of(runs, field, sign, FAMILIES, DATASETS, drop_setup=s)) for s in setups]
    out["P0_leave_one_setup_out"] = {"effect_sd_range": [min(r[0] for r in left_out), max(r[0] for r in left_out)],
                                     "p_max": max(r[3] for r in left_out)}
    all_rows = z_rows_of(runs, field, sign, FAMILIES, DATASETS)
    out["P0_permutation_p"] = permutation_p(all_rows, rng)
    out["P1_permutation_p"] = {family: permutation_p([r for r in all_rows if r[3] == family], rng) for family in FAMILIES}
    out["without_seed_1"] = family_tests(runs, field, sign, seeds=[s for s in MAIN["seeds"] if s != 1])
    return out


def hierarchical_bootstrap(real_items, phase_items, rng):
    """Two-sided p of phase minus real mean item score, resampling seeds within each arm and items (paired)."""
    real, phase = np.stack(real_items), np.stack(phase_items)
    n_items = real.shape[1]
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        items = rng.integers(0, n_items, n_items)
        draws[draw] = (phase[rng.integers(0, len(phase), len(phase))][:, items].mean()
                       - real[rng.integers(0, len(real), len(real))][:, items].mean())
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


def sensitivity(runs):
    """Per setup: Welch t-test on run scores, and the item x seed hierarchical bootstrap (appendix)."""
    rng = np.random.default_rng(0)
    rows = []
    for family, (real, phase) in FAMILIES.items():
        for dataset in DATASETS:
            for model in MODELS:
                r = runs[("main", dataset, model, MAIN_RANK, real)]
                q = runs[("main", dataset, model, MAIN_RANK, phase)]
                r_scores, q_scores = [x["test"] for x in r.values()], [x["test"] for x in q.values()]
                rows.append({"family": family, "dataset": dataset, "model": model,
                             "delta": statistics.mean(q_scores) - statistics.mean(r_scores),
                             "welch_p": float(stats.ttest_ind(q_scores, r_scores, equal_var=False).pvalue),
                             "welch_ci": list(map(float, stats.ttest_ind(q_scores, r_scores, equal_var=False)
                                                  .confidence_interval(1 - ALPHA))),
                             "bootstrap_p": hierarchical_bootstrap([x["items"] for x in r.values()],
                                                                   [x["items"] for x in q.values()], rng)})
    return rows


def best_epoch(record):
    """Epoch (0-5) of the dev check with the lowest loss; 0 means no check beat the untrained adapter."""
    best = min(range(len(record["losses"])), key=lambda index: record["losses"][index])
    return run_sweep.EPOCHS * record["steps"][best] / record["steps"][-1]


def trainable(kind, model, rank):
    """Trainable parameters over all adapted projections; phase kinds use complex rank rank/2."""
    layers, projections = ADAPTED[model]
    total = 0
    for in_features, out_features in projections:
        total += rank * (in_features + out_features)        # equal for every kind by construction
        total += {"additive": 0, "polar": 0, "stiefel": rank ** 2, "phase_stiefel": rank ** 2,
                  "dora": out_features, "phase_dora": out_features}[kind]
    return layers * total


# ---------------------------------------------------------------- post-hoc analyses
# Added after the results were seen; reported as post hoc.

def setup_effects(runs):
    """setup -> standardised phase-minus-real difference (Cohen's d, pooled sd) of the task metric, and its variance."""
    out = {}
    for family in FAMILIES:
        for dataset in DATASETS:
            for model in MODELS:
                rows = setup_rows(runs, family, dataset, model, "test", 1)
                real = [score for score, flag in rows if flag == 0]
                phase = [score for score, flag in rows if flag == 1]
                d = (statistics.mean(phase) - statistics.mean(real)) / math.sqrt(
                    (statistics.variance(real) + statistics.variance(phase)) / 2)
                n_real, n_phase = len(real), len(phase)
                variance = (n_real + n_phase) / (n_real * n_phase) + d * d / (2 * (n_real + n_phase))
                out[f"{family}|{dataset}|{model}"] = (d, variance)
    return out


def heterogeneity(runs):
    """Between-setup spread of the effect: Cochran's Q, I^2, the DerSimonian-Laird random-effects mean with a normal
    and a Hartung-Knapp t reference, a one-sample t over setups, and the family x phase F test on the z rows."""
    effects = setup_effects(runs)
    d = np.array([value[0] for value in effects.values()])
    variance = np.array([value[1] for value in effects.values()])
    k = len(d)
    weights = 1 / variance
    fixed = (weights * d).sum() / weights.sum()
    q = float((weights * (d - fixed) ** 2).sum())
    tau2 = max(0.0, (q - (k - 1)) / (weights.sum() - (weights ** 2).sum() / weights.sum()))
    random_weights = 1 / (variance + tau2)
    mu = float((random_weights * d).sum() / random_weights.sum())
    se = math.sqrt(1 / random_weights.sum())
    se_hk = math.sqrt((random_weights * (d - mu) ** 2).sum() / (k - 1) / random_weights.sum())
    z_rows = z_rows_of(runs, "test", 1, FAMILIES, DATASETS)
    y = np.array([row[0] for row in z_rows])
    phase = np.array([row[1] for row in z_rows], dtype=float)
    setups = sorted({row[2] for row in z_rows})
    reduced = np.column_stack([np.ones(len(y)), phase] + [np.array([row[2] == s for row in z_rows], float) for s in setups[1:]])
    full = np.column_stack([reduced] + [phase * np.array([row[3] == f for row in z_rows], float) for f in list(FAMILIES)[1:]])
    rss = {name: float(((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum()) for name, X in (("reduced", reduced), ("full", full))}
    df_full = len(y) - full.shape[1]
    extra = full.shape[1] - reduced.shape[1]
    f_value = ((rss["reduced"] - rss["full"]) / extra) / (rss["full"] / df_full)
    return {"setup_d": {key: value[0] for key, value in effects.items()},
            "cochran_q": q, "q_p": float(stats.chi2.sf(q, k - 1)), "i2": max(0.0, (q - (k - 1)) / q), "tau2": tau2,
            "random_effects_d": mu, "random_effects_p_normal": float(2 * stats.norm.sf(abs(mu / se))),
            "random_effects_p_hartung_knapp": float(2 * stats.t.sf(abs(mu / se_hk), k - 1)),
            "setup_level_t_p": float(stats.ttest_1samp(d, 0).pvalue),
            "family_x_phase": {"F": f_value, "df": [extra, df_full], "p": float(stats.f.sf(f_value, extra, df_full))}}


def model_interaction(runs, field, sign):
    """Per family x task: the phase effect on MODELS[0] and its change on each other model
    (y ~ phase * model with model as a factor, HC3, t reference)."""
    others = MODELS[1:]
    out = []
    for family in FAMILIES:
        for dataset in DATASETS:
            y, columns = [], []
            for model in MODELS:
                is_model = [float(model == other) for other in others]
                for score, flag in setup_rows(runs, family, dataset, model, field, sign):
                    y.append(score)
                    columns.append([1.0, flag] + is_model + [flag * value for value in is_model])
            fit = sm.OLS(np.array(y), np.array(columns)).fit(cov_type="HC3", use_t=True)
            first_interaction = 2 + len(others)
            out.append({"family": family, "dataset": dataset, "effect_first_model": float(fit.params[1]),
                        "interaction": {other: float(fit.params[first_interaction + i]) for i, other in enumerate(others)},
                        "interaction_p": {other: float(fit.pvalues[first_interaction + i]) for i, other in enumerate(others)}})
    return out


def method_vs_all(runs, target, field, sign):
    """The target method against each other method: per setup both arms are standardised as in standardised(), then
    OLS z ~ C(setup) + target flag with HC3 errors; Holm over the comparisons. Post hoc."""
    out = []
    for other in run_sweep.KINDS:
        if other == target:
            continue
        rows = []
        for dataset in DATASETS:
            for model in MODELS:
                pair = [(sign * record[field], flag)
                        for kind, flag in ((other, 0), (target, 1))
                        for record in runs[("main", dataset, model, MAIN_RANK, kind)].values()]
                rows += [(z, flag, f"{dataset}|{model}") for z, flag in standardised(pair)]
        effect, low, high, p = ols_phase([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
        ahead = sum(statistics.mean(r["test"] for r in runs[("main", d, m, MAIN_RANK, target)].values())
                    > statistics.mean(r["test"] for r in runs[("main", d, m, MAIN_RANK, other)].values())
                    for d in DATASETS for m in MODELS) if field == "test" else None
        out.append({"other": other, "effect_sd": effect, "ci": [low, high], "p": p, "ahead_in": ahead})
    adjusted = multipletests([row["p"] for row in out], alpha=ALPHA, method="holm")[1]
    for row, p_holm in zip(out, adjusted):
        row["p_holm"] = float(p_holm)
    return out


def bonferroni_all(tests):
    """Bonferroni over every test of one outcome (P0, each P1, each P2), without gates: a check of the gatekeeping."""
    entries = [("P0", tests["P0"]["p"])] + [(f"P1 {t['family']}", t["p"]) for t in tests["P1"]]
    entries += [(f"P2 {t['family']} {t['dataset']}", t["p"]) for t in tests["P2"]]
    return {name: min(1.0, p * len(entries)) for name, p in entries}


def repeats_word_4gram(text):
    words = text.split()
    return max(Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3)).values(), default=0) >= 2


def degenerate(hyp, ref):
    """A repeated word 4-gram, or more than twice the reference length in characters."""
    return repeats_word_4gram(hyp) or len(hyp) > 2 * len(ref)


def translation_checks(runs, dataset):
    """One translation dataset under the blocked decoder: corpus BLEU per run with the P2 model; per arm the share
    of replies with a repeated word 4-gram; per family x model the corpus chrF++ difference on the sentences where no
    run of either arm degenerates."""
    chrf = sacrebleu.CHRF(word_order=2)
    refs = next(iter(runs[("main", dataset, MODELS[0], MAIN_RANK, "additive")].values()))["refs"]
    out = {"reference_repeat_share": statistics.mean(map(repeats_word_4gram, refs)), "bleu": [], "repeat_share": {},
           "clean_subset": []}
    for kind in run_sweep.KINDS:
        for model in MODELS:
            arm = runs[("main", dataset, model, MAIN_RANK, kind)]
            for record in arm.values():
                record["bleu"] = sacrebleu.corpus_bleu(record["hyps"], [record["refs"]]).score
            out["repeat_share"][f"{kind}|{model}"] = statistics.mean(
                statistics.mean(map(repeats_word_4gram, record["hyps"])) for record in arm.values())
    for family, (real, phase) in FAMILIES.items():
        raw = []
        for model in MODELS:
            raw += [(score, flag, model) for score, flag in setup_rows(runs, family, dataset, model, "bleu", 1)]
            arms = {kind: runs[("main", dataset, model, MAIN_RANK, kind)].values() for kind in (real, phase)}
            clean = [index for index in range(len(refs))
                     if not any(degenerate(record["hyps"][index], refs[index]) for kind in arms for record in arms[kind])]
            subset = {kind: statistics.mean(chrf.corpus_score([record["hyps"][i] for i in clean], [[refs[i] for i in clean]]).score
                                            for record in arms[kind]) for kind in arms}
            out["clean_subset"].append({"family": family, "model": model, "sentences": len(clean),
                                        "delta_chrf": subset[phase] - subset[real]})
        effect, low, high, p = ols_of(raw)
        out["bleu"].append({"family": family, "effect": effect, "ci": [low, high], "p": p,
                            "real": statistics.mean(r[0] for r in raw if r[1] == 0),
                            "phase": statistics.mean(r[0] for r in raw if r[1] == 1)})
    return out


def grid_edges():
    """Arms whose selected learning rate is the smallest or largest value of the grid."""
    selected = json.loads((RUNS / "selected_hparams.json").read_text())
    edges = (min(run_sweep.LR_GRID), max(run_sweep.LR_GRID))
    return {key: entry["selected"]["lr"] for key, entry in selected.items() if entry["selected"]["lr"] in edges}


def post_hoc(runs, tests):
    return {"heterogeneity": heterogeneity(runs),
            "model_interaction": model_interaction(runs, "test", 1),
            "model_interaction_test_loss": model_interaction(runs, "test_loss", -1),
            "bonferroni_all": bonferroni_all(tests),
            "translation": {dataset: translation_checks(runs, dataset) for dataset in TRANSLATION},
            "grid_edge_lr": grid_edges(),
            "phase_lora_vs_all": {"test": method_vs_all(runs, "polar", "test", 1),
                                  "test_loss": method_vs_all(runs, "polar", "test_loss", -1)}}


# ---------------------------------------------------------------- tables and figures

def arm_stats(runs, block, dataset, model, rank, kind, field="test"):
    values = [record[field] for record in runs[(block, dataset, model, rank, kind)].values()]
    return statistics.mean(values), statistics.stdev(values)


def fmt_p(p):
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.0e}"


def write_tables(runs, base, tests, loss_tests):
    columns = [(dataset, model) for dataset in DATASETS for model in MODELS]
    header = " & ".join(f"{TASK_NAMES[d]} {MODEL_NAMES[m]}" for d, m in columns)

    # main: mean (sd) over 5 seeds; base row; the better arm of each pair in bold
    lines = ["\\begin{tabular}{l" + "c" * len(columns) + "}", "\\toprule", f"Method & {header} \\\\", "\\midrule",
             "Base model & " + " & ".join(f"{base[(d, m)]:.1f}" for d, m in columns) + " \\\\", "\\midrule"]
    for family, (real, phase) in FAMILIES.items():
        for kind in (real, phase):
            other = phase if kind == real else real
            cells = []
            for dataset, model in columns:
                mean, sd = arm_stats(runs, "main", dataset, model, MAIN_RANK, kind)
                better = mean > arm_stats(runs, "main", dataset, model, MAIN_RANK, other)[0]
                cells.append(f"\\textbf{{{mean:.1f}}} ({sd:.1f})" if better else f"{mean:.1f} ({sd:.1f})")
            lines.append(f"{NAMES[kind]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if family != list(FAMILIES)[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")

    # tests: P2 effect [95 % CI] p (Holm p) per family x task; P1 per family; P0 headline
    def test_table(results, unit_note, name):
        lines = ["\\begin{tabular}{l" + "c" * len(DATASETS) + "c}", "\\toprule",
                 "Family & " + " & ".join(TASK_NAMES[d] for d in DATASETS) + " & All tasks (sd units) \\\\", "\\midrule"]
        for family in FAMILIES:
            cells = []
            for dataset in DATASETS:
                t = next(t for t in results["P2"] if t["family"] == family and t["dataset"] == dataset)
                cells.append(f"{t['effect']:+.2f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}], $p_H$={fmt_p(t['p_holm'])}")
            f1 = next(t for t in results["P1"] if t["family"] == family)
            cells.append(f"{f1['effect_sd']:+.2f} [{f1['ci'][0]:+.2f}, {f1['ci'][1]:+.2f}], $p_H$={fmt_p(f1['p_holm'])}")
            lines.append(f"{family} & " + " & ".join(cells) + " \\\\")
        p0 = results["P0"]
        lines += ["\\midrule", f"\\multicolumn{{{len(DATASETS) + 1}}}{{l}}{{All families, all tasks ({unit_note})}} & "
                  f"{p0['effect_sd']:+.2f} [{p0['ci'][0]:+.2f}, {p0['ci'][1]:+.2f}], $p$={fmt_p(p0['p'])} \\\\",
                  "\\bottomrule", "\\end{tabular}"]
        (OUT / name).write_text("\n".join(lines) + "\n")
    test_table(tests, "sd units", "tab_tests.tex")
    test_table(loss_tests, "sd units; positive = lower test loss", "tab_loss_tests.tex")

    # translation decoding, one table per dataset: greedy (seed 1) chrF++ and runaway share; blocked chrF++ and
    # runaway share (5 seeds)
    for dataset in TRANSLATION:
        lines = ["\\begin{tabular}{l" + "cccc" * len(MODELS) + "}", "\\toprule",
                 "".join(f" & \\multicolumn{{4}}{{c}}{{{MODEL_NAMES[m]}}}" for m in MODELS) + " \\\\",
                 "Method" + " & greedy & runaway \\% & blocked & runaway \\%" * len(MODELS) + " \\\\",
                 "\\midrule",
                 "Base model & " + " & ".join(f"{base[(f'greedy|{dataset}', m)]:.1f} & {base[(f'runaway|{dataset}', m)]:.1f} & "
                                             f"{base[(dataset, m)]:.1f} & --" for m in MODELS) + " \\\\"]
        for kind in run_sweep.KINDS:
            cells = []
            for model in MODELS:
                arm = runs[("main", dataset, model, MAIN_RANK, kind)]
                cells += [f"{arm[1]['greedy']:.1f}", f"{arm[1]['runaway']:.1f}",
                          f"{statistics.mean(x['test'] for x in arm.values()):.1f}",
                          f"{statistics.mean(x['runaway_blocked'] for x in arm.values()):.1f}"]
            lines.append(f"{NAMES[kind]} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / f"tab_{dataset}.tex").write_text("\n".join(lines) + "\n")


def write_figures(runs):
    """Standardised effect (phase minus real, sd units) with 95 % CI per family x task."""
    plt.rcParams.update({"font.size": 8, "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    labels, effects, lows, highs = [], [], [], []
    for family in FAMILIES:
        for dataset in DATASETS:
            rows = []
            for model in MODELS:
                rows += [(z, flag, model) for z, flag in standardised(setup_rows(runs, family, dataset, model, "test", 1))]
            effect, low, high, _ = ols_phase([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
            labels.append(f"{family} {PLAIN_TASK_NAMES[dataset]}")
            effects.append(effect)
            lows.append(effect - low)
            highs.append(high - effect)
    positions = np.arange(len(labels))[::-1]
    ax.errorbar(effects, positions, xerr=[lows, highs], fmt="o", color="black", markersize=3, capsize=2, linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("phase minus real (sd units)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_effects.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, toks, caps = load()
    base = load_base(toks, caps)
    if len(BLOCK_RULES) != 1:
        sys.exit(f"blocked scores use {len(BLOCK_RULES)} different repetition rules: {sorted(BLOCK_RULES)}")
    tests = family_tests(runs, "test", 1)
    loss_tests = family_tests(runs, "test_loss", -1)
    pooled = {"test": pooled_sensitivity(runs, "test", 1), "test_loss": pooled_sensitivity(runs, "test_loss", -1)}
    write_tables(runs, base, tests, loss_tests)
    write_figures(runs)
    numbers = {
        "block_rule": json.loads(next(iter(BLOCK_RULES))),
        "tests": tests, "loss_tests": loss_tests, "sensitivity": sensitivity(runs), "pooled_sensitivity": pooled,
        "bootstrap_draws": BOOTSTRAP_DRAWS,     # a bootstrap_p of 0 means p < 1 / draws
        "post_hoc": post_hoc(runs, tests),
        "base": {f"{d}|{m}": v for (d, m), v in base.items()},
        "arms": {"|".join(map(str, key)): {"mean": statistics.mean(r["test"] for r in seeds.values()),
                                           "sd": statistics.stdev(r["test"] for r in seeds.values()),
                                           "loss_mean": statistics.mean(r["test_loss"] for r in seeds.values()),
                                           "lr": next(iter(seeds.values()))["lr"],
                                           "diverged": sum(r["diverged_at"] is not None for r in seeds.values()),
                                           "best_epoch_mean": statistics.mean(best_epoch(r) for r in seeds.values()),
                                           "length_ratio_mean": (statistics.mean(r["length_ratio"] for r in seeds.values())
                                                                 if key[1] in TRANSLATION else None)}
                 for key, seeds in runs.items()},
        "trainable": {f"{m}|{k}": trainable(k, m, MAIN_RANK) for m in MODELS for k in run_sweep.KINDS},
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, allow_nan=False))
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
