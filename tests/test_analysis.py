"""The tests of paper_analysis: the OLS with setup fixed effects equals statsmodels' formula
interface, the standardisation is per setup, and a planted effect is found with the right sign."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

spec = importlib.util.spec_from_file_location("paper_analysis", Path(__file__).resolve().parents[1] / "scripts" / "paper_analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_ols_phase_equals_the_formula_model_with_hc3():
    rng = np.random.default_rng(0)
    groups = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    phase = [0, 1] * 15
    y = [rng.normal(0.8 * f + {"a": 0, "b": 3, "c": -2}[g], 1 + (g == "b")) for f, g in zip(phase, groups)]
    effect, low, high, p = analysis.ols_phase(y, phase, groups)
    fit = smf.ols("y ~ C(g) + phase", data=pd.DataFrame({"y": y, "g": groups, "phase": phase})).fit(cov_type="HC3", use_t=True)
    assert np.isclose(effect, fit.params["phase"]) and np.isclose(p, fit.pvalues["phase"])
    assert np.allclose([low, high], fit.conf_int(0.05).loc["phase"])


def test_standardised_scores_have_zero_mean_and_pooled_unit_spread():
    rows = [(10.0, 0), (12.0, 0), (11.0, 0), (20.0, 1), (22.0, 1), (21.0, 1)]
    z = analysis.standardised(rows)
    assert np.isclose(np.mean([s for s, _ in z]), 0)
    real = [s for s, f in z if f == 0]
    assert np.isclose(np.std(real, ddof=1), 1.0), "pooled within-arm sd is the unit"


def test_a_planted_effect_is_found_and_a_null_is_not():
    rng = np.random.default_rng(1)
    groups = [g for g in "abcdef" for _ in range(10)]
    phase = [0, 1] * 30
    planted = analysis.ols_phase([rng.normal(1.0 * f, 1) for f in phase], phase, groups)
    null = analysis.ols_phase([rng.normal(0.0, 1) for _ in phase], phase, groups)
    assert planted[0] > 0 and planted[3] < 0.01
    assert null[3] > 0.05


def fake_runs(effect_of_family, seed=0):
    """Main-block runs of every setup with 5 seeds per arm and a planted phase effect per family (sd units)."""
    rng = np.random.default_rng(seed)
    runs = {}
    for family, (real, phase) in analysis.FAMILIES.items():
        for dataset in analysis.DATASETS:
            for model in analysis.MODELS:
                for kind, flag in ((real, 0), (phase, 1)):
                    runs[("main", dataset, model, analysis.MAIN_RANK, kind)] = {
                        s: {"test": rng.normal(flag * effect_of_family[family], 1)} for s in range(1, 6)}
    return runs


def test_random_effects_equal_statsmodels_and_family_heterogeneity_is_found():
    from statsmodels.stats.meta_analysis import combine_effects
    runs = fake_runs({"LoRA": 2.0, "DoRA": 2.0, "PoLAR": -1.0})
    result = analysis.heterogeneity(runs)
    effects = analysis.setup_effects(runs)
    reference = combine_effects(np.array([v[0] for v in effects.values()]), np.array([v[1] for v in effects.values()]),
                                method_re="dl")
    assert np.isclose(result["random_effects_d"], reference.mean_effect_re)
    assert np.isclose(result["tau2"], reference.tau2) and np.isclose(result["cochran_q"], reference.q)
    assert result["family_x_phase"]["p"] < 1e-3
    assert analysis.heterogeneity(fake_runs({"LoRA": 1.0, "DoRA": 1.0, "PoLAR": 1.0}, seed=1))["family_x_phase"]["p"] > 0.01


def test_a_repeated_word_4gram_is_found_only_when_present():
    assert analysis.repeats_word_4gram("a b c d x a b c d")
    assert not analysis.repeats_word_4gram("a b c d a b c e")
    assert not analysis.repeats_word_4gram("a b c")
    assert analysis.degenerate("x" * 21, "y" * 10) and not analysis.degenerate("x" * 20, "y" * 10)
