"""run_sweep selection and run naming: the main runs must use the grid's best setting."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

from phase_lora.run_config import LAM_KINDS, TRAIN_CODE, code_version, parse, run_tag

spec = importlib.util.spec_from_file_location(
    "run_sweep", Path(__file__).resolve().parents[1] / "scripts" / "run_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

EVERY = {model: {dataset: 10 for dataset in sweep.DATASETS} for model in sweep.MODELS}


def fake_score(settings):
    """Lowest at lr 1e-3 and, for PoLAR kinds, lam 5e-3."""
    score = abs(math.log10(settings["lr"]) + 3)
    if settings["lam"] is not None:
        score += abs(math.log10(settings["lam"]) - math.log10(5e-3))
    return 2.0 + score


def write_grid(tmp_path, diverge_best=False, version=None):
    version = version or code_version(TRAIN_CODE)
    for argv in sweep.grid_runs(EVERY):
        settings = vars(parse(argv))
        best = fake_score(settings)
        diverged = 7 if diverge_best and best == 2.0 else None
        (tmp_path / f"{run_tag(settings)}.json").write_text(
            json.dumps({"best_eval": best, "diverged_at": diverged, "code_version": version}))


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(sweep, "SELECTION", tmp_path / "selected_hparams.json")
    return tmp_path


def test_selects_lowest_dev_loss(runs_dir):
    write_grid(runs_dir)
    chosen = sweep.select(EVERY)
    arms = [arm for block in sweep.BLOCKS for arm in sweep.block_arms(block)]
    assert len(chosen) == len(arms) == len(set(arms)), "one selection per arm, no arm in two blocks"
    for key, entry in chosen.items():
        assert entry["selected"]["lr"] == 1e-3, key
        expected_lam = 5e-3 if key.split("|")[2] in LAM_KINDS else None
        assert entry["selected"]["lam"] == expected_lam, key
    main = list(sweep.main_runs(EVERY, chosen))
    assert len(main) == sum(len(list(sweep.block_arms(block))) * len(block["seeds"]) for block in sweep.BLOCKS)
    assert all(vars(parse(argv))["lr"] == 1e-3 for argv in main)


def test_diverged_run_never_selected(runs_dir):
    write_grid(runs_dir, diverge_best=True)
    for entry in sweep.select(EVERY).values():
        assert entry["selected"]["best_eval"] > 2.0


def test_results_from_other_code_stop_the_sweep(runs_dir):
    write_grid(runs_dir, version="000000000000")
    with pytest.raises(SystemExit, match="other code"):
        sweep.select(EVERY)


def test_incomplete_grid_stops_main(runs_dir):
    write_grid(runs_dir)
    next(runs_dir.glob("*.json")).unlink()
    with pytest.raises(SystemExit, match="grid incomplete"):
        sweep.select(EVERY)


def test_tag_changes_with_any_setting():
    argv = next(sweep.grid_runs(EVERY))
    base = run_tag(vars(parse(argv)))
    position = argv.index("--patience") + 1
    changed = argv[:position] + ["3"] + argv[position + 1:]
    assert run_tag(vars(parse(changed))) != base
    assert run_tag(vars(parse(list(argv)))) == base


def test_stop_signal_during_start_stops_the_run(tmp_path, monkeypatch):
    """A stop that lands while Popen runs must still end the new child, not orphan it."""
    real_popen = sweep.subprocess.Popen
    started = []

    def popen_then_signal(_command, **kwargs):
        child = real_popen(["sleep", "60"], **kwargs)
        started.append(child)
        sweep.STOP.set()           # as the handler does when the signal lands here
        return child

    monkeypatch.setattr(sweep.subprocess, "Popen", popen_then_signal)
    try:
        with pytest.raises(SystemExit):
            sweep.run_one(["--kind", "additive"], tmp_path / "run.log")
    finally:
        sweep.STOP.clear()
    assert started[0].poll() is not None


def test_a_run_that_fails_twice_stops_the_sweep_and_one_that_recovers_does_not(runs_dir, monkeypatch):
    monkeypatch.setattr(sweep, "FAILURES", runs_dir / "sweep_failures.log")
    runs = list(sweep.grid_runs(EVERY))[:3]
    attempts = []

    def flaky(argv, log_path):                 # the first run fails once, the second always
        attempts.append(argv)
        if argv == runs[1]:
            return 1
        return 1 if argv == runs[0] and attempts.count(argv) == 1 else 0

    monkeypatch.setattr(sweep, "run_one", flaky)
    with pytest.raises(SystemExit, match="1 runs failed twice"):
        sweep.execute(runs, "grid", dry_run=False)
    assert attempts == runs + [runs[0], runs[1]], "each failed run gets exactly one retry"
    attempts.clear()

    def healthy(argv, log_path):
        attempts.append(argv)
        return 0

    monkeypatch.setattr(sweep, "run_one", healthy)
    sweep.execute(runs, "grid", dry_run=False)
    assert attempts == runs


def test_list_phase_prints_every_main_run_once_and_filters(runs_dir, monkeypatch, capsys):
    write_grid(runs_dir)
    sweep.select(EVERY)
    monkeypatch.setattr(sweep, "eval_every", lambda model: EVERY[model])

    def listed(*extra):
        monkeypatch.setattr("sys.argv", ["run_sweep.py", "--phase", "list", *extra])
        sweep.main()
        return capsys.readouterr().out.split()

    everything = listed()
    expected = sum(len(list(sweep.block_arms(block))) * len(block["seeds"]) for block in sweep.BLOCKS)
    assert len(everything) == len(set(everything)) == expected == 360
    seed1_translation = listed("--datasets", "mafand_tsn", "--seeds", "1")
    assert len(seed1_translation) == 3 * 6, "3 models x 6 kinds"
    assert all("_mafand_tsn_" in path and "-base_s1_" in path for path in seed1_translation)
    assert set(seed1_translation) <= set(everything)


def test_an_ignored_hangup_stays_ignored(monkeypatch):
    # nohup sets SIGHUP to ignore; the sweep must not replace that with its stop handler
    import signal
    monkeypatch.setattr("sys.argv", ["run_sweep.py", "--phase", "grid", "--dry-run"])
    monkeypatch.setattr(sweep, "eval_every", lambda model: EVERY[model])
    monkeypatch.setattr(sweep, "execute", lambda runs, label, dry_run: None)
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        sweep.main()
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)
