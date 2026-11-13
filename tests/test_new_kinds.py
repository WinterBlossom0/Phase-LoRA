"""The dora, phase_dora and phase_stiefel kinds: they attach to a real Qwen3 graph, train, keep their
parameter counts, and the batched-chart path with gradient checkpointing gives the plain path's
gradients. Also the repetition block of the scorer and the record of valid older code versions."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from phase_lora.run_config import LEGACY_VERSIONS, train_version_ok
from phase_lora.train_utils import attach, make_optimizer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_test", ROOT / "scripts" / "score_test.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)

CONFIG = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64)
NEW_KINDS = ["phase_stiefel", "dora", "phase_dora"]
RANK = 8


def build(kind, checkpointing):
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(CONFIG).to(torch.bfloat16)
    base.requires_grad_(False)
    torch.manual_seed(1)
    model = attach(base, kind, RANK, RANK, lam=0.1)
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()
    return model


def trainable(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def expected_count(kind):
    # q_proj (64 -> 64) and v_proj (64 -> 32) in each of 2 layers
    total = 0
    for in_f, out_f in ((64, 64), (64, 32)):
        total += RANK * (in_f + out_f)
        total += {"phase_stiefel": RANK ** 2, "dora": out_f, "phase_dora": out_f}[kind]
    return CONFIG.num_hidden_layers * total


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_new_kind_counts_gradients_and_one_step(kind):
    ids = torch.randint(0, 128, (2, 12))
    labels = ids.clone()
    labels[:, :4] = -100
    plain, fast = build(kind, checkpointing=False), build(kind, checkpointing=True)
    assert sum(p.numel() for p in trainable(fast).values()) == expected_count(kind)
    # the plain path: no batched charts, so each module builds its own factors
    plain._forward_pre_hooks.clear()
    with torch.no_grad():
        for p_plain, p_fast in zip(trainable(plain).values(), trainable(fast).values()):
            noise = 0.05 * torch.randn_like(p_plain)
            p_plain.add_(noise)
            p_fast.add_(noise)
    for model in (plain, fast):
        model(input_ids=ids, labels=labels).loss.backward()
    for (name, p_plain), p_fast in zip(trainable(plain).items(), trainable(fast).values()):
        scale = p_plain.grad.abs().max().item() + 1e-12
        assert (p_plain.grad - p_fast.grad).abs().max().item() / scale < 1e-3, name
        assert p_fast.grad.abs().max() > 0, f"{name} gets no gradient"
    params = list(trainable(fast).values())
    make_optimizer(params, 1e-2, 1.0).step()


def test_block_stops_the_third_use_of_a_4_gram_but_not_the_prompt():
    prompt = [[7, 1, 2, 3, 9]]                 # the prompt holds 1 2 3 9 once; it must not count
    block = score.NgramThirdUseBlock(prompt_width=5)
    reply = []
    for token in [1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3]:
        reply.append(token)
        scores = block(torch.tensor([prompt[0] + reply]), torch.zeros(1, 16))
    # reply so far: 1 2 3 9 1 2 3 9 1 2 3 -> "1 2 3 9" used twice, so 9 is now forbidden
    assert scores[0, 9] == -math.inf
    assert torch.isfinite(scores[0, :9]).all() and torch.isfinite(scores[0, 10:]).all()
    second = score.NgramThirdUseBlock(prompt_width=5)
    for length in range(1, 8):                 # 1 2 3 9 1 2 3: one earlier use only
        scores = second(torch.tensor([prompt[0] + [1, 2, 3, 9, 1, 2, 3][:length]]), torch.zeros(1, 16))
    assert torch.isfinite(scores).all(), "the second use must stay allowed"


def forbidden_by_definition(reply, vocab):
    """Tokens whose choice would make some 4-gram of the reply occur a third time."""
    return {token for token in range(vocab)
            if sum(tuple(reply[i:i + 4]) == tuple(reply[-3:] + [token]) for i in range(len(reply) - 3)) >= 2}


def test_block_in_generation_equals_the_definition_on_every_row_and_step():
    # a small vocabulary forces many repeats; rows decode in one batch, as in generate_norepeat
    generator = torch.Generator().manual_seed(0)
    vocab, rows, steps, width = 3, 6, 60, 4
    prompts = torch.randint(vocab, (rows, width), generator=generator)
    block = score.NgramThirdUseBlock(prompt_width=width)
    ids = prompts.clone()
    for _ in range(steps):
        scores = block(ids, torch.rand(rows, vocab, generator=generator))
        for row in range(rows):
            reply = ids[row, width:].tolist()
            banned = {token for token in range(vocab) if scores[row, token] == -math.inf}
            assert banned == forbidden_by_definition(reply, vocab)
        if torch.isinf(scores).all(dim=1).any():
            break                              # every token banned: the definition allows no next token
        ids = torch.cat([ids, scores.argmax(dim=1, keepdim=True)], dim=1)
    for row in range(rows):
        reply = ids[row, width:].tolist()
        uses = [sum(tuple(reply[i:i + 4]) == tuple(reply[j:j + 4]) for i in range(len(reply) - 3))
                for j in range(len(reply) - 3)]
        assert max(uses) <= 2, "a 4-gram occurs three times"


def test_block_counts_tokens_that_arrive_several_at_once_and_refuses_a_new_batch():
    reply = [1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3]
    step_by_step = score.NgramThirdUseBlock(prompt_width=0)
    for length in range(len(reply) + 1):
        expected = step_by_step(torch.tensor([reply[:length]]), torch.zeros(1, 16))
    at_once = score.NgramThirdUseBlock(prompt_width=0)
    at_once(torch.tensor([reply[:2]]), torch.zeros(1, 16))
    assert torch.equal(at_once(torch.tensor([reply]), torch.zeros(1, 16)), expected)
    with pytest.raises(ValueError):
        at_once(torch.tensor([reply[:3]]), torch.zeros(1, 16))
    with pytest.raises(ValueError):
        at_once(torch.tensor([reply, reply]), torch.zeros(2, 16))


def test_a_blocked_score_is_reused_only_from_a_scorer_recorded_for_its_mode(tmp_path):
    greedy_only = next(v for v, entry in LEGACY_VERSIONS["score"].items() if "modes" not in entry)
    every_mode = next(v for v, entry in LEGACY_VERSIONS["score"].items() if "modes" in entry)
    out = tmp_path / "run.json"
    out.write_text(json.dumps({"versions": [greedy_only, "train1"]}))
    assert score.is_current(out, "current", "train1", "mafand_tsn", "greedy")
    assert not score.is_current(out, "current", "train1", "mafand_tsn", "norepeat")
    assert not score.is_current(out, "current", "train2", "mafand_tsn", "greedy")
    assert not score.is_current(out, "current", "train1", "mafand_tsn", "loss")
    out.write_text(json.dumps({"versions": [every_mode, "train1"]}))
    assert score.is_current(out, "current", "train1", "mafand_tsn", "norepeat")
    assert score.is_current(out, "current", "train1", "arc", "loss")
    out.write_text(json.dumps({"versions": ["000000000000", "train1"]}))
    assert not score.is_current(out, "current", "train1", "mafand_tsn", "greedy")
    out.write_text(json.dumps({"versions": ["current", "train1"]}))
    assert score.is_current(out, "current", "train1", "mafand_tsn", "norepeat")


def test_row_losses_add_up_to_the_models_own_loss():
    # right-padded rows with masked prompts, as collate builds them for the dev loss
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(CONFIG).eval()
    rows = [(torch.randint(128, (n,)), torch.cat([torch.full((p,), -100), torch.randint(128, (n - p,))]))
            for n, p in [(9, 4), (6, 2), (12, 7)]]
    rows = [(ids, torch.where(labels == -100, labels, ids)) for ids, labels in rows]
    ids, labels, mask = score.collate(rows, pad=0)
    with torch.no_grad():
        nll, tokens = score.row_losses(model, ids, labels, mask)
        expected = model(input_ids=ids, attention_mask=mask, labels=labels).loss.item()
    assert tokens == [5, 4, 5]
    assert math.isclose(sum(nll) / sum(tokens), expected, rel_tol=1e-5)


def test_recorded_old_versions_cover_only_their_kinds():
    old_train = next(iter(LEGACY_VERSIONS["train"]))
    assert train_version_ok(old_train, "additive")
    assert not train_version_ok(old_train, "phase_stiefel")
    assert not train_version_ok("000000000000", "additive")
    old_score = next(iter(LEGACY_VERSIONS["score"]))
    assert score.score_valid(old_score, "current", "svamp", "greedy")
    assert not score.score_valid("000000000000", "current", "svamp", "greedy")


def test_one_model_load_scores_every_missing_mode_and_only_those(tmp_path, monkeypatch):
    dirs = {mode: tmp_path / mode for mode in score.MODE_DIRS}
    for directory in dirs.values():
        directory.mkdir()
    monkeypatch.setattr(score, "MODE_DIRS", dirs)
    monkeypatch.setattr(score, "release_gpu", lambda: None)
    calls, loads = [], []

    def write(out, record, tok, model, dataset, mode, version, max_tokens):
        calls.append((record["tag"], mode))

    monkeypatch.setattr(score, "score_and_write", write)

    def load():
        loads.append(1)
        return "tok", "model"

    old_score = next(iter(LEGACY_VERSIONS["score"]))
    for tag in ("translation", "maths"):
        (dirs["greedy"] / f"{tag}.json").write_text(json.dumps({"versions": [old_score, "train1"]}))
    modes = ["greedy", "norepeat", "loss"]
    score.score_model(load, {"tag": "translation", "train_version": "train1"}, "mafand_tsn", modes, "current", 3072)
    score.score_model(load, {"tag": "maths", "train_version": "train1"}, "svamp", modes, "current", 3072)
    assert calls == [("translation", "norepeat"), ("translation", "loss"), ("maths", "loss")]
    assert len(loads) == 2, "one load per run"
    for tag, mode in calls:
        (dirs[mode] / f"{tag}.json").write_text(json.dumps({"versions": ["current", "train1"]}))
    score.score_model(load, {"tag": "maths", "train_version": "train1"}, "svamp", modes, "current", 3072)
    assert len(loads) == 2, "a run with every score current is not loaded"


def test_grouped_decode_attention_gives_the_same_padded_generation():
    # left-padded GQA batch (4 query heads share 2 key/value heads), greedy, fp32 on the CPU
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(CONFIG).eval()
    ids = torch.randint(1, 128, (3, 6))
    mask = torch.ones_like(ids)
    mask[0, :3], mask[1, :1] = 0, 0
    def run(implementation):
        model.set_attn_implementation(implementation)
        with torch.no_grad():
            return model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=12,
                                  pad_token_id=0, return_dict_in_generate=True, output_scores=True)
    plain, grouped = run("sdpa"), run(score.GROUPED_DECODE)
    assert torch.equal(plain.sequences, grouped.sequences)
    for step_plain, step_grouped in zip(plain.scores, grouped.scores):
        assert torch.allclose(step_plain, step_grouped, atol=1e-5)
