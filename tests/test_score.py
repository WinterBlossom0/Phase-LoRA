"""score_test's answer parsing, its left-padded batches and its generation settings."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_test", ROOT / "scripts" / "score_test.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


@pytest.mark.parametrize("text, value", [
    ("( 290.0 / 2.0 ) = 145", 145.0),
    ("( 3.0 - 5.0 ) = -2", -2.0),
    ("( 1.0 / 4.0 ) = 0.25", 0.25),
    ("( 8.0 + 2.0 ) = 10\nQuestion: next", 10.0),
    ("21", 21.0),
    ("( 8.0 + 2.0 )", None),
    ("I don't know", None),
    ("there are 3 apples", None),
])
def test_svamp_value(text, value):
    assert score.svamp_value(text) == value


@pytest.mark.skipif(not (ROOT / "models" / "qwen3-0.6b-base").is_dir(), reason="needs the tokenizer")
def test_prompt_batches_left_pad_and_keep_every_prompt(monkeypatch):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / "qwen3-0.6b-base")
    monkeypatch.setattr(score.torch.Tensor, "cuda", lambda self: self)
    prompts = ["short", "a much longer prompt than the first one", "mid length prompt"]
    seen = []
    for indices, ids, mask in score.prompt_batches(tok, prompts):
        for row, index in enumerate(indices):
            real = ids[row][mask[row].bool()].tolist()
            assert real == tok(prompts[index], add_special_tokens=False)["input_ids"]
            assert mask[row, -1] == 1, "left padding: the last position is always real"
            seen.append(index)
    assert sorted(seen) == [0, 1, 2]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(score.MODEL_DIR / "qwen3-0.6b-base")


def test_newline_tokens_end_a_reply_at_its_first_newline(tok):
    stops = set(score.newline_token_ids(tok))
    reply = tok("Ke a leboga.\nEnglish: next", add_special_tokens=False)["input_ids"]
    first_stop = next(i for i, token_id in enumerate(reply) if token_id in stops)
    assert "\n" not in tok.decode(reply[:first_stop]) and "\n" in tok.decode(reply[:first_stop + 1])


def test_fixed_kv_cache_holds_every_generation_task(tok):
    """A full cache evicts requests, which would make outputs depend on timing."""
    for dataset in (*score.TRANSLATION, "svamp"):
        lengths = [len(tok(prompt, add_special_tokens=False)["input_ids"])
                   for split in ("dev", "test") for prompt, _ in score.LOCAL_PAIRS[dataset](split)]
        assert score.kv_blocks_needed(lengths, score.generation_cap(tok, dataset)) <= score.CB_BLOCKS
