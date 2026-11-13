"""Checks of the pinned datasets: files match the manifest, splits do not leak, and
the loss mask scores the reply and its end token and nothing else. Skipped until
scripts/prepare_data.py has run and models/qwen3-0.6b-base exists."""
import hashlib
import json

import pytest

from phase_lora import DATA_DIR, MODEL_DIR
from phase_lora.train_utils import END_OF_TEXT, EVAL_SPLIT, LOCAL_PAIRS, build_data

MANIFEST = DATA_DIR / "MANIFEST.json"
TOKENIZER_DIR = MODEL_DIR / "qwen3-0.6b-base"
# News sentences that the published MAFAND splits share: en-tsn one in train and dev, en-lug one in train and
# dev and one in train and test. Any other shared prompt is a leak.
KNOWN_SHARED = {"mafand_tsn": {"Translate English to Setswana.\n"
                               "English: Magashule is facing corruption charges among others.\nSetswana:"},
                "mafand_lug": {"Translate English to Luganda.\n"
                               "English: Do the Hutu and Tutsi of Rwanda nurse ancient hatred?\nLuganda:",
                               "Translate English to Luganda.\n"
                               "English: He claimed that Kayumba had sent a message to Museveni seeking contact.\nLuganda:"}}

pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason="run scripts/prepare_data.py first")


def test_files_match_manifest():
    for key, entry in json.loads(MANIFEST.read_text()).items():
        path = DATA_DIR / f"{key}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], key


@pytest.mark.parametrize("name", list(LOCAL_PAIRS))
def test_no_leak_into_eval_splits(name):
    train = {prompt for prompt, _ in LOCAL_PAIRS[name]("train")}
    for split in ("dev", "test"):
        shared = train & {prompt for prompt, _ in LOCAL_PAIRS[name](split)}
        assert shared <= KNOWN_SHARED.get(name, set()), (split, sorted(shared)[:3])


def test_early_stopping_never_reads_test():
    assert all(EVAL_SPLIT[name] == "dev" for name in LOCAL_PAIRS)


def test_arc_replies_are_letters():
    for split in ("train", "dev", "test"):
        assert {reply for _, reply in LOCAL_PAIRS["arc"](split)} <= {" A", " B", " C", " D", " E"}


@pytest.mark.skipif(not TOKENIZER_DIR.is_dir(), reason="needs models/qwen3-0.6b-base")
@pytest.mark.parametrize("name", list(LOCAL_PAIRS))
def test_mask_scores_reply_and_end_only(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    pairs = LOCAL_PAIRS[name]("train")
    rows = build_data(tok, "train", 10**9, name)
    assert len(rows) == len(pairs), "a row was dropped"
    for (ids, labels), (prompt, reply) in zip(rows[:50], pairs[:50]):
        assert tok.decode(ids) == prompt + reply + END_OF_TEXT, "a row was cut or altered"
        assert tok.decode(ids[labels != -100]) == reply + END_OF_TEXT
        assert tok.decode(ids[labels == -100]) == prompt
