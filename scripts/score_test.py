"""Test-set scores for finished runs, from their best-dev weights.

  python scripts/score_test.py                         # greedy scores of every finished main run
  python scripts/score_test.py --modes greedy norepeat loss   # all three, one model load per run
  python scripts/score_test.py --runs runs/<tag>.json [...]
  python scripts/score_test.py --base --modes ...      # the untrained base models, every dataset

mafand_tsn, mafand_lug: corpus chrF++ (sacrebleu, word order 2) of greedy translations, by
            deterministic continuous batching (see generate); LFM3 by padded batches (see has_conv_layers). Mode norepeat
            instead blocks the third use of any 4-token sequence in a reply
            (see NgramThirdUseBlock) and writes to runs/test_norepeat/.
svamp:      the number after the last '=' of the greedy reply equals the answer.
arc:        the option letter with the highest next-token logit after the prompt.
loss mode:  every dataset's teacher-forced test loss, computed as train_compare computes the
            dev loss (mean loss per reply token, end token included), to runs/test_loss/.

Runs under python3 (the default env has sacrebleu). Takes the shared GPU lock for
its whole run. Per-item results go to runs/<mode dir>/<tag>.json, so two methods can be
compared question by question; a score that is current for its run is never redone.
"""

import argparse
import fcntl
import gc
import json
import math
import os
import re
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # run without installing
sys.path.insert(0, str(ROOT / "scripts"))

# same allocator settings as train_compare.py; see the reason there
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,per_process_memory_fraction:0.9"

import sacrebleu
import torch
import torch.nn.functional as F
from peft.helpers import DoraCaching
from transformers import (AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, LogitsProcessor, LogitsProcessorList)
from transformers.generation import ContinuousBatchingConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

from phase_lora import MODEL_DIR, RUNS_DIR
from phase_lora.run_config import LEGACY_VERSIONS, SCORE_CODE, code_version, train_version_ok
from phase_lora.train_utils import (END_OF_TEXT, LOCAL_PAIRS, TRANSLATION, arc_letter, attach, build_data, collate,
                                    token_batches, _local_rows)

GPU_LOCK = os.environ.get("PHASE_LORA_GPU_LOCK", str(Path.home() / ".gpu.lock"))   # the lock flock(1) takes for training
GENERATION_BATCH = 64                  # ARC rows per forward pass
# Continuous batching sizes, fixed so outputs do not depend on free GPU memory.
# 256 blocks x 256 tokens hold 64 replies at the longest test prompt plus cap (test_score).
CB_REQUESTS, CB_BLOCKS, CB_BLOCK_SIZE, CB_BATCH_TOKENS = 64, 256, 256, 2048
TEST_DIR = RUNS_DIR / "test"
NOREPEAT_DIR = RUNS_DIR / "test_norepeat"
LOSS_DIR = RUNS_DIR / "test_loss"
MODE_DIRS = {"greedy": TEST_DIR, "norepeat": NOREPEAT_DIR, "loss": LOSS_DIR}
# Repetition block: a 4-token sequence may occur twice in a reply, not a third time.
# From the train references, as tokenized for training: 13.1 % (Setswana) and 11.3 % (Luganda) repeat some 4-token
# sequence, which a plain no-repeat rule would forbid; 0.81 % (17) and 0.61 % (25) use one 3 or more times.
BLOCK_NGRAM, BLOCK_MAX_USES = 4, 2
NOREPEAT_BATCH = 128                   # the fastest batch size measured for the blocked decoder
MAX_LEFT_BYTES = 256 * 2**20           # GPU memory allowed to stay allocated between runs (one model is 1.2-3.3 GB)
MIN_FREE_RAM_KB = 1_572_864            # 1.5 GB: below this WSL can crash, so scoring stops


def watch_ram():
    """Stop this process if available RAM falls below MIN_FREE_RAM_KB."""
    while True:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        available = next(int(line.split()[1]) for line in meminfo if line.startswith("MemAvailable:"))
        if available < MIN_FREE_RAM_KB:
            print(f"stopping: available RAM {available // 1024} MB", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)


def load_run(result, device="cuda"):
    """Base model plus the run's adapter, built as train_compare built it, with its best weights."""
    args = result["args"]
    torch.manual_seed(args["seed"])          # same init as training, used if no weights were saved
    model_dir = MODEL_DIR / args["model"]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.requires_grad_(False)
    model = attach(model, args["kind"], args["rank"], args["alpha"], lam=args["lam"])
    if result["weights"] is not None:
        weights = torch.load(RUNS_DIR / result["weights"], map_location=device)
        missing_keys, unexpected = model.load_state_dict(weights, strict=False)
        names = {name for name, _ in model.named_parameters()}
        if unexpected or not set(weights) <= names:
            sys.exit(f"{result['tag']}: saved weights do not fit the rebuilt adapter: {unexpected[:3]}")
    model.eval()
    return tok, model


def load_base(model_name, device="cuda"):
    """The untrained base model, for the reference rows."""
    model_dir = MODEL_DIR / model_name
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.requires_grad_(False)
    model.eval()
    return tok, model


def prompt_batches(tok, prompts, batch_size=GENERATION_BATCH):
    """Left-padded batches of prompt ids, longest first, with each prompt's original index."""
    encoded = [tok(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    order = sorted(range(len(encoded)), key=lambda index: -len(encoded[index]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        width = max(len(encoded[index]) for index in indices)
        ids = torch.full((len(indices), width), pad)
        mask = torch.zeros(len(indices), width, dtype=torch.long)
        for row, index in enumerate(indices):
            ids[row, width - len(encoded[index]):] = torch.tensor(encoded[index])
            mask[row, width - len(encoded[index]):] = 1
        yield indices, ids.cuda(), mask.cuda()


def newline_token_ids(tok):
    """Every vocabulary token whose text contains a newline: as stop tokens they end a
    reply at its first newline, as stop_strings=["\\n"] does in model.generate."""
    return [token_id for token_id in range(len(tok)) if "\n" in tok.decode([token_id])]


def kv_blocks_needed(prompt_lengths, max_new_tokens):
    """KV-cache blocks for CB_REQUESTS of the longest requests at full length at once."""
    return CB_REQUESTS * math.ceil((max(prompt_lengths) + max_new_tokens) / CB_BLOCK_SIZE)


def has_conv_layers(model):
    """LFM3 mixes convolution and attention layers. transformers' continuous batching rejects convolution layers
    ("Invalid layer type: conv", 5.18.0), and grouped_decode_attention assumes a full-length KV cache, which
    LFM3's 256-token sliding-window layers do not keep."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return "conv" in (getattr(inner.config, "layer_types", None) or [])


@torch.no_grad()
def generate_padded(tok, model, prompts, max_new_tokens):
    """Greedy replies by left-padded batches, with the stop rules and cleanup of generate(). For models that
    continuous batching cannot run; greedy decoding picks the same tokens up to bf16 rounding."""
    end = tok.convert_tokens_to_ids(END_OF_TEXT)
    stops = [end] + newline_token_ids(tok)
    replies = [None] * len(prompts)
    for indices, ids, mask in prompt_batches(tok, prompts, NOREPEAT_BATCH):
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
                             eos_token_id=stops, pad_token_id=end)
        for row, index in enumerate(indices):
            text = tok.decode(out[row, ids.shape[1]:], skip_special_tokens=True)
            replies[index] = text.split("\n")[0].strip()
    return replies


@torch.no_grad()
def generate(tok, model, prompts, max_new_tokens):
    """Greedy replies, ended by <|endoftext|> or the first newline. No reference reply in
    any split contains a newline, so text past one is never part of an answer; stopping
    there also bounds the time a model that has not learned to stop can take.

    Continuous batching: a finished reply frees its slot for the next prompt, so a reply
    that never stops holds one slot, not a whole padded batch (2.3x faster on mafand_tsn).
    It is deterministic only with every request queued before start(), fixed cache sizes,
    and no async batching or CUDA graphs: the auto-sized default gave 126/300 identical
    outputs on a repeat; this setting gave 1500/1500 on repeats and in a new process."""
    if has_conv_layers(model):
        return generate_padded(tok, model, prompts, max_new_tokens)
    end = tok.convert_tokens_to_ids(END_OF_TEXT)
    ids = [tok(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    needed = kv_blocks_needed([len(row) for row in ids], max_new_tokens)
    if needed > CB_BLOCKS:
        # a full cache evicts requests, and eviction order would make outputs depend on timing
        sys.exit(f"generation needs {needed} KV blocks, more than the fixed {CB_BLOCKS}")
    config = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=end,
                              eos_token_id=[end] + newline_token_ids(tok))
    batching = ContinuousBatchingConfig(num_blocks=CB_BLOCKS, block_size=CB_BLOCK_SIZE, max_batch_tokens=CB_BATCH_TOKENS,
                                        max_requests_per_batch=CB_REQUESTS, use_async_batching=False,
                                        use_cuda_graph=False, allow_block_sharing=False)
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model   # PEFT wraps additive
    manager = inner.init_continuous_batching(generation_config=config, continuous_batching_config=batching)
    request_ids = manager.add_requests(inputs=ids, max_new_tokens=max_new_tokens)
    manager.start()
    results = {}
    try:
        while len(results) < len(ids):
            result = manager.get_result(timeout=1)
            if result is None and not manager.is_running():
                sys.exit(f"generation stopped after {len(results)} of {len(ids)} replies")
            if result is not None and result.is_finished():
                results[result.request_id] = result
    finally:
        manager.stop(block=True)
    failed = [result.error for result in results.values() if result.error]
    if failed:
        sys.exit(f"{len(failed)} replies failed, first: {failed[0]}")
    return [tok.decode(results[request_id].generated_tokens, skip_special_tokens=True).split("\n")[0].strip()
            for request_id in request_ids]


class NgramThirdUseBlock(LogitsProcessor):
    """Forbid any token that would make some BLOCK_NGRAM-token sequence occur a third time
    in the reply. Only the reply is counted, not the prompt, so names copied from the
    source stay allowed. One instance serves one batch of one generate() call."""

    def __init__(self, prompt_width):
        self.prompt_width = prompt_width
        self.counted = 0                 # reply length whose n-grams are already in counts
        self.counts = None               # per row: (n-1)-token prefix -> next token -> uses

    def __call__(self, input_ids, scores):
        reply = input_ids[:, self.prompt_width:].tolist()
        if self.counts is None:
            self.counts = [{} for _ in reply]
        if len(reply) != len(self.counts) or len(reply[0]) < self.counted:
            raise ValueError("NgramThirdUseBlock got a different batch; make one per generate() call")
        for row, tokens in enumerate(reply):
            # every n-gram that ends after the last call, so no step is missed or counted twice
            for end in range(max(self.counted + 1, BLOCK_NGRAM), len(tokens) + 1):
                *prefix, last = tokens[end - BLOCK_NGRAM:end]
                uses = self.counts[row].setdefault(tuple(prefix), {})
                uses[last] = uses.get(last, 0) + 1
            if len(tokens) < BLOCK_NGRAM - 1:
                continue
            for token, count in self.counts[row].get(tuple(tokens[-(BLOCK_NGRAM - 1):]), {}).items():
                if count >= BLOCK_MAX_USES:
                    scores[row, token] = -math.inf
        self.counted = len(reply[0])
        return scores


def grouped_decode_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """HF's sdpa attention, except that a one-token decode step reads the KV cache without copying it.
    With a padding mask, HF's sdpa repeats each key/value head once per query head that shares it; this function
    reads the shared heads directly. Generation is identical to HF's sdpa (test_new_kinds); on the 1.7B model it
    cut blocked En->Tsn scoring from 90 s to 61 s per run (1500/1500 identical outputs)."""
    groups = module.num_key_value_groups
    if query.shape[2] != 1 or groups == 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                                      **kwargs)
    batch, heads, _, dim = query.shape
    # query head h uses key/value head h // groups (repeat_kv's order), so the query heads of one
    # key/value head become its query positions; the (batch, 1, 1, keys) mask broadcasts over them
    grouped = query.view(batch, heads // groups, groups, dim)
    mask = None if attention_mask is None else attention_mask[:, :, :, : key.shape[-2]]
    out = F.scaled_dot_product_attention(grouped, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling)
    return out.reshape(batch, 1, heads, dim), None


GROUPED_DECODE = "grouped_decode_sdpa"
AttentionInterface.register(GROUPED_DECODE, grouped_decode_attention)
AttentionMaskInterface.register(GROUPED_DECODE, sdpa_mask)


@torch.no_grad()
def generate_norepeat(tok, model, prompts, max_new_tokens):
    """Greedy replies with NgramThirdUseBlock, by padded batches (continuous batching passes no
    token history to logits processors). Same stop rules and cleanup as generate()."""
    end = tok.convert_tokens_to_ids(END_OF_TEXT)
    stops = [end] + newline_token_ids(tok)
    replies = [None] * len(prompts)
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model   # PEFT wraps additive and dora
    previous = inner.config._attn_implementation
    if not has_conv_layers(model):
        inner.set_attn_implementation(GROUPED_DECODE)
    try:
        for indices, ids, mask in prompt_batches(tok, prompts, NOREPEAT_BATCH):
            out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
                                 eos_token_id=stops, pad_token_id=end,
                                 logits_processor=LogitsProcessorList([NgramThirdUseBlock(ids.shape[1])]))
            for row, index in enumerate(indices):
                text = tok.decode(out[row, ids.shape[1]:], skip_special_tokens=True)
                replies[index] = text.split("\n")[0].strip()
    finally:
        inner.set_attn_implementation(previous)      # the loss mode must compute as training did
    return replies


def generation_cap(tok, dataset):
    """Twice the longest training reply: room for any real reply, a bound on a runaway one."""
    longest = max(len(tok(reply, add_special_tokens=False)["input_ids"]) for _, reply in LOCAL_PAIRS[dataset]("train"))
    return 2 * longest


def svamp_value(text):
    """The reply's value: the number after the last '=' of its first line, or the line
    itself when it is a bare number. Anything else (no value given) is None."""
    line = text.split("\n")[0].strip()
    match = re.search(r"=\s*(-?\d+(?:\.\d+)?)$", line) or re.fullmatch(r"(-?\d+(?:\.\d+)?)", line)
    return float(match.group(1)) if match else None


def score_translation(dataset, tok, model, decoder="greedy"):
    pairs = LOCAL_PAIRS[dataset]("test")
    decode = generate_norepeat if decoder == "norepeat" else generate
    hyps = decode(tok, model, [prompt for prompt, _ in pairs], generation_cap(tok, dataset))
    refs = [reply.strip() for _, reply in pairs]
    chrf = sacrebleu.CHRF(word_order=2)
    items = [{"hyp": hyp, "ref": ref, "chrf++": chrf.sentence_score(hyp, [ref]).score}
             for hyp, ref in zip(hyps, refs)]
    return "chrF++", chrf.corpus_score(hyps, [refs]).score, items


def score_svamp(tok, model):
    rows = _local_rows("svamp", "test")
    hyps = generate(tok, model, [prompt for prompt, _ in LOCAL_PAIRS["svamp"]("test")], generation_cap(tok, "svamp"))
    items = []
    for row, hyp in zip(rows, hyps):
        value, gold = svamp_value(hyp), float(row["Answer"])
        items.append({"hyp": hyp, "gold": gold, "pred": value,
                      "correct": value is not None and math.isclose(value, gold, rel_tol=1e-4, abs_tol=1e-6)})
    return "accuracy", 100 * sum(item["correct"] for item in items) / len(items), items


@torch.no_grad()
def score_arc(tok, model):
    rows, pairs = _local_rows("arc", "test"), LOCAL_PAIRS["arc"]("test")
    letter_ids = {letter: tok(f" {letter}", add_special_tokens=False)["input_ids"][0] for letter in "ABCDE"}
    items = [None] * len(rows)
    for indices, ids, mask in prompt_batches(tok, [prompt for prompt, _ in pairs]):
        # left-padded, so the last position is every row's next token; positions count
        # from each row's first real token, as generate() does
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=positions).logits[:, -1].float()
        for row_index, index in enumerate(indices):
            options = [arc_letter(label) for label in rows[index]["choices"]["label"]]
            scores = {letter: logits[row_index, letter_ids[letter]].item() for letter in options}
            pred, gold = max(scores, key=scores.get), arc_letter(rows[index]["answerKey"])
            items[index] = {"pred": pred, "gold": gold, "correct": pred == gold}
    return "accuracy", 100 * sum(item["correct"] for item in items) / len(items), items


SCORERS = {**{name: partial(score_translation, name) for name in TRANSLATION}, "svamp": score_svamp, "arc": score_arc}


def row_losses(model, ids, labels, mask):
    """Per row: summed loss over its reply tokens and their count, with the shift and the fp32
    cross-entropy that the model's own loss uses; their ratio over a batch is that loss."""
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    targets = labels[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100,
                          reduction="none").view(targets.shape)
    return nll.sum(dim=1).tolist(), (targets != -100).sum(dim=1).tolist()


@torch.no_grad()
def score_loss(tok, model, dataset, split, max_tokens):
    """Teacher-forced loss on a split, in the batches train_compare uses for its dev loss."""
    rows = build_data(tok, split, len(LOCAL_PAIRS[dataset](split)), dataset)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items = [None] * len(rows)
    for batch in token_batches([len(x) for x, _ in rows], max_tokens, torch.Generator().manual_seed(0)):
        ids, labels, mask = (t.cuda() for t in collate([rows[j] for j in batch], pad))
        for index, nll, tokens in zip(batch, *row_losses(model, ids, labels, mask)):
            items[index] = {"nll": nll, "tokens": tokens}
    return "loss", sum(item["nll"] for item in items) / sum(item["tokens"] for item in items), items


def score_and_write(out, record, tok, model, dataset, mode, version, max_tokens):
    """Score one model in one mode and write its file atomically. mode: greedy or norepeat
    (a decoder), or loss (no decoding; its file has decoder None). seconds exclude the model load."""
    start = time.time()
    if mode == "loss":
        metric, score, items = score_loss(tok, model, dataset, "test", max_tokens)
    elif mode == "norepeat":
        metric, score, items = score_translation(dataset, tok, model, mode)
    else:
        metric, score, items = SCORERS[dataset](tok, model)
    seconds = time.time() - start          # inside the lock: no waiting counted
    block = {"ngram": BLOCK_NGRAM, "max_uses": BLOCK_MAX_USES} if mode == "norepeat" else None
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({**record, "metric": metric, "score": score, "seconds": seconds,
                               "decoder": None if mode == "loss" else mode, "block": block,
                               "versions": [version, record.get("train_version")], "items": items},
                              ensure_ascii=False))
    os.replace(tmp, out)
    print(f"  {metric} {score:6.2f}  {seconds:5.0f}s  {out.stem}", flush=True)


def release_gpu():
    """Free the last model; the caller must drop its own reference first."""
    # the phase kinds' batched-chart hooks refer back to the model, a cycle
    # that only gc frees; without it two 1.7B models can share the GPU and OOM
    gc.collect()
    # continuous batching generates in a new thread per run, and each thread gets a
    # 32 MB cuBLAS workspace that is never freed: measured 32 MB more after every run
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated() > MAX_LEFT_BYTES:
        sys.exit(f"{torch.cuda.memory_allocated() / 2**30:.2f} GB still allocated after a run")


def applies(mode, dataset):
    """The repetition-blocked decoder is for translation only; greedy and loss score every dataset."""
    return mode != "norepeat" or dataset in TRANSLATION


def score_model(load, record, dataset, modes, version, max_tokens):
    """Load the model once for every mode whose score is missing or stale, and score it in each."""
    tag, train_version = record["tag"], record["train_version"]
    todo = [mode for mode in modes if applies(mode, dataset)
            and not is_current(MODE_DIRS[mode] / f"{tag}.json", version, train_version, dataset, mode)]
    if not todo:
        return
    start = time.time()
    tok, model = load()
    record = {**record, "load_seconds": time.time() - start}
    # the weights are fixed while scoring, so both DoRA arms may reuse their weight norm; same values
    with DoraCaching():
        for mode in todo:
            score_and_write(MODE_DIRS[mode] / f"{tag}.json", record, tok, model, dataset, mode, version, max_tokens)
    del model
    release_gpu()


def is_current(out, version, train_version, dataset, mode):
    """A score is reused only if a valid scorer scored the same trained weights."""
    if not out.exists():
        return False
    score_version, scored_train = json.loads(out.read_text()).get("versions", [None, None])
    if scored_train != train_version:
        return False
    return score_valid(score_version, version, dataset, mode)


def score_valid(score_version, version, dataset, mode):
    """The current scorer, or a recorded older one valid for this dataset and mode. A record that names no modes
    covers greedy decoding only: the oldest recorded scorers decoded greedily."""
    legacy = LEGACY_VERSIONS["score"].get(score_version, {})
    return score_version == version or (dataset in legacy.get("datasets", []) and mode in legacy.get("modes", ["greedy"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", help="result files; default: every finished main run")
    p.add_argument("--modes", nargs="+", choices=list(MODE_DIRS), default=["greedy"],
                   help="greedy and loss score every dataset, norepeat translation only; one model load serves all")
    p.add_argument("--base", action="store_true", help="score the untrained base models on every dataset")
    p.add_argument("--models", nargs="+", help="--base: only these models; default: every model of the design")
    a = p.parse_args()
    import run_sweep
    max_tokens = run_sweep.FIXED["max-tokens"]     # the dev loss's batch budget in every run
    for directory in MODE_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=watch_ram, daemon=True).start()
    version = code_version(SCORE_CODE)
    if a.base:
        with open(GPU_LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)      # the same lock flock(1) takes for training
            for model_name in a.models or run_sweep.MODELS:
                for dataset in run_sweep.DATASETS:
                    tag = f"base_{model_name}_{dataset}"
                    record = {"tag": tag, "args": {"kind": "base", "model": model_name, "dataset": dataset},
                              "train_version": None}
                    score_model(partial(load_base, model_name), record, dataset, a.modes, version, max_tokens)
        return
    if a.runs:
        paths = [Path(path) for path in a.runs]
    else:
        paths = [path for path in sorted(RUNS_DIR.glob("*.json"))
                 if "tag" in (result := json.loads(path.read_text()))
                 and result["args"]["epochs"] == run_sweep.EPOCHS and result["args"]["seed"] in run_sweep.SEEDS]
    results = [json.loads(path.read_text()) for path in paths]
    stale = [r["tag"] for r in results if not train_version_ok(r.get("code_version"), r["args"]["kind"])]
    if stale:
        sys.exit(f"{len(stale)} runs were trained by code that is neither current nor recorded as valid, e.g. {stale[0]}")
    with open(GPU_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)          # the same lock flock(1) takes for training
        for result in results:
            record = {"tag": result["tag"], "args": result["args"], "diverged_at": result["diverged_at"],
                      "train_version": result.get("code_version")}
            score_model(partial(load_run, result), record, result["args"]["dataset"], a.modes, version, max_tokens)


if __name__ == "__main__":
    main()
