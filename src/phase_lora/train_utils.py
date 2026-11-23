"""Data prep and adapter attachment, shared by the training and benchmark scripts."""

import json
from functools import partial

import torch
from peft import LoraConfig, get_peft_model

from .paths import DATA_DIR
from .polar import PhaseDoRALinear, PolarLoRALinear
from .stiefel import PhaseStiefel, PolarStiefel, install_batched_charts

TARGETS = ("q_proj", "v_proj")


# The split that early stopping scores: dev, so the test split stays for the reported score.
EVAL_SPLIT = {"mafand_tsn": "dev", "mafand_lug": "dev", "svamp": "dev", "arc": "dev"}


def _local_rows(name, split):
    path = DATA_DIR / name / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/prepare_data.py")
    # iterate the file, not str.splitlines(): that also splits on U+2028 and U+0085,
    # which can occur unescaped inside source text and would cut a JSON row in two
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


# The local datasets use plain text, not the chat template: in both Qwen3 base models
# the chat tokens (<|im_start|>, <|im_end|>, <think>, </think>) share one untrained
# embedding row with 290 (0.6B) or 93 (1.7B) other tokens. Tied to the output layer,
# that row costs every chat-format reply >= ln(291) nats no adapter can remove, and
# greedy decoding can never emit <|im_end|>. <|endoftext|> is trained, so it ends a reply.
END_OF_TEXT = "<|endoftext|>"


# translation dataset -> (MAFAND-MT language code, language name in the prompt)
TRANSLATION = {"mafand_tsn": ("tsn", "Setswana"), "mafand_lug": ("lug", "Luganda")}


def _mafand_pairs(dataset, split):
    """(prompt, reply): an English news sentence from MAFAND-MT and its professional translation."""
    code, language = TRANSLATION[dataset]
    out = []
    for row in _local_rows(dataset, split):
        t = row["translation"]
        if t["en"].strip() and t[code].strip():
            out.append((f"Translate English to {language}.\nEnglish: {t['en'].strip()}\n{language}:",
                        f" {t[code].strip()}"))
    return out


def _svamp_pairs(split):
    """(prompt, reply): the reply is the equation and its value, e.g. ' ( 290.0 / 2.0 ) = 145'."""
    return [(f"Question: {row['question_concat'].strip()}\nAnswer:",
             f" {row['Equation'].strip()} = {row['Answer']}")
            for row in _local_rows("svamp", split)]


def arc_letter(label):
    """49 ARC-Challenge rows label their options 1-4; map to A-D so every reply is a letter."""
    return "ABCDE"[int(label) - 1] if label.isdigit() else label


def _arc_pairs(split):
    """(prompt, reply): question and lettered options; the reply is ' <letter>'."""
    out = []
    for row in _local_rows("arc", split):
        choices = row["choices"]
        options = "\n".join(f"{arc_letter(label)}. {text}"
                            for label, text in zip(choices["label"], choices["text"]))
        out.append((f"Question: {row['question'].strip()}\n{options}\nAnswer:",
                    f" {arc_letter(row['answerKey'])}"))
    return out


LOCAL_PAIRS = {**{name: partial(_mafand_pairs, name) for name in TRANSLATION}, "svamp": _svamp_pairs, "arc": _arc_pairs}


def encode_pair(tok, prompt, reply):
    """Prompt and reply tokenized apart, so generation from the prompt ids sees the
    exact prefix training saw. Only the reply and the end token are scored."""
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    reply_ids = tok(reply, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids(END_OF_TEXT)]
    return prompt_ids + reply_ids, [-100] * len(prompt_ids) + reply_ids


def build_data(tok, split, n, dataset):
    """Tokenize, masking every token that is not part of the reply. Rows are never
    cut: token_batches sizes each batch to its rows instead."""
    return [tuple(torch.tensor(seq) for seq in encode_pair(tok, prompt, reply))
            for prompt, reply in LOCAL_PAIRS[dataset](split)[:n]]


def token_batches(lengths, max_tokens, generator):
    """Split row indices into batches of at most max_tokens tokens, padding included.

    The row count per batch varies with row length, so no row is ever cut. This is
    fairseq's max_tokens batching: sort by length so each batch pads little, break
    ties in a random order, then shuffle the batch order. One generator drives both,
    so arms with the same seed train on identical batches. The memory ceiling is
    the budget itself, not whichever batch the shuffle happens to build.
    """
    longest = max(lengths)
    if longest > max_tokens:
        raise ValueError(f"a row has {longest} tokens, above max_tokens={max_tokens}; "
                         "raise the budget, rows are never cut")
    shuffled = torch.randperm(len(lengths), generator=generator).tolist()
    by_length = sorted(shuffled, key=lambda row: lengths[row])    # stable: ties keep the random order
    batches, current = [], []
    for row in by_length:
        # ascending order, so this row is the longest of the batch it joins
        if current and (len(current) + 1) * lengths[row] > max_tokens:
            batches.append(current)
            current = []
        current.append(row)
    batches.append(current)
    return [batches[position] for position in torch.randperm(len(batches), generator=generator).tolist()]


def collate(batch, pad):
    """Pad to the longest row in the batch; token_batches already bounds the total."""
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros(len(batch), n, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)], lab[i, : len(y)], att[i, : len(x)] = x, y, 1
    return ids, lab, att


def landing_modules(model):
    """The PoLAR and Phase PoLAR modules, whose X and Y gradients become the landing field before each step."""
    return [module for module in model.modules() if isinstance(module, (PolarStiefel, PhaseStiefel))]


def make_optimizer(params, lr, clip, landing=()):
    """Adam with gradient-norm clipping before each step, the same for every arm; then, for PoLAR and Phase
    PoLAR, the landing field of the clipped gradients, the order of the official PoLAR code.

    No weight decay, a choice of the owner: decay would also pull the phase angles toward 0, which the real
    arms do not get. At zero decay AdamW and Adam make the same update.
    """
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0)

    def clip_hook(*_):                                 # must return None
        total = torch.nn.utils.clip_grad_norm_(params, clip)
        # the factor that clip_grad_norm_ applied, by its own formula: 0 for an infinite norm, NaN for a NaN norm
        clip_coef = torch.clamp(clip / (total + 1e-6), max=1.0).item()
        for module in landing:
            module.apply_landing(clip_coef)

    opt.register_step_pre_hook(clip_hook)
    return opt


def attach(model, kind, r, alpha, lam=None):
    """LoRA and DoRA are PEFT's own implementations. The other kinds replace q_proj and v_proj with the
    modules of this package, since PEFT has no polar or Stiefel factors."""
    if kind in ("additive", "dora"):
        return get_peft_model(model, LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", use_dora=kind == "dora",
            target_modules=list(TARGETS), task_type="CAUSAL_LM"))
    for block in model.model.layers:
        # LFM3 convolution layers have no attention, so no q_proj or v_proj; every Qwen3 layer has attention
        if not hasattr(block, "self_attn"):
            continue
        for name in TARGETS:
            base = getattr(block.self_attn, name)
            if kind == "stiefel":                 # PoLAR, as published
                new = PolarStiefel(base, r, alpha, lam=lam)
            elif kind == "phase_stiefel":         # PoLAR at rank r, X and Y in polar coordinates
                new = PhaseStiefel(base, r // 2, alpha, lam=lam)
            elif kind == "phase_dora":            # DoRA with polar factors, complex rank r // 2 as polar
                new = PhaseDoRALinear(base, r // 2, alpha)
            elif kind == "polar":
                # complex rank r // 2 has the parameter count of real rank r: 2 (r/2)(in + out) = r (in + out)
                new = PolarLoRALinear(base, r // 2, alpha)
            else:
                raise ValueError(f"unknown kind {kind!r}")
            setattr(block.self_attn, name, new.to(base.weight.device))
    if kind == "phase_stiefel":
        install_batched_charts(model)
    return model
