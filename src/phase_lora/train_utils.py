"""Data prep and adapter attachment, shared by the training and benchmark scripts."""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from .polar import PolarLoRALinear
from .stiefel import (PhaseStelLA, PhaseStiefel, PolarStiefel, StelLA,
                      install_batched_charts)
from .unitary import OrthoLinear, install_batched_cayley

DATASET = "HuggingFaceH4/no_robots"
TARGETS = ("q_proj", "v_proj")


def assistant_spans(ids, tok, im_start, im_end):
    """Token spans of assistant turns.

    Length arithmetic on chat-template prefixes does not work here: Qwen3 injects
    a <think> block when add_generation_prompt=True that is absent from the full
    rendering, so the offsets drift. Scan the turn markers instead.
    """
    spans, j = [], 0
    while j < len(ids):
        if ids[j] != im_start:
            j += 1
            continue
        k = j + 1                                    # role name runs to the newline
        while k < len(ids) and "\n" not in tok.decode([ids[k]]):
            k += 1
        role = tok.decode(ids[j + 1 : k]).strip()
        e = k + 1
        while e < len(ids) and ids[e] != im_end:
            e += 1
        if role == "assistant":
            spans.append((k + 1, min(e + 1, len(ids))))   # keep <|im_end|>: teach it to stop
        j = e + 1
    return spans


YORUBA_HOLDOUT = 600          # last N rows reserved for eval, never trained on


def _yoruba_pairs(split, n):
    """en->yo from OPUS-100. One low-resource language: r=64 beats r=8 by 0.47 nats
    here where it *loses* by 0.12 on no_robots, so rank is a live variable."""
    ds = load_dataset("Helsinki-NLP/opus-100", "en-yo", split="train")
    cut = len(ds) - YORUBA_HOLDOUT
    idx = range(cut, len(ds)) if split == "test" else range(0, min(n, cut))
    out = []
    for ex in ds.select(idx):
        t = ex["translation"]
        if t["en"].strip() and t["yo"].strip():
            out.append([{"role": "user", "content": f"Translate to Yoruba: {t['en']}"},
                        {"role": "assistant", "content": t["yo"]}])
    return out


def build_data(tok, split, n, max_len, dataset="no_robots"):
    """Tokenize chat, masking every token that is not an assistant reply."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    if dataset == "yoruba":
        msgs = _yoruba_pairs(split, n)
    else:
        ds = load_dataset(DATASET, split=split)
        msgs = [ex["messages"] for ex in ds.select(range(min(n, len(ds))))]
    for m in msgs:
        ids = tok.apply_chat_template(m, tokenize=True)["input_ids"]
        labels = [-100] * len(ids)
        for a, b in assistant_spans(ids, tok, im_start, im_end):
            labels[a:b] = ids[a:b]
        ids, labels = ids[:max_len], labels[:max_len]
        if any(l != -100 for l in labels):
            rows.append((torch.tensor(ids), torch.tensor(labels)))
    return rows


def collate(batch, pad):
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros(len(batch), n, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)], lab[i, : len(y)], att[i, : len(x)] = x, y, 1
    return ids, lab, att


def attach(model, kind, r, alpha, eps, b_init, n_blocks=None, lam=None):
    """Additive baseline is stock PEFT -- the reference implementation, so the
    comparison is not against a reimplementation. PhaseLoRA cannot use PEFT: the
    complex model consumes adapters through effective_weight(), not their forward.

    n_blocks is not a free knob: it is what puts each orthogonal kind on LoRA's
    parameter count. A U(m) block costs m^2 where the so(2m) it sits inside costs
    2m^2 - m, so the control needs twice as many (smaller) blocks to spend the
    same budget -- without that it is a test of "more parameters help".
    """
    if n_blocks is None:
        n_blocks = {"unitary": 32, "orthogonal": 64}.get(kind, 32)
    if kind == "additive":
        return get_peft_model(model, LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=list(TARGETS), task_type="CAUSAL_LM"))
    for block in model.model.layers:
        for name in TARGETS:
            base = getattr(block.self_attn, name)
            if kind == "stiefel":                 # PoLAR, as published
                new = PolarStiefel(base, r, alpha, lam=lam)
            elif kind == "stella":                # StelLA, as published
                # d is the hidden dim of the input tokens, per their Sec. 4;
                # it only enters the gradient scaling, not the parameter count,
                # which is identical to PoLAR at the same rank.
                new = StelLA(base, r, alpha, d=model.config.hidden_size)
            elif kind == "phase_stella":          # StelLA + our complex factors
                new = PhaseStelLA(base, r // 2, alpha, d=model.config.hidden_size)
            elif kind == "phase_stiefel":         # PoLAR + our complex factors
                new = PhaseStiefel(base, r // 2, alpha, lam=lam)
            elif kind == "polar":
                # r//2 so a given --rank means the same parameter count as additive:
                # complex rank r/2 spans real rank r at 2(r/2)(in+out) = r(in+out).
                new = PolarLoRALinear(base, r // 2, alpha)
            elif kind in ("unitary", "orthogonal"):
                new = OrthoLinear(base, n_blocks=n_blocks, structured=kind == "unitary")
            else:
                raise ValueError(f"unknown kind {kind!r}")
            setattr(block.self_attn, name, new.to(base.weight.device))
    if kind in ("unitary", "orthogonal"):
        install_batched_cayley(model)
    if kind == "phase_stiefel":
        install_batched_charts(model)
    return model
