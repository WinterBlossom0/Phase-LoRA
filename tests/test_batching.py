"""token_batches: every row once, never over budget, same seed same batches, never cut."""
import random

import pytest
import torch

from phase_lora.train_utils import collate, token_batches

LENGTHS = [random.Random(0).randint(20, 988) for _ in range(2000)]
BUDGET = 3072


def batches(seed):
    return token_batches(LENGTHS, BUDGET, torch.Generator().manual_seed(seed))


def test_every_row_once_within_budget():
    plan = batches(0)
    assert sorted(row for batch in plan for row in batch) == list(range(len(LENGTHS)))
    assert all(len(batch) * max(LENGTHS[row] for row in batch) <= BUDGET for batch in plan)


def test_same_seed_same_batches():
    assert batches(0) == batches(0)
    assert batches(0) != batches(1)


def test_row_over_budget_raises():
    with pytest.raises(ValueError, match="never cut"):
        token_batches([10, BUDGET + 1], BUDGET, torch.Generator().manual_seed(0))


def test_collate_keeps_every_token():
    rows = [(torch.arange(n), torch.arange(n)) for n in (5, 9, 2)]
    ids, labels, mask = collate(rows, pad=-1)
    assert ids.shape == (3, 9)
    assert mask.sum().item() == 16 and (labels != -100).sum().item() == 16
