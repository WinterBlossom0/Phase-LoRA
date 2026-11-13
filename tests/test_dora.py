"""Phase DoRA computes what PEFT's DoRA computes when the two parametrisations hold the same dW."""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from phase_lora.polar import PhaseDoRALinear


class Tiny(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.q_proj = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x):
        return self.q_proj(x)


def matched_pair(dtype):
    torch.manual_seed(0)
    in_f, out_f, rank, alpha = 24, 40, 8, 8.0
    tiny = Tiny(in_f, out_f).to(dtype)
    phase = PhaseDoRALinear(nn.Linear(in_f, out_f, bias=False).to(dtype), r=rank // 2, alpha=alpha)
    phase.base.weight.data.copy_(tiny.q_proj.weight.data)
    peft_model = get_peft_model(tiny, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                                                 target_modules=["q_proj"], use_dora=True))
    layer = peft_model.base_model.model.q_proj
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.2)
        layer.lora_B["default"].weight.normal_(std=0.2)
        a, b = layer.lora_A["default"].weight.float(), layer.lora_B["default"].weight.float()
        half = rank // 2
        # PEFT: dW = scale * B_peft @ A_peft. Phase: dW = scale * (Ar Br^T + Ai Bi^T).
        Ar, Ai, Br, Bi = b[:, :half], b[:, half:], a[:half].T, a[half:].T
        phase.rho_a.copy_(Ar.hypot(Ai)); phase.phi_a.copy_(torch.atan2(Ai, Ar))
        phase.rho_b.copy_(Br.hypot(Bi)); phase.phi_b.copy_(torch.atan2(Bi, Br))
        layer.lora_magnitude_vector["default"].weight.mul_(1.1)
        phase.magnitude.copy_(layer.lora_magnitude_vector["default"].weight)
    return peft_model, layer, phase


def test_phase_dora_matches_peft_dora_forward_and_magnitude_gradient():
    peft_model, layer, phase = matched_pair(torch.float32)
    x = torch.randn(3, 5, 24)
    out_peft, out_phase = peft_model(x), phase(x)
    assert torch.allclose(out_peft, out_phase, atol=1e-5), (out_peft - out_phase).abs().max()
    out_peft.square().sum().backward()
    out_phase.square().sum().backward()
    assert torch.allclose(layer.lora_magnitude_vector["default"].weight.grad, phase.magnitude.grad, atol=1e-4)


def test_phase_dora_matches_peft_dora_in_bf16():
    peft_model, _, phase = matched_pair(torch.bfloat16)
    x = torch.randn(3, 5, 24).to(torch.bfloat16)
    out_peft, out_phase = peft_model(x), phase(x)
    assert out_peft.dtype == out_phase.dtype == torch.bfloat16
    assert torch.allclose(out_peft.float(), out_phase.float(), atol=2e-2), (out_peft - out_phase).abs().max()


def test_phase_dora_and_peft_dora_start_from_the_same_magnitude_in_bf16():
    torch.manual_seed(0)
    tiny = Tiny(24, 40).to(torch.bfloat16)
    base = nn.Linear(24, 40, bias=False).to(torch.bfloat16)
    base.weight.data.copy_(tiny.q_proj.weight.data)           # before init: the magnitude reads this weight
    phase = PhaseDoRALinear(base, r=4, alpha=8.0)
    peft_model = get_peft_model(tiny, LoraConfig(r=8, lora_alpha=8.0, lora_dropout=0.0, bias="none",
                                                 target_modules=["q_proj"], use_dora=True))
    peft_magnitude = peft_model.base_model.model.q_proj.lora_magnitude_vector["default"].weight
    assert torch.equal(peft_magnitude.float(), phase.magnitude), "the two arms start from different magnitudes"
    x = torch.randn(3, 24).to(torch.bfloat16)
    assert torch.equal(peft_model(x), phase(x)), "the two arms start from different models"


def test_phase_dora_starts_at_the_base_model():
    layer = PhaseDoRALinear(nn.Linear(24, 40), r=4, alpha=8.0)
    x = torch.randn(2, 24)
    assert torch.allclose(layer(x), layer.base(x), atol=1e-5)
    params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert params == 8 * (24 + 40) + 40, "LoRA factors at rank 8 plus one magnitude per output row"


def test_dora_caching_gives_the_same_outputs_and_never_outlives_a_mode_change():
    from peft.helpers import DoraCaching
    peft_model, layer, phase = matched_pair(torch.float32)
    peft_model.eval(); phase.eval()
    x = torch.randn(3, 5, 24)
    with torch.no_grad():
        plain_peft, plain_phase = peft_model(x), phase(x)
        assert phase.cached_norm_scale is None, "no caching without the switch"
        with DoraCaching():
            for _ in range(2):                 # the second call reads the caches
                assert torch.equal(peft_model(x), plain_peft)
                assert torch.equal(phase(x), plain_phase)
            assert phase.cached_norm_scale is not None and phase.cached_factors is not None
            phase.train(); phase.eval()
            assert phase.cached_norm_scale is None and phase.cached_factors is None, "a mode change clears the cache"
            phase.rho_a.mul_(2)
            assert not torch.equal(phase(x), plain_phase), "a stale scale would hide the new weights"
    phase.train()
    with DoraCaching():
        phase(x)
    assert phase.cached_norm_scale is None and phase.cached_factors is None, "training never caches"


def test_phase_lora_computes_what_peft_lora_computes_for_the_same_dw():
    # the LoRA arms differ only in the coordinates of the factors, not in the implementation
    from phase_lora.polar import PolarLoRALinear
    torch.manual_seed(0)
    in_f, out_f, rank, alpha = 24, 40, 8, 8.0
    tiny = Tiny(in_f, out_f)
    phase = PolarLoRALinear(nn.Linear(in_f, out_f, bias=False), r=rank // 2, alpha=alpha)
    phase.base.weight.data.copy_(tiny.q_proj.weight.data)
    peft_model = get_peft_model(tiny, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                                                 target_modules=["q_proj"]))
    layer = peft_model.base_model.model.q_proj
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(std=0.2)
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        half = rank // 2
        Ar, Ai, Br, Bi = b[:, :half], b[:, half:], a[:half].T, a[half:].T
        phase.rho_a.copy_(Ar.hypot(Ai)); phase.phi_a.copy_(torch.atan2(Ai, Ar))
        phase.rho_b.copy_(Br.hypot(Bi)); phase.phi_b.copy_(torch.atan2(Bi, Br))
    x = torch.randn(3, 5, in_f)
    assert torch.allclose(peft_model(x), phase(x), atol=1e-5)
