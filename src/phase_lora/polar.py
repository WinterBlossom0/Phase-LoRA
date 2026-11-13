"""Phase LoRA: LoRA's factors stored in polar coordinates.

    dW = Re(A B^H),   A = rho_a e^(i phi_a),  B = rho_b e^(i phi_b)

Re(A B^H) = Ar Br^T + Ai Bi^T, so complex rank r gives a real update of rank at most 2r with 2r(in + out)
parameters: the function class and the parameter count of real LoRA at rank 2r. Only the coordinates that the
optimiser updates differ. The factors are polar, not Cartesian: Cartesian (re, im) factors are two real tensors,
and Adam would update them exactly as it updates real LoRA at rank 2r.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import dora as peft_dora


class PolarLoRALinear(nn.Module):
    """dW = scale * Re(A B^H) in polar coordinates, at complex rank r: real LoRA's class at rank 2r."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_f, in_f = base.weight.shape

        # B multiplies the input (forward is (x @ B) @ A^T), so B takes PEFT's lora_A init: random, with
        # lora_A's per-entry variance 1/(3 in) (kaiming_uniform, a = sqrt(5)) in each real component.
        # A takes lora_B's init, zero. Hayou et al. (2024) show that the random side changes LoRA training.
        b = torch.randn(in_f, r, 2) / math.sqrt(3 * in_f)
        self.rho_b = nn.Parameter(b.norm(dim=-1))
        self.phi_b = nn.Parameter(torch.atan2(b[..., 1], b[..., 0]))
        # rho_a = 0 makes dW exactly 0 at init, as LoRA's lora_B = 0 does. The gradient of rho_a has no
        # factor rho_a, so the first step moves rho_a (checked in demo).
        self.rho_a = nn.Parameter(torch.zeros(out_f, r))
        self.phi_a = nn.Parameter(torch.rand(out_f, r) * 2 * math.pi)
        # alpha / (2r) equals real LoRA's alpha / r at rank 2r
        self.scale = alpha / (2 * r)

    @property
    def bias(self):
        return self.base.bias

    def factors(self):
        """The equivalent real rank-2r factors (Ar|Ai), (Br|Bi)."""
        return (self.rho_a * self.phi_a.cos(), self.rho_a * self.phi_a.sin(),
                self.rho_b * self.phi_b.cos(), self.rho_b * self.phi_b.sin())

    def delta_weight(self) -> torch.Tensor:
        Ar, Ai, Br, Bi = self.factors()
        return self.scale * (Ar @ Br.T + Ai @ Bi.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Ar, Ai, Br, Bi = self.factors()
        # Two low-rank products; dW is never built. fp32, as PEFT runs its LoRA branch
        # (autocast_adapter_dtype), so every arm uses the same adapter precision.
        xf = x.float()
        d = (xf @ Br) @ Ar.T + (xf @ Bi) @ Ai.T
        return (F.linear(x, self.base.weight, self.base.bias) + self.scale * d).to(x.dtype)


class PhaseDoRALinear(PolarLoRALinear):
    """DoRA (Liu et al., ICML 2024) with the polar factors: W' = m * (W + dW) / ||W + dW||_row.

    The arithmetic follows PEFT 0.20's DoraLinearLayer.forward step for step, so the only
    difference from `--kind dora` is the coordinates of dW: the row norm is detached (DoRA Sec. 4.3),
    computed in fp32, and the output is base + ((m/n - 1) * base + (m/n) * scale * lora).
    The factors and the row-norm scale are cached under PEFT's own DoRA caching switch and rules
    (eval mode only), as PEFT caches its LoRA weight and norm, so one switch serves both DoRA arms
    at test time; training never sets it.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__(base, r, alpha)
        # PEFT's magnitude init: the row norm of the weight in its own dtype (bf16 here), the
        # same call PEFT makes, then cast to fp32 as PEFT's adapter autocast does; dW is 0 at init
        self.magnitude = nn.Parameter(torch.linalg.norm(base.weight.detach(), dim=1).float())
        self.cached_factors = self.cached_norm_scale = None

    def train(self, mode: bool = True):
        self.cached_factors = self.cached_norm_scale = None    # a mode change can come before a weight change
        return super().train(mode)

    def caching(self) -> bool:
        # read at call time: PEFT's DoraCaching context sets the module attribute
        return peft_dora.ENABLE_DORA_CACHING and not self.training and not torch.is_grad_enabled()

    def cached_or_new_factors(self):
        if self.cached_factors is not None:
            return self.cached_factors
        factors = self.factors()
        if self.caching():
            self.cached_factors = factors
        return factors

    def norm_scale(self) -> torch.Tensor:
        """m / ||W + dW||_row as a (1, out) row; the norm is detached."""
        if self.cached_norm_scale is not None:
            return self.cached_norm_scale
        weight = self.base.weight.float() + self.delta_weight().detach()
        scale = (self.magnitude / torch.linalg.norm(weight, dim=1).detach()).view(1, -1)
        if self.caching():
            self.cached_norm_scale = scale
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Ar, Ai, Br, Bi = self.cached_or_new_factors()
        base_result = F.linear(x, self.base.weight)              # bias added at the end, as PEFT does
        norm_scale = self.norm_scale()
        xf = x.float()
        lora = (xf @ Br) @ Ar.T + (xf @ Bi) @ Ai.T
        result = base_result + ((norm_scale - 1) * base_result + norm_scale * lora * self.scale)
        if self.base.bias is not None:
            result = result + self.base.bias
        return result.to(x.dtype)


def demo():
    torch.manual_seed(0)
    base = nn.Linear(64, 96)
    lay = PolarLoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(5, 64)

    # dW = 0 at init, as in LoRA.
    assert lay.delta_weight().abs().max() < 1e-8
    assert torch.allclose(lay(x), base(x), atol=1e-5)

    # The parameter count equals real LoRA's at rank 2r.
    n = sum(p.numel() for p in lay.parameters() if p.requires_grad)
    assert n == 2 * 4 * (64 + 96) == 8 * (64 + 96), n

    # forward() and delta_weight() give the same output.
    with torch.no_grad():
        lay.rho_a.normal_(std=0.1)
    assert torch.allclose(lay(x), base(x) + x @ lay.delta_weight().T, atol=1e-5)

    # dW has rank at most 2r: it is a sum of 2r outer products.
    assert torch.linalg.matrix_rank(lay.delta_weight(), tol=1e-6) <= 8

    # Same init as PEFT's LoRA: the input-side factor random with per-entry variance
    # 1/(3 in) in each real component, the output-side factor zero.
    wide = PolarLoRALinear(nn.Linear(4096, 96), r=64)
    _, _, Br, Bi = wide.factors()
    assert abs(torch.cat([Br, Bi]).var().item() * 3 * 4096 - 1) < 0.05, "input-side variance"
    assert wide.rho_a.abs().max() == 0, "output side must start at zero"

    # At init only rho_a has a nonzero gradient: every other gradient has a factor rho_a = 0, as lora_A's
    # gradient has a factor lora_B = 0 in LoRA. After rho_a leaves zero, all four gradients are nonzero.
    fresh = PolarLoRALinear(nn.Linear(64, 96), r=4)
    fresh(x).pow(2).sum().backward()
    assert fresh.rho_a.grad.abs().max() > 1e-9, "rho_a must carry the first step"
    for name in ("rho_b", "phi_a", "phi_b"):
        assert getattr(fresh, name).grad.abs().max() < 1e-12, f"{name} unexpectedly live"

    fresh.zero_grad()
    with torch.no_grad():
        fresh.rho_a.normal_(std=0.05)          # rho_a away from zero, as after a first step
    fresh(x).pow(2).sum().backward()
    for name in ("rho_a", "phi_a", "rho_b", "phi_b"):
        g = getattr(fresh, name).grad
        assert g is not None and g.abs().max() > 1e-9, f"{name} still dead after a step"

    # A phase change of pi flips the sign of dW at an unchanged modulus.
    with torch.no_grad():
        fresh.rho_a.fill_(0.1)
        before = fresh.delta_weight().clone()
        r_before = fresh.rho_a.clone()
        fresh.phi_a += math.pi
    assert torch.allclose(fresh.delta_weight(), -before, atol=1e-6), "phase pi is not a sign flip"
    assert torch.allclose(fresh.rho_a, r_before), "modulus moved during the flip"

    print("ok")


if __name__ == "__main__":
    demo()
