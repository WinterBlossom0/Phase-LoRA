"""Polar LoRA: the same rank-2r updates as real LoRA, reached along different paths.

    dW = Re(A B^H),   A = rho_a e^(i phi_a),  B = rho_b e^(i phi_b)

Expanding Re(A B^H) gives Ar Br^T + Ai Bi^T, so the *image* is exactly real rank-2r
and the parameter count is exactly 2r(in+out) -- identical to real LoRA at rank 2r.
Nothing about the hypothesis class changes. Only the chart does.

That is the entire point. Cartesian complex factors would change nothing at all:
Wirtinger descent on (re, im) is plain SGD on two real tensors, and Adam sees the
same coordinates, so complex-in-Cartesian reproduces real rank-2r step for step.
Polar is where the geometry actually differs.

Why it might matter, in one line: in real LoRA d/dB = G A^T, so a rank-1 component
can only reverse sign by dragging a factor through zero -- and the *other* factor's
gradient is proportional to it, so it stalls exactly at the crossing. In polar the
same reversal is phi -> phi + pi at constant rho, and the phase gradient
(prop. rho sin) peaks precisely where the modulus gradient (prop. rho cos) vanishes.
The two relay instead of collapsing together.

The counter-argument, which this is meant to test rather than assume: BaLoRA argues
LoRA's GL(r) gauge freedom already hurts conditioning, and polar adds a U(1) per
component on top of it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarLoRALinear(nn.Module):
    """dW = scale * Re(A B^H) in polar coordinates. Matches real LoRA rank 2r."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_f, in_f = base.weight.shape

        # Complex-Gaussian A, stored polar: same per-entry scale as LoRA's kaiming A
        # once split across two components, so the two runs start at the same size.
        a = torch.randn(out_f, r, 2) / math.sqrt(2 * in_f)
        self.rho_a = nn.Parameter(a.norm(dim=-1))
        self.phi_a = nn.Parameter(torch.atan2(a[..., 1], a[..., 0]))
        # rho_b = 0 makes dW exactly 0 at init, as LoRA's B = 0 does. Phase is still
        # live: d/d rho_b is proportional to rho_a, not to rho_b, so nothing is stuck.
        self.rho_b = nn.Parameter(torch.zeros(in_f, r))
        self.phi_b = nn.Parameter(torch.rand(in_f, r) * 2 * math.pi)
        # effective rank is 2r, so alpha/(2r) puts this on real LoRA's alpha/r at rank 2r
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
        Ar, Ai, Br, Bi = (t.to(x.dtype) for t in self.factors())
        # Two low-rank products, never materialising dW -- same shape of work as LoRA.
        d = (x @ Br) @ Ar.T + (x @ Bi) @ Ai.T
        return F.linear(x, self.base.weight, self.base.bias) + self.scale * d


def demo():
    torch.manual_seed(0)
    base = nn.Linear(64, 96)
    lay = PolarLoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(5, 64)

    # Exact identity at init, like LoRA.
    assert lay.delta_weight().abs().max() < 1e-8
    assert torch.allclose(lay(x), base(x), atol=1e-5)

    # Parameter count is exactly real LoRA at rank 2r, not approximately.
    n = sum(p.numel() for p in lay.parameters() if p.requires_grad)
    assert n == 2 * 4 * (64 + 96) == 8 * (64 + 96), n

    # forward() and delta_weight() must agree, or the fast path is lying.
    with torch.no_grad():
        lay.rho_b.normal_(std=0.1)
    assert torch.allclose(lay(x), base(x) + x @ lay.delta_weight().T, atol=1e-5)

    # The image really is real rank-2r: dW is a sum of 2r outer products.
    assert torch.linalg.matrix_rank(lay.delta_weight(), tol=1e-6) <= 8

    # At init only rho_b is live: every other gradient carries a factor of rho_b = 0.
    # This is LoRA's own pathology, not a new one -- there B = 0 makes dL/dA = B^T G = 0.
    # One step lifts rho_b off zero and the remaining three wake up.
    fresh = PolarLoRALinear(nn.Linear(64, 96), r=4)
    fresh(x).pow(2).sum().backward()
    assert fresh.rho_b.grad.abs().max() > 1e-9, "rho_b must carry the first step"
    for name in ("rho_a", "phi_a", "phi_b"):
        assert getattr(fresh, name).grad.abs().max() < 1e-12, f"{name} unexpectedly live"

    fresh.zero_grad()
    with torch.no_grad():
        fresh.rho_b.normal_(std=0.05)          # what one optimizer step would do
    fresh(x).pow(2).sum().backward()
    for name in ("rho_a", "phi_a", "rho_b", "phi_b"):
        g = getattr(fresh, name).grad
        assert g is not None and g.abs().max() > 1e-9, f"{name} still dead after a step"

    # The mechanism itself: rotate one component's phase by pi and its contribution
    # flips sign while its modulus -- and so the other factor's gradient -- is intact.
    with torch.no_grad():
        fresh.rho_b.fill_(0.1)
        before = fresh.delta_weight().clone()
        r_before = fresh.rho_a.clone()
        fresh.phi_a += math.pi
    assert torch.allclose(fresh.delta_weight(), -before, atol=1e-6), "phase pi is not a sign flip"
    assert torch.allclose(fresh.rho_a, r_before), "modulus moved during the flip"

    print("ok")


if __name__ == "__main__":
    demo()
