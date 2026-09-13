"""Unitary finetuning: W' = U W with U block-diagonal orthogonal.

Every published "unitary" PEFT method (OFT, BOFT, HOFT, Quantum-PEFT, Cayley
unitary adapters) is really *orthogonal* -- Quantum-PEFT says so in as many
words: "real-valued quantum operations over SO(N), i.e. not complex-valued
operations over SU(N)". The unitary group is twice as big:

    u(n) = so(n)          + i*Sym(n)
           n(n-1)/2 dims    n(n+1)/2 dims     = n^2

so OFT explores slightly under half of a group it could have had for free,
under the same norm-preservation guarantee that is its whole selling point.

Going complex outright costs ~2x (measured on this model), which is not worth
it.  But U(m) sits inside SO(2m): the real embedding of a complex m x m matrix
A + iS is the 2m x 2m block [[A,-S],[S,A]], and when A is antisymmetric and S is
symmetric that block is *real skew-symmetric*.  So a unitary adapter is a real
Cayley transform of a structured skew-symmetric matrix -- real arithmetic, real
matmuls, no complex dtype anywhere.

What that buys, at matched parameter count: a U(m) block is fixed by m^2
numbers where a general so(2m) block needs 2m^2 - m, so the same budget spans
blocks twice as wide.  Richness of the group traded for width of the mixing --
the same axis BOFT walks with butterfly factors, from the other end.

`structured=False` gives plain so(b) blocks: the matched-parameter control,
without which this measures "more parameters help" rather than "U(1) helps".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _triu_index(n, offset, device):
    return torch.triu_indices(n, n, offset=offset, device=device)


class OrthoLinear(nn.Module):
    """W' = U W, U block-diagonal orthogonal, U = I at init so W' == W exactly.

    structured: each block is constrained to the U(m) subgroup of SO(2m).
    """

    def __init__(self, base: nn.Linear, n_blocks: int = 32, structured: bool = True):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_f, _ = base.weight.shape
        if out_f % n_blocks:
            raise ValueError(f"n_blocks={n_blocks} must divide out_features={out_f}")
        b = out_f // n_blocks                      # real block size
        self.n_blocks, self.b, self.structured = n_blocks, b, structured
        dev = base.weight.device

        if structured:
            if b % 2:
                raise ValueError(f"structured blocks need even size, got {b}")
            m = b // 2                             # complex block size
            self.m = m
            # A antisymmetric: strict upper triangle. S symmetric: upper + diagonal.
            self.register_buffer("_ia", _triu_index(m, 1, dev), persistent=False)
            self.register_buffer("_is", _triu_index(m, 0, dev), persistent=False)
            self.a = nn.Parameter(torch.zeros(n_blocks, m * (m - 1) // 2))
            self.s = nn.Parameter(torch.zeros(n_blocks, m * (m + 1) // 2))
        else:
            self.register_buffer("_ik", _triu_index(b, 1, dev), persistent=False)
            self.k = nn.Parameter(torch.zeros(n_blocks, b * (b - 1) // 2))
        self._U = None

    @property
    def bias(self):
        return self.base.bias

    def _scatter(self, vec, idx, n):
        M = vec.new_zeros(vec.shape[0], n, n)
        M[:, idx[0], idx[1]] = vec
        return M

    def _skew(self):
        """The block-diagonal skew-symmetric generator, (n_blocks, b, b)."""
        if not self.structured:
            M = self._scatter(self.k, self._ik, self.b)
            return M - M.transpose(-1, -2)
        Ma = self._scatter(self.a, self._ia, self.m)
        A = Ma - Ma.transpose(-1, -2)
        Ms = self._scatter(self.s, self._is, self.m)
        # + transpose double-counts the diagonal, so take it back out once.
        S = Ms + Ms.transpose(-1, -2) - torch.diag_embed(Ms.diagonal(dim1=-2, dim2=-1))
        return torch.cat([torch.cat([A, -S], -1),
                          torch.cat([S,  A], -1)], -2)

    def orthogonal_blocks(self) -> torch.Tensor:
        """Cayley transform: U = (I - K)^-1 (I + K), exactly orthogonal for K skew.

        Neumann is the usual way to dodge the solve, but it is unusable here:
        ||K|| runs to 3.4 at std 0.05, so the series does not converge and the
        orthogonality it is supposed to protect is the first thing lost.
        """
        if self._U is not None:                     # filled by batch_cayley
            return self._U
        K = self._skew().float()
        I = torch.eye(self.b, device=K.device, dtype=K.dtype).expand_as(K)
        return torch.linalg.solve(I - K, I + K)

    def effective_weight(self) -> torch.Tensor:
        W = self.base.weight.float().view(self.n_blocks, self.b, -1)
        return torch.bmm(self.orthogonal_blocks(), W).view(self.base.weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight().to(x.dtype), self.base.bias)


def n_params(out_f, n_blocks=32, structured=True):
    b = out_f // n_blocks
    return n_blocks * ((b // 2) ** 2 if structured else b * (b - 1) // 2)


def batch_cayley(model) -> None:
    """One batched solve per distinct block size, for every adapter in the model.

    A 64x64 solve is pure launch overhead -- 32 of them cost 0.89 ms while 1792
    cost 3.49 ms -- so doing this per-module costs ~50 ms/step against ~3.5 ms
    for the whole model at once. Runs as a pre-forward hook on the root module,
    which also means gradient checkpointing reuses the result instead of
    recomputing all 56 solves during the backward recompute.
    """
    groups = {}
    for m in model.modules():
        if isinstance(m, OrthoLinear):
            groups.setdefault(m.b, []).append(m)
    for b, mods in groups.items():
        K = torch.cat([m._skew() for m in mods]).float()
        I = torch.eye(b, device=K.device, dtype=K.dtype).expand_as(K)
        U = torch.linalg.solve(I - K, I + K)
        i = 0
        for m in mods:
            m._U = U[i : i + m.n_blocks]
            i += m.n_blocks


def install_batched_cayley(model):
    """Attach the hook, and only then -- an OrthoLinear left with a stale _U from
    a previous step would silently train the wrong weights."""
    def pre(_mod, _args, _kwargs=None):
        batch_cayley(model)
    model.register_forward_pre_hook(pre, with_kwargs=True)
    return model


def demo():
    torch.manual_seed(0)
    base = nn.Linear(64, 128)
    lay = OrthoLinear(base, n_blocks=4, structured=True)
    x = torch.randn(3, 64)

    # Zero init is an exact identity, in weight space and output space.
    assert torch.allclose(lay.effective_weight(), base.weight.float(), atol=1e-6)
    assert torch.allclose(lay(x), base(x), atol=1e-5)

    n = sum(p.numel() for p in lay.parameters() if p.requires_grad)
    assert n == n_params(128, 4) == 4 * 16 ** 2, n

    with torch.no_grad():
        lay.a.normal_(std=0.1); lay.s.normal_(std=0.1)
    U = lay.orthogonal_blocks()
    I = torch.eye(lay.b).expand_as(U)

    # Cayley output really is orthogonal.
    assert torch.allclose(U.transpose(-1, -2) @ U, I, atol=1e-5)

    # ...and really lies in U(m): the complex embedding is exactly the set of
    # real matrices commuting with J, so this is what makes it "unitary".
    m = lay.m
    J = torch.zeros(lay.b, lay.b)
    J[:m, m:] = -torch.eye(m); J[m:, :m] = torch.eye(m)
    assert torch.allclose(U @ J, J @ U, atol=1e-5), "block escaped U(m)"

    # The control must NOT commute with J, or it is not a control.
    ctrl = OrthoLinear(nn.Linear(64, 128), n_blocks=4, structured=False)
    with torch.no_grad():
        ctrl.k.normal_(std=0.1)
    Uc = ctrl.orthogonal_blocks()
    assert not torch.allclose(Uc @ J, J @ Uc, atol=1e-3)

    # No saddle: unlike the complexified model, gradients are nonzero at init.
    # Pairing existing real channels leaves no conjugation symmetry to make the
    # loss even in S, so descent moves off the orthogonal slice immediately.
    fresh = OrthoLinear(nn.Linear(64, 128), n_blocks=4)
    fresh(x).pow(2).sum().backward()
    assert fresh.s.grad.abs().max() > 1e-6, "S direction is on a saddle"
    assert fresh.a.grad.abs().max() > 1e-6, "A direction is on a saddle"

    # Batching across modules must be bit-comparable to solving one at a time,
    # and must survive two shapes in the same model (that is why it groups by b).
    stack = nn.Sequential(OrthoLinear(nn.Linear(32, 128), n_blocks=4),
                          OrthoLinear(nn.Linear(32, 64), n_blocks=4))
    for mod in stack:
        with torch.no_grad():
            mod.a.normal_(std=0.05); mod.s.normal_(std=0.05)
    solo = [mod.orthogonal_blocks() for mod in stack]
    batch_cayley(stack)
    for mod, ref in zip(stack, solo):
        assert torch.allclose(mod.orthogonal_blocks(), ref, atol=1e-5)
    assert len({mod.b for mod in stack}) == 2, "test needs two block sizes to be a test"

    print("ok")


if __name__ == "__main__":
    demo()
