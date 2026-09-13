"""PoLAR (Lion et al., NeurIPS 2025) and its phase reparametrisation.

PoLAR replaces LoRA's `dW = BA` with

    dW = (alpha/r) X Theta Y^T ,   X in St(m,r), Y in St(n,r), Theta in R^{r x r}

and trains X, Y with the *landing* field of Ablin & Peyre rather than a
retraction, because every retraction needs an SVD or an inverse -- sequential,
GPU-hostile, and brittle in bf16. Iterates are therefore never exactly on the
manifold; they provably land on it. From Alg. 2 of the paper:

    psi(X) := Skew(grad_X L . X^T),      Skew(M) = (M - M^T)/2
    N(X)   := ||X^T X - I_r||_F^2   =>   grad N = 4 X (X^T X - I)
    Gamma(X) <- psi(X) X + lambda grad N(X)
    X <- X - eta rho(Gamma(X))           rho = Adam

`Landing` implements exactly that as an identity-forward / transform-backward
autograd node, so the field replaces the Euclidean gradient and Adam then runs
on top of it, as in the paper.

`PhaseStiefel` is PoLAR with our factors: X and Y complex in polar coordinates,
the product taken real,

    dW = (alpha/2r) Re(X Theta Y^H) = [Xr|Xi] Theta~ [Yr|Yi]^T,
    Theta~ = [[Theta_r, Theta_i], [-Theta_i, Theta_r]]

The landing penalty is applied to the *stacked* [Xr|Xi] in R^{m x 2r} against
I_{2r}, not to a complex Stiefel constraint X^H X = I. That is deliberate: it
leaves the feasible set for the direction factors identical to real PoLAR at
rank 2r, so only the chart changes -- the property that made the polar
reparametrisation win. Constraining to complex Stiefel would instead shrink the
feasible set, which is precisely what lost in the unitary experiment
(2.5445 vs 2.5004 against its own matched control).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Landing(torch.autograd.Function):
    """Identity forward; replaces the gradient with PoLAR's landing field."""

    @staticmethod
    def forward(ctx, X, lam):
        ctx.save_for_backward(X)
        ctx.lam = lam
        return X

    @staticmethod
    def backward(ctx, G):
        (X,) = ctx.saved_tensors
        M = G @ X.transpose(-1, -2)                       # m x m
        psi = 0.5 * (M - M.transpose(-1, -2))             # Skew
        I = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        gradN = 4.0 * X @ (X.transpose(-1, -2) @ X - I)   # d/dX ||X^T X - I||_F^2
        return psi @ X + ctx.lam * gradN, None


def _stiefel_init(m, r, gen=None):
    """Uniform on St(m,r): QR of a Gaussian, sign-fixed so R has positive diagonal."""
    A = torch.randn(m, r, generator=gen)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


class PolarStiefel(nn.Module):
    """PoLAR as published: real X, Y on Stiefel, unconstrained Theta, Theta_0 = 0."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 8.0, *, lam: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.X = nn.Parameter(_stiefel_init(out_f, r))
        self.Y = nn.Parameter(_stiefel_init(in_f, r))
        self.theta = nn.Parameter(torch.zeros(r, r))      # Theta_0 = 0 => dW = 0
        self.scale, self.lam = alpha / r, lam

    @property
    def bias(self):
        return self.base.bias

    def directions(self):
        return Landing.apply(self.X, self.lam), Landing.apply(self.Y, self.lam)

    def delta_weight(self):
        X, Y = self.directions()
        return self.scale * (X @ self.theta @ Y.T)

    def forward(self, x):
        X, Y = (t.to(x.dtype) for t in self.directions())
        d = ((x @ Y) @ self.theta.to(x.dtype).T) @ X.T
        return F.linear(x, self.base.weight, self.base.bias) + self.scale * d


class PhaseStiefel(nn.Module):
    """Phase PoLAR: PoLAR with complex factors in polar form and a real product.

    Complex rank r spans real rank 2r, so r here matches PoLAR's 2r in both
    image and (to within 2r^2 vs 4r^2 on Theta) parameter count.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, *, lam: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.r = r
        for name, d in (("x", out_f), ("y", in_f)):
            # Init the stacked real form on St(d, 2r), then read off polar coords:
            # the run starts exactly where real PoLAR at rank 2r would.
            S = _stiefel_init(d, 2 * r)
            re, im = S[:, :r], S[:, r:]
            self.register_parameter(f"rho_{name}", nn.Parameter(re.hypot(im)))
            self.register_parameter(f"phi_{name}", nn.Parameter(torch.atan2(im, re)))
        # Theta is complex too. A real Theta gives blockdiag(Theta, Theta), which
        # forces both channels to take the identical scaling; the embedding
        # [[Tr, Ti], [-Ti, Tr]] lets them rotate into each other. With the
        # direction factors already orthonormal, this middle block is the only
        # place phase can still act.
        self.theta_r = nn.Parameter(torch.zeros(r, r))
        self.theta_i = nn.Parameter(torch.zeros(r, r))
        self.scale, self.lam = alpha / (2 * r), lam
        self._S_x = self._S_y = self._Tt = None   # filled by batch_charts

    @property
    def bias(self):
        return self.base.bias

    def _stack(self, name):
        S = getattr(self, f"_S_{name}")
        if S is not None:                 # batched across the whole model
            return S
        rho, phi = getattr(self, f"rho_{name}"), getattr(self, f"phi_{name}")
        return torch.cat([rho * phi.cos(), rho * phi.sin()], dim=1)   # d x 2r

    def theta_tilde(self):
        """Real embedding of complex Theta: [[Tr, Ti], [-Ti, Tr]]."""
        if self._Tt is not None:
            return self._Tt
        return torch.cat([torch.cat([self.theta_r,  self.theta_i], 1),
                          torch.cat([-self.theta_i, self.theta_r], 1)], 0)

    def directions(self):
        return (Landing.apply(self._stack("x"), self.lam),
                Landing.apply(self._stack("y"), self.lam))

    def delta_weight(self):
        X, Y = self.directions()
        return self.scale * (X @ self.theta_tilde() @ Y.T)

    def forward(self, x):
        X, Y = (t.to(x.dtype) for t in self.directions())
        d = ((x @ Y) @ self.theta_tilde().to(x.dtype).T) @ X.T
        return F.linear(x, self.base.weight, self.base.bias) + self.scale * d


def batch_charts(model) -> None:
    """One cos/sin per distinct shape for every PhaseStiefel in the model.

    Building each direction from (rho, phi) per module doubles the autograd graph
    (26 nodes vs PoLAR's 13), and every extra node is a tiny elementwise op on a
    d x r tensor where launch cost dwarfs the arithmetic -- 56 modules x 2
    (checkpoint recompute) of those was the whole 1.29x. Batching is bitwise
    identical: an elementwise op on stacked data gives each element the same value
    it had alone.
    """
    groups = {}
    for m in model.modules():
        if isinstance(m, PhaseStiefel):
            for nm in ("x", "y"):
                groups.setdefault(tuple(getattr(m, f"rho_{nm}").shape), []).append((m, nm))
    for items in groups.values():
        rho = torch.stack([getattr(m, f"rho_{nm}") for m, nm in items])
        phi = torch.stack([getattr(m, f"phi_{nm}") for m, nm in items])
        S = torch.cat([rho * phi.cos(), rho * phi.sin()], dim=-1)
        for i, (m, nm) in enumerate(items):
            setattr(m, f"_S_{nm}", S[i])

    # blockdiag(Theta, Theta) for every module in one shot -- pure value
    # placement, so bitwise identical to 56 separate torch.block_diag calls.
    mods = [m for m in model.modules() if isinstance(m, PhaseStiefel)]
    for r_val in {m.r for m in mods}:
        sel = [m for m in mods if m.r == r_val]
        Tr = torch.stack([m.theta_r for m in sel])                  # (k, r, r)
        Ti = torch.stack([m.theta_i for m in sel])
        Tt = torch.cat([torch.cat([Tr,  Ti], -1),
                        torch.cat([-Ti, Tr], -1)], -2)              # (k, 2r, 2r)
        for i, m in enumerate(sel):
            m._Tt = Tt[i]


def install_batched_charts(model):
    def pre(_mod, _args, _kwargs=None):
        batch_charts(model)
    model.register_forward_pre_hook(pre, with_kwargs=True)
    return model


def n_params(out_f, in_f, r, phase):
    return (2 * (out_f + in_f) * r + 2 * r * r) if phase else ((out_f + in_f) * r + r * r)


def demo():
    torch.manual_seed(0)
    m, n = 96, 64
    x = torch.randn(5, n)

    for cls, r in ((PolarStiefel, 8), (PhaseStiefel, 4)):
        base = nn.Linear(n, m)
        lay = cls(base, r=r, alpha=8.0, lam=0.1)

        # Theta_0 = 0 means dW = 0 exactly, as in the paper.
        assert lay.delta_weight().abs().max() < 1e-6, cls.__name__
        assert torch.allclose(lay(x), base(x), atol=1e-5)

        # Directions start exactly on the Stiefel manifold.
        S = lay.X if cls is PolarStiefel else lay._stack("x")
        k = S.shape[1]
        assert torch.allclose(S.T @ S, torch.eye(k), atol=1e-5), "X_0 not on Stiefel"

        # forward() must agree with the explicit dW.
        with torch.no_grad():
            (lay.theta if cls is PolarStiefel else lay.theta_r).normal_(std=0.1)
        assert torch.allclose(lay(x), base(x) + x @ lay.delta_weight().T, atol=1e-4)

        got = sum(p.numel() for p in lay.parameters() if p.requires_grad)
        assert got == n_params(m, n, r, cls is PhaseStiefel), (cls.__name__, got)

    # Landing field: on-manifold the penalty vanishes and only Skew survives;
    # off-manifold it must point back toward orthonormality.
    X = _stiefel_init(m, 8).requires_grad_(True)
    lam = 0.1
    Landing.apply(X, lam).pow(2).sum().backward()
    on = X.grad.clone()
    X2 = (_stiefel_init(m, 8) * 1.5).requires_grad_(True)   # scaled off the manifold
    Landing.apply(X2, lam).pow(2).sum().backward()
    pen = lam * 4.0 * X2 @ (X2.T @ X2 - torch.eye(8))
    assert (X2.grad - pen).abs().max() < (X2.grad.abs().max()), "penalty term missing"
    assert on.abs().max() < X2.grad.abs().max(), "on-manifold grad should be smaller"

    # Skew part really is skew-symmetric.
    G = torch.randn(m, 8)
    M = G @ X.detach().T
    assert torch.allclose(0.5 * (M - M.T), -(0.5 * (M - M.T)).T, atol=1e-6)

    # The phase model's dW is real rank-2r and matches the complex form directly.
    lay = PhaseStiefel(nn.Linear(n, m), r=4, alpha=8.0, lam=0.1)
    with torch.no_grad():
        lay.theta_r.normal_(std=0.1); lay.theta_i.normal_(std=0.1)
        lay.rho_y.normal_(std=0.3)
    Xc = (lay._stack("x")[:, :4] + 1j * lay._stack("x")[:, 4:]).detach()
    Yc = (lay._stack("y")[:, :4] + 1j * lay._stack("y")[:, 4:]).detach()
    Tc = (lay.theta_r + 1j * lay.theta_i).detach()
    ref = lay.scale * (Xc @ Tc @ Yc.conj().T).real
    assert torch.allclose(lay.delta_weight(), ref, atol=1e-5), "not Re(X Theta Y^H)"
    assert torch.linalg.matrix_rank(lay.delta_weight(), tol=1e-6) <= 8

    print("ok")


# --------------------------------------------------------------------------
# StelLA (Li et al., Sony AI, arXiv 2510.01938)
# --------------------------------------------------------------------------
# Same three-factor U S V^T on Stiefel as PoLAR, and the same 15% overhead, but
# the constraint is enforced *exactly* every step by a polar retraction instead
# of being paid down by a penalty. PoLAR's stated reason for the penalty is that
# retraction needs an SVD (their Tab. 4: landing 3-18x faster) -- but that is
# measured per layer. Batched across the 56 same-shaped adapters here it is one
# call, which is the same fix that took our own charts from 1.29x to 1.017x.
#
# Structurally it cannot reuse Landing: lines 8-9 of their Alg. 1 run *after*
# the optimizer has moved the weights, so it is optimizer hooks, not autograd.
#
#   pre-step   grad_Y <- g - Y g^T Y                        (Riemannian, eq. 2)
#   step       Y~ <- Adam(Y, grad_Y)                        (any Euclidean opt)
#   post-step  D  <- pi_Y(s * (Y~ - Y)),  pi_Y(D) = D - Y symm(Y^T D)   (eq. 3)
#              Y  <- uf(Y + D)                              (polar retr., eq. 4)
#
# s is the gradient scaling: sqrt(d/m) for U, sqrt(d/n) for V. Entries of U and
# V are O(1/sqrt(m)) and O(1/sqrt(n)), but Adam normalises coordinate variance
# to Theta(1), so the taller factor learns sqrt(n/m) times faster than the other
# and holds the pair back. It has to scale the *perturbation*, not the raw
# gradient -- Adam is scale-invariant in the gradient, so scaling that is a
# no-op. This is LoRA+'s idea applied to the Stiefel factors, and it targets the
# same coupling our phase chart does, head-on rather than through the geometry.


def _uf(M):
    """Orthogonal factor of the polar decomposition, uf(M) = U V^T.

    Via M (M^T M)^{-1/2} rather than an SVD of M. Identical for full-column-rank
    M, and the cost moves from an (m x r) SVD to an (r x r) eigendecomposition --
    r = 8 against m = 2048 here. Conditioning is a non-issue: M is a retraction
    step away from orthonormal, so M^T M is within a hair of I.
    """
    w, Q = torch.linalg.eigh(M.transpose(-1, -2) @ M)
    return M @ (Q * w.clamp_min(1e-12).rsqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)


class StelLA(nn.Module):
    """dW = (alpha/r) U S V^T, U and V exactly on Stiefel, S_0 = I.

    S_0 = I means dW != 0 at init -- deliberate, and the one place StelLA breaks
    with LoRA convention. Their Tab. 5 ablation puts zero init 0.2 behind it, and
    "pseudo-zero" (subtracting the init back out of W) 2.5 behind: the small S
    starves U and V of gradient early, which is LoRA's B=0 stall in another guise.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 8.0, d: int = None):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.U = nn.Parameter(_stiefel_init(out_f, r))
        self.V = nn.Parameter(_stiefel_init(in_f, r))
        self.S = nn.Parameter(torch.eye(r))
        self.scale = alpha / r
        d = in_f if d is None else d
        self.gs = {"U": math.sqrt(d / out_f), "V": math.sqrt(d / in_f)}

    @property
    def bias(self):
        return self.base.bias

    def delta_weight(self):
        return self.scale * (self.U @ self.S @ self.V.T)

    def forward(self, x):
        U, S, V = (t.to(x.dtype) for t in (self.U, self.S, self.V))
        d = ((x @ V) @ S.T) @ U.T
        return F.linear(x, self.base.weight, self.base.bias) + self.scale * d


def install_stella(model, opt):
    """Wire Alg. 1 lines 6 and 8-9 onto an existing optimizer.

    Everything is grouped by shape and done on stacked tensors: 3 batched ops per
    step here rather than 112 tiny ones on (d x r) slivers, where launch cost
    dwarfs the arithmetic. Same lesson as batch_cayley and batch_charts, and it
    is what makes the exact constraint cost what PoLAR's penalty costs -- StelLA's
    own Tab. 12 measures 14-25x for the batched retraction alone.
    """
    items = [(m, nm) for m in model.modules() if isinstance(m, StelLA) for nm in ("U", "V")]
    if not items:
        raise ValueError("no StelLA modules found")
    packs = {}
    for m, nm in items:
        packs.setdefault(tuple(getattr(m, nm).shape), []).append((m, nm))
    packs = [(its, torch.tensor([m.gs[nm] for m, nm in its],
                                device=getattr(its[0][0], its[0][1]).device).view(-1, 1, 1))
             for its in packs.values()]
    prev = {}

    def pre(*_):
        for k, (its, _) in enumerate(packs):
            Y = torch.stack([getattr(m, nm).detach() for m, nm in its])
            prev[k] = Y
            if getattr(*its[0]).grad is None:
                continue
            G = torch.stack([getattr(m, nm).grad for m, nm in its])
            R = G - Y @ (G.transpose(-1, -2) @ Y)                # eq. 2
            for i, (m, nm) in enumerate(its):
                getattr(m, nm).grad.copy_(R[i])

    def post(*_):
        for k, (its, gs) in enumerate(packs):
            Y0 = prev[k]
            D = gs * (torch.stack([getattr(m, nm).detach() for m, nm in its]) - Y0)
            D = D - Y0 @ _symm(Y0.transpose(-1, -2) @ D)         # eq. 3
            R = _uf(Y0 + D)                                      # eq. 4
            for i, (m, nm) in enumerate(its):
                getattr(m, nm).data.copy_(R[i])

    opt.register_step_pre_hook(pre)
    opt.register_step_post_hook(post)
    return model


def _symm(A):
    return 0.5 * (A + A.transpose(-1, -2))


def demo_stella():
    torch.manual_seed(0)
    m, n, r = 96, 64, 8
    base = nn.Linear(n, m)
    lay = StelLA(base, r=r, alpha=8.0, d=n)
    x = torch.randn(5, n)

    # Exactly on Stiefel at init, and S_0 = I means dW is deliberately NOT zero.
    for nm in ("U", "V"):
        Y = getattr(lay, nm)
        assert torch.allclose(Y.T @ Y, torch.eye(r), atol=1e-5), nm
    assert lay.delta_weight().abs().max() > 1e-3, "S_0 = I must perturb the model"
    assert torch.allclose(lay(x), base(x) + x @ lay.delta_weight().T, atol=1e-4)

    # Same parameter count as PoLAR at the same rank -- that is the whole point
    # of running it as a control against our PoLAR numbers.
    got = sum(p.numel() for p in lay.parameters() if p.requires_grad)
    assert got == n_params(m, n, r, phase=False), got

    # eq. 2 lands in the tangent space: Y^T D + D^T Y = 0.
    Y = _stiefel_init(m, r)
    G = torch.randn(m, r)
    D = G - Y @ (G.T @ Y)
    assert (Y.T @ D + D.T @ Y).abs().max() < 1e-5, "Riemannian grad off tangent"
    # ...and so does eq. 3, for an arbitrary perturbation.
    P = torch.randn(m, r)
    D = P - Y @ _symm(Y.T @ P)
    assert (Y.T @ D + D.T @ Y).abs().max() < 1e-5, "projection off tangent"

    # The property PoLAR does not have: after real optimizer steps, U and V are
    # still orthonormal to machine precision, not approaching it asymptotically.
    net = nn.ModuleList([StelLA(nn.Linear(n, m), r=r, d=n),
                         StelLA(nn.Linear(n, m), r=r, d=n),
                         StelLA(nn.Linear(n, 32), r=r, d=n)])   # 3rd U shape differs
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=1e-2)
    install_stella(net, opt)
    for _ in range(5):
        sum(mod(x).pow(2).sum() for mod in net).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    for mod in net:
        for nm in ("U", "V"):
            Y = getattr(mod, nm)
            assert torch.allclose(Y.T @ Y, torch.eye(r), atol=1e-4), \
                f"{nm} drifted off Stiefel: {(Y.T @ Y - torch.eye(r)).abs().max():.2e}"
    assert len({tuple(mod.U.shape) for mod in net}) == 2, "test needs two shapes to be a test"

    # Batched retraction must agree with retracting one at a time...
    Ms = [_stiefel_init(m, r) + 0.01 * torch.randn(m, r) for _ in range(4)]
    solo = torch.stack([_uf(M) for M in Ms])
    assert torch.allclose(_uf(torch.stack(Ms)), solo, atol=1e-5)
    # ...and the (M^T M)^{-1/2} form must agree with the SVD it stands in for.
    for M in Ms:
        Us, _, Vh = torch.linalg.svd(M, full_matrices=False)
        assert torch.allclose(_uf(M), Us @ Vh, atol=1e-5), "uf != polar factor"
    assert torch.allclose(_uf(Ms[0]).T @ _uf(Ms[0]), torch.eye(r), atol=1e-5)

    print("ok")




# --------------------------------------------------------------------------
# Phase StelLA: StelLA with our complex factors
# --------------------------------------------------------------------------
# Same relationship to StelLA that Phase PoLAR has to PoLAR. U and V are complex
# in polar coordinates and the product is taken real,
#
#   dW = (alpha/2r) Re(U S V^H) = [Ur|Ui] S~ [Vr|Vi]^T,  S~ = [[Sr, Si], [-Si, Sr]]
#
# with the Stiefel constraint on the stacked [Ur|Ui] in R^{m x 2r}, so the
# feasible set is identical to real StelLA at rank 2r and only the chart differs.
#
# The wrinkle StelLA has that PoLAR did not: its geometry runs *after* the
# optimizer, in stacked space, while Adam's moments -- the thing the chart is
# supposed to change -- live in (rho, phi). So the hooks have to move gradients
# through the chart Jacobian and pull the retracted point back:
#
#   S = [rho*cos(phi) | rho*sin(phi)]
#   d(rho,phi) -> d(Sr,Si):  gSr = grho c - (gphi/rho) s,  gSi = grho s + (gphi/rho) c
#   d(Sr,Si) -> d(rho,phi):  grho = gSr c + gSi s,  gphi = rho(-gSr s + gSi c)
#
# gphi carries a factor of rho, so gphi/rho is finite as rho -> 0; the clamp is
# only there to keep a denormal from turning into an inf.


def _to_stacked(rho, phi):
    return torch.cat([rho * phi.cos(), rho * phi.sin()], dim=-1)


def _from_stacked(S, r):
    re, im = S[..., :r], S[..., r:]
    return re.hypot(im), torch.atan2(im, re)


def _grad_to_stacked(grho, gphi, rho, phi):
    c, s = phi.cos(), phi.sin()
    # gphi carries a factor of rho *with its sign*, so the guard has to keep the
    # sign too -- clamping a negative rho up to +eps inverts the phase gradient.
    # In training rho >= 0 (the retraction rebuilds it with hypot every step),
    # but the chart is smooth through negative rho and this must not depend on it.
    safe = torch.where(rho >= 0, rho.clamp_min(1e-12), rho.clamp_max(-1e-12))
    t = gphi / safe
    return torch.cat([grho * c - t * s, grho * s + t * c], dim=-1)


def _grad_from_stacked(G, rho, phi, r):
    c, s = phi.cos(), phi.sin()
    Gr, Gi = G[..., :r], G[..., r:]
    return Gr * c + Gi * s, rho * (Gi * c - Gr * s)


class PhaseStelLA(nn.Module):
    """StelLA with complex factors in polar form and a real product.

    Complex rank r spans real rank 2r, so r here matches StelLA's 2r in image and
    -- to within 2r^2 against 4r^2 on S -- in parameter count, exactly as
    PhaseStiefel matches PolarStiefel.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, d: int = None):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.r = r
        for name, dim in (("u", out_f), ("v", in_f)):
            S = _stiefel_init(dim, 2 * r)      # start where real StelLA at rank 2r starts
            rho, phi = _from_stacked(S, r)
            self.register_parameter(f"rho_{name}", nn.Parameter(rho))
            self.register_parameter(f"phi_{name}", nn.Parameter(phi))
        self.s_r = nn.Parameter(torch.eye(r))   # StelLA's S_0 = I, complexified
        self.s_i = nn.Parameter(torch.zeros(r, r))
        self.scale = alpha / (2 * r)
        d = in_f if d is None else d
        self.gs = {"u": math.sqrt(d / out_f), "v": math.sqrt(d / in_f)}
        self._S_u = self._S_v = self._St = None   # filled by batch_phase_charts

    @property
    def bias(self):
        return self.base.bias

    def stacked(self, name):
        S = getattr(self, f"_S_{name}")           # batched across the whole model
        if S is not None:
            return S
        return _to_stacked(getattr(self, f"rho_{name}"), getattr(self, f"phi_{name}"))

    def s_tilde(self):
        if self._St is not None:
            return self._St
        return torch.cat([torch.cat([self.s_r,  self.s_i], 1),
                          torch.cat([-self.s_i, self.s_r], 1)], 0)

    def delta_weight(self):
        return self.scale * (self.stacked("u") @ self.s_tilde() @ self.stacked("v").T)

    def forward(self, x):
        U, V = (t.to(x.dtype) for t in (self.stacked("u"), self.stacked("v")))
        d = ((x @ V) @ self.s_tilde().to(x.dtype).T) @ U.T
        return F.linear(x, self.base.weight, self.base.bias) + self.scale * d


def install_phase_stella(model, opt):
    """StelLA's Alg. 1 on the stacked factors, with Adam's state left in (rho, phi)."""
    items = [(m, nm) for m in model.modules() if isinstance(m, PhaseStelLA) for nm in ("u", "v")]
    if not items:
        raise ValueError("no PhaseStelLA modules found")
    packs = {}
    for m, nm in items:
        packs.setdefault(tuple(getattr(m, f"rho_{nm}").shape), []).append((m, nm))
    packs = [(its, torch.tensor([m.gs[nm] for m, nm in its],
                                device=getattr(its[0][0], f"rho_{its[0][1]}").device).view(-1, 1, 1))
             for its in packs.values()]
    prev = {}

    def _stack(its, attr):
        return torch.stack([getattr(getattr(m, f"{attr}_{nm}"), "data"
                                    if attr in ("rho", "phi") else "grad")
                            for m, nm in its])

    def pre(*_):
        for k, (its, _) in enumerate(packs):
            rho = torch.stack([getattr(m, f"rho_{nm}").detach() for m, nm in its])
            phi = torch.stack([getattr(m, f"phi_{nm}").detach() for m, nm in its])
            Y = _to_stacked(rho, phi)
            prev[k] = Y
            if getattr(its[0][0], f"rho_{its[0][1]}").grad is None:
                continue
            grho = torch.stack([getattr(m, f"rho_{nm}").grad for m, nm in its])
            gphi = torch.stack([getattr(m, f"phi_{nm}").grad for m, nm in its])
            G = _grad_to_stacked(grho, gphi, rho, phi)
            R = G - Y @ (G.transpose(-1, -2) @ Y)                    # eq. 2, stacked
            r = rho.shape[-1]
            nrho, nphi = _grad_from_stacked(R, rho, phi, r)          # back to the chart
            for i, (m, nm) in enumerate(its):
                getattr(m, f"rho_{nm}").grad.copy_(nrho[i])
                getattr(m, f"phi_{nm}").grad.copy_(nphi[i])

    def post(*_):
        for k, (its, gs) in enumerate(packs):
            Y0 = prev[k]
            rho = torch.stack([getattr(m, f"rho_{nm}").detach() for m, nm in its])
            phi = torch.stack([getattr(m, f"phi_{nm}").detach() for m, nm in its])
            D = gs * (_to_stacked(rho, phi) - Y0)
            D = D - Y0 @ _symm(Y0.transpose(-1, -2) @ D)             # eq. 3
            nrho, nphi = _from_stacked(_uf(Y0 + D), rho.shape[-1])   # eq. 4, back to chart
            for i, (m, nm) in enumerate(its):
                getattr(m, f"rho_{nm}").data.copy_(nrho[i])
                getattr(m, f"phi_{nm}").data.copy_(nphi[i])

    opt.register_step_pre_hook(pre)
    opt.register_step_post_hook(post)
    return model


def batch_phase_charts(model) -> None:
    """One cos/sin per distinct shape for every PhaseStelLA, as batch_charts does.

    The forward rebuilds [Ur|Ui] from (rho, phi) per module, and gradient
    checkpointing pays for it twice; 56 modules of tiny elementwise ops on
    (d x r) slivers is pure launch cost. Batching is bitwise identical -- an
    elementwise op on stacked data gives each element the value it had alone.
    """
    groups = {}
    for m in model.modules():
        if isinstance(m, PhaseStelLA):
            for nm in ("u", "v"):
                groups.setdefault(tuple(getattr(m, f"rho_{nm}").shape), []).append((m, nm))
    for its in groups.values():
        S = _to_stacked(torch.stack([getattr(m, f"rho_{nm}") for m, nm in its]),
                        torch.stack([getattr(m, f"phi_{nm}") for m, nm in its]))
        for i, (m, nm) in enumerate(its):
            setattr(m, f"_S_{nm}", S[i])

    mods = [m for m in model.modules() if isinstance(m, PhaseStelLA)]
    for r_val in {m.r for m in mods}:
        sel = [m for m in mods if m.r == r_val]
        Sr = torch.stack([m.s_r for m in sel])
        Si = torch.stack([m.s_i for m in sel])
        St = torch.cat([torch.cat([Sr,  Si], -1),
                        torch.cat([-Si, Sr], -1)], -2)
        for i, m in enumerate(sel):
            m._St = St[i]


def demo_phase_stella():
    torch.manual_seed(0)
    m, n, r = 96, 64, 4
    base = nn.Linear(n, m)
    lay = PhaseStelLA(base, r=r, alpha=8.0, d=n)
    x = torch.randn(5, n)

    # The stacked factors start exactly on St(d, 2r): the chart round-trips.
    for nm in ("u", "v"):
        S = lay.stacked(nm)
        assert torch.allclose(S.T @ S, torch.eye(2 * r), atol=1e-5), nm
    # S_0 = I, so like StelLA this deliberately perturbs the model at init.
    assert lay.delta_weight().abs().max() > 1e-3
    assert torch.allclose(lay(x), base(x) + x @ lay.delta_weight().T, atol=1e-4)

    # Same parameter count as Phase PoLAR at the same rank.
    got = sum(p.numel() for p in lay.parameters() if p.requires_grad)
    assert got == n_params(m, n, r, phase=True), got

    # dW really is Re(U S V^H), real rank 2r.
    with torch.no_grad():
        lay.s_i.normal_(std=0.1); lay.rho_v.normal_(std=0.3)
    Uc = (lay.stacked("u")[:, :r] + 1j * lay.stacked("u")[:, r:]).detach()
    Vc = (lay.stacked("v")[:, :r] + 1j * lay.stacked("v")[:, r:]).detach()
    Sc = (lay.s_r + 1j * lay.s_i).detach()
    assert torch.allclose(lay.delta_weight(), lay.scale * (Uc @ Sc @ Vc.conj().T).real, atol=1e-5)
    assert torch.linalg.matrix_rank(lay.delta_weight(), tol=1e-6) <= 2 * r

    # The Jacobian round-trip is the load-bearing part: the stacked gradient the
    # pre-hook reconstructs from (rho, phi) must equal the gradient a directly
    # stacked parameterisation would have produced at the same point.
    lay.zero_grad()
    out = lay(x).pow(2).sum()
    out.backward()
    for nm in ("u", "v"):
        rho, phi = getattr(lay, f"rho_{nm}").detach(), getattr(lay, f"phi_{nm}").detach()
        got_g = _grad_to_stacked(getattr(lay, f"rho_{nm}").grad,
                                 getattr(lay, f"phi_{nm}").grad, rho, phi)
        ref = _to_stacked(rho, phi).clone().requires_grad_(True)
        other = lay.stacked("v" if nm == "u" else "u").detach()
        U, V = (ref, other) if nm == "u" else (other, ref)
        (base(x) + lay.scale * (((x @ V) @ lay.s_tilde().detach().T) @ U.T)).pow(2).sum().backward()
        assert torch.allclose(got_g, ref.grad, atol=1e-4), \
            f"chart Jacobian wrong for {nm}: {(got_g - ref.grad).abs().max():.2e}"
    # ...and it inverts.
    g = torch.randn(m, 2 * r)
    rho, phi = _from_stacked(_stiefel_init(m, 2 * r), r)
    a, b = _grad_from_stacked(g, rho, phi, r)
    assert torch.allclose(_grad_to_stacked(a, b, rho, phi), g, atol=1e-4)

    # After real optimizer steps the stacked factors are still on the manifold.
    net = nn.ModuleList([PhaseStelLA(nn.Linear(n, m), r=r, d=n),
                         PhaseStelLA(nn.Linear(n, m), r=r, d=n),
                         PhaseStelLA(nn.Linear(n, 32), r=r, d=n)])
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=1e-2)
    install_phase_stella(net, opt)
    for _ in range(5):
        sum(mod(x).pow(2).sum() for mod in net).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    for mod in net:
        for nm in ("u", "v"):
            S = mod.stacked(nm)
            assert torch.allclose(S.T @ S, torch.eye(2 * r), atol=1e-4), \
                f"{nm} drifted: {(S.T @ S - torch.eye(2 * r)).abs().max():.2e}"
    assert len({tuple(mod.rho_u.shape) for mod in net}) == 2, "test needs two shapes"

    # The mechanism itself survives the retraction: phi + pi is a sign flip at
    # constant modulus, which is the one thing the real chart cannot do.
    lay = PhaseStelLA(nn.Linear(n, m), r=r, d=n)
    with torch.no_grad():
        before, rho_before = lay.delta_weight().clone(), lay.rho_u.clone()
        lay.phi_u += math.pi
    assert torch.allclose(lay.delta_weight(), -before, atol=1e-5), "phase pi is not a sign flip"
    assert torch.allclose(lay.rho_u, rho_before), "modulus moved during the flip"

    print("ok")

if __name__ == "__main__":
    demo()
    demo_stella()
    demo_phase_stella()
