"""PoLAR (Lion et al., NeurIPS 2025) and its phase reparametrisation.

PoLAR replaces LoRA's `dW = BA` with

    dW = (alpha/r) X Theta Y^T ,   X in St(m,r), Y in St(n,r), Theta in R^{r x r}

and trains X, Y with the landing field of Ablin & Peyre instead of a retraction (the paper's reason: a
retraction needs an SVD or a matrix inverse). The iterates are near the manifold, not on it; Ablin & Peyre
prove that they converge to it. From Alg. 2 of the paper:

    psi(X) := Skew(grad_X L . X^T),      Skew(M) = (M - M^T)/2
    N(X)   := ||X^T X - I_r||_F^2   =>   grad N = 4 X (X^T X - I)
    Gamma(X) <- psi(X) X + lambda grad N(X)
    X <- X - eta rho(Gamma(X))           rho = Adam

`Landing` implements this as an autograd node with an identity forward: the field replaces the Euclidean
gradient, and Adam steps on the field, as in the paper.

`PhaseStiefel` is PoLAR at rank 2r with its two direction factors in polar coordinates:
X = [Xr|Xi] with Xr = rho cos(phi), Xi = rho sin(phi), the same for Y,

    dW = (alpha/2r) [Xr|Xi] Theta [Yr|Yi]^T,   Theta in R^{2r x 2r}, Theta_0 = 0

The landing penalty is applied to the stacked [Xr|Xi] in R^{m x 2r} against
I_{2r}, not to a complex Stiefel constraint X^H X = I, and Theta is a free real
matrix. A complex Theta, embedded as [[Tr, Ti], [-Ti, Tr]], would give every singular value of dW twice.
With these two choices the function class and the parameter count equal real PoLAR's at rank 2r (checked in
demo); only the coordinates of X and Y differ.
"""

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
        X, Y = self.directions()
        # fp32 adapter branch, as PEFT runs LoRA's (see PolarLoRALinear.forward)
        d = ((x.float() @ Y) @ self.theta.T) @ X.T
        return (F.linear(x, self.base.weight, self.base.bias) + self.scale * d).to(x.dtype)


class PhaseStiefel(nn.Module):
    """Phase PoLAR: PoLAR at rank 2r with X and Y in polar coordinates.

    r is the complex rank; the function class and the parameter count equal PoLAR's at rank 2r.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, *, lam: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.r = r
        for name, d in (("x", out_f), ("y", in_f)):
            # The stacked real form starts on St(d, 2r), as real PoLAR's factors do at rank 2r;
            # rho and phi are its polar coordinates.
            S = _stiefel_init(d, 2 * r)
            re, im = S[:, :r], S[:, r:]
            self.register_parameter(f"rho_{name}", nn.Parameter(re.hypot(im)))
            self.register_parameter(f"phi_{name}", nn.Parameter(torch.atan2(im, re)))
        self.theta = nn.Parameter(torch.zeros(2 * r, 2 * r))   # PoLAR's Theta_0 = 0 at rank 2r
        self.scale, self.lam = alpha / (2 * r), lam
        self._S_x = self._S_y = None   # filled by batch_charts

    @property
    def bias(self):
        return self.base.bias

    def _stack(self, name):
        S = getattr(self, f"_S_{name}")
        if S is not None:                 # batched across the whole model
            return S
        rho, phi = getattr(self, f"rho_{name}"), getattr(self, f"phi_{name}")
        return torch.cat([rho * phi.cos(), rho * phi.sin()], dim=1)   # d x 2r

    def directions(self):
        return (Landing.apply(self._stack("x"), self.lam),
                Landing.apply(self._stack("y"), self.lam))

    def delta_weight(self):
        X, Y = self.directions()
        return self.scale * (X @ self.theta @ Y.T)

    def forward(self, x):
        X, Y = self.directions()
        d = ((x.float() @ Y) @ self.theta.T) @ X.T      # fp32 adapter branch
        return (F.linear(x, self.base.weight, self.base.bias) + self.scale * d).to(x.dtype)


def batch_charts(model) -> None:
    """One cos/sin per distinct shape for every PhaseStiefel in the model.

    Building each direction from (rho, phi) per module adds small elementwise autograd nodes to every module;
    one batched op per shape replaces them. The ops are elementwise, so each module gets the value it gets
    unbatched (checked in demo).
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


def install_batched_charts(model):
    def pre(_mod, _args, _kwargs=None):
        batch_charts(model)
    model.register_forward_pre_hook(pre, with_kwargs=True)
    return model


def n_params(out_f, in_f, r, phase):
    """Trainable parameters of PoLAR at rank r, or of Phase PoLAR at complex rank r (real rank 2r)."""
    real_rank = 2 * r if phase else r
    return (out_f + in_f) * real_rank + real_rank ** 2


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
            lay.theta.normal_(std=0.1)
        assert torch.allclose(lay(x), base(x) + x @ lay.delta_weight().T, atol=1e-4)

        got = sum(p.numel() for p in lay.parameters() if p.requires_grad)
        assert got == n_params(m, n, r, cls is PhaseStiefel), (cls.__name__, got)

    # Landing field: the penalty term is zero on the manifold and present off it.
    X = _stiefel_init(m, 8).requires_grad_(True)
    lam = 0.1
    Landing.apply(X, lam).pow(2).sum().backward()
    on = X.grad.clone()
    X2 = (_stiefel_init(m, 8) * 1.5).requires_grad_(True)   # scaled off the manifold
    Landing.apply(X2, lam).pow(2).sum().backward()
    pen = lam * 4.0 * X2 @ (X2.T @ X2 - torch.eye(8))
    assert (X2.grad - pen).abs().max() < (X2.grad.abs().max()), "penalty term missing"
    assert on.abs().max() < X2.grad.abs().max(), "on-manifold grad should be smaller"

    # The Skew part is skew-symmetric.
    G = torch.randn(m, 8)
    M = G @ X.detach().T
    assert torch.allclose(0.5 * (M - M.T), -(0.5 * (M - M.T)).T, atol=1e-6)

    # Phase PoLAR has PoLAR's class at rank 2r: a Theta with 2r distinct singular values gives a dW with
    # exactly those; a complex middle block gives each singular value twice.
    lay = PhaseStiefel(nn.Linear(n, m), r=4, alpha=8.0, lam=0.1)
    with torch.no_grad():
        lay.theta.copy_(torch.diag(torch.arange(1.0, 9.0)))
    singular = torch.linalg.svdvals(lay.delta_weight() / lay.scale)[:8]   # descending; the rest are 0
    assert torch.allclose(singular.flip(0), torch.arange(1.0, 9.0), atol=1e-4), singular

    # Batching the charts gives each module its own value.
    net = nn.ModuleList([PhaseStiefel(nn.Linear(n, m), r=4, lam=0.1), PhaseStiefel(nn.Linear(n, 32), r=4, lam=0.1)])
    with torch.no_grad():
        for mod in net:
            mod.theta.normal_(std=0.1)
    alone = [mod(x) for mod in net]
    batch_charts(net)
    for mod, out in zip(net, alone):
        assert torch.allclose(mod(x), out, atol=1e-6), "batched chart changed a module"

    print("ok")


if __name__ == "__main__":
    demo()
