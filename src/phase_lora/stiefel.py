"""PoLAR (Lion et al., NeurIPS 2025) and its phase reparametrisation.

PoLAR replaces LoRA's `dW = BA` with

    dW = (alpha/r) X Theta Y^T ,   X in St(m,r), Y in St(n,r), Theta in R^{r x r}

and trains X, Y with the landing field of Ablin & Peyre instead of a retraction (the paper's reason: a
retraction needs an SVD or a matrix inverse). The iterates are near the manifold, not on it; Ablin & Peyre
prove that they converge to it. The field, as the official code computes it (github.com/kcc-lion/polar,
callbacks/landing_callback.py, landing_step):

    Gamma(X) = (G X^T - X G^T) X + lambda X (X^T X - I),   G = grad_X L
    X <- X - eta rho(Gamma(X))                             rho = Adam

The order is the official one, for both arms: the Euclidean gradients of all trained parameters are clipped,
then `apply_landing` replaces the gradient of X and Y by the field of the clipped gradient, just before the
optimizer step (`train_utils.make_optimizer`). Theta keeps its clipped Euclidean gradient. Phase PoLAR forms the
same field on its stacked X = [rho cos phi | rho sin phi] and passes it back to rho and phi by the chain rule.

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


def landing_field(X, G, lam):
    """PoLAR's landing field of the gradient G at X (official code, landing_step): (G X^T - X G^T) X + lam X (X^T X - I)."""
    M = G @ X.transpose(-1, -2)
    I = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    return (M - M.transpose(-1, -2)) @ X + lam * (X @ (X.transpose(-1, -2) @ X - I))


class _Record(torch.autograd.Function):
    """Identity; its backward stores the incoming gradient. An autograd node runs in every backward pass,
    gradient checkpointing included; a tensor hook on a factor built outside the checkpointed layers does not."""

    @staticmethod
    def forward(ctx, S, store, name):
        ctx.store, ctx.name = store, name
        return S.view_as(S)

    @staticmethod
    def backward(ctx, G):
        ctx.store[ctx.name] = G
        return G, None, None


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
        return self.X, self.Y

    @torch.no_grad()
    def apply_landing(self, clip_coef):
        """Gradients of X and Y -> landing field. They are already clipped, so clip_coef is not needed here."""
        for factor in (self.X, self.Y):
            if factor.grad is not None:
                factor.grad.copy_(landing_field(factor, factor.grad, self.lam))

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
        self._grad = {}                # Euclidean gradient of the stacked X and Y from the last backward pass

    @property
    def bias(self):
        return self.base.bias

    def _chart(self, name):
        rho, phi = getattr(self, f"rho_{name}"), getattr(self, f"phi_{name}")
        return torch.cat([rho * phi.cos(), rho * phi.sin()], dim=1)   # d x 2r

    def capture(self, name, S):
        """S, passed through a node that records its Euclidean gradient in the backward pass."""
        return _Record.apply(S, self._grad, name) if S.requires_grad else S

    def _stack(self, name):
        S = getattr(self, f"_S_{name}")
        if S is not None:                 # batched across the whole model; batch_charts set the capture
            return S
        return self.capture(name, self._chart(name))

    def directions(self):
        return self._stack("x"), self._stack("y")

    def apply_landing(self, clip_coef):
        """rho, phi gradients -> chain rule of the landing field of the clipped stacked gradient.
        clip_coef is the factor by which the clipping scaled every gradient, so clip_coef * G is the clipped G."""
        for name in ("x", "y"):
            G = self._grad.pop(name, None)
            if G is None:
                continue
            rho, phi = getattr(self, f"rho_{name}"), getattr(self, f"phi_{name}")
            with torch.enable_grad():
                S = self._chart(name)
                field = landing_field(S.detach(), clip_coef * G, self.lam)
                grad_rho, grad_phi = torch.autograd.grad(S, (rho, phi), grad_outputs=field)
            rho.grad, phi.grad = grad_rho, grad_phi

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
            setattr(m, f"_S_{nm}", m.capture(nm, S[i]))


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
    lam = 0.1
    X = _stiefel_init(m, 8)
    G = torch.randn(m, 8)
    assert torch.allclose(landing_field(X, torch.zeros_like(G), lam), torch.zeros(m, 8), atol=1e-5), "penalty on St"
    X2 = _stiefel_init(m, 8) * 1.5                                  # scaled off the manifold
    pen = lam * X2 @ (X2.T @ X2 - torch.eye(8))
    assert torch.allclose(landing_field(X2, torch.zeros_like(G), lam), pen, atol=1e-5), "penalty term missing"

    # On the manifold the field is (G X^T - X G^T) X, so X^T field = X^T G - G^T X is skew-symmetric.
    projected = X.T @ landing_field(X, G, lam)
    assert torch.allclose(projected, -projected.T, atol=1e-5), "skew term"
    assert torch.allclose(projected, X.T @ G - G.T @ X, atol=1e-5), "skew term"

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
