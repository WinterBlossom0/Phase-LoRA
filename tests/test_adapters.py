"""The adapters' own self-checks, run under pytest (CPU only)."""
from phase_lora.polar import demo as polar_demo
from phase_lora.stiefel import demo as polar_stiefel_demo


def test_polar():
    polar_demo()


def test_polar_stiefel():
    polar_stiefel_demo()


def test_landing_follows_the_official_order_for_both_arms():
    """Official PoLAR step: clip the Euclidean gradients of all trained parameters, then replace the X and Y
    gradients by (G X^T - X G^T) X + lambda X (X^T X - I) of the clipped G. Phase PoLAR forms the same field on
    its stacked X and passes it to rho and phi by the chain rule; Theta keeps its clipped gradient."""
    import torch
    from torch import nn
    from phase_lora.stiefel import PhaseStiefel, PolarStiefel
    from phase_lora.train_utils import make_optimizer

    def official_field(X, G, lam):
        M = G @ X.T
        return (M - M.T) @ X + lam * X @ (X.T @ X - torch.eye(X.shape[1]))

    torch.manual_seed(0)
    lam = 0.1
    x = torch.randn(5, 64)
    for (cls, rank), clip in [((PolarStiefel, 8), 1e-3), ((PhaseStiefel, 4), 1e-3),     # clipping active
                              ((PolarStiefel, 8), 1e3), ((PhaseStiefel, 4), 1e3)]:      # clipping inactive
        layer = cls(nn.Linear(64, 96), rank, 8.0, lam=lam)
        with torch.no_grad():
            layer.theta.normal_(std=0.1)
            for name, p in layer.named_parameters():       # off the manifold, so the penalty term counts
                if p.requires_grad and name != "theta":
                    p.add_(0.2 * torch.randn_like(p))
        params = [p for p in layer.parameters() if p.requires_grad]
        weights = torch.randn(5, 96)

        # reference: Euclidean gradients, clip coefficient, official field, chain rule for the phase arm
        stacks = {}
        if cls is PolarStiefel:
            stacks = {"x": layer.X, "y": layer.Y}
        else:
            stacks = {name: layer._chart(name) for name in ("x", "y")}
        X, Y = stacks["x"], stacks["y"]
        out = ((x @ Y) @ layer.theta.T) @ X.T
        euclid = torch.autograd.grad((out * weights).sum(), params + ([] if cls is PolarStiefel else [X, Y]))
        param_grads, stack_grads = euclid[:len(params)], euclid[len(params):]
        total = torch.sqrt(sum((g ** 2).sum() for g in param_grads))
        coef = min(1.0, clip / (float(total) + 1e-6))
        expected = {}
        if cls is PolarStiefel:
            for name, p, g in zip(("X", "Y", "theta"), params, param_grads):
                expected[name] = official_field(p.detach(), coef * g, lam) if name in ("X", "Y") else coef * g
        else:
            named = dict(zip([n for n, _ in layer.named_parameters() if not n.startswith("base")], param_grads))
            expected["theta"] = coef * named["theta"]
            for name, G in zip(("x", "y"), stack_grads):
                S = layer._chart(name)
                field = official_field(S.detach(), coef * G, lam)
                g_rho, g_phi = torch.autograd.grad(S, (getattr(layer, f"rho_{name}"), getattr(layer, f"phi_{name}")), field)
                expected[f"rho_{name}"], expected[f"phi_{name}"] = g_rho, g_phi

        # implementation: plain backward, then the optimizer hook (clip, then landing); lr 0 keeps the weights
        layer.zero_grad()
        (layer(x) - nn.functional.linear(x, layer.base.weight, layer.base.bias)).mul(weights).sum().backward()
        make_optimizer(params, 0.0, clip, landing=[layer]).step()
        for name, p in layer.named_parameters():
            if name in expected:
                # relative error: the correct code is ~1e-7; landing before clipping gives ~2e-3
                error = (p.grad - expected[name]).abs().max() / expected[name].abs().max()
                assert error < 1e-5, (cls.__name__, clip, name, float(error))
