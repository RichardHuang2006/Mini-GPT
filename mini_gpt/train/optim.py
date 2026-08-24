"""Optimizers and parameter groups.

Parameters are partitioned by tensor shape into two groups:

* ``hidden`` -- 2D hidden matrices (attention and MLP projections). Weight decay
  applies here, and this group is what moves to Muon.
* ``misc``   -- embeddings, RMSNorm gains, and any 1D parameter. No weight decay;
  always on AdamW.

The tied embedding / output weight is one parameter, counted once, and lands in
``misc``.

Muon conditions the momentum update for matrix-shaped weights by orthogonalizing
it with a fixed number of Newton-Schulz iterations (matmuls only, no SVD), then
applies it scaled by a shape-dependent factor.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

HIDDEN = "hidden"
MISC = "misc"


def classify_parameters(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Split model parameters into the ``hidden`` and ``misc`` groups.

    Deduplicates by identity so a tied weight is assigned exactly once.
    """
    groups: dict[str, list[nn.Parameter]] = {HIDDEN: [], MISC: []}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_embedding = "embed" in name
        if p.ndim >= 2 and not is_embedding:
            groups[HIDDEN].append(p)
        else:
            groups[MISC].append(p)
    return groups


def build_param_groups(model: nn.Module, cfg) -> list[dict]:
    """AdamW-style param groups with per-group peak LR and weight decay."""
    groups = classify_parameters(model)
    return [
        {
            "params": groups[HIDDEN],
            "weight_decay": cfg.weight_decay,
            "lr": cfg.lr_adamw,
            "name": HIDDEN,
        },
        {
            "params": groups[MISC],
            "weight_decay": 0.0,
            "lr": cfg.lr_adamw,
            "name": MISC,
        },
    ]


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """A single AdamW over both groups (the pure-AdamW A/B baseline)."""
    return torch.optim.AdamW(build_param_groups(model, cfg), betas=cfg.adam_betas)


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize a 2D matrix via a quintic Newton-Schulz iteration.

    Pushes the singular values of ``grad`` toward 1 using matmuls only. The
    quintic coefficients are Keller Jordan's tuned constants; five steps bring a
    well-conditioned matrix's singular values into roughly [0.7, 1.13].
    """
    assert grad.ndim == 2, "Newton-Schulz expects a 2D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    # Operate on the smaller dimension by transposing tall matrices.
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transpose:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Muon: orthogonalized-momentum optimizer for 2D hidden matrices."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                if update.ndim == 2:
                    ortho = zeropower_via_newtonschulz5(update, steps=ns_steps).type_as(p)
                    # Shape-scaled step: keeps the update RMS comparable across
                    # matrix aspect ratios.
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(ortho, alpha=-lr * scale)
                else:
                    # Fallback for any non-matrix param routed here.
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
# Combined optimizer (Muon on hidden + AdamW on misc)
# ---------------------------------------------------------------------------

class CombinedOptimizer:
    """Drives several optimizers as one, exposing the API the trainer needs.

    ``param_groups`` concatenates the underlying optimizers' group dicts by
    reference, so a scheduler mutating ``group['lr']`` reaches the real
    optimizers.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict]:
        return [g for opt in self.optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for opt, sub in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(sub)


def build_optimizers(model: nn.Module, cfg):
    """Build the optimizer(s) for a run.

    ``cfg.use_muon`` True  -> Muon(hidden) + AdamW(misc), wrapped as one.
    ``cfg.use_muon`` False -> a single AdamW over both groups (the A/B baseline).
    """
    groups = classify_parameters(model)
    adamw = torch.optim.AdamW(
        [{"params": groups[MISC], "weight_decay": 0.0, "lr": cfg.lr_adamw, "name": MISC}],
        betas=cfg.adam_betas,
    )
    if not cfg.use_muon:
        adamw.add_param_group(
            {"params": groups[HIDDEN], "weight_decay": cfg.weight_decay, "lr": cfg.lr_adamw, "name": HIDDEN}
        )
        return adamw

    muon = Muon(
        [{"params": groups[HIDDEN], "lr": cfg.lr_muon, "weight_decay": cfg.weight_decay, "name": HIDDEN}],
        momentum=0.95,
        ns_steps=5,
    )
    return CombinedOptimizer([muon, adamw])
