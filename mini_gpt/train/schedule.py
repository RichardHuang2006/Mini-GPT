"""Learning-rate schedule.

Linear warmup to the peak, then cosine decay to a floor fraction of the peak.
Implemented as one multiplier applied to every param group's base LR, so
distinct per-group peak LRs (AdamW vs. Muon) keep their ratio through the whole
schedule. This is a small standalone scheduler rather than ``LambdaLR`` so it
also drives the ``CombinedOptimizer`` wrapper (which is not a torch
``Optimizer`` instance).
"""

from __future__ import annotations

import math
from typing import Any


def lr_multiplier(step: int, warmup: int, max_steps: int, floor_frac: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if step >= max_steps:
        return floor_frac
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor_frac + (1.0 - floor_frac) * cosine


class WarmupCosineScheduler:
    """Warmup-then-cosine LR schedule over any object with ``param_groups``."""

    def __init__(self, optimizer, warmup: int, max_steps: int, floor_frac: float):
        self.optimizer = optimizer
        self.warmup = warmup
        self.max_steps = max_steps
        self.floor_frac = floor_frac
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_step = 0
        self._apply()

    def _apply(self) -> None:
        m = lr_multiplier(self.last_step, self.warmup, self.max_steps, self.floor_frac)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * m

    def step(self) -> None:
        self.last_step += 1
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_step = state["last_step"]
        self.base_lrs = state["base_lrs"]
        self._apply()


def build_scheduler(optimizer, cfg) -> WarmupCosineScheduler:
    return WarmupCosineScheduler(optimizer, cfg.warmup_steps, cfg.max_steps, cfg.lr_floor_frac)
