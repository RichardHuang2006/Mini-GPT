"""Training loop: micro-batching, accumulation, autocast.

Each optimizer step accumulates gradients over ``grad_accum`` micro-batches, so
the global batch is independent of the memory-bound micro-batch size. bf16
autocast on CUDA; CPU runs in float32, which is deterministic for tests.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from mini_gpt.data.sampler import ShardSampler

BatchFn = Callable[[], tuple[torch.Tensor, torch.Tensor]]

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class DataStream:
    """Adapts a ``ShardSampler`` into ``(inputs, targets)`` torch batches.

    A drawn window ``[B, context+1]`` becomes ``x = w[:, :-1]``, ``y = w[:, 1:]``.
    Owns the sampler so its position is part of the checkpoint.
    """

    def __init__(self, sampler: ShardSampler, batch_size: int, device: torch.device | str = "cpu"):
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = torch.device(device)

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.sampler.next_batch(self.batch_size)  # [B, ctx+1] int64
        t = torch.from_numpy(np.ascontiguousarray(window)).to(self.device)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def state_dict(self) -> dict[str, Any]:
        return self.sampler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.sampler.load_state_dict(state)


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def maybe_compile(model: nn.Module, cfg):
    """Return ``torch.compile(model)`` when enabled, else the model unchanged.

    Compilation is lazy (on the first forward), so a backend failure surfaces at
    call time; callers that must tolerate a missing compiler should guard the
    first step.
    """
    if getattr(cfg, "compile", False):
        return torch.compile(model)
    return model


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        get_batch: BatchFn,
        cfg,
        *,
        device: torch.device | str = "cpu",
        grad_accum: int | None = None,
        raw_model: nn.Module | None = None,
    ):
        self.model = model
        # Uncompiled module for parameters()/state_dict(), so a compiled `model`
        # does not leak an `_orig_mod.` prefix into checkpoints.
        self.raw_model = raw_model if raw_model is not None else model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.get_batch = get_batch
        self.cfg = cfg
        self.device = torch.device(device)
        self.grad_accum = grad_accum if grad_accum is not None else cfg.grad_accum
        self.amp_dtype = _DTYPES.get(cfg.dtype, torch.float32)
        self.step = 0

    def train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(self.grad_accum):
            x, y = self.get_batch()
            with _autocast_ctx(self.device, self.amp_dtype):
                _, loss = self.model(x, y)
            (loss / self.grad_accum).backward()
            total += loss.item()
        if self.cfg.grad_clip:
            nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return total / self.grad_accum

    def train(self, num_steps: int) -> list[float]:
        return [self.train_step() for _ in range(num_steps)]

    # --------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
