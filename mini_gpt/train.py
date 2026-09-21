"""Pretraining: AdamW + Muon, the LR schedule, the loop, and checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mini_gpt.config import Config, get_config
from mini_gpt.data import ShardSampler
from mini_gpt.model import MiniGPT

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def seed_everything(seed: int = 0, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and Torch, optionally forcing deterministic algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this set before first use for reproducible matmuls.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few ops have no deterministic implementation, and a
        # warning beats crashing the run.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def classify_parameters(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Split parameters into Muon-eligible 2D matrices ("hidden") and the rest.

    Deduplicated by identity, so the tied embedding weight is assigned once.
    """
    groups: dict[str, list[nn.Parameter]] = {"hidden": [], "misc": []}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_embedding = "embed" in name
        if p.ndim >= 2 and not is_embedding:
            groups["hidden"].append(p)
        else:
            groups["misc"].append(p)
    return groups


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize a 2D matrix with a quintic Newton-Schulz iteration."""
    assert grad.ndim == 2, "Newton-Schulz expects a 2D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315  # Keller Jordan's tuned quintic constants
    x = grad.float()
    transpose = x.size(0) > x.size(1)  # operate on the smaller dimension
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
    """SGD-momentum whose update is orthogonalized by Newton-Schulz."""

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
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                if update.ndim == 2:
                    ortho = zeropower_via_newtonschulz5(update, steps=group["ns_steps"]).type_as(p)
                    # Keeps the update RMS comparable across matrix aspect ratios.
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.add_(ortho, alpha=-lr * scale)
                else:  # fallback for any non-matrix param routed here
                    p.add_(update, alpha=-lr)
        return loss


def build_optimizers(model: nn.Module, cfg: Config) -> list[torch.optim.Optimizer]:
    """[Muon(hidden), AdamW(misc)], or a single AdamW when cfg.use_muon is off."""
    groups = classify_parameters(model)
    adamw = torch.optim.AdamW(
        [{"params": groups["misc"], "weight_decay": 0.0, "lr": cfg.lr_adamw, "name": "misc"}],
        betas=cfg.adam_betas,
    )
    if not cfg.use_muon:
        adamw.add_param_group(
            {"params": groups["hidden"], "weight_decay": cfg.weight_decay,
             "lr": cfg.lr_adamw, "name": "hidden"}
        )
        return [adamw]

    muon = Muon(
        [{"params": groups["hidden"], "lr": cfg.lr_muon,
          "weight_decay": cfg.weight_decay, "name": "hidden"}],
        momentum=0.95,
        ns_steps=5,
    )
    return [muon, adamw]


def lr_multiplier(step: int, warmup: int, max_steps: int, floor_frac: float) -> float:
    """Linear warmup, then cosine decay to floor_frac of the peak."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if step >= max_steps:
        return floor_frac
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor_frac + (1.0 - floor_frac) * cosine


class WarmupCosine:
    """Warmup-then-cosine schedule over a list of optimizers.

    One multiplier scales every group's base LR, so the distinct AdamW and Muon
    peaks keep their ratio across the whole schedule.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer], warmup: int,
                 max_steps: int, floor_frac: float):
        self.optimizers = optimizers
        self.warmup = warmup
        self.max_steps = max_steps
        self.floor_frac = floor_frac
        self.base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
        self.last_step = 0
        self._apply()

    def _apply(self) -> None:
        m = lr_multiplier(self.last_step, self.warmup, self.max_steps, self.floor_frac)
        for opt, bases in zip(self.optimizers, self.base_lrs, strict=True):
            for group, base in zip(opt.param_groups, bases, strict=True):
                group["lr"] = base * m

    def step(self) -> None:
        self.last_step += 1
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for opt in self.optimizers for g in opt.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_step = state["last_step"]
        self.base_lrs = state["base_lrs"]
        self._apply()


class DataStream:
    """Turns sampled [B, context+1] windows into (inputs, next-token targets)."""

    def __init__(self, sampler: ShardSampler, batch_size: int,
                 device: torch.device | str = "cpu"):
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


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    """bf16/fp16 autocast on CUDA; a no-op on CPU (float32 everywhere)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def maybe_compile(model: nn.Module, cfg: Config) -> nn.Module:
    """torch.compile when cfg.compile, else the model unchanged."""
    return torch.compile(model) if cfg.compile else model


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    scheduler: WarmupCosine,
    step: int,
    sampler_state: dict | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save weights, optimizer/scheduler/sampler state and host RNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "scheduler": scheduler.state_dict(),
            "step": step,
            "sampler": sampler_state,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "extra": extra or {},
        },
        str(path),
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer] | None = None,
    scheduler: WarmupCosine | None = None,
) -> dict[str, Any]:
    """Restore state in place; returns the payload (for step / sampler state)."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizers is not None:
        for opt, state in zip(optimizers, payload["optimizers"], strict=True):
            opt.load_state_dict(state)
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    return payload


def load_model_weights(model: nn.Module, path: str | Path, device: str = "cpu") -> None:
    """Load just the weights from a checkpoint (or a bare state dict)."""
    payload = torch.load(str(path), map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)


@torch.no_grad()
def estimate_val_loss(model: nn.Module, val_data: DataStream, iters: int = 20) -> float:
    """Mean next-token cross-entropy over `iters` held-out batches."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = val_data.batch()
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


def train(
    cfg: Config,
    data_dir: str,
    out_dir: str,
    *,
    steps: int | None = None,
    resume: str | None = None,
    device: str = "cpu",
    log_every: int = 50,
    ckpt_every: int = 1000,
    eval_every: int = 0,
    eval_iters: int = 20,
) -> list[float]:
    """Pretrain MiniGPT on packed shards; returns the per-step loss history."""
    seed_everything(cfg.seed)
    max_steps = steps if steps is not None else cfg.max_steps
    out = Path(out_dir)

    train_data = DataStream(
        ShardSampler(data_dir, context=cfg.context, split="train", seed=cfg.seed),
        cfg.micro_batch,
        device=device,
    )
    val_data = None
    with contextlib.suppress(ValueError):  # tiny corpus with no val shard
        val_data = DataStream(
            ShardSampler(data_dir, context=cfg.context, split="val", seed=cfg.seed),
            cfg.micro_batch,
            device=device,
        )

    raw_model = MiniGPT(cfg).to(device)
    raw_model.fused_loss = cfg.use_triton
    model = maybe_compile(raw_model, cfg)  # raw_model keeps clean state_dict keys

    # The schedule always spans cfg.max_steps; `steps` only bounds this run, so
    # an interrupted run and its continuation share one LR trajectory.
    optimizers = build_optimizers(raw_model, cfg)
    scheduler = WarmupCosine(optimizers, cfg.warmup_steps, cfg.max_steps, cfg.lr_floor_frac)
    amp_dtype = DTYPES.get(cfg.dtype, torch.float32)

    step = 0
    if resume:
        payload = load_checkpoint(resume, model=raw_model, optimizers=optimizers,
                                  scheduler=scheduler)
        step = payload["step"]
        if payload.get("sampler"):
            train_data.load_state_dict(payload["sampler"])
        print(f"resumed from {resume} at step {step}")

    losses: list[float] = []
    while step < max_steps:
        model.train()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # grad_accum micro-batches per step, so the global batch size is
        # independent of what fits in memory at once.
        total = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_data.batch()  # [B, T]
            with autocast_ctx(torch.device(device), amp_dtype):
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            total += loss.item()

        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        for opt in optimizers:  # Muon then AdamW
            opt.step()
        scheduler.step()
        step += 1
        losses.append(total / cfg.grad_accum)

        if log_every and step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:>7} loss {losses[-1]:.4f} lr {lr:.2e}")
        if eval_every and val_data is not None and step % eval_every == 0:
            val_loss = estimate_val_loss(model, val_data, iters=eval_iters)
            print(f"step {step:>7} val loss {val_loss:.4f} ppl {math.exp(val_loss):.3f}")
        if ckpt_every and step % ckpt_every == 0:
            save_checkpoint(
                out / f"ckpt_{step}.pt", model=raw_model, optimizers=optimizers,
                scheduler=scheduler, step=step, sampler_state=train_data.state_dict(),
            )

    if val_data is not None:
        val_loss = estimate_val_loss(model, val_data, iters=eval_iters)
        print(f"final val loss {val_loss:.4f} ppl {math.exp(val_loss):.3f}")
    save_checkpoint(
        out / "ckpt_final.pt", model=raw_model, optimizers=optimizers,
        scheduler=scheduler, step=step, sampler_state=train_data.state_dict(),
    )
    print(f"done: {step} steps -> {out}")
    return losses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pretrain Mini-GPT on packed shards.")
    ap.add_argument("--tier", default="mini", help="config preset: nano | mini | small")
    ap.add_argument("--data", required=True, help="packed shard dir (with manifest.json)")
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--steps", type=int, default=None, help="override cfg.max_steps")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--micro-batch", type=int, default=None, help="override cfg.micro_batch")
    ap.add_argument("--grad-accum", type=int, default=None, help="override cfg.grad_accum")
    ap.add_argument("--optimizer", default="muon", choices=["muon", "adamw"],
                    help="muon: Muon on hidden matrices + AdamW elsewhere; "
                    "adamw: the pure-AdamW baseline")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    ap.add_argument("--no-triton", action="store_true", help="disable the fused kernels")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="log held-out val loss + perplexity every N steps (0 = off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    overrides: dict[str, Any] = {"use_muon": args.optimizer == "muon"}
    if args.micro_batch is not None:
        overrides["micro_batch"] = args.micro_batch
    if args.grad_accum is not None:
        overrides["grad_accum"] = args.grad_accum
    if args.no_compile:
        overrides["compile"] = False
    if args.no_triton:
        overrides["use_triton"] = False
    if args.steps is not None:
        # Keep the LR schedule in sync with the run length: cosine decay spans
        # max_steps, so an unsynced --steps would end the run at a high LR.
        overrides["max_steps"] = args.steps
        overrides["warmup_steps"] = min(
            get_config(args.tier).warmup_steps, max(1, args.steps // 10)
        )
    cfg = get_config(args.tier, **overrides)

    train(
        cfg,
        args.data,
        args.out,
        steps=args.steps,
        resume=args.resume,
        device=args.device,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        eval_every=args.eval_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
