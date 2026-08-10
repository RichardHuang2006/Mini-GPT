"""Throughput / memory / roofline harness.

Measures tokens/s, achieved TFLOPs (and MFU if a device peak is supplied), and
peak VRAM for a tier, and records the largest feasible micro-batch -- the
naive-cross-entropy baseline that chunked CE is meant to beat. Uses random token
data so the benchmark has no data dependency.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import Config, get_config  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.train.optim import build_optimizers  # noqa: E402
from mini_gpt.train.schedule import build_scheduler  # noqa: E402


def flops_per_token(cfg: Config) -> float:
    """First-order training FLOPs/token: 6*N (fwd+bwd) plus the attention term.

    ``6*N`` is the standard parameter-count approximation; the attention score/
    value matmuls add ~12 * n_layers * context * d_model that ``6*N`` misses.
    """
    n = cfg.param_count().non_embedding
    attn = 12 * cfg.n_layers * cfg.context * cfg.d_model
    return 6 * n + attn


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def measure(
    cfg: Config,
    *,
    steps: int = 20,
    warmup: int = 3,
    device: str | None = None,
    micro_batch: int | None = None,
    peak_tflops: float | None = None,
    fused_loss: bool = False,
) -> dict[str, Any]:
    device = device or _default_device()
    mb = micro_batch or cfg.micro_batch
    seed_everything(cfg.seed)

    model = MiniGPT(cfg).to(device)
    # fused_loss=True streams the vocabulary (chunked CE) instead of materializing
    # the [mb*ctx, V] logits -- the memory difference this benchmark exists to show.
    model.fused_loss = fused_loss
    opt = build_optimizers(model, cfg)
    sched = build_scheduler(opt, cfg)

    x = torch.randint(0, cfg.vocab_size, (mb, cfg.context + 1), device=device)
    xin, y = x[:, :-1].contiguous(), x[:, 1:].contiguous()

    def one_step():
        opt.zero_grad(set_to_none=True)
        _, loss = model(xin, y)
        loss.backward()
        opt.step()
        sched.step()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tokens = steps * mb * cfg.context
    tps = tokens / dt
    achieved_tflops = flops_per_token(cfg) * tokens / dt / 1e12
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
    )
    mfu = (achieved_tflops / peak_tflops) if peak_tflops else None

    return {
        "tier": cfg.name,
        "device": device,
        "micro_batch": mb,
        "context": cfg.context,
        "steps": steps,
        "tokens_per_s": tps,
        "achieved_tflops": achieved_tflops,
        "peak_vram_mb": peak_vram_mb,
        "mfu": mfu,
        "fused_loss": fused_loss,
    }


def max_microbatch(
    cfg: Config,
    *,
    device: str | None = None,
    start: int = 1,
    fused_loss: bool = False,
    cap: int | None = None,
) -> int:
    """Largest micro-batch that fits.

    With ``fused_loss=False`` this is the naive-CE baseline; with ``True`` it is
    the chunked-CE path, which should fit a several-fold larger micro-batch.
    On CUDA, doubles until OOM (or ``cap``); on CPU there is
    no hard memory wall to probe, so the configured micro-batch is returned
    unchanged. ``cap`` bounds the probe so a small model with a tiny logits
    tensor does not march up to absurd batch sizes before hitting the wall.
    """
    device = device or _default_device()
    if device != "cuda":
        return cfg.micro_batch
    mb, best = start, start  # pragma: no cover - needs a GPU
    while cap is None or mb <= cap:  # pragma: no cover - needs a GPU
        try:
            measure(
                cfg, steps=1, warmup=0, device=device, micro_batch=mb, fused_loss=fused_loss
            )
            best, mb = mb, mb * 2
        except RuntimeError as e:
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                return best
            raise
    return best  # pragma: no cover - needs a GPU


def format_table(stats: dict[str, Any]) -> str:
    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:,.2f}"
        return str(v)

    rows = [
        ("tier", stats["tier"]),
        ("device", stats["device"]),
        ("loss path", "chunked CE" if stats.get("fused_loss") else "naive CE"),
        ("micro-batch x context", f"{stats['micro_batch']} x {stats['context']}"),
        ("tokens / s", fmt(stats["tokens_per_s"])),
        ("achieved TFLOPs", fmt(stats["achieved_tflops"])),
        ("MFU", fmt(stats["mfu"])),
        ("peak VRAM (MB)", fmt(stats["peak_vram_mb"])),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k.ljust(width)}  {v}" for k, v in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark Mini-GPT throughput/memory.")
    ap.add_argument("--tier", default="nano")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--peak-tflops", type=float, default=None, help="device peak for MFU")
    ap.add_argument("--fused-loss", action="store_true", help="use chunked CE")
    ap.add_argument(
        "--scan-microbatch",
        action="store_true",
        help="probe the largest micro-batch with naive vs chunked CE",
    )
    args = ap.parse_args(argv)

    cfg = get_config(args.tier)

    if args.scan_microbatch:
        naive = max_microbatch(cfg, device=args.device, fused_loss=False, cap=args.micro_batch)
        chunked = max_microbatch(cfg, device=args.device, fused_loss=True, cap=args.micro_batch)
        rise = f"{chunked / naive:.1f}x" if naive else "n/a"
        print(f"bench: {cfg.name} -- largest feasible micro-batch")
        print(f"  naive CE     {naive}")
        print(f"  chunked CE   {chunked}  ({rise} the naive baseline)")
        return 0

    stats = measure(
        cfg,
        steps=args.steps,
        device=args.device,
        micro_batch=args.micro_batch,
        peak_tflops=args.peak_tflops,
        fused_loss=args.fused_loss,
    )
    print(f"bench: {cfg.name}")
    print(format_table(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
