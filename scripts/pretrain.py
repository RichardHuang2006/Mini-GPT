"""Pretraining entry point.

Builds the model, optimizer, schedule, data sampler, and trainer from a tier
preset and a packed shard directory, then runs the loop with periodic
checkpoints and validation-loss logging.

    python scripts/pretrain.py --tier nano --data data/packed --out out/nano
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/pretrain.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import get_config  # noqa: E402
from mini_gpt.data.anneal import AnnealDataStream, anneal_switch_step  # noqa: E402
from mini_gpt.data.sampler import ShardSampler  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.train.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from mini_gpt.train.loop import DataStream, Trainer, maybe_compile  # noqa: E402
from mini_gpt.train.optim import build_optimizers  # noqa: E402
from mini_gpt.train.schedule import build_scheduler  # noqa: E402


def build_training(cfg, data_dir: str, device: str, *, anneal_dir: str | None = None, steps: int | None = None):
    """Assemble (trainer, train_data, val_data) for a tier + packed data dir.

    When ``anneal_dir`` is given, ``train_data`` is an ``AnnealDataStream`` that
    switches to the anneal mix for the last ``cfg.anneal_frac`` of ``steps``.
    """
    raw_model = MiniGPT(cfg).to(device)
    # Fused chunked-CE loss when kernels are on: the batch-size unlock.
    raw_model.fused_loss = getattr(cfg, "use_triton", False)
    model = maybe_compile(raw_model, cfg)
    optimizer = build_optimizers(raw_model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    train_sampler = ShardSampler(data_dir, context=cfg.context, split="train", seed=cfg.seed)
    train_data: DataStream | AnnealDataStream = DataStream(
        train_sampler, cfg.micro_batch, device=device
    )

    if anneal_dir is not None:
        total = steps if steps is not None else cfg.max_steps
        anneal_sampler = ShardSampler(
            anneal_dir, context=cfg.context, split="train", seed=cfg.seed + 1
        )
        anneal_data = DataStream(anneal_sampler, cfg.micro_batch, device=device)
        train_data = AnnealDataStream(
            train_data,
            anneal_data,
            switch_step=anneal_switch_step(total, cfg.anneal_frac),
            grad_accum=cfg.grad_accum,
        )

    trainer = Trainer(
        model, optimizer, scheduler, train_data.batch, cfg, device=device, raw_model=raw_model
    )

    val_data = None
    try:
        val_sampler = ShardSampler(data_dir, context=cfg.context, split="val", seed=cfg.seed)
        val_data = DataStream(val_sampler, cfg.micro_batch, device=device)
    except ValueError:
        pass  # no val shard (tiny corpus)
    return trainer, train_data, val_data


@torch.no_grad()
def evaluate(trainer: Trainer, val_data: DataStream, iters: int = 20) -> float:
    trainer.model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = val_data.batch()
        _, loss = trainer.model(x, y)
        total += loss.item()
    trainer.model.train()
    return total / iters


def eval_perplexity(trainer: Trainer, cfg, data_dir: str, device: str, n_windows: int) -> float:
    """Held-out perplexity via the eval harness.

    Scored on the ``val`` split -- shards the sampler keeps disjoint from training
    -- so this is the same number ``scripts/eval.py`` reports, logged in-loop.
    """
    from mini_gpt.eval.harness import perplexity_from_split

    ppl = perplexity_from_split(
        trainer.model, data_dir, context=cfg.context, device=device,
        n_windows=n_windows, seed=cfg.seed, split="val",
    )
    trainer.model.train()
    return ppl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pretrain Mini-GPT.")
    ap.add_argument("--tier", default="mini")
    ap.add_argument("--data", required=True, help="packed shard directory (with manifest.json)")
    ap.add_argument("--anneal-data", default=None, help="anneal-mix dir; switches in for the last anneal_frac")
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--steps", type=int, default=None, help="override cfg.max_steps")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=0, help="log held-out perplexity every N steps (0=off)")
    ap.add_argument("--eval-windows", type=int, default=128, help="val windows per perplexity eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = get_config(args.tier)
    seed_everything(cfg.seed)
    steps = args.steps if args.steps is not None else cfg.max_steps

    trainer, train_data, val_data = build_training(
        cfg, args.data, args.device, anneal_dir=args.anneal_data, steps=steps
    )
    out = Path(args.out)

    if args.resume:
        load_checkpoint(args.resume, trainer, train_data)
        print(f"resumed from {args.resume} at step {trainer.step}")

    is_anneal = isinstance(train_data, AnnealDataStream)
    announced = train_data.switched if is_anneal else True
    while trainer.step < steps:
        loss = trainer.train_step()
        if is_anneal and train_data.switched and not announced:
            announced = True
            print(f"step {trainer.step:>7} anneal mix ON (last {cfg.anneal_frac:.0%} of budget)")
        if trainer.step % args.log_every == 0:
            lr = trainer.scheduler.get_last_lr()[0]
            print(f"step {trainer.step:>7} loss {loss:.4f} lr {lr:.2e}")
        if args.eval_every and val_data is not None and trainer.step % args.eval_every == 0:
            ppl = eval_perplexity(trainer, cfg, args.data, args.device, args.eval_windows)
            print(f"step {trainer.step:>7} val ppl {ppl:.3f}")
        if trainer.step % args.ckpt_every == 0:
            save_checkpoint(out / f"ckpt_{trainer.step}.pt", trainer, train_data)

    if val_data is not None:
        print(f"final val loss {evaluate(trainer, val_data):.4f}")
    save_checkpoint(out / "ckpt_final.pt", trainer, train_data)
    print(f"done: {trainer.step} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
