"""Context extension 1024 -> 2048.

A short continued-training phase at a longer context with a rescaled RoPE base.
No weight is tied to a context length -- RoPE carries position at runtime and
there is no learned positional table -- so a base checkpoint loads directly into a
longer-context config. The only changes are ``context`` and the NTK-rescaled
``rope_base``, which the continued training adapts to.

    python scripts/extend_context.py --tier mini --data data/packed_2k \
        --init out/mini/ckpt_final.pt --out out/mini_2k --new-context 2048 --steps 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import Config, get_config  # noqa: E402
from mini_gpt.data.sampler import ShardSampler  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.model.rope import scale_rope_base  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.train.checkpoint import save_checkpoint  # noqa: E402
from mini_gpt.train.loop import DataStream, Trainer, maybe_compile  # noqa: E402
from mini_gpt.train.optim import build_optimizers  # noqa: E402
from mini_gpt.train.schedule import build_scheduler  # noqa: E402


def extended_config(cfg: Config, new_context: int) -> Config:
    """A copy of ``cfg`` at ``new_context`` with an NTK-rescaled RoPE base."""
    new_base = scale_rope_base(cfg.rope_base, cfg.context, new_context, cfg.head_dim)
    # The window is unchanged, so windowed layers keep bounded attention cost at
    # the longer context while full layers pay the quadratic price.
    return cfg.with_overrides(context=new_context, rope_base=new_base)


def load_base_weights(model: MiniGPT, init_path: str, device: str) -> None:
    """Load weights from a pretrain checkpoint; optimizer state starts fresh."""
    payload = torch.load(init_path, map_location=device, weights_only=False)
    state = payload["trainer"]["model"] if "trainer" in payload else payload["model"]
    model.load_state_dict(state)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extend a base model's context length.")
    ap.add_argument("--tier", default="mini")
    ap.add_argument("--data", required=True, help="packed dir with shards long enough for new-context")
    ap.add_argument("--init", required=True, help="base checkpoint to initialize from")
    ap.add_argument("--out", default="out/extended")
    ap.add_argument("--new-context", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    base_cfg = get_config(args.tier)
    cfg = extended_config(base_cfg, args.new_context)
    seed_everything(cfg.seed)
    print(
        f"extend: context {base_cfg.context} -> {cfg.context}, "
        f"rope_base {base_cfg.rope_base:.0f} -> {cfg.rope_base:.0f}"
    )

    raw_model = MiniGPT(cfg).to(args.device)
    load_base_weights(raw_model, args.init, args.device)
    raw_model.fused_loss = getattr(cfg, "use_triton", False)
    model = maybe_compile(raw_model, cfg)

    optimizer = build_optimizers(raw_model, cfg)
    scheduler = build_scheduler(optimizer, cfg.with_overrides(max_steps=args.steps, warmup_steps=min(cfg.warmup_steps, args.steps // 10 + 1)))

    sampler = ShardSampler(args.data, context=cfg.context, split="train", seed=cfg.seed)
    data = DataStream(sampler, cfg.micro_batch, device=args.device)
    trainer = Trainer(model, optimizer, scheduler, data.batch, cfg, device=args.device, raw_model=raw_model)

    while trainer.step < args.steps:
        loss = trainer.train_step()
        if trainer.step % args.log_every == 0:
            print(f"step {trainer.step:>7} loss {loss:.4f}")

    save_checkpoint(Path(args.out) / "ckpt_extended.pt", trainer, data)
    print(f"done: extended to context {cfg.context} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
