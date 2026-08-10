"""GRPO entry point.

Loads an SFT (instruct) checkpoint and runs GRPO on the countdown / arithmetic
task, logging the mean reward per step -- the curve whose *rise* is the milestone
that proves the RL loop works. The GSM8K reward is available through the same
interface but is honestly flat at `mini`.

    python scripts/grpo.py --tier mini --tokenizer out/tok.json \
        --init out/mini_sft/ckpt_sft.pt --out out/mini_grpo --steps 500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import get_config  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.posttrain.grpo import Completion, run_grpo  # noqa: E402
from mini_gpt.posttrain.rewards import arithmetic_reward  # noqa: E402
from mini_gpt.tokenizer import MiniTokenizer  # noqa: E402


def build_arithmetic_bank(n: int, *, max_new_tokens: int, seed: int = 0):
    """A prompt bank of ``(messages, reward_fn)`` for simple arithmetic targets."""
    rng = random.Random(seed)
    bank = []
    for _ in range(n):
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        target = a + b
        messages = [{"role": "user", "content": f"What is {a} + {b}?"}]

        def reward_fn(c: Completion, target=target) -> float:
            return arithmetic_reward(
                c.text,
                target=target,
                terminated=c.terminated,
                n_new_tokens=c.n_new,
                max_new_tokens=max_new_tokens,
            ).total

        bank.append((messages, reward_fn))
    return bank


def load_base_weights(model: MiniGPT, init_path: str, device: str) -> None:
    payload = torch.load(init_path, map_location=device, weights_only=False)
    state = payload["trainer"]["model"] if "trainer" in payload else payload["model"]
    model.load_state_dict(state)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GRPO on countdown/arithmetic.")
    ap.add_argument("--tier", default="mini")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--init", default=None, help="SFT checkpoint (else random init)")
    ap.add_argument("--out", default="out/grpo")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = get_config(args.tier)
    seed_everything(cfg.seed)
    tok = MiniTokenizer.load(args.tokenizer)

    model = MiniGPT(cfg).to(args.device)
    if args.init:
        load_base_weights(model, args.init, args.device)

    bank = build_arithmetic_bank(256, max_new_tokens=args.max_new_tokens, seed=cfg.seed)
    rewards = run_grpo(
        model,
        tok,
        bank,
        steps=args.steps,
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        max_new_tokens=args.max_new_tokens,
        lr=args.lr,
        seed=cfg.seed,
        device=args.device,
    )
    first = sum(rewards[:10]) / min(10, len(rewards))
    last = sum(rewards[-10:]) / min(10, len(rewards))
    print(f"GRPO done: mean reward {first:.3f} -> {last:.3f} over {len(rewards)} steps")

    torch.save({"model": model.state_dict(), "rewards": rewards}, str(Path(args.out) / "ckpt_grpo.pt"))
    print(f"wrote {args.out}/ckpt_grpo.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
