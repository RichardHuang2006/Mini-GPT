"""Supervised fine-tuning entry point.

Loads a pretrained (base or annealed) checkpoint, continues training on
chat-formatted conversations with an assistant-only masked loss and packed,
attention-isolated conversations, and writes an instruct checkpoint.

    python scripts/sft.py --tier mini --tokenizer out/tok.json \
        --init out/mini/ckpt_final.pt --data data/chat.jsonl --out out/mini_sft --steps 2000

Chat data is a JSONL file of ``{"messages": [{"role","content"}, ...]}``;
``--data synthetic`` uses the offline arithmetic conversations instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import get_config  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.generate import generate_reply  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.posttrain.sft import (  # noqa: E402
    pack_conversations,
    synthetic_sft_conversations,
    train_sft,
)
from mini_gpt.tokenizer import MiniTokenizer  # noqa: E402
from mini_gpt.train.checkpoint import save_checkpoint  # noqa: E402
from mini_gpt.train.loop import Trainer, maybe_compile  # noqa: E402
from mini_gpt.train.optim import build_optimizers  # noqa: E402
from mini_gpt.train.schedule import build_scheduler  # noqa: E402


def load_conversations(path: str) -> list[list[dict]]:
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line)["messages"])
    return convs


def load_base_weights(model: MiniGPT, init_path: str, device: str) -> None:
    payload = torch.load(init_path, map_location=device, weights_only=False)
    state = payload["trainer"]["model"] if "trainer" in payload else payload["model"]
    model.load_state_dict(state)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Supervised fine-tune a base checkpoint.")
    ap.add_argument("--tier", default="mini")
    ap.add_argument("--tokenizer", required=True, help="trained tokenizer json")
    ap.add_argument("--init", default=None, help="base checkpoint (else random init)")
    ap.add_argument("--data", default="synthetic", help="chat JSONL path, or 'synthetic'")
    ap.add_argument("--out", default="out/sft")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = get_config(args.tier)
    seed_everything(cfg.seed)
    tok = MiniTokenizer.load(args.tokenizer)

    raw_model = MiniGPT(cfg).to(args.device)
    if args.init:
        load_base_weights(raw_model, args.init, args.device)
    raw_model.fused_loss = getattr(cfg, "use_triton", False)
    model = maybe_compile(raw_model, cfg)

    convs = (
        synthetic_sft_conversations(5000, seed=cfg.seed)
        if args.data == "synthetic"
        else load_conversations(args.data)
    )
    packed = pack_conversations(convs, tok, seq_len=cfg.context)
    print(f"SFT: {len(convs)} conversations -> {len(packed)} packed rows of {packed.seq_len}")

    losses = train_sft(model, packed, cfg, device=args.device, steps=args.steps, raw_model=raw_model)
    print(f"SFT done: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    reply = generate_reply(model, tok, [{"role": "user", "content": "What is 2 + 3?"}], max_new_tokens=16)
    print(f"sample reply: {reply!r}")

    # Reuse the Trainer checkpoint format so the SFT model loads like any other.
    opt = build_optimizers(raw_model, cfg)
    sched = build_scheduler(opt, cfg)
    trainer = Trainer(model, opt, sched, lambda: None, cfg, device=args.device, raw_model=raw_model)
    save_checkpoint(Path(args.out) / "ckpt_sft.pt", trainer)
    print(f"wrote {args.out}/ckpt_sft.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
