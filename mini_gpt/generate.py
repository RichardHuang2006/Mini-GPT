"""Autoregressive generation: greedy, temperature, and top-k sampling.

No KV cache -- the full sequence is re-run each step, keeping the loop obvious.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
from torch import nn

from mini_gpt.tokenizer import MiniTokenizer


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    seed: int | None = None,
    context: int | None = None,
) -> torch.Tensor:
    """Extend idx [B, T0] by up to max_new_tokens tokens; returns [B, T0 + n].

    temperature == 0 is greedy and exactly reproducible; > 0 samples from the
    tempered softmax, reproducibly when `seed` is given. A row emitting eos_id
    is frozen, and the loop exits once every row has finished.

    Only the last `context` tokens reach the forward pass, so generation can
    run past the trained window. Defaults to model.cfg.context.
    """
    model.eval()
    device = idx.device
    if context is None:
        context = getattr(getattr(model, "cfg", None), "context", None)

    gen = None
    if temperature > 0 and seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx if context is None or idx.shape[1] <= context else idx[:, -context:]
        logits, _ = model(idx_cond)              # [B, T, V]
        next_logits = logits[:, -1, :].float()   # [B, V]: last position only

        if temperature <= 0.0:
            next_tok = next_logits.argmax(dim=-1)  # greedy: [B]
        else:
            next_logits = next_logits / temperature
            if top_k is not None:
                # Keep only the k largest logits; everything else -> -inf.
                k = min(top_k, next_logits.shape[-1])
                kth = next_logits.topk(k, dim=-1).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

        if eos_id is not None:
            # A finished row keeps emitting eos, freezing its content.
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)

        idx = torch.cat([idx, next_tok[:, None]], dim=1)

        if eos_id is not None:
            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break

    return idx


@torch.no_grad()
def generate_reply(
    model: nn.Module,
    tokenizer: MiniTokenizer,
    messages: Sequence[dict],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Prompt through the chat template; returns the decoded new tokens,
    truncated at the first <|eos|>."""
    # Lazy: posttrain.py imports this module for GRPO rollouts, so a top-level
    # import would be circular.
    from mini_gpt.posttrain import build_prompt

    prompt = build_prompt(messages, tokenizer)
    idx = torch.tensor([prompt], dtype=torch.long, device=next(model.parameters()).device)
    out = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
        seed=seed,
    )
    new_tokens = out[0, len(prompt):].tolist()
    if tokenizer.eos_id in new_tokens:
        new_tokens = new_tokens[: new_tokens.index(tokenizer.eos_id)]
    return tokenizer.decode(new_tokens)


# --- CLI ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from mini_gpt.config import get_config
    from mini_gpt.model import MiniGPT
    from mini_gpt.train import load_model_weights, seed_everything

    ap = argparse.ArgumentParser(description="Generate text from a Mini-GPT checkpoint.")
    ap.add_argument("--ckpt", required=True, help="checkpoint from train.py / posttrain.py")
    ap.add_argument("--tokenizer", required=True, help="trained tokenizer json")
    ap.add_argument("--tier", default="mini", help="config the checkpoint was trained with")
    ap.add_argument("--prompt", default=None, help="raw text prompt")
    ap.add_argument("--chat", default=None, help="a user message, rendered via the chat template")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    if (args.prompt is None) == (args.chat is None):
        ap.error("give exactly one of --prompt (raw text) or --chat (user message)")

    cfg = get_config(args.tier)
    seed_everything(cfg.seed)
    tok = MiniTokenizer.load(args.tokenizer)
    model = MiniGPT(cfg).to(args.device)
    load_model_weights(model, args.ckpt, device=args.device)

    if args.chat is not None:
        text = generate_reply(
            model, tok, [{"role": "user", "content": args.chat}],
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, seed=args.seed,
        )
        print(text)
        return 0

    idx = torch.tensor([tok.encode(args.prompt, add_bos=True)], dtype=torch.long,
                       device=args.device)
    out = generate(
        model, idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, eos_id=tok.eos_id, seed=args.seed,
    )
    print(tok.decode(out[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
