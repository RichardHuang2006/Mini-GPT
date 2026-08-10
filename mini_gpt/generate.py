"""Autoregressive generation.

Greedy and temperature sampling, shared by the SFT smoke test, the evaluation
harness, and GRPO rollouts -- one generator so sampled completions are formatted
exactly like training data. Generation is a plain re-run of the forward pass on
the growing sequence (no KV cache yet; a paged-KV server is future scope);
correctness and determinism matter here, speed does not.

Determinism: with ``temperature == 0`` sampling is greedy (argmax) and exactly
reproducible; with ``temperature > 0`` a per-call ``torch.Generator`` seeded by
``seed`` makes the draw reproducible too. Each sequence stops contributing new
content once it emits ``<|eos|>``; the whole batch halts when all sequences have.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from mini_gpt.chat_template import Message, build_prompt
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
) -> torch.Tensor:
    """Extend ``idx`` ``[B, T0]`` by up to ``max_new_tokens`` tokens.

    Returns the full ``[B, T0 + n]`` sequence (prompt included). Rows that emit
    ``eos_id`` are padded with ``eos_id`` thereafter and the loop exits early once
    every row has finished.
    """
    model.eval()
    device = idx.device
    was_compiled = hasattr(model, "_orig_mod")

    gen = None
    if temperature > 0 and seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        next_logits = logits[:, -1, :].float()

        if temperature <= 0.0:
            next_tok = next_logits.argmax(dim=-1)
        else:
            next_logits = next_logits / temperature
            if top_k is not None:
                k = min(top_k, next_logits.shape[-1])
                kth = next_logits.topk(k, dim=-1).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

        if eos_id is not None:
            # A finished row keeps emitting eos so its content is frozen.
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)

        idx = torch.cat([idx, next_tok[:, None]], dim=1)

        if eos_id is not None:
            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break

    _ = was_compiled  # (kept for parity; compiled modules generate the same way)
    return idx


@torch.no_grad()
def generate_reply(
    model: nn.Module,
    tokenizer: MiniTokenizer,
    messages: Sequence[Message | dict],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Prompt the model through the chat template and decode the assistant reply.

    Uses ``build_prompt`` (the single source of the chat format), so a sampled
    reply is formatted identically to SFT training data. Returns the decoded text
    of the new tokens, stopped at the first ``<|eos|>``.
    """
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
