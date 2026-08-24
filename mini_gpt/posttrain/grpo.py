"""GRPO: group-relative policy optimization.

For each prompt the policy samples a group of ``G`` completions, each scored by a
reward function. The advantage is the reward normalized within its group, using
the group mean as the baseline instead of a learned critic, which is what makes
GRPO cheap enough for a single card. The policy update is the clipped PPO
surrogate over completion tokens only.

Degenerate case: a zero-variance group (all rewards equal) has zero advantage and
so produces no gradient, pushing the model neither toward nor away from
equally-good samples.

Sampling reuses ``mini_gpt.generate`` verbatim, so a GRPO rollout is formatted
exactly like SFT training data and there is only one sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from mini_gpt.chat_template import Message, build_prompt
from mini_gpt.generate import generate
from mini_gpt.tokenizer import MiniTokenizer

# A reward function scores one completion given its decoded text and metadata.
RewardFn = Callable[["Completion"], float]


@dataclass
class Completion:
    group: int          # which prompt-group this completion belongs to
    prompt_len: int
    tokens: list[int]   # full sequence: prompt tokens + completion (eos kept)
    text: str           # decoded completion text (new tokens only, eos stripped)
    terminated: bool
    n_new: int


# ----------------------------------------------------------------- sampling

@torch.no_grad()
def sample_groups(
    model: nn.Module,
    tokenizer: MiniTokenizer,
    prompts: Sequence[Sequence[Message | dict]],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> list[Completion]:
    """Sample ``group_size`` completions for each prompt via ``generate``.

    The same ``seed`` reproduces the same completions; each prompt uses a distinct
    sub-seed so groups differ.
    """
    eos = tokenizer.eos_id
    out: list[Completion] = []
    for gi, msgs in enumerate(prompts):
        prompt = build_prompt(msgs, tokenizer)
        idx = torch.tensor([prompt] * group_size, dtype=torch.long, device=device)
        gen = generate(
            model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos,
            seed=seed + gi,
        )
        for row in gen:
            full = row.tolist()
            new = full[len(prompt):]
            terminated = eos in new
            if terminated:
                new = new[: new.index(eos) + 1]  # keep the eos
            text_tokens = new[:-1] if terminated else new
            out.append(
                Completion(
                    group=gi,
                    prompt_len=len(prompt),
                    tokens=prompt + new,
                    text=tokenizer.decode(text_tokens),
                    terminated=terminated,
                    n_new=len(new),
                )
            )
    return out


def collate(completions: list[Completion], pad_id: int, device: torch.device | str = "cpu"):
    """Pad completions to ``[B, L]`` and build the completion-token target mask.

    Returns ``(seqs [B, L], comp_mask [B, L], groups [B])`` where ``comp_mask`` is
    ``True`` on positions that are generated completion tokens (prompt and pad are
    ``False``), so the policy loss touches only the tokens the model produced.
    """
    b = len(completions)
    L = max(len(c.tokens) for c in completions)
    seqs = torch.full((b, L), pad_id, dtype=torch.long)
    comp_mask = torch.zeros((b, L), dtype=torch.bool)
    groups = torch.empty(b, dtype=torch.long)
    for i, c in enumerate(completions):
        t = torch.tensor(c.tokens, dtype=torch.long)
        seqs[i, : len(t)] = t
        comp_mask[i, c.prompt_len : len(t)] = True  # generated tokens only
        groups[i] = c.group
    return seqs.to(device), comp_mask.to(device), groups.to(device)


# ------------------------------------------------------- advantage + update

def group_advantages(rewards: torch.Tensor, groups: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Reward normalized within each group; zero for a zero-variance group.

    ``A_i = (r_i - mean_g) / (std_g + eps)`` over the group ``g`` of completion
    ``i``. A group with all-equal rewards has ``std_g == 0`` and a mean-centred
    numerator of ``0``, so every advantage is exactly ``0`` and no update occurs.
    """
    groups = groups.to(rewards.device)  # rewards is built on CPU; groups may be on GPU
    adv = torch.zeros_like(rewards, dtype=torch.float32)
    for g in groups.unique():
        m = groups == g
        r = rewards[m].float()
        std = r.std(unbiased=False)
        centered = r - r.mean()
        adv[m] = centered / (std + eps) if std > 0 else torch.zeros_like(centered)
    return adv


def token_logprobs(model: nn.Module, seqs: torch.Tensor, target_mask: torch.Tensor):
    """Per-token log-probs of ``seqs`` under ``model`` and the shifted token mask.

    Returns ``(logp [B, L-1], mask [B, L-1])`` where ``logp[:, t]`` is the log-prob
    the model assigns to the actual token ``seqs[:, t+1]`` and ``mask`` marks which
    of those targets are completion tokens.
    """
    logits, _ = model(seqs)  # targets=None -> full logits (fused_loss path is bypassed)
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = seqs[:, 1:]
    tok_logp = logp.gather(-1, tgt[:, :, None]).squeeze(-1)
    return tok_logp, target_mask[:, 1:]


def grpo_loss(
    model: nn.Module,
    seqs: torch.Tensor,
    comp_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_logp: torch.Tensor,
    *,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO surrogate on completion tokens, weighted by group advantage."""
    new_logp, m = token_logprobs(model, seqs, comp_mask)
    ratio = torch.exp(new_logp - old_logp)
    adv = advantages[:, None]
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    m = m.float()
    return -(surr * m).sum() / m.sum().clamp(min=1.0)


def grpo_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    seqs: torch.Tensor,
    comp_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    grad_clip: float = 1.0,
) -> float:
    """One GRPO optimizer step. Returns the surrogate loss value."""
    with torch.no_grad():
        old_logp, _ = token_logprobs(model, seqs, comp_mask)
        old_logp = old_logp.detach()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = grpo_loss(model, seqs, comp_mask, advantages, old_logp, clip_eps=clip_eps)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach())


# ------------------------------------------------------------------- loop

def run_grpo(
    model: nn.Module,
    tokenizer: MiniTokenizer,
    prompt_bank: Sequence[tuple[Sequence[Message | dict], RewardFn]],
    *,
    steps: int,
    group_size: int = 8,
    prompts_per_step: int = 4,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    top_k: int | None = 40,
    lr: float = 1e-4,
    clip_eps: float = 0.2,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Run GRPO and return the mean reward per step.

    ``prompt_bank`` is a list of ``(messages, reward_fn)`` pairs; each step samples
    ``prompts_per_step`` of them, rolls out ``group_size`` completions each, scores
    them, computes group-relative advantages, and takes one clipped update.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    pad_id = tokenizer.pad_id
    mean_rewards: list[float] = []
    n = len(prompt_bank)
    cursor = 0

    for step in range(steps):
        batch = [prompt_bank[(cursor + j) % n] for j in range(prompts_per_step)]
        cursor += prompts_per_step
        prompts = [p for p, _ in batch]
        reward_fns = [fn for _, fn in batch]

        comps = sample_groups(
            model,
            tokenizer,
            prompts,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed + step * 1000,
            device=device,
        )
        rewards = torch.tensor([reward_fns[c.group](c) for c in comps], dtype=torch.float32)
        seqs, comp_mask, groups = collate(comps, pad_id, device=device)
        advantages = group_advantages(rewards, groups).to(device)
        grpo_step(
            model, optimizer, seqs, comp_mask, advantages, clip_eps=clip_eps, grad_clip=grad_clip
        )
        mean_rewards.append(float(rewards.mean()))

    return mean_rewards
