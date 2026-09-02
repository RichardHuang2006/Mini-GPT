"""Post-training: the chat template, supervised fine-tuning, and GRPO.

Two stages turn a next-token model into an assistant -- SFT (assistant-only
loss over packed conversations), then GRPO (group-relative rewards, no learned
critic) on GSM8K or offline arithmetic. Built on tokenizer.py's role tokens,
generate.py's sampling, and train.py's optimizers and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence

import torch
from torch import nn

from mini_gpt.config import Config, get_config
from mini_gpt.generate import generate
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import MiniTokenizer
from mini_gpt.train import (
    WarmupCosine,
    autocast_ctx,
    build_optimizers,
    load_model_weights,
    save_checkpoint,
    seed_everything,
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# --- 1. Chat message representation ------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]

# The token opening each role's turn. A "tool" message carries a tool result;
# an assistant's tool call is inline <|tool_call|> content in its own turn.
ROLE_TOKEN: dict[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool_result|>",
}


@dataclass
class Message:
    role: Role
    content: str


def _coerce(messages: Sequence[Message | dict]) -> list[Message]:
    return [m if isinstance(m, Message) else Message(m["role"], m["content"]) for m in messages]


# --- 2. Conversation rendering + 3. assistant-only loss mask -----------------
# One function maps messages to token IDs, so the SFT loss mask and the
# GRPO/inference prompt format cannot drift apart. Layout, after a leading
# <|bos|>: `<|role|> <content> <|eos|>` per turn. The mask covers an assistant
# turn's content and closing <|eos|> -- never the role header, which is prompt.

@dataclass
class RenderedChat:
    ids: list[int]        # token IDs, len L
    loss_mask: list[int]  # parallel 0/1 mask, 1 only on assistant-authored tokens

    def __post_init__(self) -> None:
        assert len(self.ids) == len(self.loss_mask), "ids and mask must align"


def render_chat(
    messages: Sequence[Message | dict],
    tokenizer: MiniTokenizer,
    *,
    add_generation_prompt: bool = False,
) -> RenderedChat:
    """Render a conversation to token IDs plus the assistant-only loss mask.

    add_generation_prompt appends a content-free trailing <|assistant|> header
    (not part of the loss) to prime a reply, for inference and GRPO rollouts.
    """
    msgs = _coerce(messages)

    ids: list[int] = [tokenizer.bos_id]
    mask: list[int] = [0]

    for msg in msgs:
        if msg.role not in ROLE_TOKEN:
            raise ValueError(f"unknown role {msg.role!r}; expected one of {sorted(ROLE_TOKEN)}")
        role_id = tokenizer.token_to_id(ROLE_TOKEN[msg.role])
        assert role_id is not None

        ids.append(role_id)   # prompt prefix, never predicted
        mask.append(0)

        content_ids = tokenizer.encode(msg.content)
        is_assistant = msg.role == "assistant"
        ids.extend(content_ids)
        mask.extend([1 if is_assistant else 0] * len(content_ids))

        # The closing <|eos|> is in the loss for assistant turns only, so the
        # model learns where to stop.
        ids.append(tokenizer.eos_id)
        mask.append(1 if is_assistant else 0)

    if add_generation_prompt:
        role_id = tokenizer.token_to_id(ROLE_TOKEN["assistant"])
        assert role_id is not None
        ids.append(role_id)
        mask.append(0)

    return RenderedChat(ids=ids, loss_mask=mask)


def build_prompt(messages: Sequence[Message | dict], tokenizer: MiniTokenizer) -> list[int]:
    """Token IDs for a generation prompt. The one entry point generation, GRPO
    and eval share, so sampled text is formatted exactly like SFT data."""
    return render_chat(messages, tokenizer, add_generation_prompt=True).ids


# --- 4. Packed SFT examples and segment IDs ----------------------------------
# Several short conversations share one fixed-length row, keeping throughput
# near pretraining's. Each gets a distinct segment id, which model.py turns
# into the attention mask, so nothing attends across a boundary. Tail padding
# takes PAD_SEGMENT and a zero loss mask.

PAD_SEGMENT = -1


def synthetic_sft_conversations(n: int, *, seed: int = 0) -> list[list[dict]]:
    """Deterministic arithmetic chat conversations for offline dev and tests."""
    rng = random.Random(seed)
    ops = ("+", "-", "*")
    convs: list[list[dict]] = []
    for _ in range(n):
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        op = rng.choice(ops)
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        conv = [
            {"role": "user", "content": f"What is {a} {op} {b}?"},
            {"role": "assistant", "content": f"{ans}"},
        ]
        if rng.random() < 0.3:  # sometimes a second turn
            c = rng.randint(1, 20)
            conv += [
                {"role": "user", "content": f"Now add {c}."},
                {"role": "assistant", "content": f"{ans + c}"},
            ]
        convs.append(conv)
    return convs


@dataclass
class PackedSFT:
    """Batched, packed SFT tensors, all [N, seq_len] int64."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    segment_ids: torch.Tensor

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]

    def to(self, device) -> "PackedSFT":
        return PackedSFT(
            self.input_ids.to(device),
            self.targets.to(device),
            self.loss_mask.to(device),
            self.segment_ids.to(device),
        )

    def microbatches(self, micro_batch: int) -> Iterator[tuple[torch.Tensor, ...]]:
        for i in range(0, len(self), micro_batch):
            yield (
                self.input_ids[i : i + micro_batch],
                self.targets[i : i + micro_batch],
                self.loss_mask[i : i + micro_batch],
                self.segment_ids[i : i + micro_batch],
            )


def pack_conversations(
    conversations: Sequence[Sequence[Message | dict]],
    tokenizer: MiniTokenizer,
    *,
    seq_len: int,
) -> PackedSFT:
    """Render and greedily pack conversations into [N, seq_len] tensors.

    Each row holds whole conversations (one longer than seq_len+1 is
    truncated), and is seq_len+1 tokens before the next-token shift.
    """
    capacity = seq_len + 1
    pad_id = tokenizer.pad_id

    rows_ids: list[list[int]] = []
    rows_mask: list[list[int]] = []
    rows_seg: list[list[int]] = []

    buf_ids: list[int] = []
    buf_mask: list[int] = []
    buf_seg: list[int] = []
    seg = 0

    def flush() -> None:
        nonlocal buf_ids, buf_mask, buf_seg, seg
        if not buf_ids:
            return
        pad = capacity - len(buf_ids)
        rows_ids.append(buf_ids + [pad_id] * pad)
        rows_mask.append(buf_mask + [0] * pad)
        rows_seg.append(buf_seg + [PAD_SEGMENT] * pad)
        buf_ids, buf_mask, buf_seg, seg = [], [], [], 0

    for conv in conversations:
        r = render_chat(conv, tokenizer)
        ids, mask = r.ids, r.loss_mask
        if len(ids) > capacity:
            ids, mask = ids[:capacity], mask[:capacity]
        if len(buf_ids) + len(ids) > capacity:
            flush()
        buf_ids.extend(ids)
        buf_mask.extend(mask)
        buf_seg.extend([seg] * len(ids))
        seg += 1
    flush()

    ids_t = torch.tensor(rows_ids, dtype=torch.long)
    mask_t = torch.tensor(rows_mask, dtype=torch.long)
    seg_t = torch.tensor(rows_seg, dtype=torch.long)

    # Next-token shift: position i predicts i+1, supervised iff that predicted
    # token is an assistant one (mask[1:]). Segment ids align with the input
    # positions -- the queries -- hence [:-1].
    return PackedSFT(
        input_ids=ids_t[:, :-1].contiguous(),
        targets=ids_t[:, 1:].contiguous(),
        loss_mask=mask_t[:, 1:].contiguous(),
        segment_ids=seg_t[:, :-1].contiguous(),
    )


# --- 5. SFT loss + 6. SFT training loop --------------------------------------

def sft_loss(model: nn.Module, mb: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Cross-entropy over assistant tokens only, with cross-conversation
    attention blocked by the segment mask (both applied inside model.forward)."""
    ids, tgt, mask, seg = mb
    _, loss = model(ids, tgt, loss_mask=mask, segment_ids=seg)
    return loss


def train_sft(
    model: nn.Module,
    packed: PackedSFT,
    cfg: Config,
    *,
    device: torch.device | str = "cpu",
    steps: int | None = None,
    raw_model: nn.Module | None = None,
) -> list[float]:
    """Run SFT for `steps` optimizer steps over `packed` (cycled), reusing the
    pretraining optimizer/schedule stack. Returns per-step losses."""
    device = torch.device(device)
    raw = raw_model if raw_model is not None else model
    optimizers = build_optimizers(raw, cfg)
    steps = steps if steps is not None else cfg.max_steps
    scheduler = WarmupCosine(optimizers, min(cfg.warmup_steps, max(1, steps // 10)),
                             steps, cfg.lr_floor_frac)
    amp_dtype = _DTYPES.get(cfg.dtype, torch.float32)

    packed = packed.to(device)
    micro = list(packed.microbatches(cfg.micro_batch))
    if not micro:
        raise ValueError("no SFT micro-batches -- empty packed set")

    losses: list[float] = []
    cursor = 0
    for _ in range(steps):
        model.train()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg.grad_accum):
            mb = micro[cursor % len(micro)]
            cursor += 1
            with autocast_ctx(device, amp_dtype):
                loss = sft_loss(model, mb)
            (loss / cfg.grad_accum).backward()
            total += loss.item()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
        scheduler.step()
        losses.append(total / cfg.grad_accum)
    return losses


# --- 7. GSM8K answer extraction + 8. rewards ---------------------------------
# GSM8K solutions end with "#### <answer>", so the extractor prefers that
# delimiter and falls back to the last integer: a model that learns the
# delimiter is rewarded, a bare number still parses. Format-shaping terms give
# partial signal before the model is ever correct.

_HASH_ANSWER = re.compile(r"####\s*(-?\d+)")
_INT = re.compile(r"-?\d+")


@dataclass
class RewardResult:
    """A reward split into its correctness and format components (for logging)."""

    total: float
    correct: float
    format: float


def extract_final_int(text: str) -> int | None:
    """The answer after a '####' delimiter (GSM8K convention), else the last
    integer in the text, else None."""
    m = _HASH_ANSWER.search(text)
    if m:
        return int(m.group(1))
    ints = _INT.findall(text)
    return int(ints[-1]) if ints else None


def _format_score(text: str, *, terminated: bool, n_new_tokens: int, max_new_tokens: int) -> float:
    """Partial credit in [0, 1] for well-formedness, independent of correctness."""
    parseable = 1.0 if extract_final_int(text) is not None else 0.0
    term = 1.0 if terminated else 0.0
    within = 1.0 if 0 < n_new_tokens <= max_new_tokens else 0.0
    return (parseable + term + within) / 3.0


def answer_match_reward(
    text: str,
    *,
    gold: int,
    terminated: bool,
    n_new_tokens: int,
    max_new_tokens: int,
    w_correct: float = 1.0,
    w_format: float = 0.5,
) -> RewardResult:
    """Exact-match reward on the extracted final integer, plus format shaping.
    Scores both GSM8K (gold from the dataset) and offline arithmetic."""
    pred = extract_final_int(text)
    correct = 1.0 if (pred is not None and pred == gold) else 0.0
    fmt = _format_score(
        text, terminated=terminated, n_new_tokens=n_new_tokens, max_new_tokens=max_new_tokens
    )
    return RewardResult(total=w_correct * correct + w_format * fmt, correct=correct, format=fmt)


def gsm8k_reward(
    text: str,
    *,
    gold: int,
    terminated: bool,
    n_new_tokens: int,
    max_new_tokens: int,
) -> RewardResult:
    """GSM8K reward: final-integer exact match against the gold answer."""
    return answer_match_reward(
        text, gold=gold, terminated=terminated,
        n_new_tokens=n_new_tokens, max_new_tokens=max_new_tokens,
    )


# --- 9. Group sampling -------------------------------------------------------

# Scores one completion from its decoded text and metadata.
RewardFn = Callable[["Completion"], float]


@dataclass
class Completion:
    group: int          # which prompt-group this completion belongs to
    prompt_len: int
    tokens: list[int]   # full sequence: prompt tokens + completion (eos kept)
    text: str           # decoded completion text (new tokens only, eos stripped)
    terminated: bool
    n_new: int


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
    """Sample group_size completions per prompt. The same seed reproduces the
    same completions; each prompt takes a distinct sub-seed so its group
    differs from the others'."""
    eos = tokenizer.eos_id
    out: list[Completion] = []
    for gi, msgs in enumerate(prompts):
        prompt = build_prompt(msgs, tokenizer)
        idx = torch.tensor([prompt] * group_size, dtype=torch.long, device=device)
        gen = generate(
            model, idx, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, eos_id=eos, seed=seed + gi,
        )
        for row in gen:
            full = row.tolist()
            new = full[len(prompt):]
            terminated = eos in new
            if terminated:
                new = new[: new.index(eos) + 1]  # keep the eos itself
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
    """Pad completions to [B, L] and build the completion-token mask.

    Returns (seqs [B, L], comp_mask [B, L], groups [B]). comp_mask is True only
    on generated tokens, so the policy loss touches only what the model wrote.
    """
    b = len(completions)
    L = max(len(c.tokens) for c in completions)
    seqs = torch.full((b, L), pad_id, dtype=torch.long)
    comp_mask = torch.zeros((b, L), dtype=torch.bool)
    groups = torch.empty(b, dtype=torch.long)
    for i, c in enumerate(completions):
        t = torch.tensor(c.tokens, dtype=torch.long)
        seqs[i, : len(t)] = t
        comp_mask[i, c.prompt_len : len(t)] = True
        groups[i] = c.group
    return seqs.to(device), comp_mask.to(device), groups.to(device)


# --- 10. Group-relative advantage normalization ------------------------------

def group_advantages(rewards: torch.Tensor, groups: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """A_i = (r_i - mean_g) / (std_g + eps) within completion i's group g.

    The group mean is the baseline: that is what replaces a learned critic. A
    zero-variance group gives exactly zero advantage and so zero gradient --
    equally-good samples push the model neither way.
    """
    groups = groups.to(rewards.device)
    adv = torch.zeros_like(rewards, dtype=torch.float32)
    for g in groups.unique():
        m = groups == g
        r = rewards[m].float()
        std = r.std(unbiased=False)
        centered = r - r.mean()
        adv[m] = centered / (std + eps) if std > 0 else torch.zeros_like(centered)
    return adv


# --- 11. Token log-probabilities ---------------------------------------------

def token_logprobs(model: nn.Module, seqs: torch.Tensor, target_mask: torch.Tensor):
    """Per-token log-probs of seqs under model, plus the shifted token mask.

    Returns (logp [B, L-1], mask [B, L-1]): logp[:, t] is the log-prob of the
    actual next token seqs[:, t+1], and mask marks the completion targets.
    """
    logits, _ = model(seqs)  # targets=None -> full logits path
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)   # [B, L-1, V]
    tgt = seqs[:, 1:]                                          # [B, L-1]
    tok_logp = logp.gather(-1, tgt[:, :, None]).squeeze(-1)    # [B, L-1]
    return tok_logp, target_mask[:, 1:]


# --- 12. Clipped GRPO objective + 13. GRPO update ----------------------------

def grpo_loss(
    model: nn.Module,
    seqs: torch.Tensor,
    comp_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_logp: torch.Tensor,
    *,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """The clipped PPO surrogate over completion tokens, weighted by the
    group-relative advantage:

        ratio = exp(logp_new - logp_old)
        loss  = -mean( min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) )

    Clipping keeps the policy near the one that sampled; min() makes the
    objective a pessimistic bound.
    """
    new_logp, m = token_logprobs(model, seqs, comp_mask)
    ratio = torch.exp(new_logp - old_logp)
    adv = advantages[:, None]  # one advantage per sequence, broadcast to tokens
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
    """The GRPO loop over a prompt_bank of (messages, reward_fn) pairs; returns
    the mean reward per step. Each step draws prompts_per_step prompts, rolls
    out group_size completions each, scores them, normalizes within groups, and
    takes one clipped update."""
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
            model, tokenizer, prompts, group_size=group_size,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            seed=seed + step * 1000, device=device,
        )
        rewards = torch.tensor([reward_fns[c.group](c) for c in comps], dtype=torch.float32)
        seqs, comp_mask, groups = collate(comps, pad_id, device=device)
        advantages = group_advantages(rewards, groups).to(device)
        grpo_step(model, optimizer, seqs, comp_mask, advantages,
                  clip_eps=clip_eps, grad_clip=grad_clip)
        mean_rewards.append(float(rewards.mean()))

    return mean_rewards


# --- Prompt banks: GSM8K and offline arithmetic ------------------------------

def _bank_entry(question: str, gold: int, max_new_tokens: int) -> tuple[list[dict], RewardFn]:
    messages = [{"role": "user", "content": question}]

    def reward_fn(c: Completion, gold: int = gold) -> float:
        return gsm8k_reward(
            c.text, gold=gold, terminated=c.terminated,
            n_new_tokens=c.n_new, max_new_tokens=max_new_tokens,
        ).total

    return messages, reward_fn


def build_gsm8k_bank(
    *, max_new_tokens: int, split: str = "train", limit: int | None = None,
) -> list[tuple[list[dict], RewardFn]]:
    """A GRPO prompt bank from GSM8K (downloads `openai/gsm8k` on first use).

    Rows are {"question", "answer"}, the answer ending '#### <gold>'. Commas
    are stripped ('1,200' -> '1200') before extraction.
    """
    from datasets import load_dataset  # lazy: only the gsm8k task needs it

    ds = load_dataset("openai/gsm8k", "main", split=split)
    bank = []
    for row in ds:
        gold = extract_final_int(row["answer"].replace(",", ""))
        if gold is None:
            continue
        bank.append(_bank_entry(row["question"], gold, max_new_tokens))
        if limit is not None and len(bank) >= limit:
            break
    return bank


def build_arithmetic_bank(
    n: int, *, max_new_tokens: int, seed: int = 0,
) -> list[tuple[list[dict], RewardFn]]:
    """An offline bank of single-step additions: the same GRPO machinery
    without a dataset download."""
    rng = random.Random(seed)
    bank = []
    for _ in range(n):
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        bank.append(_bank_entry(f"What is {a} + {b}?", a + b, max_new_tokens))
    return bank


# --- 14. CLI: `sft` and `grpo` -----------------------------------------------

def load_conversations(path: str) -> list[list[dict]]:
    """Chat JSONL: one {"messages": [{"role", "content"}, ...]} per line."""
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line)["messages"])
    return convs


def _build_model(tier: str, init: str | None, device: str, **overrides: Any) -> tuple[Config, MiniGPT]:
    cfg = get_config(tier, **overrides)
    seed_everything(cfg.seed)
    model = MiniGPT(cfg).to(device)
    if init:
        load_model_weights(model, init, device=device)
    return cfg, model


def main_sft(args: argparse.Namespace) -> int:
    cfg, model = _build_model(args.tier, args.init, args.device)
    # SFT batches are full-context, so the [B*T, V] logits tensor is worth
    # avoiding here too.
    model.fused_loss = getattr(cfg, "use_triton", False)
    tok = MiniTokenizer.load(args.tokenizer)

    convs = (
        synthetic_sft_conversations(5000, seed=cfg.seed)
        if args.data == "synthetic"
        else load_conversations(args.data)
    )
    packed = pack_conversations(convs, tok, seq_len=cfg.context)
    print(f"SFT: {len(convs)} conversations -> {len(packed)} packed rows of {packed.seq_len}")

    losses = train_sft(model, packed, cfg, device=args.device, steps=args.steps)
    print(f"SFT done: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    # train.py's checkpoint format, so this loads like any other checkpoint.
    optimizers = build_optimizers(model, cfg)
    scheduler = WarmupCosine(optimizers, cfg.warmup_steps, args.steps, cfg.lr_floor_frac)
    save_checkpoint(
        Path(args.out) / "ckpt_sft.pt", model=model, optimizers=optimizers,
        scheduler=scheduler, step=args.steps,
    )
    print(f"wrote {args.out}/ckpt_sft.pt")
    return 0


def main_grpo(args: argparse.Namespace) -> int:
    cfg, model = _build_model(args.tier, args.init, args.device)
    tok = MiniTokenizer.load(args.tokenizer)

    if args.task == "gsm8k":
        bank = build_gsm8k_bank(max_new_tokens=args.max_new_tokens, limit=args.limit)
        print(f"GRPO on GSM8K: {len(bank)} prompts")
    else:
        bank = build_arithmetic_bank(256, max_new_tokens=args.max_new_tokens, seed=cfg.seed)
        print(f"GRPO on offline arithmetic: {len(bank)} prompts")

    rewards = run_grpo(
        model, tok, bank,
        steps=args.steps, group_size=args.group_size,
        prompts_per_step=args.prompts_per_step, max_new_tokens=args.max_new_tokens,
        lr=args.lr, seed=cfg.seed, device=args.device,
    )
    first = sum(rewards[:10]) / min(10, len(rewards))
    last = sum(rewards[-10:]) / min(10, len(rewards))
    print(f"GRPO done: mean reward {first:.3f} -> {last:.3f} over {len(rewards)} steps")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "rewards": rewards}, str(out / "ckpt_grpo.pt"))
    print(f"wrote {args.out}/ckpt_grpo.pt")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Post-train Mini-GPT: SFT or GRPO.")
    sub = ap.add_subparsers(dest="command", required=True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    sft = sub.add_parser("sft", help="supervised fine-tuning on chat conversations")
    sft.add_argument("--tier", default="mini")
    sft.add_argument("--tokenizer", required=True, help="trained tokenizer json")
    sft.add_argument("--init", default=None, help="base checkpoint (else random init)")
    sft.add_argument("--data", default="synthetic",
                     help="chat JSONL path, or 'synthetic' for the offline conversations")
    sft.add_argument("--out", default="out/sft")
    sft.add_argument("--steps", type=int, default=2000)
    sft.add_argument("--device", default=default_device)
    sft.set_defaults(fn=main_sft)

    grpo = sub.add_parser("grpo", help="GRPO on GSM8K or offline arithmetic")
    grpo.add_argument("--tier", default="mini")
    grpo.add_argument("--tokenizer", required=True)
    grpo.add_argument("--init", default=None, help="SFT checkpoint (else random init)")
    grpo.add_argument("--task", default="gsm8k", choices=["gsm8k", "arithmetic"],
                      help="gsm8k downloads the dataset; arithmetic is fully offline")
    grpo.add_argument("--limit", type=int, default=None, help="cap the GSM8K prompt bank size")
    grpo.add_argument("--out", default="out/grpo")
    grpo.add_argument("--steps", type=int, default=500)
    grpo.add_argument("--group-size", type=int, default=8)
    grpo.add_argument("--prompts-per-step", type=int, default=8)
    grpo.add_argument("--max-new-tokens", type=int, default=64)
    grpo.add_argument("--lr", type=float, default=1e-5)
    grpo.add_argument("--device", default=default_device)
    grpo.set_defaults(fn=main_grpo)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
