"""Supervised fine-tuning.

Continues training on chat-formatted conversations rendered through the same
``render_chat`` template used at inference. Two properties are pinned by
``test_posttrain.py``:

* Assistant-only loss. The template's loss mask is ``1`` only on
  assistant-authored tokens, so the model is never trained to predict user or
  system text. Next-token training shifts by one: predicting token ``t`` is
  supervised iff ``t`` is an assistant token, i.e. ``mask[1:]``.
* Packed, isolated conversations. Several short conversations share one
  fixed-length sequence to keep throughput near pretraining's, each with a
  distinct ``segment_id`` so attention cannot cross a conversation boundary (the
  segment mask is applied in ``MiniGPT.forward``).

Tail padding gets ``segment_id = PAD_SEGMENT`` and a zero loss mask, so it
neither contributes to the loss nor is attended to by real tokens; it can still
attend to itself, keeping the softmax row well-defined.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import nn

from mini_gpt.chat_template import Message, render_chat
from mini_gpt.tokenizer import MiniTokenizer

PAD_SEGMENT = -1
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def synthetic_sft_conversations(n: int, *, seed: int = 0) -> list[list[dict]]:
    """Deterministic chat conversations for offline dev and tests.

    Arithmetic Q/A in the role-tagged format, exercising the packing, masking, and
    generation path end to end without an instruction dataset. Some are
    multi-turn.
    """
    import random

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
        if rng.random() < 0.3:  # a second turn
            c = rng.randint(1, 20)
            conv += [
                {"role": "user", "content": f"Now add {c}."},
                {"role": "assistant", "content": f"{ans + c}"},
            ]
        convs.append(conv)
    return convs


@dataclass
class PackedSFT:
    """Batched, packed SFT tensors, all ``[N, seq_len]`` (int64 except targets)."""

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
    """Render and greedily pack conversations into ``[N, seq_len]`` tensors.

    Each packed row holds one or more whole conversations (a conversation longer
    than ``seq_len+1`` is truncated). A row is ``seq_len+1`` tokens before the
    next-token shift; the shift then yields ``seq_len`` input/target positions.
    """
    capacity = seq_len + 1  # room for the +1 shift
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
        if len(ids) > capacity:  # a single conversation longer than the window
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

    # Next-token shift. Predicting position i uses the target at i+1, supervised
    # iff that predicted token is an assistant token (mask[1:]). Segment ids align
    # with the input positions (queries), hence the left slice.
    return PackedSFT(
        input_ids=ids_t[:, :-1].contiguous(),
        targets=ids_t[:, 1:].contiguous(),
        loss_mask=mask_t[:, 1:].contiguous(),
        segment_ids=seg_t[:, :-1].contiguous(),
    )


def sft_loss(model: nn.Module, mb: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Cross-entropy over assistant tokens only, with cross-conversation blocking."""
    ids, tgt, mask, seg = mb
    _, loss = model(ids, tgt, loss_mask=mask, segment_ids=seg)
    return loss


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def train_sft(
    model: nn.Module,
    packed: PackedSFT,
    cfg,
    *,
    device: torch.device | str = "cpu",
    steps: int | None = None,
    raw_model: nn.Module | None = None,
) -> list[float]:
    """Run SFT for ``steps`` optimizer steps over ``packed`` (cycled).

    Reuses the pretraining optimizer/schedule stack; one optimizer step
    accumulates ``cfg.grad_accum`` micro-batches. Returns per-step losses.
    """
    from mini_gpt.train.optim import build_optimizers
    from mini_gpt.train.schedule import build_scheduler

    device = torch.device(device)
    raw = raw_model if raw_model is not None else model
    opt = build_optimizers(raw, cfg)
    sched = build_scheduler(opt, cfg)
    amp_dtype = _DTYPES.get(cfg.dtype, torch.float32)
    steps = steps if steps is not None else cfg.max_steps

    packed = packed.to(device)
    micro = list(packed.microbatches(cfg.micro_batch))
    if not micro:
        raise ValueError("no SFT micro-batches -- empty packed set")

    losses: list[float] = []
    cursor = 0
    for _ in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg.grad_accum):
            mb = micro[cursor % len(micro)]
            cursor += 1
            with _autocast_ctx(device, amp_dtype):
                loss = sft_loss(model, mb)
            (loss / cfg.grad_accum).backward()
            total += loss.item()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        losses.append(total / cfg.grad_accum)
    return losses
