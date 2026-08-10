"""Chat template + SFT loss mask.

One deterministic function maps a list of role-tagged messages to a token
sequence and, for SFT, a parallel loss mask that is ``1`` only on
assistant-authored tokens. Keeping the template in a single function -- rather
than string-formatted at each call site -- is what makes the SFT loss mask and
the GRPO prompt formatting provably consistent: both call ``render_chat`` /
``build_prompt`` here, so there is exactly one copy of the format.

Layout of one turn::

    <|role|>  <content tokens>  <|eos|>

The whole sequence opens with ``<|bos|>``. For an assistant turn, the loss mask
covers the content tokens and the closing ``<|eos|>`` (the tokens the model must
learn to produce) but NOT the ``<|role|>`` header (which is always given as a
prompt, never predicted). Every non-assistant token is masked out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from mini_gpt.tokenizer import MiniTokenizer

Role = Literal["system", "user", "assistant", "tool"]

# Map a message role to the special token that opens its turn. A "tool" message
# carries a tool result; an assistant tool call is emitted inline (see below).
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
    out: list[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
        else:
            out.append(Message(role=m["role"], content=m["content"]))
    return out


@dataclass
class RenderedChat:
    ids: list[int]
    loss_mask: list[int]

    def __post_init__(self) -> None:
        assert len(self.ids) == len(self.loss_mask), "ids and mask must align"


def render_chat(
    messages: Sequence[Message | dict],
    tokenizer: MiniTokenizer,
    *,
    add_generation_prompt: bool = False,
) -> RenderedChat:
    """Render a conversation to token IDs plus an assistant-only loss mask.

    ``add_generation_prompt=True`` appends a trailing ``<|assistant|>`` header
    (with no content) so the model is primed to generate a reply -- used at
    inference and for GRPO rollouts. Its header token is not part of the loss.
    """
    msgs = _coerce(messages)

    ids: list[int] = [tokenizer.bos_id]
    mask: list[int] = [0]

    for msg in msgs:
        if msg.role not in ROLE_TOKEN:
            raise ValueError(f"unknown role {msg.role!r}; expected one of {sorted(ROLE_TOKEN)}")
        role_id = tokenizer.token_to_id(ROLE_TOKEN[msg.role])
        assert role_id is not None

        # Role header: always a prompt prefix, never predicted.
        ids.append(role_id)
        mask.append(0)

        content_ids = tokenizer.encode(msg.content)
        is_assistant = msg.role == "assistant"

        ids.extend(content_ids)
        mask.extend([1 if is_assistant else 0] * len(content_ids))

        # Closing marker; part of the loss only for assistant turns so the model
        # learns where to stop.
        ids.append(tokenizer.eos_id)
        mask.append(1 if is_assistant else 0)

    if add_generation_prompt:
        role_id = tokenizer.token_to_id(ROLE_TOKEN["assistant"])
        assert role_id is not None
        ids.append(role_id)
        mask.append(0)

    return RenderedChat(ids=ids, loss_mask=mask)


def build_prompt(messages: Sequence[Message | dict], tokenizer: MiniTokenizer) -> list[int]:
    """Token IDs for a generation prompt (adds the trailing assistant header).

    The single entry point GRPO and eval use to format a prompt for sampling,
    guaranteeing it matches the SFT training format.
    """
    return render_chat(messages, tokenizer, add_generation_prompt=True).ids
