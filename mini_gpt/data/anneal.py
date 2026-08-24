"""The anneal mix and the mid-run data switch.

The last ~2% of the pretraining token budget switches from the general web mix to
one folding in math and instruction-formatted data, giving the 39M model nonzero
arithmetic competence before GRPO, without which the RL phase has no gradient
signal.

Two pieces:

* mix content -- ``synthetic_math_docs`` (offline arithmetic and instruction
  text) plus ``mix_docs`` to interleave it with base text at a ratio;
  ``scripts/anneal.py`` packs the result into a shard directory.
* mix switch -- ``AnnealDataStream`` composes the base and anneal ``DataStream``s
  and flips between them at a fixed optimizer step. The switch step is a pure
  function of the token budget and the stream position is fully checkpointed, so
  a run resumes across the switch without replaying or skipping tokens.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Iterator

from mini_gpt.train.loop import DataStream


# --------------------------------------------------------------- mix content

def synthetic_math_docs(n: int, *, seed: int = 0) -> Iterator[str]:
    """Deterministic arithmetic / instruction-formatted docs for offline dev.

    Mirrors the shape of the real anneal mix (short Q/A and instruction spans) so
    the packing and switch logic runs without a math corpus. The countdown-style
    arithmetic is the same task GRPO rewards.
    """
    rng = random.Random(seed)
    ops = ("+", "-", "*")
    for _ in range(n):
        kind = rng.random()
        if kind < 0.6:
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            op = rng.choice(ops)
            ans = {"+": a + b, "-": a - b, "*": a * b}[op]
            yield f"Question: What is {a} {op} {b}?\nAnswer: {ans}."
        elif kind < 0.85:
            nums = [rng.randint(1, 20) for _ in range(rng.randint(2, 4))]
            target = sum(nums)
            joined = " + ".join(str(x) for x in nums)
            yield f"Instruction: Add these numbers: {joined}.\nResponse: {target}."
        else:
            a = rng.randint(2, 12)
            b = rng.randint(2, 12)
            yield f"Question: Compute {a} times {b}.\nAnswer: {a * b}."


def mix_docs(
    base_docs: Iterable[str],
    math_docs: Iterable[str],
    *,
    math_frac: float,
    seed: int = 0,
) -> Iterator[str]:
    """Interleave ``base_docs`` and ``math_docs`` so ~``math_frac`` are math.

    A seeded per-document coin decides the source, so the mix is reproducible.
    The base stream is the backbone: it runs until empty with math interleaved,
    and math stops contributing once exhausted.
    """
    assert 0.0 <= math_frac <= 1.0
    rng = random.Random(seed)
    base_it = iter(base_docs)
    math_it = iter(math_docs)
    math_done = False
    for doc in base_it:
        if not math_done and rng.random() < math_frac:
            try:
                yield next(math_it)
            except StopIteration:
                math_done = True
        yield doc


# ----------------------------------------------------------------- switch

def anneal_switch_step(max_steps: int, anneal_frac: float) -> int:
    """Optimizer step at which the last ``anneal_frac`` of the budget begins."""
    assert 0.0 <= anneal_frac < 1.0
    return int(round(max_steps * (1.0 - anneal_frac)))


class AnnealDataStream:
    """A ``DataStream`` that switches from ``base`` to ``anneal`` at a step.

    ``batch()`` matches ``DataStream.batch``, so the ``Trainer`` is oblivious to
    the switch. The current optimizer step is derived from micro-batches drawn
    (``batches // grad_accum``), so one step's ``grad_accum`` micro-batches always
    come from a single source and the switch lands on a step boundary. Both
    sub-sampler positions and the draw counter are checkpointed, so resume
    reproduces the stream and switches at the same absolute step.
    """

    def __init__(
        self,
        base: DataStream,
        anneal: DataStream,
        *,
        switch_step: int,
        grad_accum: int,
    ):
        self.base = base
        self.anneal = anneal
        self.switch_step = switch_step
        self.grad_accum = max(1, grad_accum)
        self.batches = 0
        self.switched = False  # latched True once the anneal source is first used

    @property
    def current_step(self) -> int:
        return self.batches // self.grad_accum

    @property
    def in_anneal(self) -> bool:
        return self.current_step >= self.switch_step

    def batch(self):
        source = self.anneal if self.in_anneal else self.base
        just_switched = False
        if source is self.anneal and not self.switched:
            self.switched = True
            just_switched = True
        self._just_switched = just_switched
        self.batches += 1
        return source.batch()

    def took_switch(self) -> bool:
        """True exactly on the first micro-batch drawn from the anneal mix."""
        return getattr(self, "_just_switched", False)

    # --------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "switched": self.switched,
            "switch_step": self.switch_step,
            "grad_accum": self.grad_accum,
            "base": self.base.state_dict(),
            "anneal": self.anneal.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.batches = state["batches"]
        self.switched = state["switched"]
        self.switch_step = state["switch_step"]
        self.grad_accum = state["grad_accum"]
        self.base.load_state_dict(state["base"])
        self.anneal.load_state_dict(state["anneal"])
