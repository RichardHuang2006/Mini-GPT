"""Seeded, resumable windowed sampler over packed shards.

Draws ``context + 1``-length windows (inputs plus the shifted target) from the
memory-mapped shards of one split. Given a seed and the manifest, the stream of
windows is exactly reproducible -- the precondition for the golden-loss-curve
regression and for resuming a checkpoint mid-run without replaying or skipping
tokens.

A window lies entirely within one shard (no cross-shard concatenation): the shard
is chosen with probability proportional to its number of valid start positions,
then a uniform start inside it. Both draws come from one seeded ``numpy``
generator, so the sequence is a pure function of (seed, windows drawn).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mini_gpt.data.pack import load_manifest, open_shard


class ShardSampler:
    def __init__(
        self,
        data_dir: str | Path,
        *,
        context: int,
        split: str = "train",
        seed: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.context = context
        self.window = context + 1
        self.split = split
        self.seed = seed

        manifest = load_manifest(self.data_dir)
        infos = manifest.shards_for(split)
        if not infos:
            raise ValueError(f"no {split!r} shards in {self.data_dir}")

        self.shard_names = [s.name for s in infos]
        self._shards = [open_shard(self.data_dir / s.name) for s in infos]

        # Valid start positions per shard: len - window + 1 (0 if too short).
        starts = np.array(
            [max(0, len(m) - self.window + 1) for m in self._shards], dtype=np.int64
        )
        if starts.sum() == 0:
            raise ValueError(
                f"no shard in split {split!r} is long enough for a window of {self.window}"
            )
        self._start_counts = starts
        self._probs = starts / starts.sum()

        self.rng = np.random.default_rng(seed)
        self.position = 0  # number of windows drawn so far

    # ------------------------------------------------------------ sampling
    def next_window(self) -> np.ndarray:
        """Return one ``[window]`` int64 array (input+target share the buffer)."""
        shard_idx = int(self.rng.choice(len(self._shards), p=self._probs))
        start = int(self.rng.integers(0, self._start_counts[shard_idx]))
        self.position += 1
        chunk = self._shards[shard_idx][start : start + self.window]
        return np.asarray(chunk, dtype=np.int64)

    def next_batch(self, batch_size: int) -> np.ndarray:
        """Return a ``[batch_size, window]`` int64 array of stacked windows."""
        return np.stack([self.next_window() for _ in range(batch_size)])

    # --------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "split": self.split,
            "context": self.context,
            "position": self.position,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        assert state["context"] == self.context, "context mismatch on resume"
        assert state["split"] == self.split, "split mismatch on resume"
        self.position = state["position"]
        self.rng.bit_generator.state = state["rng_state"]
