"""Pack text into uint16 token shards + a manifest.

Documents are BPE-encoded and appended to flat ``.bin`` shards of a fixed token
count, stored as ``uint16``: lossless because the vocab is < 65536, and half the
disk and page-cache pressure of int32. An ``<|eos|>`` is inserted between
documents so the model sees document boundaries. ``manifest.json`` records the
tokenizer fingerprint, dtype, and per-shard token counts with a train/val split,
so the sampler knows what it is reading and against which tokenizer.

Shards are written and read in native byte order (single-machine project); the
manifest records the dtype so a reader never guesses.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from mini_gpt.tokenizer import MiniTokenizer

DTYPE = np.uint16
SHARD_TEMPLATE = "shard_{:05d}.bin"
MANIFEST_NAME = "manifest.json"


@dataclass
class ShardInfo:
    name: str
    tokens: int
    split: str  # "train" | "val"


@dataclass
class Manifest:
    tokenizer_fingerprint: str
    vocab_size: int
    dtype: str
    add_eos: bool
    total_tokens: int
    shards: list[ShardInfo]

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        d = json.loads(text)
        d["shards"] = [ShardInfo(**s) for s in d["shards"]]
        return cls(**d)

    def shards_for(self, split: str) -> list[ShardInfo]:
        return [s for s in self.shards if s.split == split]


def pack_corpus(
    docs: Iterable[str],
    tokenizer: MiniTokenizer,
    out_dir: str | Path,
    *,
    shard_tokens: int = 100_000_000,
    val_shards: int = 1,
    add_eos: bool = True,
) -> Manifest:
    """Encode ``docs`` and write fixed-size uint16 shards + a manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    eos = tokenizer.eos_id
    buffer: list[int] = []
    shard_index = 0
    shard_infos: list[ShardInfo] = []

    def flush(chunk: list[int]) -> None:
        nonlocal shard_index
        arr = np.asarray(chunk, dtype=DTYPE)
        # uint16 must be lossless for these IDs.
        assert np.array_equal(arr.astype(np.int64), np.asarray(chunk, dtype=np.int64)), (
            "token id exceeded uint16 range -- vocab must be < 65536"
        )
        name = SHARD_TEMPLATE.format(shard_index)
        arr.tofile(out / name)
        shard_infos.append(ShardInfo(name=name, tokens=int(arr.size), split="train"))
        shard_index += 1

    for doc in docs:
        buffer.extend(tokenizer.encode(doc))
        if add_eos:
            buffer.append(eos)
        while len(buffer) >= shard_tokens:
            flush(buffer[:shard_tokens])
            del buffer[:shard_tokens]

    if buffer:
        flush(buffer)

    if not shard_infos:
        raise ValueError("no tokens produced -- empty corpus?")

    # Tag the last `val_shards` shards as validation, keeping >=1 train shard.
    n_val = max(0, min(val_shards, len(shard_infos) - 1))
    for s in shard_infos[len(shard_infos) - n_val :]:
        s.split = "val"

    manifest = Manifest(
        tokenizer_fingerprint=tokenizer.fingerprint(),
        vocab_size=tokenizer.vocab_size,
        dtype=np.dtype(DTYPE).name,
        add_eos=add_eos,
        total_tokens=sum(s.tokens for s in shard_infos),
        shards=shard_infos,
    )
    (out / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def load_manifest(data_dir: str | Path) -> Manifest:
    text = (Path(data_dir) / MANIFEST_NAME).read_text(encoding="utf-8")
    return Manifest.from_json(text)


def open_shard(path: str | Path) -> np.memmap:
    """Memory-map a shard as read-only uint16."""
    return np.memmap(path, dtype=DTYPE, mode="r")


def read_all_tokens(data_dir: str | Path, split: str | None = None) -> np.ndarray:
    """Concatenate shards (optionally one split) into one array -- for tests."""
    manifest = load_manifest(data_dir)
    shards = manifest.shards if split is None else manifest.shards_for(split)
    parts = [np.asarray(open_shard(Path(data_dir) / s.name)) for s in shards]
    return np.concatenate(parts) if parts else np.empty(0, dtype=DTYPE)


def verify_against_tokenizer(data_dir: str | Path, tokenizer: MiniTokenizer) -> bool:
    """True iff the manifest's fingerprint matches ``tokenizer``."""
    return load_manifest(data_dir).tokenizer_fingerprint == tokenizer.fingerprint()
