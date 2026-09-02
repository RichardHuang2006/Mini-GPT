"""Dataset preparation: text sources -> tokenizer -> packed uint16 shards.

Three stages, in file order: fetch text into .jsonl parts, pack it into flat
uint16 shards plus a manifest, then sample fixed-length windows for training.
Shards are encoded with tokenizer.py's 32K BPE.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from mini_gpt.tokenizer import DEFAULT_VOCAB_SIZE, MiniTokenizer

# Token IDs are stored as uint16: lossless because the vocab is < 65,536, and
# half the disk and page-cache footprint of int32.
DTYPE = np.uint16
PART_TEMPLATE = "part_{:05d}.jsonl"
SHARD_TEMPLATE = "shard_{:05d}.bin"
MANIFEST_NAME = "manifest.json"


# --- 1. Text sources ---------------------------------------------------------
# Each source is an iterator of document strings. ClimbMix's official release
# ships GPT-2 token IDs, hence two HuggingFace paths plus an offline one.

def synthetic_docs(n: int, *, seed: int = 0) -> Iterator[str]:
    """Deterministic pseudo-text for offline development and tests: not real
    language, but varied enough to give the BPE trainer real merges."""
    rng = random.Random(seed)
    words = (
        "the quick brown fox jumps over a lazy dog while tensor gradients flow "
        "through attention heads and rotary embeddings rotate query keys softly "
        "muon orthogonalizes momentum and adamw decays weights on embeddings"
    ).split()
    for _ in range(n):
        length = rng.randint(20, 120)
        yield " ".join(rng.choice(words) for _ in range(length)) + "."


def iter_climbmix_raw(streaming: bool = True) -> Iterator[str]:  # pragma: no cover - network
    """Stream the community raw-text mirror of ClimbMix (OptimalScale/ClimbMix)."""
    from datasets import load_dataset

    ds = load_dataset("OptimalScale/ClimbMix", split="train", streaming=streaming)
    for row in ds:
        text = row.get("text")
        if text:
            yield text


def iter_climbmix_tokens(streaming: bool = True) -> Iterator[str]:  # pragma: no cover - network
    """Stream NVIDIA's token-ID release (nvidia/Nemotron-ClimbMix), GPT-2
    detokenized because training a new 32K BPE needs raw text. Needs the
    optional `tiktoken` package."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("nvidia/Nemotron-ClimbMix", split="train", streaming=streaming)
    for row in ds:
        ids = row.get("tokens") or row.get("input_ids")
        if ids:
            yield enc.decode(list(ids))


def resolve_source(source: str | Callable[[], Iterable[str]] | Iterable[str]) -> Iterable[str]:
    """Map a source name (or an injected iterable, used by tests) to documents."""
    if callable(source):
        return source()
    if isinstance(source, str):
        if source == "synthetic":
            return synthetic_docs(1_000_000)
        if source == "hf-raw":
            return iter_climbmix_raw()
        if source == "hf-tokens":
            return iter_climbmix_tokens()
        raise ValueError(f"unknown source {source!r}")
    return source


# --- 2. Text parts on disk (fetch) -------------------------------------------

def part_path(out_dir: str | Path, index: int) -> Path:
    return Path(out_dir) / PART_TEMPLATE.format(index)


def existing_part_indices(out_dir: str | Path) -> list[int]:
    out = Path(out_dir)
    if not out.exists():
        return []
    idx = []
    for p in out.glob("part_*.jsonl"):
        try:
            idx.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(idx)


def write_parts(
    out_dir: str | Path,
    texts: Iterable[str],
    *,
    parts: int,
    docs_per_part: int,
    overwrite: bool = False,
) -> list[Path]:
    """Group `texts` into `parts` jsonl files of `docs_per_part` docs each.

    Idempotent: existing part indices are kept unless `overwrite`, so an
    interrupted download continues instead of re-fetching.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    have = set() if overwrite else set(existing_part_indices(out))
    written: list[Path] = []
    it = iter(texts)

    for index in range(parts):
        path = part_path(out, index)
        if index in have:
            written.append(path)
            continue
        docs = []
        for _ in range(docs_per_part):
            try:
                docs.append(next(it))
            except StopIteration:
                break
        if not docs:
            break  # source exhausted
        # Atomic via temp file: an interrupted run must not leave a half-written
        # part that the idempotency check would take for complete.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        tmp.replace(path)
        written.append(path)

    return written


def read_parts(parts_dir: str | Path) -> Iterator[str]:
    """Yield every document across all parts, in part order. Re-reads from disk
    per call, so the corpus streams twice -- once to train the tokenizer, once
    to pack -- without ever sitting in RAM."""
    for index in existing_part_indices(parts_dir):
        with part_path(parts_dir, index).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)["text"]


# --- 3. Packed uint16 shards + manifest (pack) -------------------------------
# One long token stream cut into flat .bin shards. The manifest records the
# train/val split and the tokenizer fingerprint, so a stale pairing fails loud.

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
        return json.dumps(asdict(self), indent=2)

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
    """Encode `docs` into fixed-size uint16 shards plus a manifest.

    An <|eos|> between documents marks the boundaries; the last `val_shards`
    shards become the held-out split, always leaving one train shard.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    eos = tokenizer.eos_id
    buffer: list[int] = []
    shard_index = 0
    shard_infos: list[ShardInfo] = []

    def flush(chunk: list[int]) -> None:
        nonlocal shard_index
        arr = np.asarray(chunk, dtype=DTYPE)
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
    return Manifest.from_json((Path(data_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def open_shard(path: str | Path) -> np.memmap:
    """Memory-map a shard read-only: windows are sliced from the OS page cache,
    so no shard is ever fully loaded into RAM."""
    return np.memmap(path, dtype=DTYPE, mode="r")


def read_all_tokens(data_dir: str | Path, split: str | None = None) -> np.ndarray:
    """Concatenate shards (optionally one split) into one array -- for tests."""
    manifest = load_manifest(data_dir)
    shards = manifest.shards if split is None else manifest.shards_for(split)
    parts = [np.asarray(open_shard(Path(data_dir) / s.name)) for s in shards]
    return np.concatenate(parts) if parts else np.empty(0, dtype=DTYPE)


def verify_against_tokenizer(data_dir: str | Path, tokenizer: MiniTokenizer) -> bool:
    """True iff the manifest's fingerprint matches `tokenizer`."""
    return load_manifest(data_dir).tokenizer_fingerprint == tokenizer.fingerprint()


# --- 4. Windowed sampling (sample) -------------------------------------------

class ShardSampler:
    """Seeded, restartable sampler of `context + 1`-token windows over one
    split: the first `context` are inputs, the same window shifted by one is
    targets.

    A window lies entirely within one shard, chosen with probability
    proportional to its valid start positions, then a uniform start inside it.
    Both draws come from one seeded generator, so the stream is a pure function
    of (seed, draws so far) -- which is what reproduces the data order exactly
    when a run picks up from a checkpoint.
    """

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
        self.window = context + 1  # inputs + the one-token-shifted targets
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

    def next_window(self) -> np.ndarray:
        """One `[context + 1]` int64 window (inputs and targets share it)."""
        shard_idx = int(self.rng.choice(len(self._shards), p=self._probs))
        start = int(self.rng.integers(0, self._start_counts[shard_idx]))
        self.position += 1
        return np.asarray(self._shards[shard_idx][start : start + self.window], dtype=np.int64)

    def next_batch(self, batch_size: int) -> np.ndarray:
        """A `[batch_size, context + 1]` int64 array of stacked windows."""
        return np.stack([self.next_window() for _ in range(batch_size)])

    # --- persistence: rides along in checkpoints so a restarted run carries
    # on through the window stream instead of replaying or skipping data.
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


# --- 5. CLI: fetch -> train tokenizer -> pack --------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prepare data: fetch text, train the 32K BPE, pack uint16 shards."
    )
    ap.add_argument(
        "--source",
        default="synthetic",
        choices=["synthetic", "hf-raw", "hf-tokens"],
        help="synthetic: offline; hf-raw: ClimbMix raw-text mirror (large download); "
        "hf-tokens: official NVIDIA token release, GPT-2-detokenized (large download)",
    )
    ap.add_argument("--parts-dir", default="data/parts", help="where text parts are written/read")
    ap.add_argument("--parts", type=int, default=4, help="number of text parts to fetch")
    ap.add_argument("--docs-per-part", type=int, default=20_000)
    ap.add_argument("--tokenizer", default="data/tok.json", help="output tokenizer json")
    ap.add_argument("--data", default="data/packed", help="output packed-shard directory")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--shard-tokens", type=int, default=50_000_000, help="tokens per shard")
    ap.add_argument("--val-shards", type=int, default=1, help="held-out shards for perplexity")
    ap.add_argument("--overwrite", action="store_true", help="re-fetch parts even if present")
    args = ap.parse_args(argv)

    # 1. fetch text parts
    written = write_parts(
        args.parts_dir,
        resolve_source(args.source),
        parts=args.parts,
        docs_per_part=args.docs_per_part,
        overwrite=args.overwrite,
    )
    print(f"parts: {len(written)} in {args.parts_dir}")

    # 2. train and save the byte-level BPE
    tok = MiniTokenizer.train(
        read_parts(args.parts_dir), vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    tok.save(args.tokenizer)
    print(f"tokenizer: {tok.vocab_size} tokens -> {args.tokenizer}")

    # 3. pack to uint16 shards + manifest
    manifest = pack_corpus(
        read_parts(args.parts_dir),
        tok,
        args.data,
        shard_tokens=args.shard_tokens,
        val_shards=args.val_shards,
    )
    n_train = len(manifest.shards_for("train"))
    n_val = len(manifest.shards_for("val"))
    print(f"packed: {n_train} train + {n_val} val shard(s) -> {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
