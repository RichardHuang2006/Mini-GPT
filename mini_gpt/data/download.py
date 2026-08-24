"""Download + detokenize ClimbMix into raw-text parts.

NVIDIA's official ``nvidia/Nemotron-ClimbMix`` ships GPT-2 token IDs in jsonl
parts, not raw text, and training a 32K BPE needs text. This stage writes UTF-8
``.jsonl`` parts of ``{"text": ...}`` documents from one of three sources:

* ``"hf-raw"``    -- the community raw-text mirror ``OptimalScale/ClimbMix``.
* ``"hf-tokens"`` -- the official token-ID release, detokenized with GPT-2.
* an injected iterable / callable of strings -- used by tests and offline dev, so
  the download and idempotency logic runs without a network.

Writing parts is idempotent: an already-present part index is skipped, so a
re-run continues rather than re-downloading.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Iterable, Iterator

PART_TEMPLATE = "part_{:05d}.jsonl"


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


def _write_part(path: Path, docs: list[str]) -> None:
    # Write atomically via a temp file: an interrupted run must not leave a
    # half-written part that the idempotency check would take for complete.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_parts(
    out_dir: str | Path,
    texts: Iterable[str],
    *,
    parts: int,
    docs_per_part: int,
    overwrite: bool = False,
) -> list[Path]:
    """Group ``texts`` into ``parts`` files of ``docs_per_part`` docs each.

    Skips part indices that already exist (unless ``overwrite``), so a re-run is
    idempotent and continues where a previous one stopped.
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
        docs = list(_take(it, docs_per_part))
        if not docs:
            break  # source exhausted
        _write_part(path, docs)
        written.append(path)

    return written


def read_parts(out_dir: str | Path) -> Iterator[str]:
    """Yield every document (as text) across all parts, in part order."""
    for index in existing_part_indices(out_dir):
        with part_path(out_dir, index).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)["text"]


def _take(it: Iterator[str], n: int) -> Iterator[str]:
    for _ in range(n):
        try:
            yield next(it)
        except StopIteration:
            return


# ---------------------------------------------------------------- sources ---

def synthetic_docs(n: int, *, seed: int = 0) -> Iterator[str]:
    """Deterministic pseudo-text for offline dev and tests.

    Not language, but a recombined controlled vocabulary, varied enough to give
    the BPE trainer real merges and the sampler real windows.
    """
    rng = random.Random(seed)
    words = (
        "the quick brown fox jumps over a lazy dog while tensor gradients flow "
        "through attention heads and rotary embeddings rotate query keys softly "
        "muon orthogonalizes momentum and adamw decays weights on embeddings"
    ).split()
    for _ in range(n):
        length = rng.randint(20, 120)
        doc = " ".join(rng.choice(words) for _ in range(length))
        yield doc + "."


def iter_hf_raw_text(streaming: bool = True) -> Iterator[str]:  # pragma: no cover - network
    """Stream the community raw-text ClimbMix mirror (OptimalScale/ClimbMix)."""
    from datasets import load_dataset

    ds = load_dataset("OptimalScale/ClimbMix", split="train", streaming=streaming)
    for row in ds:
        text = row.get("text")
        if text:
            yield text


def iter_hf_tokenized(streaming: bool = True) -> Iterator[str]:  # pragma: no cover - network
    """Stream the official token-ID ClimbMix release and GPT-2-detokenize it."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("nvidia/Nemotron-ClimbMix", split="train", streaming=streaming)
    for row in ds:
        ids = row.get("tokens") or row.get("input_ids")
        if ids:
            yield enc.decode(list(ids))


def _resolve_source(source) -> Iterable[str]:
    if callable(source):
        return source()
    if isinstance(source, str):
        if source == "synthetic":
            return synthetic_docs(1_000_000)
        if source == "hf-raw":
            return iter_hf_raw_text()
        if source == "hf-tokens":
            return iter_hf_tokenized()
        raise ValueError(f"unknown source {source!r}")
    return source  # assume an iterable of strings


def download(
    out_dir: str | Path,
    *,
    parts: int,
    docs_per_part: int = 10_000,
    source: str | Callable[[], Iterable[str]] | Iterable[str] = "hf-raw",
    overwrite: bool = False,
) -> list[Path]:
    """Download (or generate) ``parts`` jsonl text parts into ``out_dir``."""
    return write_parts(
        out_dir,
        _resolve_source(source),
        parts=parts,
        docs_per_part=docs_per_part,
        overwrite=overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download/generate ClimbMix text parts.")
    ap.add_argument("--out", default="data/climbmix", help="output directory")
    ap.add_argument("--parts", type=int, required=True, help="number of parts to fetch")
    ap.add_argument("--docs-per-part", type=int, default=10_000)
    ap.add_argument(
        "--source",
        default="hf-raw",
        choices=["hf-raw", "hf-tokens", "synthetic"],
        help="hf-raw: OptimalScale mirror; hf-tokens: official + GPT-2 detok; synthetic: offline",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    written = download(
        args.out,
        parts=args.parts,
        docs_per_part=args.docs_per_part,
        source=args.source,
        overwrite=args.overwrite,
    )
    print(f"wrote/kept {len(written)} parts in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
