"""Data-prep entry point: text parts -> tokenizer -> packed uint16 shards.

Turns a text source into everything the training scripts need: a trained
byte-level BPE (``--tokenizer``) and a packed shard directory with a manifest and
a held-out val split (``--data``). Sources mirror ``mini_gpt.data.download``:
``synthetic`` (offline, deterministic), ``hf-raw`` (the ClimbMix raw-text mirror),
or ``hf-tokens`` (the official token release, detokenized with GPT-2).

    python scripts/prepare_data.py --source synthetic --parts 4 --docs-per-part 20000 \
        --tokenizer data/tok.json --data data/packed --shard-tokens 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_gpt.data import download, pack  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.tokenizer import MiniTokenizer  # noqa: E402


def iter_docs(parts_dir: str | Path) -> Iterator[str]:
    """Yield the ``text`` field of every document across the jsonl parts.

    Re-reads from disk per call, so the corpus can be streamed twice -- once to
    train the tokenizer, once to pack -- without holding it in RAM.
    """
    for part in sorted(Path(parts_dir).glob("part_*.jsonl")):
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)["text"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare tokenizer + packed shards.")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "hf-raw", "hf-tokens"])
    ap.add_argument("--parts-dir", default="data/parts", help="where text parts are written/read")
    ap.add_argument("--parts", type=int, default=4, help="number of text parts to fetch")
    ap.add_argument("--docs-per-part", type=int, default=20_000)
    ap.add_argument("--tokenizer", default="data/tok.json", help="output tokenizer json")
    ap.add_argument("--data", default="data/packed", help="output packed-shard directory")
    ap.add_argument("--vocab-size", type=int, default=32_768)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--shard-tokens", type=int, default=50_000_000, help="tokens per shard")
    ap.add_argument("--val-shards", type=int, default=1, help="held-out shards for perplexity")
    ap.add_argument("--overwrite", action="store_true", help="re-fetch parts even if present")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    seed_everything(args.seed)

    # text parts (idempotent: existing parts are kept unless --overwrite)
    written = download.download(
        args.parts_dir,
        parts=args.parts,
        docs_per_part=args.docs_per_part,
        source=args.source,
        overwrite=args.overwrite,
    )
    print(f"parts: {len(written)} in {args.parts_dir}")

    # train + save the byte-level BPE
    tok = MiniTokenizer.train(
        iter_docs(args.parts_dir), vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    tok.save(args.tokenizer)
    print(f"tokenizer: {tok.vocab_size} tokens -> {args.tokenizer}")

    # pack to uint16 shards + manifest (with a held-out val split)
    manifest = pack.pack_corpus(
        iter_docs(args.parts_dir),
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
