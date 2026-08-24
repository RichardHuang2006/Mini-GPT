"""Build the anneal-mix shard directory.

Folds math / instruction-formatted data into the base web text at a ratio and
packs the result into ``uint16`` shards with the same tokenizer as the base
pretraining corpus; the manifest fingerprint must match for the sampler to read
it. ``scripts/pretrain.py --anneal-data <this dir>`` then switches to it for the
last ``anneal_frac`` of the run.

    python scripts/anneal.py --base data/climbmix --tokenizer out/tok.json \
        --out data/anneal --math-frac 0.5 --source synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_gpt.data import download, pack  # noqa: E402
from mini_gpt.data.anneal import mix_docs, synthetic_math_docs  # noqa: E402
from mini_gpt.tokenizer import MiniTokenizer  # noqa: E402


def build_anneal_shards(
    base_docs,
    tokenizer: MiniTokenizer,
    out_dir: str | Path,
    *,
    math_frac: float = 0.5,
    math_docs=None,
    seed: int = 0,
    shard_tokens: int = 100_000_000,
    val_shards: int = 0,
) -> pack.Manifest:
    """Mix base text with math/instruct docs and pack to shards."""
    if math_docs is None:
        # Upper bound only; ``mix_docs`` pulls just as many as it needs.
        math_docs = synthetic_math_docs(1_000_000, seed=seed)
    mixed = mix_docs(base_docs, math_docs, math_frac=math_frac, seed=seed)
    return pack.pack_corpus(
        mixed, tokenizer, out_dir, shard_tokens=shard_tokens, val_shards=val_shards
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the anneal-mix shard directory.")
    ap.add_argument("--base", required=True, help="base text parts dir (from download.py)")
    ap.add_argument("--tokenizer", required=True, help="trained tokenizer json (must match base)")
    ap.add_argument("--out", required=True, help="output shard directory")
    ap.add_argument("--math-frac", type=float, default=0.5)
    ap.add_argument("--source", default="synthetic", choices=["synthetic"], help="math source")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tok = MiniTokenizer.load(args.tokenizer)
    base_docs = download.read_parts(args.base)
    manifest = build_anneal_shards(
        base_docs,
        tok,
        args.out,
        math_frac=args.math_frac,
        seed=args.seed,
        shard_tokens=args.shard_tokens,
    )
    print(f"anneal mix: {manifest.total_tokens:,} tokens -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
