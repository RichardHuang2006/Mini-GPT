"""Evaluation entry point.

Scores one or more checkpoints on every metric and writes ``results.json`` plus a
Markdown table. Perplexity is measured on the held-out ``val`` split of the packed
data, a shard the model never trained on. The multiple-choice and HumanEval task
sets are read from JSONL when given; otherwise tiny built-in samples keep the
command runnable end to end.

    python scripts/eval.py --tier mini --tokenizer out/tok.json --data out/shards \
        --ckpt base:out/mini/ckpt_final.pt --ckpt sft:out/mini_sft/ckpt_sft.pt \
        --ckpt grpo:out/mini_grpo/ckpt_grpo.pt --out out/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import get_config  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.eval import harness  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.tokenizer import MiniTokenizer  # noqa: E402

# Built-in samples so the harness runs without external downloads. They carry no
# signal; they exist so every metric is scored end to end.
_SAMPLE_MC = [
    {"prompt": "The sky is", "choices": [" blue", " loud"], "answer": 0},
    {"prompt": "Two plus two equals", "choices": [" four", " purple"], "answer": 0},
]
_SAMPLE_HUMANEVAL = [
    {
        "prompt": "def add(a, b):\n    ",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    }
]


def _load_jsonl(path: str | None):
    if not path:
        return None
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_weights(model: MiniGPT, path: str, device: str) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "trainer" in payload:
        state = payload["trainer"]["model"]
    elif "model" in payload:
        state = payload["model"]
    else:
        state = payload
    model.load_state_dict(state)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score checkpoints (ppl/ARC/MMLU/HumanEval).")
    ap.add_argument("--tier", default="mini")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", default=None, help="packed-shard dir for held-out perplexity")
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME:PATH",
                    help="checkpoint to score (repeatable); NAME is a row label")
    ap.add_argument("--arc-easy", default=None, help="ARC-Easy JSONL")
    ap.add_argument("--arc-challenge", default=None, help="ARC-Challenge JSONL")
    ap.add_argument("--mmlu", default=None, help="MMLU JSONL")
    ap.add_argument("--humaneval", default=None, help="HumanEval JSONL")
    ap.add_argument("--n-ppl-windows", type=int, default=256)
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = get_config(args.tier)
    seed_everything(cfg.seed)
    tok = MiniTokenizer.load(args.tokenizer)

    arc_easy = _load_jsonl(args.arc_easy) or _SAMPLE_MC
    arc_challenge = _load_jsonl(args.arc_challenge) or _SAMPLE_MC
    mmlu = _load_jsonl(args.mmlu) or _SAMPLE_MC
    humaneval = _load_jsonl(args.humaneval) or _SAMPLE_HUMANEVAL

    windows = None
    if args.data:
        from mini_gpt.data.sampler import ShardSampler

        sampler = ShardSampler(args.data, context=cfg.context, split="val", seed=cfg.seed)
        windows = sampler.next_batch(args.n_ppl_windows)

    ckpts = args.ckpt or ["base:"]
    checkpoints: dict[str, dict[str, float]] = {}
    for spec in ckpts:
        name, _, path = spec.partition(":")
        model = MiniGPT(cfg).to(args.device)
        if path:
            _load_weights(model, path, args.device)
        checkpoints[name] = harness.evaluate(
            model,
            tok,
            perplexity_windows=windows,
            arc_easy=arc_easy,
            arc_challenge=arc_challenge,
            mmlu=mmlu,
            humaneval=humaneval,
            device=args.device,
        )
        print(f"scored {name}: {checkpoints[name]}")

    out = Path(args.out)
    harness.write_results(checkpoints, out / "results.json")
    table = harness.format_tables(checkpoints)
    (out / "results.md").write_text(table)
    print(f"\nwrote {out}/results.json and {out}/results.md\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
