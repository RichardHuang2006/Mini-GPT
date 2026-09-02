"""Evaluation: held-out perplexity, ARC, MMLU, and HumanEval.

What this file teaches
    The three standard ways a small LM is scored:
      1. perplexity -- exp(mean next-token NLL) on held-out text: the model's
         effective branching factor, and the only metric with strong signal at
         this scale;
      2. multiple choice (ARC-Easy, ARC-Challenge, MMLU) -- score each choice
         by the log-likelihood the model assigns to it given the question,
         normalized by choice length so short answers don't win by brevity;
      3. code generation (HumanEval) -- run the model's completion against the
         task's unit tests in an isolated subprocess with a timeout, so a
         generated infinite loop counts as a failure instead of hanging the
         harness. pass@1 is the fraction of problems whose tests exit cleanly.

    Implemented support vs. executed runs: this file implements the evaluators
    and dataset loaders. It records numbers only in the results.json/results.md
    that a run you launch writes -- no scores are baked in anywhere.

Read first
    model.py (the forward pass), generate.py (sampling for HumanEval),
    data.py (the held-out val split perplexity draws from).

Inputs and outputs
    In:  checkpoints, a tokenizer, and per-metric task sets (a JSONL file,
         'hf' to download the real dataset, or 'sample' for tiny built-ins).
    Out: results.json and a Markdown table (results.md), one row per checkpoint.

Representative command (offline; add --arc-easy hf etc. to download real sets):
    python evaluate.py --tier nano --tokenizer data/tok.json --data data/packed \
        --ckpt base:out/nano/ckpt_final.pt --arc-easy sample --mmlu sample \
        --humaneval sample --out out/eval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tokenizer import MiniTokenizer

# Chance-level baselines, reported next to every score. None means "not an
# accuracy" (perplexity is a magnitude).
CHANCE_BASELINES: dict[str, float | None] = {
    "perplexity": None,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "mmlu": 0.25,
    "humaneval": 0.0,
}


# =============================================================================
# 1. Held-out perplexity
# =============================================================================

@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    windows: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
) -> float:
    """Token-weighted perplexity over [N, T+1] windows (inputs + shifted targets).

    perplexity = exp(total_nll / total_tokens). Windows must come from a split
    the model never trained on; the caller draws them from a val ShardSampler.
    """
    model.eval()
    w = torch.as_tensor(np.asarray(windows), dtype=torch.long)
    total_nll = 0.0
    total_tokens = 0
    for i in range(0, w.shape[0], batch_size):
        batch = w[i : i + batch_size].to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits, _ = model(x)  # [B, T, V]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum"
        )
        total_nll += float(nll)
        total_tokens += y.numel()
    return math.exp(total_nll / max(1, total_tokens))


def perplexity_from_split(
    model: nn.Module,
    data_dir: str | Path,
    *,
    context: int,
    device: torch.device | str = "cpu",
    n_windows: int = 256,
    seed: int = 0,
    split: str = "val",
) -> float:
    """Perplexity on n_windows sampled from the held-out split. The manifest's
    fixed train/val shard split guarantees these tokens were never trained on."""
    from data import ShardSampler

    sampler = ShardSampler(data_dir, context=context, split=split, seed=seed)
    return evaluate_perplexity(model, sampler.next_batch(n_windows), device=device)


# =============================================================================
# 2. Multiple choice (ARC, MMLU): length-normalized log-likelihood scoring
#
# Every question is a dict {"prompt": str, "choices": [str, ...], "answer": int}.
# =============================================================================

@torch.no_grad()
def _continuation_logprob(
    model: nn.Module,
    context_ids: Sequence[int],
    cont_ids: Sequence[int],
    *,
    device: torch.device | str,
) -> float:
    """Summed log-prob the model assigns to cont_ids following context_ids."""
    ids = list(context_ids) + list(cont_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)                              # [1, len(ids), V]
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    start = len(context_ids)
    total = 0.0
    for i, tok in enumerate(cont_ids):
        # Logits at position p predict the token at p+1, hence the -1.
        total += float(logp[start + i - 1, tok])
    return total


@torch.no_grad()
def evaluate_multiple_choice(
    model: nn.Module,
    tokenizer: MiniTokenizer,
    questions: Sequence[dict],
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Accuracy under length-normalized log-likelihood: each choice's summed
    log-prob is divided by its token count, and the argmax is the prediction."""
    model.eval()
    if not questions:
        return 0.0
    correct = 0
    for q in questions:
        ctx = tokenizer.encode(q["prompt"])
        scored = []
        for choice in q["choices"]:
            cont = tokenizer.encode(choice)
            if not cont:
                scored.append(-math.inf)
                continue
            scored.append(_continuation_logprob(model, ctx, cont, device=device) / len(cont))
        correct += int(int(np.argmax(scored)) == q["answer"])
    return correct / len(questions)


# =============================================================================
# 3. HumanEval: sandboxed pass@1
# =============================================================================

@dataclass
class HumanEvalProblem:
    prompt: str        # the function signature + docstring the model completes
    test: str          # defines `check(fn)`, asserting against the completion
    entry_point: str   # the function name to pass to check()


def run_code_sandbox(program: str, *, timeout: float = 5.0) -> bool:
    """Run `program` in an isolated subprocess; True iff it exits cleanly.

    Subprocess isolation means a crash, a failed assertion (non-zero exit), or
    an infinite loop (timeout) is contained -- the harness itself never hangs.
    """
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(program)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


@torch.no_grad()
def evaluate_humaneval(
    model: nn.Module,
    tokenizer: MiniTokenizer,
    problems: Sequence[HumanEvalProblem | dict],
    *,
    device: torch.device | str = "cpu",
    max_new_tokens: int = 256,
    timeout: float = 5.0,
) -> float:
    """pass@1: greedily complete each prompt, then execute prompt + completion
    + the task's check() in a sandbox. Score = fraction passing."""
    from generate import generate

    model.eval()
    probs = [p if isinstance(p, HumanEvalProblem) else HumanEvalProblem(**p) for p in problems]
    if not probs:
        return 0.0

    passed = 0
    for p in probs:
        prompt_ids = tokenizer.encode(p.prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = generate(model, idx, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id)
        completion = tokenizer.decode(out[0, len(prompt_ids):].tolist())
        program = f"{p.prompt}{completion}\n\n{p.test}\n\ncheck({p.entry_point})\n"
        if run_code_sandbox(program, timeout=timeout):
            passed += 1
    return passed / len(probs)


# =============================================================================
# 4. Orchestration + results.json + Markdown table
# =============================================================================

def evaluate(
    model: nn.Module,
    tokenizer: MiniTokenizer,
    *,
    perplexity_windows: np.ndarray | torch.Tensor | None = None,
    arc_easy: Sequence[dict] | None = None,
    arc_challenge: Sequence[dict] | None = None,
    mmlu: Sequence[dict] | None = None,
    humaneval: Sequence[HumanEvalProblem | dict] | None = None,
    device: torch.device | str = "cpu",
    humaneval_max_new_tokens: int = 256,
    humaneval_timeout: float = 5.0,
) -> dict[str, float]:
    """Score every metric whose inputs were supplied. Returns a flat dict."""
    results: dict[str, float] = {}
    if perplexity_windows is not None:
        results["perplexity"] = evaluate_perplexity(model, perplexity_windows, device=device)
    if arc_easy is not None:
        results["arc_easy"] = evaluate_multiple_choice(model, tokenizer, arc_easy, device=device)
    if arc_challenge is not None:
        results["arc_challenge"] = evaluate_multiple_choice(
            model, tokenizer, arc_challenge, device=device
        )
    if mmlu is not None:
        results["mmlu"] = evaluate_multiple_choice(model, tokenizer, mmlu, device=device)
    if humaneval is not None:
        results["humaneval"] = evaluate_humaneval(
            model, tokenizer, humaneval, device=device,
            max_new_tokens=humaneval_max_new_tokens, timeout=humaneval_timeout,
        )
    return results


def write_results(checkpoints: dict[str, dict[str, float]], path: str | Path) -> None:
    """Persist {checkpoint_name: {metric: value}} to results.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoints": checkpoints, "chance_baselines": CHANCE_BASELINES}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_results(path: str | Path) -> dict[str, dict[str, float]]:
    return json.loads(Path(path).read_text())["checkpoints"]


# Display order, label, and whether lower is better.
_METRICS: list[tuple[str, str, bool]] = [
    ("perplexity", "Perplexity", True),
    ("arc_easy", "ARC-Easy", False),
    ("arc_challenge", "ARC-Challenge", False),
    ("mmlu", "MMLU", False),
    ("humaneval", "HumanEval", False),
]

# base -> SFT -> GRPO is the training order; unknown names sort after, stably.
_ROW_ORDER = {"base": 0, "sft": 1, "grpo": 2}


def _header_label(key: str, label: str, lower_better: bool) -> str:
    arrow = " (lower is better)" if lower_better else ""
    chance = CHANCE_BASELINES.get(key)
    if chance is None:
        return f"{label}{arrow}"
    return f"{label} (chance {chance:.0%})"


def _fmt(key: str, value: float | None) -> str:
    if value is None:
        return "—"
    if key == "perplexity":
        return f"{value:.2f}"
    return f"{value:.1%}"


def format_tables(checkpoints: dict[str, dict[str, float]]) -> str:
    """Render the Markdown results table: rows are checkpoints in training
    order, columns are metrics labelled with their chance baselines."""
    present = [m for m in _METRICS if any(m[0] in v for v in checkpoints.values())]
    headers = ["Checkpoint"] + [_header_label(k, lbl, lb) for k, lbl, lb in present]

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for name in sorted(checkpoints, key=lambda n: (_ROW_ORDER.get(n, 99), n)):
        metrics = checkpoints[name]
        cells = [name] + [_fmt(k, metrics.get(k)) for k, _, _ in present]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# =============================================================================
# 5. Task-set loading: JSONL fixtures, tiny built-in samples, or the real
#    datasets from HuggingFace ('hf' -- a download).
# =============================================================================

# Tiny built-in samples: they carry no signal; they exist so every metric can
# be exercised end to end with zero downloads.
SAMPLE_MC = [
    {"prompt": "The sky is", "choices": [" blue", " loud"], "answer": 0},
    {"prompt": "Two plus two equals", "choices": [" four", " purple"], "answer": 0},
]
SAMPLE_HUMANEVAL = [
    {
        "prompt": "def add(a, b):\n    ",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    }
]


def load_arc(subset: str, *, split: str = "test", limit: int | None = None) -> list[dict]:
    """ARC from HuggingFace (allenai/ai2_arc; subset 'ARC-Easy' or
    'ARC-Challenge') -> the internal question format. Downloads on first use."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    out = []
    for row in ds:
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            continue
        out.append({
            "prompt": f"Question: {row['question']}\nAnswer:",
            "choices": [f" {t}" for t in row["choices"]["text"]],
            "answer": labels.index(row["answerKey"]),
        })
        if limit is not None and len(out) >= limit:
            break
    return out


def load_mmlu(*, split: str = "test", limit: int | None = None) -> list[dict]:
    """MMLU from HuggingFace (cais/mmlu, 'all') -> the internal question
    format. Downloads on first use."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)
    out = []
    for row in ds:
        out.append({
            "prompt": f"Question: {row['question']}\nAnswer:",
            "choices": [f" {t}" for t in row["choices"]],
            "answer": int(row["answer"]),
        })
        if limit is not None and len(out) >= limit:
            break
    return out


def load_humaneval(*, limit: int | None = None) -> list[dict]:
    """HumanEval from HuggingFace (openai/openai_humaneval): rows already carry
    prompt / test / entry_point. Downloads on first use."""
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for row in ds:
        out.append({
            "prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"],
        })
        if limit is not None and len(out) >= limit:
            break
    return out


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _resolve_taskset(spec: str | None, metric: str, limit: int | None) -> list[dict] | None:
    """Map a CLI flag value to a task set: None (skip) | 'sample' (built-in) |
    'hf' (download the real dataset) | a JSONL path."""
    if spec is None:
        return None
    if spec == "sample":
        return SAMPLE_HUMANEVAL if metric == "humaneval" else SAMPLE_MC
    if spec == "hf":
        if metric == "arc_easy":
            return load_arc("ARC-Easy", limit=limit)
        if metric == "arc_challenge":
            return load_arc("ARC-Challenge", limit=limit)
        if metric == "mmlu":
            return load_mmlu(limit=limit)
        return load_humaneval(limit=limit)
    return _load_jsonl(spec)


# =============================================================================
# 6. Command-line interface
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    from config import get_config
    from model import MiniGPT
    from train import load_model_weights, seed_everything

    taskset_help = "JSONL path, 'sample' (tiny built-in), or 'hf' (download the real set)"
    ap = argparse.ArgumentParser(
        description="Score checkpoints on perplexity / ARC / MMLU / HumanEval."
    )
    ap.add_argument("--tier", default="mini")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", default=None, help="packed shard dir for held-out perplexity")
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME:PATH",
                    help="checkpoint to score (repeatable); NAME is the row label")
    ap.add_argument("--arc-easy", default=None, help=taskset_help)
    ap.add_argument("--arc-challenge", default=None, help=taskset_help)
    ap.add_argument("--mmlu", default=None, help=taskset_help)
    ap.add_argument("--humaneval", default=None, help=taskset_help)
    ap.add_argument("--limit", type=int, default=None, help="cap per-metric question count")
    ap.add_argument("--n-ppl-windows", type=int, default=256)
    ap.add_argument("--humaneval-timeout", type=float, default=5.0)
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = get_config(args.tier)
    seed_everything(cfg.seed)
    tok = MiniTokenizer.load(args.tokenizer)

    arc_easy = _resolve_taskset(args.arc_easy, "arc_easy", args.limit)
    arc_challenge = _resolve_taskset(args.arc_challenge, "arc_challenge", args.limit)
    mmlu = _resolve_taskset(args.mmlu, "mmlu", args.limit)
    humaneval = _resolve_taskset(args.humaneval, "humaneval", args.limit)

    windows = None
    if args.data:
        from data import ShardSampler

        sampler = ShardSampler(args.data, context=cfg.context, split="val", seed=cfg.seed)
        windows = sampler.next_batch(args.n_ppl_windows)

    ckpts = args.ckpt or ["base:"]
    checkpoints: dict[str, dict[str, float]] = {}
    for spec in ckpts:
        name, _, path = spec.partition(":")
        model = MiniGPT(cfg).to(args.device)
        if path:
            load_model_weights(model, path, device=args.device)
        checkpoints[name] = evaluate(
            model, tok,
            perplexity_windows=windows,
            arc_easy=arc_easy, arc_challenge=arc_challenge, mmlu=mmlu, humaneval=humaneval,
            device=args.device, humaneval_timeout=args.humaneval_timeout,
        )
        print(f"scored {name}: {checkpoints[name]}")

    out = Path(args.out)
    write_results(checkpoints, out / "results.json")
    table = format_tables(checkpoints)
    (out / "results.md").write_text(table)
    print(f"\nwrote {out}/results.json and {out}/results.md\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
