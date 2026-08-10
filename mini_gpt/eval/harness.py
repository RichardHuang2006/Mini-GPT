"""Evaluation harness: perplexity / ARC / MMLU / HumanEval.

One command scores a checkpoint on every metric and writes ``results.json`` plus
Markdown tables. The design is explicit about which numbers carry information at
39M: held-out **perplexity** and **ARC-Easy** are the only two with real signal;
MMLU is a chance-level regression tripwire and HumanEval is effectively zero.
Every score is printed next to its chance baseline so a 25% MMLU is never
mistaken for a result.

Scoring methods:

* **Perplexity** -- token-weighted ``exp(mean NLL)`` over windows drawn from a
  shard split the model never trained on (the fixed train/val split guarantees
  this; see ``mini_gpt.data.sampler``).
* **Multiple choice** (ARC, MMLU) -- each choice scored by its **length-normalized
  log-likelihood** given the question; the argmax is the prediction.
* **HumanEval** -- the model's completion is executed in an **isolated subprocess
  with a timeout** against the task's unit test; pass@1 is the fraction that exit
  cleanly. Sandboxing keeps a generated infinite loop or crash from taking the
  harness down with it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mini_gpt.tokenizer import MiniTokenizer

# Chance-level baselines. ``None`` means "no baseline"
# (perplexity is a magnitude, not an accuracy).
CHANCE_BASELINES: dict[str, float | None] = {
    "perplexity": None,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "mmlu": 0.25,
    "humaneval": 0.0,
}


# ============================================================ perplexity

@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    windows: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
) -> float:
    """Token-weighted perplexity over ``[N, T+1]`` windows (input+shifted target).

    Windows must come from a split the model never trained on -- the caller draws
    them from a validation ``ShardSampler``.
    """
    model.eval()
    w = torch.as_tensor(np.asarray(windows), dtype=torch.long)
    total_nll = 0.0
    total_tokens = 0
    for i in range(0, w.shape[0], batch_size):
        batch = w[i : i + batch_size].to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits, _ = model(x)  # targets=None -> full logits
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
    """Perplexity on ``n_windows`` sampled from the held-out ``split``."""
    from mini_gpt.data.sampler import ShardSampler

    sampler = ShardSampler(data_dir, context=context, split=split, seed=seed)
    windows = sampler.next_batch(n_windows)
    return evaluate_perplexity(model, windows, device=device)


# ==================================================== multiple choice

@torch.no_grad()
def _continuation_logprob(
    model: nn.Module,
    context_ids: Sequence[int],
    cont_ids: Sequence[int],
    *,
    device: torch.device | str,
) -> float:
    """Summed log-prob the model assigns to ``cont_ids`` following ``context_ids``."""
    ids = list(context_ids) + list(cont_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    start = len(context_ids)
    total = 0.0
    for i, tok in enumerate(cont_ids):
        # logits at position p predict the token at p+1.
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
    """Accuracy under length-normalized log-likelihood scoring.

    Each question is ``{"prompt": str, "choices": [str, ...], "answer": int}``.
    Length normalization keeps a short choice from winning purely for being short.
    """
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
            lp = _continuation_logprob(model, ctx, cont, device=device)
            scored.append(lp / len(cont))  # length-normalized
        pred = int(np.argmax(scored))
        correct += int(pred == q["answer"])
    return correct / len(questions)


# ============================================================= humaneval

@dataclass
class HumanEvalProblem:
    prompt: str        # the function signature + docstring the model completes
    test: str          # defines `def check(candidate): ...` with asserts
    entry_point: str   # the function name to pass to check()


def run_code_sandbox(program: str, *, timeout: float = 5.0) -> bool:
    """Run ``program`` in an isolated subprocess; ``True`` iff it exits cleanly.

    A generated infinite loop hits the timeout and counts as a failure rather than
    hanging the harness; a crash or failed assertion is a non-zero exit.
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
    """pass@1 over ``problems``; ~0% at `mini` but the loop is wired end-to-end."""
    from mini_gpt.generate import generate

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


# ============================================================ orchestrate

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
    """Score every metric for which inputs were supplied. Returns a flat dict."""
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
            model,
            tokenizer,
            humaneval,
            device=device,
            max_new_tokens=humaneval_max_new_tokens,
            timeout=humaneval_timeout,
        )
    return results


def write_results(checkpoints: dict[str, dict[str, float]], path: str | Path) -> None:
    """Persist ``{name: {metric: value}}`` to ``results.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoints": checkpoints, "chance_baselines": CHANCE_BASELINES}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_results(path: str | Path) -> dict[str, dict[str, float]]:
    payload = json.loads(Path(path).read_text())
    return payload["checkpoints"]


# ===================================================================== tables

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
    """Render the Markdown results table with chance baselines in the header.

    Rows are checkpoints in training order (base, SFT, GRPO); columns are metrics,
    each labelled with its chance baseline so a chance-level score reads as such.
    """
    present = [m for m in _METRICS if any(m[0] in v for v in checkpoints.values())]
    headers = ["Checkpoint"] + [_header_label(k, lbl, lb) for k, lbl, lb in present]

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]

    names = sorted(checkpoints, key=lambda n: (_ROW_ORDER.get(n, 99), n))
    for name in names:
        metrics = checkpoints[name]
        cells = [name] + [_fmt(k, metrics.get(k)) for k, _, _ in present]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def tables_from_results(path: str | Path) -> str:
    """Regenerate the Markdown table from a ``results.json``."""
    return format_tables(load_results(path))
