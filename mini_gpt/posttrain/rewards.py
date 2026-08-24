"""GRPO reward functions.

A reward is a correctness term (exact match on the extracted final answer) plus
lightweight format terms (the completion terminated, produced a parseable answer,
stayed within length). Format shaping supplies partial signal before the model is
ever correct, which keeps a countdown run from stalling at zero reward in its
first steps.

Three tasks share the ``RewardResult`` interface:

* ``arithmetic_reward`` -- the final integer must equal the target.
* ``countdown_reward`` -- an expression using the given operands (each at most
  once) that evaluates to the target; the RL task with usable signal at 39M.
* ``gsm8k_reward`` -- the same final-integer match against a gold answer. Wired
  for pipeline parity; correctness is near-zero at `mini`, since a model solving
  ~0% of grade-school word problems gives GRPO no gradient.

The final answer comes from a ``#### <answer>`` delimiter when present (GSM8K
convention), else the last integer in the text, so a model that learns the
delimiter is rewarded while a bare number still parses.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass

_HASH_ANSWER = re.compile(r"####\s*(-?\d+)")
_INT = re.compile(r"-?\d+")
_EXPR = re.compile(r"[0-9]+(?:\s*[-+*/()]\s*[0-9]+)+")


@dataclass
class RewardResult:
    """A reward broken into its correctness and format components (for logging)."""

    total: float
    correct: float
    format: float


# ---------------------------------------------------------------- extraction

def extract_final_int(text: str) -> int | None:
    """The answer after a ``####`` delimiter, else the last integer in ``text``."""
    m = _HASH_ANSWER.search(text)
    if m:
        return int(m.group(1))
    ints = _INT.findall(text)
    return int(ints[-1]) if ints else None


def _format_score(text: str, *, terminated: bool, n_new_tokens: int, max_new_tokens: int) -> float:
    """Partial credit in [0, 1] for well-formedness, independent of correctness."""
    parseable = 1.0 if extract_final_int(text) is not None else 0.0
    term = 1.0 if terminated else 0.0
    within = 1.0 if 0 < n_new_tokens <= max_new_tokens else 0.0
    return (parseable + term + within) / 3.0


# ----------------------------------------------------------------- arithmetic

def arithmetic_reward(
    text: str,
    *,
    target: int,
    terminated: bool,
    n_new_tokens: int,
    max_new_tokens: int,
    w_correct: float = 1.0,
    w_format: float = 0.5,
) -> RewardResult:
    pred = extract_final_int(text)
    correct = 1.0 if (pred is not None and pred == target) else 0.0
    fmt = _format_score(
        text, terminated=terminated, n_new_tokens=n_new_tokens, max_new_tokens=max_new_tokens
    )
    return RewardResult(total=w_correct * correct + w_format * fmt, correct=correct, format=fmt)


# ------------------------------------------------------------------ countdown

def _safe_eval(expr: str) -> float | None:
    """Evaluate a ``+ - * / ()`` integer expression, or ``None`` if invalid."""
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            return a / b if b != 0 else float("nan")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        raise ValueError("disallowed node")

    try:
        return ev(node)
    except (ValueError, ZeroDivisionError):
        return None


def _operands(expr: str) -> list[int]:
    return [int(x) for x in _INT.findall(expr)]


def countdown_reward(
    text: str,
    *,
    numbers: list[int],
    target: int,
    terminated: bool,
    n_new_tokens: int,
    max_new_tokens: int,
    w_correct: float = 1.0,
    w_format: float = 0.5,
) -> RewardResult:
    """Reward reaching ``target`` from ``numbers`` (each used at most once)."""
    m = _EXPR.search(text)
    correct = 0.0
    if m:
        expr = m.group(0)
        value = _safe_eval(expr)
        used = Counter(_operands(expr))
        allowed = Counter(numbers)
        uses_valid = all(used[k] <= allowed.get(k, 0) for k in used)
        if value is not None and uses_valid and abs(value - target) < 1e-6:
            correct = 1.0
    fmt = _format_score(
        text, terminated=terminated, n_new_tokens=n_new_tokens, max_new_tokens=max_new_tokens
    )
    return RewardResult(total=w_correct * correct + w_format * fmt, correct=correct, format=fmt)


# ---------------------------------------------------------------------- gsm8k

def gsm8k_reward(
    text: str,
    *,
    gold: int,
    terminated: bool,
    n_new_tokens: int,
    max_new_tokens: int,
    w_correct: float = 1.0,
    w_format: float = 0.5,
) -> RewardResult:
    """Same final-integer match as arithmetic; near-zero correctness at `mini`.

    Kept identical in shape to the other tasks so the harness is complete for the
    `small` tier.
    """
    return arithmetic_reward(
        text,
        target=gold,
        terminated=terminated,
        n_new_tokens=n_new_tokens,
        max_new_tokens=max_new_tokens,
        w_correct=w_correct,
        w_format=w_format,
    )
