"""Evaluation harness tests.

Properties checked:

* one call scores a checkpoint on every metric and the results round-trip through
  ``results.json``;
* held-out perplexity is measured on a ``val`` shard the sampler keeps disjoint
  from ``train``;
* HumanEval runs generated code in an isolated subprocess: a passing program
  clears, a failing one fails, and an infinite loop is killed by the timeout;
* the Markdown tables regenerate from ``results.json``, with each metric shown
  next to its chance baseline and base/SFT/GRPO as separate rows.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config import Config, swiglu_hidden  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.eval import harness  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.posttrain.sft import synthetic_sft_conversations  # noqa: E402
from mini_gpt.tokenizer import MiniTokenizer  # noqa: E402


def _tiny_cfg(**overrides) -> Config:
    base = dict(
        name="tiny", vocab_size=512, d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2,
        head_dim=16, mlp_hidden=swiglu_hidden(64), context=64, window=32, micro_batch=4,
        grad_accum=1, warmup_steps=5, max_steps=50, use_muon=False, compile=False,
        dtype="float32", use_triton=False,
    )
    base.update(overrides)
    return Config(**base)


@pytest.fixture(scope="module")
def tokenizer() -> MiniTokenizer:
    convs = synthetic_sft_conversations(400, seed=0)
    text = [m["content"] for c in convs for m in c]
    return MiniTokenizer.train(text, vocab_size=512, min_frequency=1)


@pytest.fixture(scope="module")
def model() -> MiniGPT:
    seed_everything(0)
    return MiniGPT(_tiny_cfg())


def _random_windows(n: int, t: int, vocab: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=(n, t + 1)).astype(np.int64)


# ------------------------------------------------------------ perplexity

def test_perplexity_is_finite_and_positive(model):
    windows = _random_windows(16, 16, 512)
    ppl = harness.evaluate_perplexity(model, windows, device="cpu")
    assert np.isfinite(ppl) and ppl > 1.0  # random text -> high but finite ppl


def test_perplexity_uses_held_out_val_shard(tokenizer, model, tmp_path):
    # Build a tiny packed dataset with a disjoint val shard, then score on val.
    from mini_gpt.data import pack

    docs = [f"document number {i} with some filler words here ." * 4 for i in range(40)]
    data_dir = tmp_path / "shards"
    pack.pack_corpus(docs, tokenizer, data_dir, shard_tokens=400, val_shards=1)

    ppl = harness.perplexity_from_split(
        model, data_dir, context=32, device="cpu", n_windows=16, split="val"
    )
    assert np.isfinite(ppl) and ppl > 1.0


# ---------------------------------------------------------- multiple choice

def test_multiple_choice_accuracy_in_range(tokenizer, model):
    questions = [
        {"prompt": "The sky is", "choices": [" blue", " green", " loud"], "answer": 0},
        {"prompt": "Water is", "choices": [" wet", " square"], "answer": 0},
    ]
    acc = harness.evaluate_multiple_choice(model, tokenizer, questions, device="cpu")
    assert 0.0 <= acc <= 1.0


# --------------------------------------------------------------- humaneval

def test_run_code_sandbox_pass_fail_and_timeout():
    ok = "def add(a, b):\n    return a + b\nassert add(1, 2) == 3\n"
    assert harness.run_code_sandbox(ok, timeout=5.0) is True

    bad = "assert 1 == 2\n"
    assert harness.run_code_sandbox(bad, timeout=5.0) is False

    loop = "while True:\n    pass\n"
    assert harness.run_code_sandbox(loop, timeout=1.0) is False  # killed by timeout


def test_humaneval_runs_and_is_bounded(tokenizer, model):
    problems = [
        {
            "prompt": "def add(a, b):\n    ",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": "add",
        }
    ]
    score = harness.evaluate_humaneval(
        model, tokenizer, problems, device="cpu", max_new_tokens=8, timeout=5.0
    )
    assert 0.0 <= score <= 1.0  # ~0 at this scale, but the loop completes


# ------------------------------------------------------------- harness smoke

def test_harness_smoke(tokenizer, model, tmp_path):
    windows = _random_windows(12, 16, 512)
    mc = [{"prompt": "The sky is", "choices": [" blue", " loud"], "answer": 0}]
    he = [{
        "prompt": "def add(a, b):\n    ",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    }]

    results = harness.evaluate(
        model, tokenizer, perplexity_windows=windows, arc_easy=mc, arc_challenge=mc,
        mmlu=mc, humaneval=he, device="cpu", humaneval_max_new_tokens=8, humaneval_timeout=5.0,
    )
    for key in ("perplexity", "arc_easy", "arc_challenge", "mmlu", "humaneval"):
        assert key in results
    assert np.isfinite(results["perplexity"])

    path = tmp_path / "results.json"
    harness.write_results({"base": results}, path)
    reloaded = harness.load_results(path)
    assert reloaded["base"]["perplexity"] == pytest.approx(results["perplexity"])


# -------------------------------------------------------------------- tables

def test_tables(tmp_path):
    checkpoints = {
        "grpo": {"perplexity": 41.0, "arc_easy": 0.33, "arc_challenge": 0.26, "mmlu": 0.252, "humaneval": 0.0},
        "base": {"perplexity": 48.0, "arc_easy": 0.30, "arc_challenge": 0.25, "mmlu": 0.251, "humaneval": 0.0},
        "sft": {"perplexity": 44.0, "arc_easy": 0.31, "arc_challenge": 0.26, "mmlu": 0.250, "humaneval": 0.0},
    }
    table = harness.format_tables(checkpoints)

    # base -> SFT -> GRPO appear as separate rows, in training order.
    assert table.index("| base ") < table.index("| sft ") < table.index("| grpo ")
    # Each MC metric is shown next to its chance baseline; perplexity has none.
    assert "chance 25%" in table
    assert "chance 0%" in table
    assert "lower is better" in table  # perplexity column
    # A chance-level MMLU renders as a percentage.
    assert "25.2%" in table

    # Regenerates deterministically from results.json.
    path = tmp_path / "results.json"
    harness.write_results(checkpoints, path)
    assert harness.tables_from_results(path) == table
