# Mini-GPT

A from-scratch GPT pipeline — tokenizer, pretraining, Triton kernels, SFT, GRPO,
and evaluation — organized as an **executable textbook**: nine Python modules in
one `mini_gpt/` package plus a single test suite in `tests/`, each module
teaching one major concept, readable in order. The layout matches the sibling
Mini-* repositories (a source package beside a `tests/` folder).

## 1. What this implements

- A **32,768-vocabulary byte-level BPE tokenizer** with conversation and tool
  special tokens (`<|pad|> <|bos|> <|eos|> <|system|> <|user|> <|assistant|>
  <|tool_call|> <|tool_result|>`) at fixed IDs.
- Data preparation for **NVIDIA ClimbMix** (the official
  `nvidia/Nemotron-ClimbMix` token release, GPT-2 detokenized, and the
  `OptimalScale/ClimbMix` raw-text mirror) into **packed uint16 token shards**
  with reproducible batch sampling, up to a **2,048-token context**.
- A **grouped-query-attention Transformer** with **RoPE**, per-head **QK-norm**,
  **RMSNorm**, a **SwiGLU MLP**, and **sliding-window attention** with periodic
  full-context layers — tensor shapes annotated at every step.
- Pretraining with **AdamW and Muon** (Newton–Schulz orthogonalized momentum),
  warmup + cosine decay, gradient accumulation, and bf16 autocast.
- Four fused **Triton kernels** (RMSNorm, RoPE, SwiGLU, chunked cross-entropy)
  differential-tested against eager PyTorch references, with CPU fallback.
- Post-training: **supervised fine-tuning** (assistant-only loss over packed,
  attention-isolated conversations) and **GRPO** with GSM8K rewards.
- Scoring: **perplexity, ARC-Easy, ARC-Challenge, MMLU, HumanEval**.

Everything above is *implemented and runnable*; this README marks which commands
download datasets or cost real compute. **No benchmark scores or completed
training runs are claimed anywhere in this repository** — results exist only in
the `results.json` files that runs you launch produce.

## 2. End-to-end pipeline

```
text source (ClimbMix or synthetic)
        │  data.py: fetch -> train 32K BPE -> pack uint16 shards (train/val)
        ▼
packed shards + tok.json
        │  train.py: AdamW + Muon, warmup+cosine, grad accumulation,
        │            bf16 autocast, val perplexity, checkpoints
        ▼
base checkpoint ──── generate.py (greedy / temperature / top-k sampling)
        │  posttrain.py sft: chat template, assistant-only loss, packed convs
        ▼
instruct checkpoint
        │  posttrain.py grpo: group sampling, group-relative advantages,
        │                     clipped update, GSM8K reward
        ▼
final checkpoint ─── evaluate.py: perplexity / ARC / MMLU / HumanEval
```

## 3. Repository structure and reading order

Each module teaches one concept and carries a short docstring saying what it
does.

| # | File | Responsibility |
| --- | --- | --- |
| 1 | `mini_gpt/config.py` | The `Config` dataclass and the `nano` / `mini` / `small` presets. |
| 2 | `mini_gpt/tokenizer.py` | 32K byte-level BPE with fixed-ID conversation/tool special tokens. |
| 3 | `mini_gpt/data.py` | Fetch text (ClimbMix or synthetic), pack uint16 shards, sample windows. |
| 4 | `mini_gpt/model.py` | The eager Transformer: GQA, RoPE, QK-norm, RMSNorm, SwiGLU, windows. |
| 5 | `mini_gpt/train.py` | AdamW + Muon, LR schedule, training loop, checkpoints, seeding. |
| 6 | `mini_gpt/generate.py` | Greedy / temperature / top-k sampling with context enforcement. |
| 7 | `mini_gpt/kernels.py` | Triton kernels + eager references + automatic fallback. |
| 8 | `mini_gpt/posttrain.py` | Chat template, SFT, GSM8K rewards, GRPO, `sft`/`grpo` CLIs. |
| 9 | `mini_gpt/evaluate.py` | Perplexity, ARC/MMLU multiple choice, sandboxed HumanEval, reports. |
| 10 | `tests/test_minigpt.py` | The whole test suite, ordered like the reading order. |

## 4. Model architecture and tensor shapes

Decoder-only causal Transformer, pre-norm residual wiring, tied input/output
embeddings. Grouped-query attention uses fewer KV heads than query heads;
sliding-window layers see only the last `window` tokens while every
`full_attn_every`-th layer keeps full context. Shape conventions used
throughout `model.py`:

| Tensor | Shape |
| --- | --- |
| Token IDs | `[batch, sequence]` |
| Embeddings | `[batch, sequence, model_dimension]` |
| Queries | `[batch, query_heads, sequence, head_dimension]` |
| Keys / values | `[batch, key_value_heads, sequence, head_dimension]` |
| Attention scores | `[batch, query_heads, query_sequence, key_sequence]` |
| Logits | `[batch, sequence, vocabulary_size]` |

Tier presets (all share one `Config`; parameter counts are computed by
`Config.param_count()` and asserted against the built model in the tests):

| Tier | Params (total / non-emb) | `d_model` | Layers | Q/KV heads | Context | Window | Tokens/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nano` | 12.5M / 4.1M | 256 | 6 | 4 / 2 | 512 | 256 | 131K |
| `mini` | 39.3M / 22.6M | 512 | 8 | 8 / 2 | 1024 | 256 | 524K |
| `small` | 100.7M / 75.5M | 768 | 12 | 12 / 4 | **2048** | 256 | 1.05M |

`nano` is the CPU-friendly smoke tier; `small` is the largest configuration,
with the full 2,048-token context.

## 5. Tokenizer and special tokens

`tokenizer.py` trains a byte-level BPE (HuggingFace `tokenizers` Rust backend)
with a 32,768 vocabulary — byte-level, so *every* input round-trips exactly:
no `[UNK]`, no out-of-vocabulary failure. Eight special tokens are registered
before training so they hold fixed IDs 0–7 forever:

```
0 <|pad|>   1 <|bos|>   2 <|eos|>   3 <|system|>
4 <|user|>  5 <|assistant|>  6 <|tool_call|>  7 <|tool_result|>
```

The tokenizer's `fingerprint()` (a content hash) is recorded in every packed
shard directory's manifest, so pairing shards with the wrong tokenizer fails
loudly instead of training on garbage.

## 6. Preparing synthetic data (offline, seconds)

One-time setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m mini_gpt.data --source synthetic --parts 2 --docs-per-part 2000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 100000
```

Writes jsonl text parts, trains the BPE, and packs uint16 shards with one
held-out validation shard. Fully deterministic and network-free.

## 7. Preparing ClimbMix-derived data

> **Downloads a large dataset from HuggingFace.** Start small to verify the
> path, then scale `--parts` up.

```bash
# small first taste (~1000 documents from the raw-text mirror):
python -m mini_gpt.data --source hf-raw --parts 1 --docs-per-part 1000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 1000000

# EXPENSIVE: a real pretraining corpus (tens of GB, hours of download):
python -m mini_gpt.data --source hf-raw --parts 64 --docs-per-part 20000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 50000000
```

`--source hf-raw` streams the `OptimalScale/ClimbMix` raw-text mirror;
`--source hf-tokens` streams NVIDIA's official `nvidia/Nemotron-ClimbMix`
token-ID release and detokenizes it with GPT-2 (needs the optional `tiktoken`
package). Part-writing is idempotent: an interrupted fetch continues where it
stopped.

## 8. Pretraining with AdamW and Muon

Parameters are split by shape: **Muon** (momentum orthogonalized by
Newton–Schulz iteration, `mini_gpt/train.py:zeropower_via_newtonschulz5`)
updates the 2D hidden matrices; **AdamW** updates embeddings and norm gains.
`--optimizer adamw` gives the pure-AdamW baseline. Warmup then cosine decay;
gradient accumulation; bf16 autocast on CUDA; gradient clipping.

```bash
# CPU smoke run (minutes):
python -m mini_gpt.train --tier nano --data data/packed --out out/nano \
    --steps 30 --micro-batch 4 --grad-accum 2 --device cpu --no-compile

# EXPENSIVE: CUDA training with periodic held-out perplexity:
python -m mini_gpt.train --tier mini --data data/packed --out out/mini --eval-every 1000

# continue from a checkpoint (same data stream and LR schedule):
python -m mini_gpt.train --tier mini --data data/packed --out out/mini \
    --resume out/mini/ckpt_1000.pt

# pure-AdamW baseline for an A/B comparison:
python -m mini_gpt.train --tier mini --data data/packed --out out/mini_adamw --optimizer adamw
```

Checkpoints hold model weights, both optimizers' state, the scheduler, the
data-sampler position, and the step counter.

## 9. Triton kernels and eager fallbacks

`kernels.py` contains four operations, each with an eager PyTorch reference,
a fused implementation, and one dispatch rule — *Triton kernel iff the input is
on CUDA and Triton imported; eager reference otherwise* — so `use_triton=True`
is safe on a CPU-only machine:

- **RMSNorm** — one-pass fused forward, hand-derived analytic backward.
- **RoPE** — the rotation as one elementwise kernel; the backward is the same
  kernel with the sine sign flipped.
- **SwiGLU** — fused `SiLU(a) * b` gating, forward and backward.
- **Chunked cross-entropy** — streams the 32K vocabulary in tiles with an
  online softmax so the full `[batch·seq, vocab]` logits tensor is never
  materialized; exact to `F.cross_entropy` for any chunk size. (Pure PyTorch:
  the win is memory, not a faster kernel.)

Forward values *and* backward gradients are compared against the references on
CUDA by the test suite, and the chunked-CE peak-memory property is asserted
directly.

## 10. Text generation

```bash
python -m mini_gpt.generate --ckpt out/nano/ckpt_final.pt --tokenizer data/tok.json \
    --tier nano --prompt "the quick brown" --max-new-tokens 40 \
    --temperature 0.8 --top-k 40 --seed 0

# chat-formatted prompt through the same template SFT trains on:
python -m mini_gpt.generate --ckpt out/nano_sft/ckpt_sft.pt --tokenizer data/tok.json \
    --tier nano --chat "What is 2 + 3?" --max-new-tokens 16
```

Greedy (`--temperature 0`), temperature, and top-k sampling; stops at
`<|eos|>`; seeded sampling is reproducible; the loop feeds the model at most
its trained context (2,048 tokens for `small`), cropping older tokens.

## 11. Supervised fine-tuning

`posttrain.py` renders conversations through one chat template
(`<|role|> content <|eos|>` per turn), builds a loss mask that covers **only
assistant-authored tokens**, and packs several conversations per row with
segment IDs — the attention mask blocks any lookback across conversation
boundaries.

```bash
# offline smoke run on synthetic arithmetic conversations (CPU, ~a minute):
python -m mini_gpt.posttrain sft --tier nano --tokenizer data/tok.json \
    --init out/nano/ckpt_final.pt --data synthetic --out out/nano_sft --steps 20

# real chat data: a JSONL of {"messages": [{"role": ..., "content": ...}]}
python -m mini_gpt.posttrain sft --tier mini --tokenizer data/tok.json \
    --init out/mini/ckpt_final.pt --data chats.jsonl --out out/mini_sft --steps 2000
```

## 12. GRPO on GSM8K

GRPO samples a *group* of completions per prompt, scores each with a reward,
and normalizes rewards **within the group** (mean as baseline, std as scale) —
no learned critic. The update is the clipped PPO surrogate over completion
tokens only. A group whose rewards are all equal has zero advantage and
produces exactly zero gradient. The GSM8K reward extracts the final integer
(preferring the `#### N` delimiter) and adds small format-shaping terms.

```bash
# GRPO on GSM8K (DOWNLOADS openai/gsm8k; EXPENSIVE at real scale):
python -m mini_gpt.posttrain grpo --tier mini --tokenizer data/tok.json \
    --init out/mini_sft/ckpt_sft.pt --task gsm8k --out out/mini_grpo --steps 500

# offline smoke run on the arithmetic task (no downloads, CPU-sized):
python -m mini_gpt.posttrain grpo --tier nano --tokenizer data/tok.json \
    --init out/nano_sft/ckpt_sft.pt --task arithmetic --out out/nano_grpo \
    --steps 10 --group-size 4 --prompts-per-step 2 --max-new-tokens 8
```

Mean reward per step is logged; the checkpoint stores the reward history. At
these model scales GSM8K correctness is expected to be near zero — the
machinery is what is demonstrated.

## 13. Perplexity, ARC, MMLU, and HumanEval

- **Perplexity** — `exp(mean NLL)` over windows from the held-out `val` shard
  split (also logged in-loop by `train.py --eval-every`).
- **ARC-Easy / ARC-Challenge / MMLU** — each choice scored by its
  length-normalized log-likelihood given the question; argmax is the
  prediction; chance is 25%.
- **HumanEval** — completions execute in an isolated subprocess with a timeout
  against the task's unit tests; pass@1. An infinite loop times out and counts
  as a failure instead of hanging the harness.

Each task-set flag takes a JSONL path, `sample` (tiny built-in fixtures, no
downloads), or `hf` (**downloads the real dataset**).

```bash
# perplexity only (offline, needs the packed val split):
python -m mini_gpt.evaluate --tier nano --tokenizer data/tok.json --data data/packed \
    --ckpt base:out/nano/ckpt_final.pt --out out/eval

# ARC (downloads allenai/ai2_arc):
python -m mini_gpt.evaluate --tier mini --tokenizer data/tok.json \
    --ckpt base:out/mini/ckpt_final.pt --arc-easy hf --arc-challenge hf --out out/eval

# MMLU (downloads cais/mmlu; --limit caps question count):
python -m mini_gpt.evaluate --tier mini --tokenizer data/tok.json \
    --ckpt base:out/mini/ckpt_final.pt --mmlu hf --limit 500 --out out/eval

# HumanEval (downloads openai/openai_humaneval; executes generated code):
python -m mini_gpt.evaluate --tier mini --tokenizer data/tok.json \
    --ckpt base:out/mini/ckpt_final.pt --humaneval hf --out out/eval

# several checkpoints in one table (base -> sft -> grpo rows):
python -m mini_gpt.evaluate --tier mini --tokenizer data/tok.json --data data/packed \
    --ckpt base:out/mini/ckpt_final.pt --ckpt sft:out/mini_sft/ckpt_sft.pt \
    --ckpt grpo:out/mini_grpo/ckpt_grpo.pt --arc-easy hf --out out/eval
```

Every run writes `results.json` plus a Markdown table (`results.md`) with
chance baselines in the headers.

## 14. Testing

```bash
python -m pytest -q                                  # full suite
CUDA_VISIBLE_DEVICES="" python -m pytest -q          # CPU-only: CUDA tests skip
python -m pytest -k "triton or chunked" -q           # kernel tests only
```

The suite covers: tokenizer round-trips and stable special IDs; causal and
sliding-window masking; GQA head expansion; QK-norm bounds; RoPE norm
preservation; tied weights; the 2,048-context configuration; finite
forward/backward; one-batch overfitting; AdamW and Muon updates; Newton–Schulz
orthogonalization; checkpoint save/load, and a restarted run matching an
uninterrupted one;
Triton forward *and* gradient comparisons on CUDA (skipped cleanly without a
GPU); chunked-CE exactness for any chunk size; assistant-only masking;
packed-conversation isolation; GSM8K extraction and rewards; zero-variance
GRPO groups; GRPO update direction; multiple-choice scoring; HumanEval sandbox
pass/fail/timeout; and results JSON round-trips.

## 15. Full historical implementation

The complete pre-simplification pipeline (~50 files, including data annealing,
context extension with NTK RoPE rescaling, a benchmark harness, and scaling-law
plotting) is preserved locally on the **`full-pipeline`** branch:

```bash
git checkout full-pipeline
```

## License

MIT. See [LICENSE](LICENSE).
