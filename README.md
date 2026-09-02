# Mini-GPT

A from-scratch GPT pipeline — tokenizer, pretraining, Triton kernels, SFT, GRPO,
and evaluation — reorganized as an **executable textbook**: ten Python files at
the repository root, each teaching one major concept, readable in order.

## 1. What Mini-GPT teaches

How a modern small language model is actually built and trained, end to end,
with nothing hidden behind a framework:

- how a **byte-level BPE tokenizer** is trained and why special tokens get
  fixed IDs;
- how text becomes **packed uint16 token shards** and reproducible training
  batches;
- every module of a modern decoder-only **Transformer** — grouped-query
  attention, RoPE, QK-norm, RMSNorm, SwiGLU, sliding windows — with tensor
  shapes annotated at each step;
- how **AdamW and Muon** split the parameters and drive the training loop;
- how **Triton kernels** slot into PyTorch with eager fallbacks and
  differential tests;
- how a base model becomes an assistant via **SFT** and improves with **GRPO**
  reinforcement learning on GSM8K;
- how models are scored: **perplexity, ARC, MMLU, HumanEval**.

## 2. Resume-level feature summary

This repository implements:

- A **32,768-vocabulary byte-level BPE tokenizer** with conversation and tool
  special tokens (`<|pad|> <|bos|> <|eos|> <|system|> <|user|> <|assistant|>
  <|tool_call|> <|tool_result|>`), with data preparation paths for **NVIDIA
  ClimbMix** (the official `nvidia/Nemotron-ClimbMix` token release, GPT-2
  detokenized, and the `OptimalScale/ClimbMix` raw-text mirror), supporting a
  maximum context length of **2,048 tokens** (the `small` tier).
- A **grouped-query-attention Transformer** with **RoPE**, per-head
  **QK-norm**, **RMSNorm**, a **SwiGLU MLP**, and **sliding-window attention**
  with periodic full-context layers, trained with **AdamW and Muon**
  (Newton–Schulz orthogonalized momentum).
- Post-training: **supervised fine-tuning** (assistant-only loss over packed,
  attention-isolated conversations) and **GRPO** with GSM8K rewards;
  **perplexity tracking** during training; evaluators for **ARC-Easy,
  ARC-Challenge, MMLU, and HumanEval**.
- Both **Triton and PyTorch**: four fused Triton kernels (RMSNorm, RoPE,
  SwiGLU, chunked cross-entropy dispatch) differential-tested against eager
  PyTorch references, with automatic CPU fallback.

Everything above is *implemented and runnable*; this README marks which
commands download datasets or cost real compute. No benchmark scores or
completed training runs are claimed anywhere in this repository — results exist
only in the `results.json` files that runs you launch produce.

## 3. End-to-end pipeline

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

## 4. Repository structure

| File | One-line responsibility |
| --- | --- |
| `config.py` | The `Config` dataclass and the `nano` / `mini` / `small` presets. |
| `tokenizer.py` | 32K byte-level BPE with fixed-ID conversation/tool special tokens. |
| `data.py` | Fetch text (ClimbMix or synthetic), pack uint16 shards, sample windows. |
| `model.py` | The eager Transformer: GQA, RoPE, QK-norm, RMSNorm, SwiGLU, windows. |
| `kernels.py` | Triton kernels + eager references + automatic fallback. |
| `train.py` | AdamW + Muon, LR schedule, training loop, checkpoints, seeding. |
| `generate.py` | Greedy / temperature / top-k sampling with context enforcement. |
| `posttrain.py` | Chat template, SFT, GSM8K rewards, GRPO, `sft`/`grpo` CLIs. |
| `evaluate.py` | Perplexity, ARC/MMLU multiple choice, sandboxed HumanEval, reports. |
| `test_minigpt.py` | The whole test suite, ordered like the reading order. |

## 5. Recommended file-reading order

1. `config.py` — every knob in one dataclass.
2. `tokenizer.py` — text ↔ token IDs.
3. `data.py` — token IDs → shards → training windows.
4. `model.py` — the Transformer itself.
5. `train.py` — optimizers, schedule, loop, checkpoints.
6. `generate.py` — sampling from a trained model.
7. `kernels.py` — the same math, fused in Triton.
8. `posttrain.py` — SFT and GRPO.
9. `evaluate.py` — how the model is scored.
10. `test_minigpt.py` — every guarantee, as assertions.

Each file's docstring states what it teaches, what to read first, its inputs
and outputs, and a representative command.

## 6. Model architecture and tensor shapes

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

`nano` is the CPU-friendly smoke tier; `small` is the resume-scale
configuration with the full 2,048-token context.

## 7. Tokenizer and special tokens

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

Setup first (one-time):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 8. Preparing synthetic data (offline, seconds)

```bash
python data.py --source synthetic --parts 2 --docs-per-part 2000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 100000
```

Writes jsonl text parts, trains the BPE, and packs uint16 shards with one
held-out validation shard. Fully deterministic and network-free.

## 9. Preparing ClimbMix-derived data

> **Downloads a large dataset from HuggingFace.** Start small to verify the
> path, then scale `--parts` up.

```bash
# small first taste (~1000 documents from the raw-text mirror):
python data.py --source hf-raw --parts 1 --docs-per-part 1000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 1000000

# EXPENSIVE: a real pretraining corpus (tens of GB, hours of download):
python data.py --source hf-raw --parts 64 --docs-per-part 20000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 50000000
```

`--source hf-raw` streams the `OptimalScale/ClimbMix` raw-text mirror;
`--source hf-tokens` streams NVIDIA's official `nvidia/Nemotron-ClimbMix`
token-ID release and detokenizes it with GPT-2 (needs the optional `tiktoken`
package). Part-writing is idempotent: an interrupted fetch continues where it
stopped.

## 10. Pretraining with AdamW and Muon

Parameters are split by shape: **Muon** (momentum orthogonalized by
Newton–Schulz iteration, `train.py:zeropower_via_newtonschulz5`) updates the 2D
hidden matrices; **AdamW** updates embeddings and norm gains. `--optimizer
adamw` gives the pure-AdamW baseline. Warmup then cosine decay; gradient
accumulation; bf16 autocast on CUDA; gradient clipping.

```bash
# CPU smoke run (minutes):
python train.py --tier nano --data data/packed --out out/nano \
    --steps 30 --micro-batch 4 --grad-accum 2 --device cpu --no-compile

# EXPENSIVE: CUDA training with periodic held-out perplexity:
python train.py --tier mini --data data/packed --out out/mini --eval-every 1000

# resume from a checkpoint (continues the same data stream and LR schedule):
python train.py --tier mini --data data/packed --out out/mini \
    --resume out/mini/ckpt_1000.pt

# pure-AdamW baseline for an A/B comparison:
python train.py --tier mini --data data/packed --out out/mini_adamw --optimizer adamw
```

Checkpoints hold model weights, both optimizers' state, the scheduler, the
data-sampler position, and the step counter.

## 11. Triton kernels and eager fallbacks

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
CUDA by the test suite; the chunked-CE peak-memory property is asserted
directly. No speed or memory figures are quoted here beyond what those tests
measure on your machine.

## 12. Text generation

```bash
python generate.py --ckpt out/nano/ckpt_final.pt --tokenizer data/tok.json \
    --tier nano --prompt "the quick brown" --max-new-tokens 40 \
    --temperature 0.8 --top-k 40 --seed 0

# chat-formatted prompt through the same template SFT trains on:
python generate.py --ckpt out/nano_sft/ckpt_sft.pt --tokenizer data/tok.json \
    --tier nano --chat "What is 2 + 3?" --max-new-tokens 16
```

Greedy (`--temperature 0`), temperature, and top-k sampling; stops at
`<|eos|>`; seeded sampling is reproducible; the loop feeds the model at most
its trained context (2,048 tokens for `small`), cropping older tokens.

## 13. Supervised fine-tuning

`posttrain.py` renders conversations through one chat template
(`<|role|> content <|eos|>` per turn), builds a loss mask that covers **only
assistant-authored tokens**, and packs several conversations per row with
segment IDs — the attention mask blocks any lookback across conversation
boundaries.

```bash
# offline smoke run on synthetic arithmetic conversations (CPU, ~a minute):
python posttrain.py sft --tier nano --tokenizer data/tok.json \
    --init out/nano/ckpt_final.pt --data synthetic --out out/nano_sft --steps 20

# real chat data: a JSONL of {"messages": [{"role": ..., "content": ...}]}
python posttrain.py sft --tier mini --tokenizer data/tok.json \
    --init out/mini/ckpt_final.pt --data chats.jsonl --out out/mini_sft --steps 2000
```

## 14. GRPO on GSM8K

GRPO samples a *group* of completions per prompt, scores each with a reward,
and normalizes rewards **within the group** (mean as baseline, std as scale) —
no learned critic. The update is the clipped PPO surrogate over completion
tokens only. A group whose rewards are all equal has zero advantage and
produces exactly zero gradient. The GSM8K reward extracts the final integer
(preferring the `#### N` delimiter) and adds small format-shaping terms.

```bash
# GRPO on GSM8K (DOWNLOADS openai/gsm8k; EXPENSIVE at real scale):
python posttrain.py grpo --tier mini --tokenizer data/tok.json \
    --init out/mini_sft/ckpt_sft.pt --task gsm8k --out out/mini_grpo --steps 500

# offline smoke run on the arithmetic task (no downloads, CPU-sized):
python posttrain.py grpo --tier nano --tokenizer data/tok.json \
    --init out/nano_sft/ckpt_sft.pt --task arithmetic --out out/nano_grpo \
    --steps 10 --group-size 4 --prompts-per-step 2 --max-new-tokens 8
```

Mean reward per step is logged; the checkpoint stores the reward history.
Note: at these model scales GSM8K correctness is expected to be near zero —
the machinery is what is demonstrated, and no reward curves or scores are
claimed in this repository.

## 15. Perplexity, ARC, MMLU, and HumanEval

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
python evaluate.py --tier nano --tokenizer data/tok.json --data data/packed \
    --ckpt base:out/nano/ckpt_final.pt --out out/eval

# ARC (downloads allenai/ai2_arc):
python evaluate.py --tier mini --tokenizer data/tok.json \
    --ckpt base:out/mini/ckpt_final.pt --arc-easy hf --arc-challenge hf --out out/eval

# MMLU (downloads cais/mmlu; --limit caps question count):
python evaluate.py --tier mini --tokenizer data/tok.json \
    --ckpt base:out/mini/ckpt_final.pt --mmlu hf --limit 500 --out out/eval

# HumanEval (downloads openai/openai_humaneval; executes generated code):
python evaluate.py --tier mini --tokenizer data/tok.json \
    --ckpt base:out/mini/ckpt_final.pt --humaneval hf --out out/eval

# several checkpoints in one table (base -> sft -> grpo rows):
python evaluate.py --tier mini --tokenizer data/tok.json --data data/packed \
    --ckpt base:out/mini/ckpt_final.pt --ckpt sft:out/mini_sft/ckpt_sft.pt \
    --ckpt grpo:out/mini_grpo/ckpt_grpo.pt --arc-easy hf --out out/eval
```

Every run writes `results.json` plus a Markdown table (`results.md`) with
chance baselines in the headers. **Implemented support vs. executed runs:**
the evaluators and dataset loaders above are implemented and tested on
fixtures; this repository ships no recorded benchmark results.

## 16. Testing

```bash
python -m pytest -q                                  # full suite
CUDA_VISIBLE_DEVICES="" python -m pytest -q          # CPU-only: CUDA tests skip
python -m pytest test_minigpt.py -k "triton or chunked" -q   # kernel tests only
```

The suite covers: tokenizer round-trips and stable special IDs; causal and
sliding-window masking; GQA head expansion; QK-norm bounds; RoPE norm
preservation; tied weights; the 2,048-context configuration; finite
forward/backward; one-batch overfitting; AdamW and Muon updates; Newton–Schulz
orthogonalization; checkpoint save/load and resume-equals-uninterrupted;
Triton forward *and* gradient comparisons on CUDA (skipped cleanly without a
GPU); chunked-CE exactness for any chunk size; assistant-only masking;
packed-conversation isolation; GSM8K extraction and rewards; zero-variance
GRPO groups; GRPO update direction; multiple-choice scoring; HumanEval sandbox
pass/fail/timeout; and results JSON round-trips.

## 17. Features removed during simplification

Removed from the learning branch because no resume-listed capability depends
on them (all remain on the `full-pipeline` branch):

- **Data annealing** (mid-run switch to a math/instruct mix) and its CLI.
- **Context-extension script** with NTK RoPE rescaling — the 2,048-token
  context is supported directly by the `small` tier configuration.
- **Benchmark harness** (`bench.py`: tokens/s, MFU, peak VRAM) and
  **scaling-law plotting**.
- **Countdown reward** task (arithmetic covers the offline smoke path; GSM8K
  is the real task).
- **Wrapper classes** (`CombinedOptimizer`, `Trainer`) — replaced by a plain
  list of optimizers and a flat, traceable training loop.
- `Makefile` and `pytest.ini` (plain commands above), per-package
  `__init__.py` files, and duplicate script entry points.
- CUDA RNG capture in checkpoints (host RNG, optimizer, scheduler, and data
  sampler state are still checkpointed; resume is exact on the tested CPU
  path).

## 18. Full historical implementation

The complete pre-simplification pipeline (~50 files, including annealing,
context extension, benchmarking, and plotting) is preserved locally on the
**`full-pipeline`** branch:

```bash
git checkout full-pipeline
```

## License

MIT. See [LICENSE](LICENSE).
