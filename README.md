# Mini-GPT

A from-scratch GPT pretraining and post-training pipeline in Python, PyTorch, and Triton, sized so
the default model trains overnight on a single 12 GB GPU. The model is a decoder-only transformer
with grouped-query attention, rotary position embeddings, per-head QK-norm, RMSNorm, a SwiGLU MLP,
sliding-window attention on alternating layers, and tied input/output embeddings. It trains under
bf16 autocast with AdamW on the embeddings and norms and Muon on the 2-D hidden matrices, over a
`uint16` token-shard pipeline built from its own 32K byte-level BPE tokenizer. Four fused Triton
kernels each have an eager PyTorch twin they are differential-tested against, and the post-training
stack — supervised fine-tuning and GRPO — plus a perplexity / ARC / MMLU / HumanEval evaluation
harness sit on top. No architectural novelty is claimed; the goal is a correct, testable,
single-GPU implementation.

## Features

- **32K-vocab byte-level BPE tokenizer** (`tokenizers` Rust BPE) with eight fixed-ID special
  tokens, including the conversation roles (`<|system|>`, `<|user|>`, `<|assistant|>`) and
  tool-call tokens (`<|tool_call|>`, `<|tool_result|>`). Trained on text derived from
  **NVIDIA ClimbMix**. Byte-level, so every input round-trips exactly: no `[UNK]`, no
  out-of-vocabulary token. Context reaches **2048 tokens** at the `small` tier and via context
  extension.
- **GQA transformer** with **RoPE**, per-head **QK-norm**, **RMSNorm** in pre-norm placement, a
  **SwiGLU MLP**, and **sliding-window attention** on non-full layers (full-context on every
  `full_attn_every`-th layer).
- **Training with AdamW and Muon**: parameters are split by shape, with Muon (momentum
  orthogonalized by Newton–Schulz iteration) on the 2-D hidden matrices and AdamW on embeddings
  and norm gains. `use_muon=False` gives a pure-AdamW A/B baseline.
- **Fused Triton kernels validated against eager reference implementations**: RMSNorm, RoPE,
  SwiGLU gating, and a chunked cross-entropy that never materializes the `[tokens × vocab]`
  logits tensor. Each falls back to the eager path off CUDA.
- **Supervised fine-tuning and GRPO**, including arithmetic, countdown, and **GSM8K** reward
  functions. SFT trains on an assistant-only loss mask over conversations packed with segment IDs;
  GRPO uses group-relative advantages with no learned critic and a clipped policy update.
- **Perplexity tracking and ARC / MMLU / HumanEval evaluation**: held-out perplexity is logged
  in-loop during pretraining and reported by the eval harness alongside length-normalized
  multiple-choice accuracy and sandboxed HumanEval pass@1.
- **Reproducibility**: a single seeding harness plus deterministic algorithms, and checkpoints that
  capture model, optimizer, scheduler, data-sampler position, and RNG state — enough to resume a
  run bit-identically.

## Architecture

### Library (`mini_gpt/`)

| Module | Contents |
| --- | --- |
| `config.py` (repo root) | The `Config` dataclass, the three tier presets, and an exact parameter-count method the built model is checked against. |
| `determinism.py` | `seed_everything`: seeds Python, NumPy, and Torch (CPU + CUDA) and enables deterministic algorithms. |
| `tokenizer.py` | The byte-level BPE with stable special-token IDs, save/load, and a content fingerprint. |
| `chat_template.py` | Renders a message list to token IDs plus an assistant-only loss mask, and builds generation prompts. The single source of the chat format. |
| `data/` | `download.py` (ClimbMix or synthetic text to JSONL parts, idempotently), `pack.py` (`uint16` shards plus a manifest pinning the tokenizer fingerprint and train/val split), `sampler.py` (seeded, resumable windowed sampler over memory-mapped shards), `anneal.py` (the math/instruct mix and the mid-run data switch). |
| `model/` | The eager transformer: `rope.py` (rotary tables, NTK base rescaling), `norm.py`, `mlp.py`, `attention.py` (GQA, QK-norm, RoPE, sliding-window and segment masks over PyTorch SDPA), `transformer.py` (pre-norm blocks, tied embeddings, GPT-2 residual-scaled init, loss-returning forward). |
| `kernels/` | The four fused Triton kernels with eager fallbacks: `rmsnorm.py`, `rope_kernel.py`, `swiglu.py`, `cross_entropy.py`. |
| `train/` | `optim.py` (parameter grouping, Muon, and a combined Muon+AdamW wrapper), `schedule.py` (warmup then cosine decay), `loop.py` (gradient accumulation, autocast, clipping, `torch.compile`), `checkpoint.py` (save/resume). |
| `generate.py` | Autoregressive generation — greedy and seeded temperature sampling with `eos` stopping — shared by SFT, GRPO rollouts, and the eval harness. |
| `posttrain/` | `sft.py` (conversation packing with segment IDs and assistant-only loss), `rewards.py` (arithmetic, countdown, and GSM8K scoring with format shaping), `grpo.py` (group sampling, group-relative advantages, clipped update). |
| `eval/harness.py` | Held-out perplexity, log-likelihood multiple choice (ARC, MMLU), sandboxed HumanEval, and `results.json` plus Markdown tables. |
| `bench.py` | Tokens/s, achieved TFLOPs, MFU, peak VRAM, and the largest feasible micro-batch with and without chunked cross-entropy. |

### Entry points (`scripts/`)

| Script | Purpose |
| --- | --- |
| `prepare_data.py` | Text parts → trained tokenizer → packed `uint16` shards with a held-out val split. |
| `pretrain.py` | The training loop, with periodic held-out perplexity and resumable checkpoints. |
| `anneal.py` | Builds the annealed shard mix (math/instruct folded into base text). |
| `extend_context.py` | Continued training at a longer context with an NTK-rescaled RoPE base. |
| `sft.py` | Supervised fine-tuning of a base checkpoint on chat data. |
| `grpo.py` | GRPO on the arithmetic/countdown task, logging mean reward per step. |
| `eval.py` | Scores base / SFT / GRPO checkpoints on every metric. |
| `plot_scaling.py` | Loss-vs-tokens and loss-vs-parameters charts with a fitted power law, from captured pretrain logs. |

### Chunked cross-entropy

A naive `F.cross_entropy` on a `mini` micro-batch materializes a `[16·1024, 32768]` logits tensor
(~1 GB in bf16) plus its fp32 softmax upcast and gradient, roughly 4.3 GB in total, which caps the
micro-batch well below what compute allows. The chunked kernel tiles the vocabulary and streams it
with an online-softmax running max and sum-of-exp, dropping peak memory from `O(B·T·V)` to
`O(B·T·chunk)`. The math is exact, so the loss and gradients match `F.cross_entropy` to
floating-point tolerance for any chunk size.

## Model tiers

All three tiers share one `Config`; nothing downstream hardcodes a dimension.

| Tier | Params (total / non-emb) | `d_model` | Layers | Q/KV heads | Context | Window | Global batch | Token budget | Steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nano` | 12.5M / 4.1M | 256 | 6 | 4 / 2 | 512 | 256 | 131K tokens | 200M | 6,000 |
| `mini` | 39.3M / 22.6M | 512 | 8 | 8 / 2 | 1024 | 256 | 524K tokens | 2B | 40,000 |
| `small` | 100.7M / 75.5M | 768 | 12 | 12 / 4 | 2048 | 256 | 1.05M tokens | 3B | 60,000 |

`nano` is the sub-hour tier that exercises the full pipeline; `mini` is the overnight default.
Tied embeddings are about 43% of `mini`'s parameters.

## Setup

```bash
make setup     # create .venv and install pinned deps
make test      # run the suite (CUDA kernel tests skip without a GPU)
make bench     # tokens/s, MFU, peak VRAM, largest micro-batch
make clean     # remove .venv, out/, and caches
make help      # list targets
```

Dependencies are pinned in `requirements.txt` (torch, triton, numpy, tokenizers, datasets,
huggingface-hub, pytest). `make test` uses whatever `python` is on `PATH`, so it works against an
already-installed torch without `make setup` first.

## Data preparation

One command fetches text, trains the tokenizer, and packs shards:

```bash
python scripts/prepare_data.py --source hf-raw --parts 64 --docs-per-part 20000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 50000000 --val-shards 1
```

Sources are `hf-raw` (the ClimbMix raw-text mirror), `hf-tokens` (NVIDIA's official token-ID
release, detokenized with GPT-2), or `synthetic` for a fully offline run. Part writing is
idempotent: an existing part index is skipped, so an interrupted fetch continues rather than
re-downloading. The packed directory carries a `manifest.json` recording the tokenizer
fingerprint, dtype, per-shard token counts, and the train/val split.

## Training

```bash
# pretrain, with periodic held-out perplexity
python scripts/pretrain.py --tier nano --data data/packed --out out/nano
python scripts/pretrain.py --tier mini --data data/packed --out out/mini --eval-every 1000

# resume from a checkpoint
python scripts/pretrain.py --tier mini --data data/packed --out out/mini \
    --resume out/mini/ckpt_20000.pt

# the last anneal_frac of the budget switches to the math/instruct mix
python scripts/anneal.py --base data/parts --tokenizer data/tok.json --out data/anneal
python scripts/pretrain.py --tier mini --data data/packed --anneal-data data/anneal --out out/mini

# stretch 1024 -> 2048 context with an NTK-rescaled RoPE base
python scripts/extend_context.py --tier mini --data data/packed_2k \
    --init out/mini/ckpt_final.pt --out out/mini_2k --new-context 2048 --steps 2000
```

## Post-training

```bash
# assistant-only SFT on packed, attention-isolated conversations
python scripts/sft.py  --tier mini --tokenizer data/tok.json \
    --init out/mini/ckpt_final.pt --out out/mini_sft --steps 2000

# GRPO on arithmetic/countdown
python scripts/grpo.py --tier mini --tokenizer data/tok.json \
    --init out/mini_sft/ckpt_sft.pt --out out/mini_grpo --steps 500
```

SFT renders conversations through the same template used at inference, so the loss mask covers
exactly the assistant tokens and nothing else. Several conversations are packed into one sequence
with distinct segment IDs, and the segment mask blocks attention across conversation boundaries.

GRPO samples a group of completions per prompt, scores each with a reward function, and normalizes
the reward within its group as the advantage — the group mean is the baseline, so there is no
learned critic. The update is the clipped PPO surrogate over completion tokens only. A group whose
rewards are all equal has zero advantage and produces no gradient. Rewards combine a correctness
term with format shaping (terminated, parseable answer, within length), which supplies partial
signal before the model is ever correct.

## Evaluation

```bash
python scripts/eval.py --tier mini --tokenizer data/tok.json --data data/packed \
    --ckpt base:out/mini/ckpt_final.pt --ckpt sft:out/mini_sft/ckpt_sft.pt \
    --ckpt grpo:out/mini_grpo/ckpt_grpo.pt --out out/eval
```

This writes `results.json` and a Markdown table with one row per checkpoint in training order.
Scoring methods:

- **Perplexity** — token-weighted `exp(mean NLL)` over windows drawn from a shard split the model
  never trained on; the fixed train/val split in the manifest guarantees disjointness. Also logged
  in-loop by `pretrain.py --eval-every`.
- **ARC-Easy / ARC-Challenge / MMLU** — each choice scored by its length-normalized log-likelihood
  given the question; the argmax is the prediction. Length normalization keeps a short choice from
  winning for being short.
- **HumanEval** — the completion is executed in an isolated subprocess with a timeout against the
  task's unit test; pass@1 is the fraction that exit cleanly. A generated infinite loop hits the
  timeout and counts as a failure rather than hanging the harness.

Every score is reported next to its chance baseline (25% for the multiple-choice sets, 0% for
HumanEval). At 39M parameters, held-out perplexity and ARC-Easy are the metrics with usable
signal; MMLU serves as a chance-level regression tripwire and HumanEval is effectively zero.

## Testing and validation

```bash
python -m pytest -q                      # full suite
python -m pytest tests/test_kernels.py   # differential kernel tests (need CUDA)
```

The suite runs under seeded RNG and deterministic algorithms. CPU-only sections always run; the
CUDA kernel comparisons skip automatically without a GPU.

**Kernel correctness** is established in two layers. Each kernel's analytic definition is checked
by an fp64 `torch.autograd.gradcheck` on CPU, and on CUDA the kernel's forward value and backward
gradient are compared against the eager implementation at fp32 tolerance, with RMSNorm also
compared at a loose bf16 tolerance. Chunked cross-entropy is additionally checked against
`F.cross_entropy` across chunk sizes from 1 to larger-than-vocab, on the loss and both input
gradients, plus a fully-masked batch that must stay finite, and its lower peak VRAM is measured
directly. A fixed-seed training run must produce the same loss curve with the fused kernels on or
off, with the deterministic SDPA backend pinned so the kernels are the only difference.

**Behavioral properties** cover what numerics alone miss:

- the eager model overfits one fixed batch to near-zero loss;
- attention matches an independent hand-written reference, QK-norm bounds the logit magnitude at
  `sqrt(head_dim)` and makes logits invariant to input scale, and the windowed mask is a strict
  subset of the full causal mask;
- the built model's unique parameter count equals `Config.param_count()` for every tier;
- the tokenizer round-trips every input exactly, including whitespace and mixed Unicode, and
  special-token IDs are stable; loading a tokenizer without them fails loudly;
- packing is lossless in `uint16`, the manifest pins the tokenizer that produced the shards, and
  train and val shards are disjoint;
- the sampler reproduces its window stream from a seed and resumes a snapshot identically;
- a checkpointed run resumes to the same parameters as an uninterrupted one, and the anneal switch
  is resumable across the boundary;
- the SFT loss equals a hand-computed cross-entropy over assistant positions only, and a token's
  logits are invariant to edits in a different packed conversation;
- generation is reproducible under a fixed seed and stops on `<|eos|>`;
- GRPO moves a positive-advantage completion's log-probability up and a negative one's down, a
  zero-variance group produces exactly zero gradient, and mean reward rises over steps on a
  learnable shaped reward;
- Newton–Schulz orthogonalization brings singular values into a bounded range, and Muon reaches a
  lower loss than the pure-AdamW baseline for the same tokens;
- `torch.compile` matches eager, and the eval harness round-trips results through `results.json`.

## License

MIT. See [LICENSE](LICENSE).
