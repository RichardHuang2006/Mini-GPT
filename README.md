# Mini-GPT

A from-scratch GPT pretraining and post-training pipeline in Python and PyTorch, sized so the
default model trains overnight on one 12 GB GPU. The model is a decoder-only transformer with
grouped-query attention, rotary position embeddings, per-head QK-norm, RMSNorm, a SwiGLU MLP,
and sliding-window attention on alternating layers, with tied embeddings. It is trained with
AdamW on the embeddings and norms and Muon on the 2-D hidden matrices, in bf16 autocast, over a
`uint16` token-shard pipeline built from its own byte-level BPE tokenizer. Four fused Triton
kernels (RMSNorm, RoPE, SwiGLU, and a chunked cross-entropy that never materializes the full
logits tensor) each have an eager PyTorch twin, and the whole post-training stack — supervised
fine-tuning and GRPO — plus a perplexity / ARC / MMLU / HumanEval evaluation harness sit on top.
The method throughout: an obviously-correct eager reference is the oracle, and every fast custom
component is validated by differential testing against it.

Three tiers share one `Config`: `nano` (~12M params, the sub-hour CI run that exercises the full
pipeline), `mini` (~39M, the default overnight run), and `small` (~124M, a documented stretch).
No architectural novelty is claimed; the point is a correct, testable, single-GPU implementation.

## Architecture

### The model and training stack (`mini_gpt/`)

- `config.py` (repo root) — the `Config` dataclass: one field per hyperparameter, the three tier
  presets, and an exact parameter-count method that the built model is checked against.
- `determinism.py` — `seed_everything`: seeds Python, NumPy, and Torch (CPU + CUDA) and enables
  deterministic algorithms, so a failure reproduces bit-identically.
- `tokenizer.py` — a byte-level BPE (Hugging Face `tokenizers`) with eight fixed special-token
  IDs. Byte-level means every input round-trips: no out-of-vocabulary token, no `[UNK]`.
- `chat_template.py` — renders a message list to token IDs and a loss mask that covers exactly
  the assistant spans, plus the generation-priming prompt. The single source of the chat format.
- `data/` — the corpus pipeline. `download.py` pulls ClimbMix (or synthetic text) to raw JSONL;
  `pack.py` encodes it to fixed-size `uint16` shards with a manifest that pins the tokenizer
  fingerprint and the train/val split; `sampler.py` is a seeded, resumable windowed sampler over
  the memory-mapped shards; `anneal.py` switches the data mix to a math/instruct blend for the
  last fraction of training.
- `model/` — the eager transformer, correct by inspection and used as the oracle. `rope.py`
  (rotary tables + NTK base rescaling for context extension), `norm.py` (RMSNorm), `mlp.py`
  (SwiGLU), `attention.py` (GQA + QK-norm + RoPE + sliding-window / segment masks over PyTorch
  SDPA), and `transformer.py` (pre-norm blocks, tied embeddings, GPT-2 residual-scaled init, and
  a forward that returns the cross-entropy loss).
- `kernels/` — the four fused Triton kernels with eager fallbacks: `rmsnorm.py`, `rope_kernel.py`,
  `swiglu.py`, and `cross_entropy.py`. Chunked cross-entropy streams the vocabulary so the
  `[tokens × vocab]` logits tensor never exists — the memory win that unlocks the batch size on a
  12 GB card.
- `train/` — `optim.py` (parameter grouping, the Muon optimizer with Newton–Schulz
  orthogonalization, and a combined Muon+AdamW wrapper), `schedule.py` (warmup + cosine decay),
  `loop.py` (gradient accumulation, autocast, clipping, `torch.compile`), and `checkpoint.py`
  (model, optimizer, scheduler, sampler position, and all RNG states — enough to resume a run
  bit-identically).
- `bench.py` — measures tokens/s, TFLOPs, MFU, peak VRAM, and the largest feasible micro-batch
  with and without chunked cross-entropy.
- `generate.py` — autoregressive generation (greedy and seeded temperature sampling, `eos`
  stopping), shared verbatim by the SFT smoke test, GRPO rollouts, and the eval harness.
- `posttrain/` — `sft.py` (multi-conversation packing with segment IDs and an assistant-only
  loss mask), `rewards.py` (arithmetic / countdown correctness plus format shaping, and a GSM8K
  scorer that is honest about being near-zero at this scale), and `grpo.py` (group sampling,
  group-relative advantages with no critic, and the clipped policy update).
- `eval/harness.py` — held-out perplexity, log-likelihood multiple-choice (ARC, MMLU), and a
  sandboxed HumanEval, emitting `results.json` and Markdown tables that print every score next to
  its chance-level baseline.

### Entry points (`scripts/`)

`pretrain.py` runs the loop with periodic held-out perplexity and resumable checkpoints;
`anneal.py` builds the annealed shard mix; `extend_context.py` continues training at a longer
context with NTK-rescaled RoPE; `sft.py` and `grpo.py` are the two post-training stages; and
`eval.py` scores base / SFT / GRPO checkpoints on every metric.

### Validation (`tests/`)

Correctness rests on differential testing against the eager model. Every Triton kernel is pinned
to its eager twin on both forward value (bf16-appropriate tolerance) and backward gradient
(`torch.autograd.gradcheck` in fp64), and a fixed-seed training run must produce the same loss
curve with kernels on or off. Above the kernels sit behavioral gates that numerics alone miss:
the eager model must **overfit one batch** to near-zero loss (the correctness gate before any
real training), the tokenizer must round-trip exactly, the SFT loss mask must cover exactly the
assistant tokens, packed conversations must not attend across their boundaries, a checkpointed
run must resume to the same trajectory, and a hand-built GRPO group with known rewards must move
the policy in the correct direction while a zero-variance group produces no update. Everything
runs under seeded RNG and deterministic algorithms.

## Build and run

```bash
make setup     # create the venv and install pinned deps
make test      # run the suite (CPU parts always; the Triton kernels need CUDA)
make bench     # tokens/s, MFU, peak VRAM, and the largest micro-batch
make clean     # remove build/cache artifacts
make help      # list the targets
```

The pipeline end to end, on the sub-hour `nano` tier and then the overnight `mini` default
(`--data` is a packed-shard directory, `--tokenizer` the trained BPE json):

```bash
# pretrain (periodic held-out perplexity via the eval harness)
python scripts/pretrain.py --tier nano --data data/packed --out out/nano
python scripts/pretrain.py --tier mini --data data/packed --out out/mini --eval-every 1000

# stretch 1024 -> 2048 context with NTK-rescaled RoPE
python scripts/extend_context.py --tier mini --data data/packed --init out/mini/ckpt_final.pt

# post-training: assistant-only SFT, then GRPO on countdown/arithmetic
python scripts/sft.py  --tier mini --tokenizer data/tok.json --init out/mini/ckpt_final.pt  --out out/mini_sft
python scripts/grpo.py --tier mini --tokenizer data/tok.json --init out/mini_sft/ckpt_sft.pt --out out/mini_grpo

# score base / SFT / GRPO on every metric -> results.json + results.md
python scripts/eval.py --tier mini --tokenizer data/tok.json --data data/packed \
    --ckpt base:out/mini/ckpt_final.pt --ckpt sft:out/mini_sft/ckpt_sft.pt --ckpt grpo:out/mini_grpo/ckpt_grpo.pt
```

The two metrics that carry information at 39M are **held-out perplexity** and **ARC-Easy**;
MMLU and HumanEval are wired up so the harness is complete for the larger `small` tier and are
reported next to their chance baselines rather than dressed up as capability.
