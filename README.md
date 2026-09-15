# Mini-GPT

A from-scratch GPT pipeline: BPE tokenizer, data packing, a GQA/RoPE/SwiGLU
Transformer, AdamW + Muon pretraining, Triton kernels, SFT, GRPO, and evaluation.

## Modules

| File | What it does |
| --- | --- |
| `mini_gpt/config.py` | `Config` dataclass and the `nano` / `mini` / `small` presets. |
| `mini_gpt/tokenizer.py` | 32K byte-level BPE with fixed-ID special tokens. |
| `mini_gpt/data.py` | Fetch text, train the BPE, pack uint16 shards. |
| `mini_gpt/model.py` | Transformer: GQA, RoPE, QK-norm, RMSNorm, SwiGLU, sliding windows. |
| `mini_gpt/train.py` | AdamW + Muon, warmup + cosine LR, checkpoints. |
| `mini_gpt/generate.py` | Greedy / temperature / top-k sampling. |
| `mini_gpt/kernels.py` | Triton RMSNorm, RoPE, SwiGLU, chunked cross-entropy (CPU fallback). |
| `mini_gpt/posttrain.py` | Chat template, SFT, GRPO on GSM8K. |
| `mini_gpt/evaluate.py` | Perplexity, ARC, MMLU, HumanEval. |
| `tests/test_minigpt.py` | Test suite. |

Tiers: `nano` (12.5M, 512 ctx), `mini` (39.3M, 1024 ctx), `small` (100.7M, 2048 ctx).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Everything runs locally in the venv. Steps are offline unless the comment says
they download a dataset from HuggingFace.

```bash
# 1. data (offline synthetic)
python -m mini_gpt.data --source synthetic --parts 2 --docs-per-part 2000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 100000

# 1b. real data instead (downloads ClimbMix; raise --parts to scale up)
python -m mini_gpt.data --source hf-raw --parts 1 --docs-per-part 1000 \
    --tokenizer data/tok.json --data data/packed --shard-tokens 1000000

# 2. pretrain
python -m mini_gpt.train --tier nano --data data/packed --out out/nano \
    --steps 30 --micro-batch 4 --grad-accum 2 --device cpu --no-compile

# 3. generate
python -m mini_gpt.generate --ckpt out/nano/ckpt_final.pt --tokenizer data/tok.json \
    --tier nano --prompt "the quick brown" --max-new-tokens 40 --temperature 0.8 --top-k 40

# 4. supervised fine-tuning
python -m mini_gpt.posttrain sft --tier nano --tokenizer data/tok.json \
    --init out/nano/ckpt_final.pt --data synthetic --out out/nano_sft --steps 20

# 5. GRPO
python -m mini_gpt.posttrain grpo --tier nano --tokenizer data/tok.json \
    --init out/nano_sft/ckpt_sft.pt --task arithmetic --out out/nano_grpo \
    --steps 10 --group-size 4 --prompts-per-step 2 --max-new-tokens 8

# 6. evaluate (add --arc-easy hf / --mmlu hf / --humaneval hf to download real sets)
python -m mini_gpt.evaluate --tier nano --tokenizer data/tok.json --data data/packed \
    --ckpt base:out/nano/ckpt_final.pt --out out/eval
```

Swap `--tier nano` for `mini` or `small` and drop `--device cpu --no-compile`
for a GPU run. Results land in `out/eval/results.json` and `results.md`.

## Tests

```bash
python -m pytest -q                          # full suite
CUDA_VISIBLE_DEVICES="" python -m pytest -q  # CPU only; CUDA tests skip
```

## License

MIT. See [LICENSE](LICENSE).
