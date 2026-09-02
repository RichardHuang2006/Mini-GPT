"""One Config dataclass holding every model and training knob.

Nothing downstream hardcodes a dimension. Three tiers: nano (CPU smoke,
context 512), mini (single-GPU, 1024), small (full scale, 2048).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import NamedTuple


def swiglu_hidden(d_model: int, multiple: int = 128) -> int:
    """SwiGLU hidden width ~= 8/3 * d_model, rounded to a multiple of 128.

    8/3 matches a plain 4*d_model GELU MLP's parameter count; the rounding
    keeps GEMM shapes tile-friendly.
    """
    target = (8 * d_model) / 3
    return int(round(target / multiple)) * multiple


class ParamCount(NamedTuple):
    total: int
    embedding: int
    non_embedding: int


@dataclass
class Config:
    # --- identity -----------------------------------------------------------
    name: str = "mini"

    # --- tokenizer / vocab --------------------------------------------------
    vocab_size: int = 32_768

    # --- model shape --------------------------------------------------------
    d_model: int = 512
    n_layers: int = 8
    n_q_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    mlp_hidden: int = 1_408
    context: int = 1_024

    # --- attention detail ---------------------------------------------------
    window: int = 256          # sliding-window span
    full_attn_every: int = 2    # every Nth layer is full-context, the rest windowed
    rope_base: float = 10_000.0  # rescaled when extending the context length
    qk_norm: bool = True

    # --- optimization -------------------------------------------------------
    micro_batch: int = 16       # sequences per forward; the largest that fits
    grad_accum: int = 32        # micro-batches per optimizer step
    lr_adamw: float = 3.0e-3    # embeddings, norms
    lr_muon: float = 2.0e-2     # 2D hidden matrices
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 300
    max_steps: int = 40_000
    lr_floor_frac: float = 0.1  # cosine decays to this fraction of peak
    use_muon: bool = True       # False -> pure-AdamW A/B baseline

    # --- data budget --------------------------------------------------------
    train_tokens: int = 2_000_000_000  # target tokens = max_steps * global batch

    # --- runtime ------------------------------------------------------------
    seed: int = 0
    dtype: str = "bfloat16"
    compile: bool = True
    use_triton: bool = True     # route through the fused kernels
    ce_chunk: int = 8192        # vocab tile for chunked cross-entropy

    def __post_init__(self) -> None:
        self.validate()

    # ---------------------------------------------------------------- checks
    def validate(self) -> None:
        assert self.d_model == self.n_q_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_q_heads * head_dim "
            f"({self.n_q_heads} * {self.head_dim})"
        )
        assert self.n_q_heads % self.n_kv_heads == 0, (
            f"n_q_heads ({self.n_q_heads}) must be a multiple of n_kv_heads "
            f"({self.n_kv_heads}) for grouped-query attention"
        )
        assert self.vocab_size < 65_536, (
            "vocab must be < 65536 so packed token streams fit in uint16"
        )
        assert self.vocab_size % 128 == 0, "vocab should tile cleanly (multiple of 128)"
        assert self.window <= self.context, "sliding window cannot exceed context"
        assert self.n_layers > 0 and self.d_model > 0

    # -------------------------------------------------------- derived values
    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def global_batch_tokens(self) -> int:
        return self.micro_batch * self.grad_accum * self.context

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Layer schedule: full-context on every ``full_attn_every``-th layer,
        sliding-window on the rest."""
        return (layer_idx + 1) % self.full_attn_every == 0

    # ------------------------------------------------------- parameter count
    def param_count(self) -> ParamCount:
        """Count fields exactly as model.py builds them: tied embeddings once,
        per-head QK-norm gains, two RMSNorm gains per layer plus a final one,
        no biases. The tests assert the built model matches."""
        d = self.d_model
        embedding = self.vocab_size * d  # tied input/output projection

        attn = (
            d * self.q_dim               # q_proj
            + d * self.kv_dim            # k_proj
            + d * self.kv_dim            # v_proj
            + self.q_dim * d             # o_proj
        )
        if self.qk_norm:
            attn += 2 * self.head_dim    # per-head q-norm and k-norm gains

        mlp = 3 * d * self.mlp_hidden    # gate, up, down (SwiGLU)
        norms = 2 * d                    # pre-attn and pre-mlp RMSNorm gains
        per_layer = attn + mlp + norms

        non_embedding = self.n_layers * per_layer + d  # + final RMSNorm
        total = non_embedding + embedding
        return ParamCount(total=total, embedding=embedding, non_embedding=non_embedding)

    def with_overrides(self, **kwargs) -> "Config":
        """Return a copy with fields replaced (re-validated)."""
        return replace(self, **kwargs)


TIERS: dict[str, Config] = {
    "nano": Config(
        name="nano",
        d_model=256,
        n_layers=6,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=64,
        mlp_hidden=swiglu_hidden(256),   # 640
        context=512,
        window=256,
        micro_batch=32,
        grad_accum=8,
        max_steps=6_000,
        train_tokens=200_000_000,
    ),
    "mini": Config(
        name="mini",
        d_model=512,
        n_layers=8,
        n_q_heads=8,
        n_kv_heads=2,
        head_dim=64,
        mlp_hidden=swiglu_hidden(512),   # 1408
        context=1_024,
        window=256,
        micro_batch=16,
        grad_accum=32,
        max_steps=40_000,
        train_tokens=2_000_000_000,
    ),
    "small": Config(
        name="small",
        d_model=768,
        n_layers=12,
        n_q_heads=12,
        n_kv_heads=4,
        head_dim=64,
        mlp_hidden=swiglu_hidden(768),   # 2048
        context=2_048,
        window=256,
        micro_batch=8,
        grad_accum=64,
        max_steps=60_000,
        train_tokens=3_000_000_000,
    ),
}


def get_config(tier: str = "mini", **overrides) -> Config:
    """Fetch a tier preset (a fresh copy), optionally overriding fields."""
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; choose from {sorted(TIERS)}")
    cfg = replace(TIERS[tier])
    return cfg.with_overrides(**overrides) if overrides else cfg
