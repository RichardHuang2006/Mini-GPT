"""The Mini-GPT architecture: a decoder-only Transformer with grouped-query
attention, RoPE, QK-norm, RMSNorm, a SwiGLU MLP, and sliding-window attention.

What this file teaches
    Every module of a modern small language model, written in plain eager
    PyTorch and readable top to bottom in the order the data flows:
    normalization -> position encoding -> attention -> MLP -> block -> model.
    Each important transformation is annotated with its tensor shape.

Read first
    config.py (all dimensions come from the Config dataclass).

Inputs and outputs
    forward(): token IDs [batch, seq] -> (logits [batch, seq, vocab], loss)
    Optional per-position loss_mask and segment_ids refine the loss and the
    attention pattern for packed chat fine-tuning (see posttrain.py).

Tensor-shape conventions used throughout
    B = batch size            T  = sequence length (<= cfg.context)
    D = cfg.d_model           Hq = cfg.n_q_heads     Hkv = cfg.n_kv_heads
    d = cfg.head_dim          V  = cfg.vocab_size

    token IDs          [B, T]
    embeddings         [B, T, D]
    queries            [B, Hq,  T, d]
    keys / values      [B, Hkv, T, d]  (expanded to Hq for the score matmul)
    attention scores   [B, Hq, T_query, T_key]
    logits             [B, T, V]

Everything here is the eager reference implementation. When cfg.use_triton is
True, RMSNorm / RoPE / SwiGLU / the loss route through the fused kernels in
kernels.py, which fall back to this same math off CUDA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import Config

# Positions whose target equals IGNORE_INDEX are excluded from the loss. The
# assistant-only SFT loss works by setting every non-assistant target to this.
IGNORE_INDEX = -100


# =============================================================================
# 1. RMSNorm
#
# LayerNorm without mean subtraction or bias: y = x / rms(x) * weight, where
# rms(x) = sqrt(mean(x^2) + eps). Used in pre-norm placement before attention
# and the MLP, plus once after the last block.
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, use_triton: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learned per-channel gain
        # When True, forward routes through the fused kernel (which falls back
        # to this same math off CUDA). False keeps this a pure eager reference.
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim] -> [..., dim], normalized over the last axis.
        if self.use_triton:
            import kernels

            return kernels.rmsnorm(x, self.weight, self.eps)
        # Reduce in float32 for stability, then cast back to the input dtype.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


# =============================================================================
# 2. Rotary position embeddings (RoPE)
#
# Position is injected by rotating each (even, odd) pair of query/key channels
# by an angle proportional to the token's position. There is no learned
# positional table, so context length is a runtime choice: any T <= cfg.context
# works with the same weights.
#
# Llama-style formulation: per-channel frequencies are duplicated across the
# two halves of the head dimension and applied as x*cos + rotate_half(x)*sin.
# =============================================================================

def precompute_rope(
    head_dim: int,
    seq_len: int,
    base: float = 10_000.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape [T, d].

    Channel pair i rotates with frequency base^(-2i/d): low channels spin fast
    (fine-grained local position), high channels spin slowly (coarse position).
    Computed in float32 for numerical stability regardless of model dtype.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)          # [T, d/2]: angle per (position, pair)
    emb = torch.cat([freqs, freqs], dim=-1)   # [T, d]: duplicated across halves
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dim and negate the first: the '90-degree
    rotation' partner used by apply_rope."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x: [B, H, T, d] by position. cos/sin: [T, d] broadcast over B, H.

    Each (x1, x2) channel pair becomes (x1*cos - x2*sin, x2*cos + x1*sin) --
    a 2D rotation by the position-dependent angle.
    """
    cos = cos[None, None, :, :]  # [1, 1, T, d]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


# =============================================================================
# 3. Grouped-query attention helpers
#
# GQA uses fewer key/value heads than query heads (Hkv < Hq): several query
# heads share one KV head, shrinking the KV projection parameters and the KV
# cache by Hq/Hkv while keeping the full number of query heads.
# =============================================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand [B, Hkv, T, d] -> [B, Hkv * n_rep, T, d] so each KV head serves
    its group of n_rep = Hq/Hkv query heads in the score matmul."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, t, d)
        .reshape(b, n_kv * n_rep, t, d)
    )


# =============================================================================
# 4. Causal and sliding-window attention masks
# =============================================================================

def build_attn_mask(
    seq_len: int,
    window: int | None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Boolean [T, T] mask, True where query position i may attend to key j.

    window=None : full causal        -- allowed iff j <= i
    window=w    : sliding window     -- allowed iff j <= i and i - j < w
    A windowed layer only sees the last w tokens, so its attention cost is
    O(T*w) instead of O(T^2); periodic full layers restore global reach.
    """
    i = torch.arange(seq_len, device=device)[:, None]  # query positions, [T, 1]
    j = torch.arange(seq_len, device=device)[None, :]  # key positions,   [1, T]
    allowed = j <= i
    if window is not None:
        allowed &= (i - j) < window
    return allowed


# =============================================================================
# 5. Grouped-query attention module
# =============================================================================

class Attention(nn.Module):
    """One attention layer: project -> QK-norm -> RoPE -> GQA expand -> SDPA.

    Per-head QK-norm RMS-normalizes queries and keys before RoPE, which bounds
    the attention-logit magnitude near sqrt(d) and makes the logits invariant
    to the scale of the residual stream -- a stability trick for training.
    """

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_q_heads // cfg.n_kv_heads  # query heads per KV head

        self.q_proj = nn.Linear(cfg.d_model, cfg.q_dim, bias=False)   # D -> Hq*d
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)  # D -> Hkv*d
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)  # D -> Hkv*d
        self.o_proj = nn.Linear(cfg.q_dim, cfg.d_model, bias=False)   # Hq*d -> D

        self.use_triton = getattr(cfg, "use_triton", False)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, use_triton=self.use_triton)
            self.k_norm = RMSNorm(self.head_dim, use_triton=self.use_triton)

        # Layer schedule: full context on every full_attn_every-th layer,
        # sliding window on the rest.
        self.is_full = cfg.is_full_attention_layer(layer_idx)
        self.window = None if self.is_full else cfg.window

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seg_equal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape  # x: [B, T, D]

        # Project and split into heads:
        # [B, T, H*d] -> [B, T, H, d] -> [B, H, T, d]
        q = self.q_proj(x).view(b, t, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Per-head QK-norm, then position via RoPE.
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_triton:
            import kernels

            q = kernels.apply_rope(q, cos, sin)
            k = kernels.apply_rope(k, cos, sin)
        else:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # GQA: expand KV heads to match the query head count.
        # [B, Hkv, T, d] -> [B, Hq, T, d]
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Causal (optionally windowed) mask; scores are [B, Hq, T, T] inside SDPA.
        mask = build_attn_mask(t, self.window, device=x.device)
        if seg_equal is not None:
            # Packed conversations: a query may attend to a key only if they
            # share a segment id. The causal diagonal is always allowed, so no
            # query row is fully masked (padding attends to itself), avoiding a
            # NaN softmax row.
            mask = mask.view(1, 1, t, t) & seg_equal.view(b, 1, t, t)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # Merge heads: [B, Hq, T, d] -> [B, T, Hq*d] -> [B, T, D]
        out = out.transpose(1, 2).reshape(b, t, self.n_q_heads * self.head_dim)
        return self.o_proj(out)


# =============================================================================
# 6. SwiGLU MLP
#
# The gated MLP: down(SiLU(gate(x)) * up(x)), no biases. The hidden width is
# ~8/3 * D (config.swiglu_hidden) so the three matrices cost about the same as
# a plain 4*D GELU MLP's two.
# =============================================================================

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, *, use_triton: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> a, b: [B, T, hidden] -> out: [B, T, D]
        a = self.gate(x)
        b = self.up(x)
        if self.use_triton:
            import kernels

            h = kernels.swiglu(a, b)
        else:
            h = F.silu(a) * b  # elementwise gate: SiLU(a) * b
        return self.down(h)


# =============================================================================
# 7. Transformer block
#
# Pre-norm residual wiring: x = x + Attn(Norm(x)); x = x + MLP(Norm(x)).
# Normalizing the branch input (rather than the residual sum) keeps the
# residual stream an unnormalized accumulator, which trains more stably.
# =============================================================================

class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        use_triton = getattr(cfg, "use_triton", False)
        self.attn_norm = RMSNorm(cfg.d_model, use_triton=use_triton)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, use_triton=use_triton)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden, use_triton=use_triton)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seg_equal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, seg_equal)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# =============================================================================
# 8. The MiniGPT model
# =============================================================================

class MiniGPT(nn.Module):
    """Decoder-only causal Transformer with tied input/output embeddings.

    forward(idx) predicts the next token at every position. With targets it
    also returns the mean cross-entropy; loss_mask restricts the loss to chosen
    positions (assistant tokens in SFT) and segment_ids isolate packed
    conversations from one another in attention.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)          # V -> D
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, use_triton=getattr(cfg, "use_triton", False))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # D -> V

        # Opt-in chunked-cross-entropy loss (kernels.py): forward computes the
        # loss without materializing the [B*T, V] logits and returns
        # logits=None. Off by default so the plain `logits` API is preserved.
        self.fused_loss = False

        self.apply(self._init_weights)
        self._scale_residual_projections()
        # Tie AFTER init so the embedding's initialization is the one kept.
        # One weight matrix serves as both the input embedding (line above)
        # and the output projection -- the classic weight-tying trick.
        self.lm_head.weight = self.embed.weight

    # ---------------------------------------------------------------- init
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        # GPT-2 residual growth control: the two projections that write into
        # the residual stream are scaled by 1/sqrt(2 * n_layers) so the
        # residual variance stays bounded with depth.
        scale = (2 * self.cfg.n_layers) ** -0.5
        for block in self.layers:
            block.attn.o_proj.weight.data.mul_(scale)
            block.mlp.down.weight.data.mul_(scale)

    # ------------------------------------------------------------- forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        b, t = idx.shape  # token IDs: [B, T]
        cos, sin = precompute_rope(
            self.cfg.head_dim, t, self.cfg.rope_base,
            device=idx.device, dtype=self.embed.weight.dtype,
        )

        # Packed-conversation mask: [B, T, T], True where positions share a
        # segment id (see posttrain.pack_conversations).
        seg_equal = None
        if segment_ids is not None:
            seg_equal = segment_ids[:, :, None] == segment_ids[:, None, :]

        x = self.embed(idx)  # [B, T] -> [B, T, D]
        for block in self.layers:
            x = block(x, cos, sin, seg_equal)
        x = self.final_norm(x)

        # Loss masking: positions with loss_mask == 0 get IGNORE_INDEX targets
        # and drop out of the mean cross-entropy (assistant-only SFT loss).
        tgt = None
        if targets is not None:
            tgt = targets.clone()
            if loss_mask is not None:
                tgt = tgt.masked_fill(loss_mask == 0, IGNORE_INDEX)

        # Memory-saving path: stream the vocabulary in chunks instead of
        # materializing the full [B*T, V] logits tensor (see kernels.py).
        if self.fused_loss and tgt is not None:
            import kernels

            loss = kernels.chunked_cross_entropy(
                x.reshape(-1, x.size(-1)),        # [B*T, D]
                self.lm_head.weight,              # [V, D] (the tied embedding)
                tgt.reshape(-1),                  # [B*T]
                ignore_index=IGNORE_INDEX,
                chunk=getattr(self.cfg, "ce_chunk", 8192),
            )
            return None, loss

        logits = self.lm_head(x)  # [B, T, D] -> [B, T, V]
        loss = None
        if tgt is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # [B*T, V]
                tgt.reshape(-1),                      # [B*T]
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    # ------------------------------------------- loss / parameter counting
    def num_parameters(self) -> int:
        """Unique parameter count: the tied embedding/output weight is one
        tensor and is counted once. Matches Config.param_count().total."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    @classmethod
    def from_config(cls, cfg: Config) -> "MiniGPT":
        return cls(cfg)
