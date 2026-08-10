"""Grouped-query attention with RoPE, QK-norm, and sliding windows.

* Queries use the full head count; keys/values a smaller one (GQA), repeated to
  match at attention time.
* Queries and keys are RMS-normalized per head before RoPE (QK-norm), which
  bounds attention-logit magnitude -- the cheapest single stabilizer for small
  models.
* A layer is either full-causal or sliding-window (window ``w``): position i
  attends to j iff ``j <= i`` and ``i - j < w``.

Attention itself uses ``F.scaled_dot_product_attention`` -- the stock op the
eager oracle is allowed to lean on; a fused-attention kernel is future work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_gpt.model.norm import RMSNorm
from mini_gpt.model.rope import apply_rope as apply_rope_eager


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``[B, n_kv, T, D]`` to ``[B, n_kv * n_rep, T, D]`` for GQA."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, t, d)
        .reshape(b, n_kv * n_rep, t, d)
    )


def build_attn_mask(
    seq_len: int,
    window: int | None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Boolean ``[T, T]`` mask, True where attention is allowed.

    ``window=None`` is full causal; otherwise causal within a sliding window.
    """
    i = torch.arange(seq_len, device=device)[:, None]
    j = torch.arange(seq_len, device=device)[None, :]
    allowed = j <= i
    if window is not None:
        allowed &= (i - j) < window
    return allowed


class Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_q_heads // cfg.n_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.d_model, cfg.q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.d_model, bias=False)

        self.use_triton = getattr(cfg, "use_triton", False)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, use_triton=self.use_triton)
            self.k_norm = RMSNorm(self.head_dim, use_triton=self.use_triton)

        # Full-context on every full_attn_every-th layer, sliding on the rest.
        self.is_full = cfg.is_full_attention_layer(layer_idx)
        self.window = None if self.is_full else cfg.window

    def _rope_fn(self):
        if self.use_triton:
            from mini_gpt.kernels import apply_rope

            return apply_rope
        return apply_rope_eager

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seg_equal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        rope = self._rope_fn()
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        mask = build_attn_mask(t, self.window, device=x.device)
        if seg_equal is not None:
            # Block attention across packed conversation boundaries: a query may
            # attend to a key only if they share a segment. The
            # causal diagonal is always allowed, so no query row is fully masked
            # (padding attends to itself), avoiding a NaN softmax row.
            mask = mask.view(1, 1, t, t) & seg_equal.view(b, 1, t, t)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(b, t, self.n_q_heads * self.head_dim)
        return self.o_proj(out)
