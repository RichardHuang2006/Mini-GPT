"""Rotary position embedding.

Position is injected by rotating query/key head pairs, so there is no learned
positional table and context length is a runtime choice. The RoPE base frequency
is the only knob context extension rescales (1024 -> 2048).

Llama-style formulation: frequencies are duplicated to the full head dimension
and applied as ``x * cos + rotate_half(x) * sin``.
"""

from __future__ import annotations

import torch


def precompute_rope(
    head_dim: int,
    seq_len: int,
    base: float = 10_000.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` each of shape ``[seq_len, head_dim]``.

    Computed in float32 for numerical stability regardless of the model dtype;
    the caller casts to the activation dtype when applying.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)          # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)   # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def scale_rope_base(base: float, old_ctx: int, new_ctx: int, head_dim: int) -> float:
    """NTK-aware RoPE base for extending context ``old_ctx -> new_ctx``.

    Stretching the base frequency by ``s = new_ctx/old_ctx``, rather than
    interpolating positions, keeps high-frequency resolution while lengthening
    the low-frequency wavelength. NTK scaling multiplies the base by
    ``s ** (d/(d-2))``; a short continued-training phase adapts the model to the
    new base. Returns ``base`` unchanged when ``new_ctx <= old_ctx``.
    """
    if new_ctx <= old_ctx:
        return base
    s = new_ctx / old_ctx
    return base * (s ** (head_dim / (head_dim - 2)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``[B, H, T, head_dim]``.

    ``cos``/``sin`` are ``[T, head_dim]`` and broadcast over batch and heads.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin
