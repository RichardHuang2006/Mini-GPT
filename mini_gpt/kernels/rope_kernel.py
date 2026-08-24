"""Fused RoPE forward + backward.

The query/key rotation ``y = x*cos + rotate_half(x)*sin`` as one elementwise
kernel with no sin/cos-expanded ``[B, H, T, D]`` tensors: ``cos`` and ``sin``
stay ``[T, D]`` and each row reads its own position.

The rotation is an orthogonal linear map per position, so the backward is the
same op with the sign of ``sin`` flipped::

    grad_x = dy*cos + rotate_half(dy)*(-sin)

so the forward kernel serves the backward too. ``cos``/``sin`` are precomputed
constants and receive no gradient.
"""

from __future__ import annotations

import torch

from mini_gpt.kernels import HAS_TRITON
from mini_gpt.model.rope import apply_rope as _apply_rope_eager  # eager reference

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_kernel(X, COS, SIN, O, T, D, HALF, BLOCK: tl.constexpr):
        # One program per (B*H*T) row; each row rotates its D-vector in place.
        row = tl.program_id(0)
        pos = row % T  # position index into cos/sin (contiguous [.., T, D] layout)

        x_ptr = X + row * D
        o_ptr = O + row * D
        cs_ptr = COS + pos * D
        sn_ptr = SIN + pos * D

        cols = tl.arange(0, BLOCK)
        mask = cols < D

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cs_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sn_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # rotate_half: first half -> -x[second half], second half -> x[first half]
        shifted = tl.where(cols < HALF, cols + HALF, cols - HALF)
        x_rot = tl.load(x_ptr + shifted, mask=mask, other=0.0).to(tl.float32)
        sign = tl.where(cols < HALF, -1.0, 1.0)
        rot = sign * x_rot

        y = x * cos + rot * sin
        tl.store(o_ptr + cols, y, mask=mask)

    def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        xc = x.contiguous()
        out = torch.empty_like(xc)
        M = b * h * t
        BLOCK = triton.next_power_of_2(d)
        _rope_kernel[(M,)](
            xc, cos.contiguous(), sin.contiguous(), out, t, d, d // 2, BLOCK=BLOCK
        )
        return out

    class _RoPETriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cos, sin):
            ctx.save_for_backward(cos, sin)
            return _rope_apply(x, cos, sin)

        @staticmethod
        def backward(ctx, grad_out):
            cos, sin = ctx.saved_tensors
            grad_x = _rope_apply(grad_out.contiguous(), cos, -sin)
            return grad_x, None, None


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``[B, H, T, head_dim]``.

    Fused Triton kernel on CUDA, the eager rotation elsewhere.
    """
    if HAS_TRITON and x.is_cuda:
        return _RoPETriton.apply(x, cos, sin)
    return _apply_rope_eager(x, cos, sin)
