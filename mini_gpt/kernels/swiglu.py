"""Fused SwiGLU gating.

Fuses the elementwise gate ``h = SiLU(a) * b`` where ``a = x·Wg`` and ``b = x·Wu``,
so the two hidden intermediates are never both held expanded in HBM -- only the
single fused product ``h`` is. The surrounding ``gate``/``up``/``down`` matmuls
stay as cuBLAS GEMMs (fusing a GEMM is out of scope).

Backward (``s = sigmoid(a)``, ``silu = a*s``)::

    da = dh * b * (s * (1 + a * (1 - s)))
    db = dh * silu

The forward saves only ``a`` and ``b`` (same size as the output), so the memory
footprint is the eager one minus the second materialized intermediate -- which is
what ``test_swiglu`` checks via a peak-memory delta.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_gpt.kernels import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_fwd_kernel(A, B, H, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(A + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + off, mask=mask, other=0.0).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(-a))
        h = (a * s) * b
        tl.store(H + off, h, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(A, B, DH, DA, DB, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(A + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + off, mask=mask, other=0.0).to(tl.float32)
        dh = tl.load(DH + off, mask=mask, other=0.0).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(-a))
        silu = a * s
        dsilu = s * (1.0 + a * (1.0 - s))
        da = dh * b * dsilu
        db = dh * silu
        tl.store(DA + off, da, mask=mask)
        tl.store(DB + off, db, mask=mask)

    class _SwiGLUTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            a = a.contiguous()
            b = b.contiguous()
            h = torch.empty_like(a)
            n = a.numel()
            grid = (triton.cdiv(n, 1024),)
            _swiglu_fwd_kernel[grid](a, b, h, n, BLOCK=1024)
            ctx.save_for_backward(a, b)
            return h

        @staticmethod
        def backward(ctx, dh):
            a, b = ctx.saved_tensors
            dh = dh.contiguous()
            da = torch.empty_like(a)
            db = torch.empty_like(b)
            n = a.numel()
            grid = (triton.cdiv(n, 1024),)
            _swiglu_bwd_kernel[grid](a, b, dh, da, db, n, BLOCK=1024)
            return da, db


def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused ``SiLU(a) * b`` gating (Triton on CUDA, eager elsewhere)."""
    if HAS_TRITON and a.is_cuda:
        return _SwiGLUTriton.apply(a, b)
    return F.silu(a) * b
