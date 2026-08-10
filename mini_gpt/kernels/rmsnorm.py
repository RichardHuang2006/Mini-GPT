"""Fused RMSNorm forward + backward.

One pass computes the mean-square reduction, ``rsqrt``, scale, and gain multiply
without round-tripping the normalized activation to HBM. The backward is a
hand-derived analytic gradient (not autograd), which is the whole point: it is
what the fp64 ``gradcheck`` and the eager-vs-kernel compare in ``test_rmsnorm``
hold to account.

Math (row ``x``, gain ``w``, ``r = rsqrt(mean(x^2) + eps)``)::

    y      = x * r * w
    g      = dy * w
    c      = sum_j g_j x_j
    dx     = r * (g - x * (r^2 * c / N))
    dw     = sum_over_rows(dy * (x * r))

Reductions run in float32 regardless of the activation dtype, matching the eager
``RMSNorm`` reference so bf16 inputs agree to tolerance.
"""

from __future__ import annotations

import torch

from mini_gpt.kernels import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X, W, Y, Rstd, stride, N, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        x_ptr = X + row * stride
        y_ptr = Y + row * stride
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)

        y = x * rstd * w
        tl.store(y_ptr + cols, y, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_dx_kernel(
        X, W, DY, Rstd, DX, stride, N, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        x_ptr = X + row * stride
        dy_ptr = DY + row * stride
        dx_ptr = DX + row * stride
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        g = dy * w
        c = tl.sum(g * x, axis=0)
        dx = rstd * (g - x * (rstd * rstd * c / N))
        tl.store(dx_ptr + cols, dx, mask=mask)

    class _RMSNormTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            shape = x.shape
            N = shape[-1]
            x2 = x.reshape(-1, N).contiguous()
            M = x2.shape[0]

            y = torch.empty_like(x2)
            rstd = torch.empty(M, device=x2.device, dtype=torch.float32)
            BLOCK = triton.next_power_of_2(N)
            _rmsnorm_fwd_kernel[(M,)](
                x2, weight, y, rstd, x2.stride(0), N, eps, BLOCK=BLOCK
            )
            ctx.save_for_backward(x2, weight, rstd)
            ctx.eps = eps
            ctx.shape = shape
            return y.reshape(shape)

        @staticmethod
        def backward(ctx, dy):
            x2, weight, rstd = ctx.saved_tensors
            N = x2.shape[-1]
            M = x2.shape[0]
            dy2 = dy.reshape(-1, N).contiguous()

            dx = torch.empty_like(x2)
            BLOCK = triton.next_power_of_2(N)
            _rmsnorm_bwd_dx_kernel[(M,)](
                x2, weight, dy2, rstd, dx, x2.stride(0), N, BLOCK=BLOCK
            )
            # Gain gradient: reduce over rows. Cheap [N] output; done in torch
            # (fp32) to keep the kernel scope to the memory-bound dx.
            x_hat = (x2.float() * rstd[:, None])
            dweight = (dy2.float() * x_hat).sum(dim=0).to(weight.dtype)
            return dx.reshape(ctx.shape), dweight, None


def _rmsnorm_eager(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xf.to(dtype) * weight


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim: ``x * rsqrt(mean(x^2)+eps) * weight``.

    Uses the fused Triton kernel on CUDA and the eager reference elsewhere; both
    paths are numerically equivalent.
    """
    if HAS_TRITON and x.is_cuda:
        return _RMSNormTriton.apply(x, weight, eps)
    return _rmsnorm_eager(x, weight, eps)
