"""Fused Triton kernels with eager PyTorch references and automatic fallback.

What this file teaches
    How custom GPU kernels slot into a PyTorch model: each operation has
      1. a plain PyTorch *reference* implementation (the ground truth),
      2. a Triton kernel wrapped in a torch.autograd.Function with a
         hand-derived backward,
      3. a public dispatcher that uses the kernel on CUDA and the reference
         everywhere else, so `use_triton=True` is safe on a CPU-only box.
    test_minigpt.py compares each kernel's forward value and backward gradient
    against its reference on CUDA (a "differential test").

Read first
    model.py (the eager modules these kernels accelerate).

Operations and shapes
    rmsnorm(x [.., N], weight [N])            -> [.., N]
    apply_rope(x [B, H, T, d], cos/sin [T, d])-> [B, H, T, d]
    swiglu(a [..], b [..])                    -> [..]   (elementwise SiLU(a)*b)
    chunked_cross_entropy(hidden [N, D], weight [V, D], targets [N]) -> scalar

Representative command (differential tests; the CUDA ones skip without a GPU):
    python -m pytest test_minigpt.py -k kernel -q

No speed or memory numbers are claimed here; the one memory property that is
guaranteed by construction -- chunked cross-entropy never materializes the full
[N, V] logits tensor -- is asserted directly by a peak-memory test on CUDA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # Triton ships with the Linux torch wheel, but keep the import soft.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - only on triton-less installs
    HAS_TRITON = False

IGNORE_INDEX = -100
DEFAULT_CHUNK = 8192

__all__ = [
    "HAS_TRITON",
    "rmsnorm",
    "rmsnorm_reference",
    "apply_rope",
    "rope_reference",
    "swiglu",
    "swiglu_reference",
    "chunked_cross_entropy",
]


# =============================================================================
# 1. RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight, over the last dim.
# =============================================================================

def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eager reference. x: [.., N], weight: [N] -> [.., N].

    Reductions run in float32 regardless of the activation dtype (matching the
    kernel), so bf16 inputs agree with the fused path to tolerance.
    """
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xf.to(dtype) * weight


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(X, W, Y, Rstd, stride, N, eps, BLOCK: tl.constexpr):
        # One program per row: compute the mean-square reduction, rsqrt, scale,
        # and gain multiply in a single pass over the row.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)  # saved for the backward pass

        tl.store(Y + row * stride + cols, x * rstd * w, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_dx_kernel(X, W, DY, Rstd, DX, stride, N, BLOCK: tl.constexpr):
        # Analytic input gradient (r = rstd, g = dy * w, c = sum_j g_j x_j):
        #   dx = r * (g - x * (r^2 * c / N))
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        g = dy * w
        c = tl.sum(g * x, axis=0)
        dx = rstd * (g - x * (rstd * rstd * c / N))
        tl.store(DX + row * stride + cols, dx, mask=mask)

    class _RMSNormTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            shape = x.shape
            N = shape[-1]
            x2 = x.reshape(-1, N).contiguous()  # [M, N]: one kernel program per row
            M = x2.shape[0]

            y = torch.empty_like(x2)
            rstd = torch.empty(M, device=x2.device, dtype=torch.float32)
            BLOCK = triton.next_power_of_2(N)
            _rmsnorm_fwd_kernel[(M,)](x2, weight, y, rstd, x2.stride(0), N, eps, BLOCK=BLOCK)
            ctx.save_for_backward(x2, weight, rstd)
            ctx.shape = shape
            return y.reshape(shape)

        @staticmethod
        def backward(ctx, dy):
            x2, weight, rstd = ctx.saved_tensors
            M, N = x2.shape
            dy2 = dy.reshape(-1, N).contiguous()

            dx = torch.empty_like(x2)
            BLOCK = triton.next_power_of_2(N)
            _rmsnorm_bwd_dx_kernel[(M,)](x2, weight, dy2, rstd, dx, x2.stride(0), N, BLOCK=BLOCK)
            # Gain gradient: a cheap [N]-sized reduction over rows, done in
            # torch fp32 to keep the kernel scoped to the memory-bound dx.
            x_hat = x2.float() * rstd[:, None]
            dweight = (dy2.float() * x_hat).sum(dim=0).to(weight.dtype)
            return dx.reshape(ctx.shape), dweight, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim. Triton kernel on CUDA, eager reference
    elsewhere; both paths compute the same value."""
    if HAS_TRITON and x.is_cuda:
        return _RMSNormTriton.apply(x, weight, eps)
    return rmsnorm_reference(x, weight, eps)


# =============================================================================
# 2. RoPE: y = x * cos + rotate_half(x) * sin, per position.
#
# The rotation is an orthogonal linear map per position, so the backward is the
# same op with the sign of sin flipped: grad_x = dy*cos + rotate_half(dy)*(-sin).
# One kernel therefore serves both directions. cos/sin are precomputed
# constants and receive no gradient.
# =============================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Eager reference. x: [B, H, T, d]; cos/sin: [T, d] broadcast over B, H."""
    return x * cos[None, None, :, :] + _rotate_half(x) * sin[None, None, :, :]


if HAS_TRITON:

    @triton.jit
    def _rope_kernel(X, COS, SIN, O, T, D, HALF, BLOCK: tl.constexpr):
        # One program per (B*H*T) row; each row rotates its d-vector in place.
        # cos/sin stay [T, d] -- no [B, H, T, d] expanded copies are built.
        row = tl.program_id(0)
        pos = row % T  # position index into cos/sin (contiguous [.., T, D] layout)

        cols = tl.arange(0, BLOCK)
        mask = cols < D

        x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(COS + pos * D + cols, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(SIN + pos * D + cols, mask=mask, other=0.0).to(tl.float32)

        # rotate_half via a shifted load: col c reads its partner channel and
        # negates it when c is in the first half.
        shifted = tl.where(cols < HALF, cols + HALF, cols - HALF)
        x_rot = tl.load(X + row * D + shifted, mask=mask, other=0.0).to(tl.float32)
        sign = tl.where(cols < HALF, -1.0, 1.0)

        tl.store(O + row * D + cols, x * cos + sign * x_rot * sin, mask=mask)

    def _rope_apply_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        xc = x.contiguous()
        out = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _rope_kernel[(b * h * t,)](
            xc, cos.contiguous(), sin.contiguous(), out, t, d, d // 2, BLOCK=BLOCK
        )
        return out

    class _RoPETriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cos, sin):
            ctx.save_for_backward(cos, sin)
            return _rope_apply_triton(x, cos, sin)

        @staticmethod
        def backward(ctx, grad_out):
            cos, sin = ctx.saved_tensors
            # Inverse rotation = same kernel with -sin.
            return _rope_apply_triton(grad_out.contiguous(), cos, -sin), None, None


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x: [B, H, T, d]. Triton on CUDA, eager elsewhere."""
    if HAS_TRITON and x.is_cuda:
        return _RoPETriton.apply(x, cos, sin)
    return rope_reference(x, cos, sin)


# =============================================================================
# 3. SwiGLU gating: h = SiLU(a) * b, elementwise.
#
# Fusing the gate means only the product h is written back to memory, not the
# SiLU(a) intermediate. The surrounding gate/up/down matmuls stay cuBLAS GEMMs.
# Backward (s = sigmoid(a)):
#   da = dh * b * (s * (1 + a * (1 - s)))       (the SiLU derivative)
#   db = dh * (a * s)                            (= dh * SiLU(a))
# =============================================================================

def swiglu_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Eager reference: SiLU(a) * b, any matching shapes."""
    return F.silu(a) * b


if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(A, B, H, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(A + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + off, mask=mask, other=0.0).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(-a))
        tl.store(H + off, (a * s) * b, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(A, B, DH, DA, DB, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(A + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + off, mask=mask, other=0.0).to(tl.float32)
        dh = tl.load(DH + off, mask=mask, other=0.0).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(-a))
        silu = a * s
        dsilu = s * (1.0 + a * (1.0 - s))
        tl.store(DA + off, dh * b * dsilu, mask=mask)
        tl.store(DB + off, dh * silu, mask=mask)

    class _SwiGLUTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            a = a.contiguous()
            b = b.contiguous()
            h = torch.empty_like(a)
            n = a.numel()
            _swiglu_fwd_kernel[(triton.cdiv(n, 1024),)](a, b, h, n, BLOCK=1024)
            ctx.save_for_backward(a, b)
            return h

        @staticmethod
        def backward(ctx, dh):
            a, b = ctx.saved_tensors
            dh = dh.contiguous()
            da = torch.empty_like(a)
            db = torch.empty_like(b)
            n = a.numel()
            _swiglu_bwd_kernel[(triton.cdiv(n, 1024),)](a, b, dh, da, db, n, BLOCK=1024)
            return da, db


def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(a) * b gating. Triton on CUDA, eager elsewhere."""
    if HAS_TRITON and a.is_cuda:
        return _SwiGLUTriton.apply(a, b)
    return swiglu_reference(a, b)


# =============================================================================
# 4. Chunked cross-entropy.
#
# A naive F.cross_entropy over next-token logits first materializes the full
# [B*T, V] logits tensor. With V = 32,768 that tensor (plus its fp32 softmax
# upcast and gradient) dominates peak memory. Here the vocabulary is processed
# in chunks with an online softmax -- a running max `m` and running
# sum-of-exponentials `s` are updated chunk by chunk, exactly like FlashAttention
# handles its softmax -- so peak memory is O(B*T*chunk) instead of O(B*T*V).
#
# The math is exact cross-entropy: loss and gradients match F.cross_entropy to
# floating-point tolerance for ANY chunk size (asserted in test_minigpt.py).
#
# This is a pure-PyTorch autograd.Function rather than a hand-written Triton
# kernel: the per-chunk work is cuBLAS GEMMs plus reductions, so the win here
# is memory, not a faster kernel. It runs identically on CPU and CUDA.
# =============================================================================

def _chunks(total: int, chunk: int):
    for start in range(0, total, chunk):
        yield start, min(start + chunk, total)


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    # Upcast only low-precision activation dtypes for a stable softmax; keep
    # float32/float64 as-is so an fp64 gradcheck exercises the exact path.
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, ignore_index, chunk):
        # hidden: [N, D] activations; weight: [V, D] (the tied embedding);
        # targets: [N] int64. Returns the mean loss over non-ignored positions.
        # Autocast is disabled so the explicit float32 upcast is honored;
        # otherwise autocast would re-downcast the matmuls and the fp32
        # running-softmax buffers would collide with bf16 logits.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            N, _ = hidden.shape
            V = weight.shape[0]
            cdtype = _compute_dtype(hidden.dtype)
            hf = hidden.to(cdtype)
            wf = weight.to(cdtype)

            valid = targets != ignore_index
            n_valid = valid.sum().clamp(min=1)

            m = torch.full((N,), float("-inf"), device=hidden.device, dtype=cdtype)
            s = torch.zeros(N, device=hidden.device, dtype=cdtype)
            z = torch.zeros(N, device=hidden.device, dtype=cdtype)  # target logits

            for c0, c1 in _chunks(V, chunk):
                logits_c = hf @ wf[c0:c1].T  # [N, c1-c0]: one vocab tile
                chunk_max = logits_c.max(dim=1).values
                new_m = torch.maximum(m, chunk_max)
                # Online softmax: rescale the old sum to the new max, add the
                # new chunk's exponentials.
                s = s * torch.exp(m - new_m) + torch.exp(logits_c - new_m[:, None]).sum(dim=1)
                m = new_m

                in_chunk = (targets >= c0) & (targets < c1) & valid
                if in_chunk.any():
                    z[in_chunk] = logits_c[in_chunk, targets[in_chunk] - c0]

            lse = m + torch.log(s)               # log-sum-exp over the full vocab, [N]
            loss = ((lse - z) * valid).sum() / n_valid  # CE = lse - target_logit

        ctx.save_for_backward(hidden, weight, targets, lse, valid)
        ctx.n_valid = n_valid
        ctx.ignore_index = ignore_index
        ctx.chunk = chunk
        return loss.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, targets, lse, valid = ctx.saved_tensors
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            V = weight.shape[0]
            chunk = ctx.chunk
            cdtype = _compute_dtype(hidden.dtype)
            hf = hidden.to(cdtype)
            wf = weight.to(cdtype)

            scale = grad_loss.to(cdtype) / ctx.n_valid
            grad_hidden = torch.zeros_like(hf)
            grad_weight = torch.zeros_like(wf)
            valid_f = valid.to(cdtype)[:, None]

            # dCE/dlogit = softmax(logits) - one_hot(target); recompute each
            # vocab tile's logits from the saved lse instead of storing them.
            for c0, c1 in _chunks(V, chunk):
                logits_c = hf @ wf[c0:c1].T
                p_c = torch.exp(logits_c - lse[:, None])  # softmax probs for this tile
                in_chunk = (targets >= c0) & (targets < c1) & valid
                if in_chunk.any():
                    p_c[in_chunk, targets[in_chunk] - c0] -= 1.0
                p_c = p_c * valid_f * scale
                grad_hidden += p_c @ wf[c0:c1]
                grad_weight[c0:c1] += p_c.T @ hf

        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), None, None, None


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    chunk: int = DEFAULT_CHUNK,
) -> torch.Tensor:
    """Mean cross-entropy of `hidden @ weight.T` against `targets`, streaming
    the vocabulary so the full [N, V] logits tensor is never materialized.

    Equivalent (to floating-point tolerance) to:
        F.cross_entropy(hidden @ weight.T, targets, ignore_index=ignore_index)

    Args:
        hidden:  [N, D] activations (e.g. flattened [B*T, D]).
        weight:  [V, D] output projection (the tied embedding weight).
        targets: [N] int64 class indices; ignore_index positions are dropped.
        chunk:   vocabulary tile size; smaller trades compute for less memory.
    """
    if hidden.dim() != 2:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    return _ChunkedCrossEntropy.apply(hidden, weight, targets, ignore_index, chunk)
