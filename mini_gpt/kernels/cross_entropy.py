"""Chunked cross-entropy.

A naive ``F.cross_entropy`` on a ``mini`` micro-batch materializes a
``[B·T, V] = [16·1024, 32768]`` logits tensor (~1 GB bf16) plus its fp32 softmax
upcast and gradient (~4.3 GB total), capping the micro-batch far below what
compute allows. Here the vocabulary is tiled and streamed with an online-softmax
running max / sum-of-exp, dropping peak memory from ``O(B·T·V)`` to
``O(B·T·chunk)``.

The math is exact cross-entropy, so the value and input gradient match
``F.cross_entropy`` to floating-point tolerance for any ``chunk``; that
independence is what ``test_chunked_ce`` asserts alongside the peak-memory delta.

Implemented as a pure-PyTorch autograd ``Function``: the per-chunk work is cuBLAS
GEMMs plus reductions, already at peak, so the win is memory rather than a hand
kernel. Runs identically on CPU and CUDA.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100
DEFAULT_CHUNK = 8192


def _chunks(total: int, chunk: int):
    for start in range(0, total, chunk):
        yield start, min(start + chunk, total)


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    # Upcast only the low-precision activation dtypes for a stable softmax; keep
    # float32/float64 as-is so an fp64 gradcheck exercises the exact path.
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, ignore_index, chunk):
        # hidden: [N, d] (activation), weight: [V, d] (tied embedding),
        # targets: [N] int64. Returns mean loss over non-ignored positions.
        # Autocast is disabled so the explicit float32 upcast below is honored:
        # otherwise autocast re-downcasts the matmuls to bf16/fp16 and the fp32
        # running-softmax buffers collide with bf16 logits.
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
                logits_c = hf @ wf[c0:c1].T  # [N, c1 - c0]
                chunk_max = logits_c.max(dim=1).values
                new_m = torch.maximum(m, chunk_max)
                s = s * torch.exp(m - new_m) + torch.exp(logits_c - new_m[:, None]).sum(dim=1)
                m = new_m

                in_chunk = (targets >= c0) & (targets < c1) & valid
                if in_chunk.any():
                    z[in_chunk] = logits_c[in_chunk, targets[in_chunk] - c0]

            lse = m + torch.log(s)  # [N]
            loss_row = (lse - z) * valid
            loss = loss_row.sum() / n_valid

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

            for c0, c1 in _chunks(V, chunk):
                logits_c = hf @ wf[c0:c1].T
                p_c = torch.exp(logits_c - lse[:, None])  # softmax prob for this chunk
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
    """Mean cross-entropy of ``hidden @ weight.T`` against ``targets``.

    Never materializes the full ``[N, V]`` logits; peak memory is ``O(N*chunk)``.
    Equivalent (to tolerance) to::

        F.cross_entropy(hidden @ weight.T, targets, ignore_index=ignore_index)

    Args:
        hidden:  ``[N, d]`` activations (e.g. flattened ``[B*T, d]``).
        weight:  ``[V, d]`` output projection (the tied embedding weight).
        targets: ``[N]`` int64 class indices; ``ignore_index`` positions dropped.
        chunk:   vocabulary tile size; smaller trades compute for less memory.
    """
    if hidden.dim() != 2:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    return _ChunkedCrossEntropy.apply(hidden, weight, targets, ignore_index, chunk)
