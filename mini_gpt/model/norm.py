"""RMSNorm.

Root-mean-square normalization with a learned per-channel gain and no bias, in
pre-norm placement throughout the model. The fused Triton version is tested
against the eager implementation here.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, use_triton: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # When True, forward routes through the fused kernel (which falls back to
        # this same math off CUDA). Defaults False so the module stays a pure
        # eager reference unless a Config opts in.
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            from mini_gpt.kernels import rmsnorm

            return rmsnorm(x, self.weight, self.eps)
        # Reduce in float32 for stability, then cast back to the input dtype.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight
