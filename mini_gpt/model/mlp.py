"""SwiGLU MLP.

``down(SiLU(gate(x)) * up(x))`` with no biases. Hidden width is ~= 8/3 * d_model
rounded to a multiple of 128 (``config.swiglu_hidden``), keeping the gated MLP's
parameter count comparable to a plain 4*d_model GELU MLP.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, *, use_triton: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.gate(x)
        b = self.up(x)
        if self.use_triton:
            from mini_gpt.kernels import swiglu

            h = swiglu(a, b)
        else:
            h = F.silu(a) * b
        return self.down(h)
