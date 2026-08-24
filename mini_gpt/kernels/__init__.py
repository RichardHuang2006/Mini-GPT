"""Fused Triton kernels.

Each kernel is a drop-in replacement for an eager op and is tested against its
eager twin in the reference model. Every public function falls back to a
pure-PyTorch path when Triton or CUDA is unavailable, so ``use_triton=True`` is
safe on a CPU box -- it runs the reference math. Dispatch is:

    Triton kernel   iff   input is on CUDA and Triton imported successfully
    eager fallback  otherwise

Both paths produce the same value, which is what the differential tests in
``tests/test_kernels.py`` assert on GPU.
"""

from __future__ import annotations

try:  # Triton ships with the Linux torch wheel, but keep the import soft.
    import triton  # noqa: F401

    HAS_TRITON = True
except Exception:  # pragma: no cover - only on triton-less installs
    HAS_TRITON = False

from mini_gpt.kernels.cross_entropy import chunked_cross_entropy
from mini_gpt.kernels.rmsnorm import rmsnorm
from mini_gpt.kernels.rope_kernel import apply_rope
from mini_gpt.kernels.swiglu import swiglu

__all__ = [
    "HAS_TRITON",
    "rmsnorm",
    "apply_rope",
    "swiglu",
    "chunked_cross_entropy",
]
