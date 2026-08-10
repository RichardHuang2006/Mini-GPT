"""Determinism harness.

Every script and test calls :func:`seed_everything` first. Reproducibility is
the precondition for the whole differential-testing strategy: a failure at step
41,382 must reproduce bit-identically or debugging is dead.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 0, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and Torch (CPU + CUDA) and, if requested, switch
    Torch to deterministic algorithms.

    Returns the seed so callers can log it. ``deterministic=False`` keeps the
    fast nondeterministic kernels for throughput runs where bit-exactness is not
    being asserted.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # CPU-only tooling (tokenizer, packing) needs no torch
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this set before use to make matmuls reproducible.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some ops (rare in this project) lack a deterministic impl;
        # do not hard-crash a run over them, just surface a warning.
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
