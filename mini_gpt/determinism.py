"""Determinism harness.

Every script and test calls :func:`seed_everything` first. Reproducibility is a
precondition for differential testing: a failure deep into a run must reproduce
bit-identically to be debuggable.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 0, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and Torch (CPU + CUDA) and, if requested, switch
    Torch to deterministic algorithms.

    Returns the seed so callers can log it. ``deterministic=False`` keeps the fast
    nondeterministic kernels, for throughput runs that assert no bit-exactness.
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
        # warn_only: a few ops lack a deterministic implementation; warn rather
        # than hard-crash the run.
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
