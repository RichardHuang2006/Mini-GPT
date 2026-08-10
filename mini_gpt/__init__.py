"""Mini-GPT: a from-scratch GPT pretraining and post-training pipeline.

An eager, obviously-correct reference model is the oracle; the fused Triton
kernels and the rest of the fast path are validated by differential testing
against it. See the README for the architecture and the module map.
"""

__version__ = "0.1.0"
