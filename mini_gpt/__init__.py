"""Mini-GPT: a from-scratch GPT pretraining and post-training pipeline.

The fused Triton kernels and the rest of the fast path are validated by
differential testing against the eager reference model. See the README for the
architecture and module map.
"""

__version__ = "0.1.0"
