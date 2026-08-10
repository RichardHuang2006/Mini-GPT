"""The eager PyTorch model -- the oracle.

Pure stock-op modules with no custom kernels. This is the known-correct
reference every fused Triton kernel is differential-tested against, so it never
imports Triton and is meant to be obviously correct rather than fast.
"""
