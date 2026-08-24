"""The eager PyTorch model -- the reference implementation.

Stock-op modules with no custom kernels. Every fused Triton kernel is
differential-tested against this, so nothing here imports Triton; clarity is
preferred over speed.
"""
