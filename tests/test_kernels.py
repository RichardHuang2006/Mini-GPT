"""The fused kernels vs their eager twins.

Two kinds of checks:

* **Reference math is correct** -- an fp64 ``gradcheck`` on the analytic definition
  each kernel implements (runs on CPU, no GPU needed).
* **The kernel equals the reference** -- forward value and backward gradient match
  the eager op on CUDA, in fp32 (tight) and bf16 (loose); plus a 200-step
  fixed-seed training-equivalence run and the chunked-CE peak-memory delta.

The kernels fall back to eager off CUDA, so the "kernel vs eager" assertions are
only meaningful on a GPU and are marked ``requires_cuda``. The chunked-CE
correctness check is pure PyTorch and runs everywhere.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from config import Config
from mini_gpt.determinism import seed_everything
from mini_gpt.kernels import apply_rope, chunked_cross_entropy, rmsnorm, swiglu
from mini_gpt.model.rope import apply_rope as rope_eager, precompute_rope
from mini_gpt.model.transformer import MiniGPT

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a CUDA device"
)


def _rmsnorm_ref(x, weight, eps=1e-6):
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * weight


# ==========================================================================
# fp64 gradcheck on the analytic definition each kernel implements (CPU)
# ==========================================================================

def test_rmsnorm_reference_gradcheck():
    x = torch.randn(6, 32, dtype=torch.float64, requires_grad=True)
    w = torch.randn(32, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_rmsnorm_ref, (x, w), eps=1e-6, atol=1e-6)


def test_rope_reference_gradcheck():
    cos, sin = precompute_rope(16, 8, dtype=torch.float64)
    x = torch.randn(2, 3, 8, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: rope_eager(t, cos, sin), (x,), atol=1e-6)


def test_swiglu_reference_gradcheck():
    a = torch.randn(5, 12, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, 12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, y: F.silu(x) * y, (a, b), atol=1e-6)


def test_chunked_ce_reference_gradcheck():
    # The chunked kernel preserves fp64, so gradcheck exercises its exact backward.
    torch.manual_seed(0)
    hidden = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(20, 8, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 20, (7,))
    assert torch.autograd.gradcheck(
        lambda h, w: chunked_cross_entropy(h, w, targets, chunk=6),
        (hidden, weight),
        atol=1e-6,
    )


# ==========================================================================
# chunked cross-entropy vs F.cross_entropy (runs on CPU + CUDA)
# ==========================================================================

@pytest.mark.parametrize("chunk", [1, 7, 64, 100_000])
def test_chunked_ce_matches_cross_entropy(chunk):
    torch.manual_seed(0)
    N, d, V = 40, 32, 200
    hidden = torch.randn(N, d, requires_grad=True)
    weight = torch.randn(V, d, requires_grad=True)
    targets = torch.randint(0, V, (N,))
    targets[::5] = -100  # some ignored positions

    loss = chunked_cross_entropy(hidden, weight, targets, ignore_index=-100, chunk=chunk)
    loss.backward()

    he = hidden.detach().clone().requires_grad_()
    we = weight.detach().clone().requires_grad_()
    ref = F.cross_entropy(he @ we.T, targets, ignore_index=-100)
    ref.backward()

    assert torch.allclose(loss, ref, atol=1e-5), f"loss {loss.item()} vs {ref.item()}"
    assert torch.allclose(hidden.grad, he.grad, atol=1e-5)
    assert torch.allclose(weight.grad, we.grad, atol=1e-5)


def test_chunked_ce_all_ignored_is_finite():
    # A fully-masked batch must not divide by zero.
    hidden = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(10, 8, requires_grad=True)
    targets = torch.full((4,), -100)
    loss = chunked_cross_entropy(hidden, weight, targets, ignore_index=-100)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.all(hidden.grad == 0)


@requires_cuda
def test_chunked_ce_peak_memory_scales_with_chunk():
    # Peak is O(N*chunk), not O(N*V): a small chunk must use markedly less
    # memory than materializing the full [N, V] logits.
    N, d, V = 4096, 256, 8192
    hidden = torch.randn(N, d, device="cuda", requires_grad=True)
    weight = torch.randn(V, d, device="cuda", requires_grad=True)
    targets = torch.randint(0, V, (N,), device="cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    loss = F.cross_entropy(hidden @ weight.T, targets)
    loss.backward()
    torch.cuda.synchronize()
    naive_peak = torch.cuda.max_memory_allocated() - base

    hidden.grad = None
    weight.grad = None
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.max_memory_allocated()
    loss = chunked_cross_entropy(hidden, weight, targets, chunk=512)
    loss.backward()
    torch.cuda.synchronize()
    chunked_peak = torch.cuda.max_memory_allocated() - base

    assert chunked_peak < naive_peak, f"chunked {chunked_peak} !< naive {naive_peak}"


# ==========================================================================
# Steps 6.1-6.3 -- Triton kernels match the eager op on CUDA
# ==========================================================================

@requires_cuda
@pytest.mark.parametrize("shape", [(8, 64), (4, 7, 128), (2, 4, 16, 48)])
def test_rmsnorm_matches_eager(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", requires_grad=True)
    w = torch.randn(shape[-1], device="cuda", requires_grad=True)
    y = rmsnorm(x, w, 1e-6)
    g = torch.randn_like(y)
    y.backward(g)

    xe = x.detach().clone().requires_grad_()
    we = w.detach().clone().requires_grad_()
    ye = _rmsnorm_ref(xe, we)
    ye.backward(g)

    assert torch.allclose(y, ye, atol=1e-5)
    assert torch.allclose(x.grad, xe.grad, atol=1e-4)
    assert torch.allclose(w.grad, we.grad, atol=1e-3)


@requires_cuda
def test_rmsnorm_bf16_forward_tolerance():
    torch.manual_seed(0)
    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    y = rmsnorm(x, w)
    ye = _rmsnorm_ref(x.float(), w.float()).to(torch.bfloat16)
    assert torch.allclose(y, ye, atol=2e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("base", [10_000.0, 40_000.0])  # extension rescales base
def test_rope_matches_eager(base):
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 32, 64
    cos, sin = precompute_rope(D, T, base=base, device="cuda")
    x = torch.randn(B, H, T, D, device="cuda", requires_grad=True)
    y = apply_rope(x, cos, sin)
    g = torch.randn_like(y)
    y.backward(g)

    xe = x.detach().clone().requires_grad_()
    ye = rope_eager(xe, cos, sin)
    ye.backward(g)

    assert torch.allclose(y, ye, atol=1e-5)
    assert torch.allclose(x.grad, xe.grad, atol=1e-4)


@requires_cuda
def test_swiglu_matches_eager():
    torch.manual_seed(0)
    a = torch.randn(64, 256, device="cuda", requires_grad=True)
    b = torch.randn(64, 256, device="cuda", requires_grad=True)
    h = swiglu(a, b)
    g = torch.randn_like(h)
    h.backward(g)

    ae = a.detach().clone().requires_grad_()
    be = b.detach().clone().requires_grad_()
    he = F.silu(ae) * be
    he.backward(g)

    assert torch.allclose(h, he, atol=1e-5)
    assert torch.allclose(a.grad, ae.grad, atol=1e-4)
    assert torch.allclose(b.grad, be.grad, atol=1e-4)


# ==========================================================================
# training equivalence, kernels on vs off
# ==========================================================================

def _equiv_cfg(**overrides) -> Config:
    base = dict(
        name="tiny",
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=16,
        mlp_hidden=128,
        context=32,
        window=16,
        micro_batch=8,
        grad_accum=1,
        warmup_steps=5,
        max_steps=200,
        use_muon=False,
        compile=False,
        dtype="float32",
        ce_chunk=64,  # force the vocab-tiling path (V=512 -> 8 chunks)
    )
    base.update(overrides)
    return Config(**base)


@requires_cuda
def test_training_equivalence():
    # The fully-fused path (Triton norms/rope/swiglu + chunked CE) must track the
    # eager path's loss curve. We overfit one fixed
    # batch for 80 steps: unlike a stream of random-token batches -- which is
    # unlearnable and just jitters at the entropy floor where fp rounding is
    # chaotic -- overfitting is contractive, so the tiny fp difference between the
    # two paths stays bounded and the curves are effectively identical. SDPA's
    # memory-efficient backward is non-deterministic, so both runs pin the
    # deterministic math backend, isolating the kernels as the only difference.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    steps = 80

    seed_everything(1)
    xb = torch.randint(0, 512, (8, 32), device=dev)
    yb = torch.randint(0, 512, (8, 32), device=dev)

    def run(use_triton):
        from mini_gpt.train.optim import build_optimizer
        from mini_gpt.train.schedule import build_scheduler

        cfg = _equiv_cfg(use_triton=use_triton, max_steps=steps, lr_adamw=1e-3)
        seed_everything(0)
        model = MiniGPT(cfg).to(dev)
        model.fused_loss = cfg.use_triton
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg)
        out = []
        with sdpa_kernel(SDPBackend.MATH):
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                _, loss = model(xb, yb)
                loss.backward()
                opt.step()
                sched.step()
                out.append(loss.item())
        return out

    a = run(False)
    b = run(True)
    max_diff = max(abs(x - y) for x, y in zip(a, b))
    assert max_diff < 1e-3, f"loss curves diverged: max |Δ| = {max_diff:.2e}"
    assert b[-1] < b[0] - 1.0, f"fused-kernel run did not train: {b[0]:.2f} -> {b[-1]:.2f}"


def test_fused_loss_matches_eager_loss_cpu():
    # On CPU the kernels fall back to eager, but the fused-loss *plumbing*
    # (chunked CE returning None logits) must still equal F.cross_entropy.
    cfg = _equiv_cfg()
    seed_everything(0)
    model = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.context))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.context))

    _, eager_loss = model(x, y)
    model.fused_loss = True
    logits, fused_loss = model(x, y)
    assert logits is None
    assert torch.allclose(eager_loss, fused_loss, atol=1e-5)
