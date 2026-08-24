"""The eager model as the reference implementation.

Every module is checked against a hand-written reference or a stock op, and the
whole model must drive one fixed batch to near-zero loss.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from config import Config, TIERS, get_config, swiglu_hidden  # noqa: E402
from mini_gpt.determinism import seed_everything  # noqa: E402
from mini_gpt.model.attention import Attention, build_attn_mask, repeat_kv  # noqa: E402
from mini_gpt.model.mlp import SwiGLU  # noqa: E402
from mini_gpt.model.norm import RMSNorm  # noqa: E402
from mini_gpt.model.rope import apply_rope, precompute_rope  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402


def tiny_config(**overrides) -> Config:
    base = dict(
        name="tiny",
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=16,
        mlp_hidden=swiglu_hidden(64),
        context=16,
        window=8,
    )
    base.update(overrides)
    return Config(**base)


# --------------------------------------------------------------------------
# RMSNorm and SwiGLU vs stock references
# --------------------------------------------------------------------------

def test_rmsnorm_matches_functional():
    seed_everything(0)
    dim = 48
    norm = RMSNorm(dim)
    norm.weight.data.normal_()
    x = torch.randn(4, 10, dim)
    ref = F.rms_norm(x, (dim,), weight=norm.weight, eps=norm.eps)
    assert torch.allclose(norm(x), ref, atol=1e-5, rtol=1e-5)


def test_swiglu_matches_manual():
    seed_everything(0)
    mlp = SwiGLU(32, 64)
    x = torch.randn(3, 5, 32)
    ref = mlp.down(F.silu(mlp.gate(x)) * mlp.up(x))
    assert torch.allclose(mlp(x), ref, atol=1e-6)


def test_mini_mlp_hidden_is_1408():
    assert get_config("mini").mlp_hidden == 1408
    assert TIERS["mini"].mlp_hidden == swiglu_hidden(512)


# --------------------------------------------------------------------------
# RoPE / GQA / QK-norm attention vs a hand reference
# --------------------------------------------------------------------------

def _reference_attention(attn: Attention, x, cos, sin):
    """Recompute attention from the module's weights, independently of SDPA.

    Returns (output, pre-softmax logits) so callers can inspect logit magnitude.
    """
    b, t, _ = x.shape
    hd, nq, nkv = attn.head_dim, attn.n_q_heads, attn.n_kv_heads

    def proj(w, n):
        return (x @ w.t()).view(b, t, n, hd).transpose(1, 2)

    q = proj(attn.q_proj.weight, nq)
    k = proj(attn.k_proj.weight, nkv)
    v = proj(attn.v_proj.weight, nkv)

    def rms(z, w):
        z32 = z.float()
        z32 = z32 * torch.rsqrt(z32.pow(2).mean(-1, keepdim=True) + 1e-6)
        return z32.to(z.dtype) * w

    if attn.qk_norm:
        q = rms(q, attn.q_norm.weight)
        k = rms(k, attn.k_norm.weight)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    k = repeat_kv(k, attn.n_rep)
    v = repeat_kv(v, attn.n_rep)

    logits = (q @ k.transpose(-1, -2)) * attn.scale
    mask = build_attn_mask(t, attn.window)
    logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = probs @ v
    out = out.transpose(1, 2).reshape(b, t, nq * hd)
    return out @ attn.o_proj.weight.t(), logits


def test_attention_matches_reference():
    seed_everything(0)
    cfg = tiny_config()
    attn = Attention(cfg, layer_idx=1)  # full-attention layer (idx+1 % 2 == 0)
    x = torch.randn(2, cfg.context, cfg.d_model)
    cos, sin = precompute_rope(cfg.head_dim, cfg.context, cfg.rope_base)
    ref_out, _ = _reference_attention(attn, x, cos, sin)
    assert torch.allclose(attn(x, cos, sin), ref_out, atol=1e-5, rtol=1e-4)


def test_sliding_window_mask_differs_from_full():
    full = build_attn_mask(10, None)
    windowed = build_attn_mask(10, window=3)
    # Both causal (upper triangle excluded).
    assert not full[0].tolist()[1:].count(True)  # row 0 attends only to itself
    # The windowed mask forbids positions farther back than the window.
    assert full[9, 0].item() and not windowed[9, 0].item()
    # Windowed is a strict subset of full.
    assert torch.all(windowed <= full)


def test_full_vs_sliding_layer_selection():
    cfg = tiny_config(full_attn_every=2)
    assert Attention(cfg, 0).window == cfg.window   # layer 0: sliding
    assert Attention(cfg, 1).window is None          # layer 1: full


def test_qk_norm_bounds_logit_magnitude():
    seed_everything(0)
    cfg = tiny_config()
    attn = Attention(cfg, layer_idx=1)
    cos, sin = precompute_rope(cfg.head_dim, cfg.context, cfg.rope_base)

    # A large-magnitude input: without QK-norm the logits would scale with it.
    x = torch.randn(2, cfg.context, cfg.d_model) * 50.0
    _, logits = _reference_attention(attn, x, cos, sin)
    finite = logits[torch.isfinite(logits)]
    # QK-norm gives q,k unit RMS => |q.k| <= head_dim, scaled => |logit| <= sqrt(hd).
    bound = math.sqrt(cfg.head_dim) + 1e-3
    assert finite.abs().max().item() <= bound


def test_qk_norm_makes_logits_scale_invariant():
    seed_everything(0)
    cfg = tiny_config()
    attn = Attention(cfg, layer_idx=1)
    cos, sin = precompute_rope(cfg.head_dim, cfg.context, cfg.rope_base)
    x = torch.randn(1, cfg.context, cfg.d_model)
    _, l1 = _reference_attention(attn, x, cos, sin)
    _, l2 = _reference_attention(attn, x * 10.0, cos, sin)
    f = torch.isfinite(l1)
    assert torch.allclose(l1[f], l2[f], atol=1e-4)


# --------------------------------------------------------------------------
# transformer shapes, tying, param count
# --------------------------------------------------------------------------

def test_forward_shapes_and_loss():
    cfg = tiny_config()
    model = MiniGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, cfg.context))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, cfg.context, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_embedding_and_head_are_tied_by_identity():
    model = MiniGPT(tiny_config())
    assert model.lm_head.weight is model.embed.weight


@pytest.mark.parametrize("name", ["nano", "mini", "small"])
def test_param_count_matches_config(name):
    cfg = TIERS[name]
    model = MiniGPT(cfg)
    assert model.num_parameters() == cfg.param_count().total


def test_loss_mask_excludes_positions():
    cfg = tiny_config()
    model = MiniGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.context))
    mask = torch.zeros_like(idx)
    _, loss = model(idx, idx, loss_mask=mask)
    # Everything masked out -> no supervised positions -> nan loss from CE.
    assert torch.isnan(loss)


# --------------------------------------------------------------------------
# overfit one batch
# --------------------------------------------------------------------------

def test_overfit_one_batch():
    seed_everything(0)
    cfg = tiny_config()
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    idx = torch.randint(0, cfg.vocab_size, (4, cfg.context + 1))
    x, y = idx[:, :-1], idx[:, 1:]

    loss_val = float("inf")
    for _ in range(400):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = loss.item()

    # A correct model drives one fixed batch to near-zero loss; a failure here
    # points at the model or the backward pass.
    assert loss_val < 0.1, f"failed to overfit: final loss {loss_val:.3f}"
