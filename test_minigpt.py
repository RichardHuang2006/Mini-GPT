"""The Mini-GPT test suite: one file, ordered like the reading order.

What this file teaches
    What each component *guarantees*, as executable assertions: the tokenizer
    round-trips exactly, attention is causal, the Triton kernels match their
    eager references (forward and backward), Muon and AdamW both learn, resume
    reproduces an uninterrupted run, GRPO pushes log-probs the right way, and
    the evaluators score what they claim to score.

Read first
    Everything else; this file is last in the reading order.

Representative commands
    python -m pytest test_minigpt.py -q                    # full suite
    CUDA_VISIBLE_DEVICES="" python -m pytest -q            # CPU-only: CUDA tests skip
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import data as data_mod
import evaluate as eval_mod
import kernels
import posttrain
from config import TIERS, Config, get_config
from generate import generate, generate_reply
from model import (
    IGNORE_INDEX,
    MiniGPT,
    RMSNorm,
    SwiGLU,
    apply_rope,
    build_attn_mask,
    precompute_rope,
    repeat_kv,
)
from tokenizer import DEFAULT_VOCAB_SIZE, SPECIAL_TOKENS, MiniTokenizer
from train import (
    DataStream,
    Muon,
    WarmupCosine,
    build_optimizers,
    classify_parameters,
    load_checkpoint,
    lr_multiplier,
    save_checkpoint,
    seed_everything,
    train,
    zeropower_via_newtonschulz5,
)

needs_cuda_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernels.HAS_TRITON),
    reason="requires CUDA and Triton",
)


def tiny_config(**overrides) -> Config:
    base = dict(
        name="tiny", vocab_size=256, d_model=64, n_layers=2, n_q_heads=4,
        n_kv_heads=2, head_dim=16, mlp_hidden=128, context=32, window=16,
        micro_batch=2, grad_accum=1, warmup_steps=2, max_steps=50,
        compile=False, use_triton=False, dtype="float32", seed=0,
    )
    base.update(overrides)
    return Config(**base)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def tok() -> MiniTokenizer:
    """A small trained tokenizer (512 vocab) -- same code path as the 32K one."""
    seed_everything(0)
    return MiniTokenizer.train(data_mod.synthetic_docs(400, seed=0), vocab_size=512)


@pytest.fixture(scope="session")
def packed(tok, tmp_path_factory):
    """A tiny packed shard directory with train + val splits."""
    out = tmp_path_factory.mktemp("packed")
    data_mod.pack_corpus(
        data_mod.synthetic_docs(120, seed=1), tok, out, shard_tokens=2_000, val_shards=1
    )
    return out


# =============================================================================
# Tokenizer
# =============================================================================

def test_default_vocab_is_32k_and_uint16_safe():
    assert DEFAULT_VOCAB_SIZE == 32_768
    for tier in TIERS.values():
        assert tier.vocab_size == 32_768
        assert tier.vocab_size < 65_536  # packed shards store uint16 IDs


def test_special_tokens_are_stable_low_ids(tok):
    # The 8 conversation/tool tokens occupy IDs 0..7 in declaration order.
    assert [tok.special_ids[t] for t in SPECIAL_TOKENS] == list(range(8))
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.eos_id == 2
    assert tok.token_to_id("<|tool_call|>") == 6
    assert tok.token_to_id("<|tool_result|>") == 7


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "  leading and trailing  ",
        "tabs\tand\nnewlines",
        "mixed Unicode: naïve café 東京 🚀",
        "",
    ],
)
def test_encode_decode_roundtrip_is_exact(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_bos_eos_wrap_and_vocab_size(tok):
    ids = tok.encode("abc", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    # The tiny fixture corpus runs out of frequent pairs before 512, so the
    # trained vocab is capped by the target but must exceed bytes + specials.
    assert 256 + len(SPECIAL_TOKENS) < tok.vocab_size <= 512


def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = MiniTokenizer.load(path)
    assert loaded.special_ids == tok.special_ids
    assert loaded.fingerprint() == tok.fingerprint()
    text = "round trip me"
    assert loaded.encode(text) == tok.encode(text)


def test_loading_tokenizer_without_specials_raises(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    backend = Tokenizer(models.BPE(unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=[], show_progress=False)
    backend.train_from_iterator(["some text to learn merges from"] * 50, trainer=trainer)
    path = tmp_path / "foreign.json"
    backend.save(str(path))
    with pytest.raises(ValueError, match="special token"):
        MiniTokenizer.load(path)


# =============================================================================
# Data (packing + sampling behaviors the training pipeline relies on)
# =============================================================================

def test_pack_is_lossless_and_fingerprinted(tok, tmp_path):
    docs = ["alpha beta", "gamma delta epsilon"]
    manifest = data_mod.pack_corpus(docs, tok, tmp_path, shard_tokens=1_000, val_shards=0)
    tokens = data_mod.read_all_tokens(tmp_path)
    expected = []
    for d in docs:
        expected.extend(tok.encode(d))
        expected.append(tok.eos_id)
    assert tokens.tolist() == expected  # uint16 round-trip is exact
    assert manifest.tokenizer_fingerprint == tok.fingerprint()
    assert data_mod.verify_against_tokenizer(tmp_path, tok)


def test_sampler_is_seeded_and_resumable(packed):
    a = data_mod.ShardSampler(packed, context=32, seed=7)
    b = data_mod.ShardSampler(packed, context=32, seed=7)
    assert np.array_equal(a.next_batch(4), b.next_batch(4))  # same seed, same stream

    state = a.state_dict()
    expected = a.next_batch(3)
    fresh = data_mod.ShardSampler(packed, context=32, seed=0)
    fresh.load_state_dict(state)
    assert np.array_equal(fresh.next_batch(3), expected)  # snapshot resumes identically


def test_train_and_val_splits_are_disjoint(packed):
    manifest = data_mod.load_manifest(packed)
    train_names = {s.name for s in manifest.shards_for("train")}
    val_names = {s.name for s in manifest.shards_for("val")}
    assert train_names and val_names and not (train_names & val_names)


# =============================================================================
# Model
# =============================================================================

def test_forward_shapes_and_finite_loss():
    cfg = tiny_config()
    seed_everything(0)
    model = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)  # [B, T, V]
    assert loss is not None and torch.isfinite(loss)


def test_causal_masking_blocks_the_future():
    cfg = tiny_config()
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    x1 = torch.randint(0, cfg.vocab_size, (1, 12))
    x2 = x1.clone()
    x2[0, 8:] = (x2[0, 8:] + 1) % cfg.vocab_size  # change only the future
    with torch.no_grad():
        l1, _ = model(x1)
        l2, _ = model(x2)
    assert torch.allclose(l1[:, :8], l2[:, :8], atol=1e-5)  # past logits unchanged
    assert not torch.allclose(l1[:, 8:], l2[:, 8:], atol=1e-5)


def test_sliding_window_mask_shape_and_subset():
    full = build_attn_mask(10, None)
    windowed = build_attn_mask(10, 3)
    i = torch.arange(10)[:, None]
    j = torch.arange(10)[None, :]
    assert torch.equal(full, j <= i)                       # causal
    assert torch.equal(windowed, (j <= i) & (i - j < 3))   # causal within window
    assert bool((windowed & ~full).sum() == 0)             # strict subset of causal
    assert bool(windowed.sum() < full.sum())


def test_sliding_window_limits_receptive_field():
    # One all-windowed layer (full_attn_every > n_layers): position i sees
    # exactly the last `window` tokens, so a change further back cannot reach it.
    cfg = tiny_config(n_layers=1, window=4, full_attn_every=99)
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    x1 = torch.randint(0, cfg.vocab_size, (1, 12))

    far, near, probe = 0, 10, 11  # probe - far >= window, probe - near < window
    x2 = x1.clone()
    x2[0, far] = (x2[0, far] + 1) % cfg.vocab_size
    x3 = x1.clone()
    x3[0, near] = (x3[0, near] + 1) % cfg.vocab_size
    with torch.no_grad():
        l1, l2, l3 = model(x1)[0], model(x2)[0], model(x3)[0]
    assert torch.allclose(l1[0, probe], l2[0, probe], atol=1e-5)      # out of window
    assert not torch.allclose(l1[0, probe], l3[0, probe], atol=1e-5)  # in window


def test_full_attention_layer_schedule():
    cfg = tiny_config(n_layers=4, full_attn_every=2)
    model = MiniGPT(cfg)
    assert [blk.attn.is_full for blk in model.layers] == [False, True, False, True]
    assert model.layers[0].attn.window == cfg.window
    assert model.layers[1].attn.window is None


def test_gqa_repeat_kv_expands_heads():
    x = torch.randn(2, 2, 5, 4)  # [B, Hkv=2, T, d]
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 5, 4)  # Hkv * n_rep = 6 query heads served
    # Each KV head's content is repeated for its group of query heads.
    assert torch.equal(out[:, 0], x[:, 0]) and torch.equal(out[:, 2], x[:, 0])
    assert torch.equal(out[:, 3], x[:, 1]) and torch.equal(out[:, 5], x[:, 1])
    assert repeat_kv(x, 1) is x


def test_qk_norm_bounds_scores_and_is_scale_invariant():
    d = 16
    norm = RMSNorm(d)  # unit gains
    q = torch.randn(2, 4, 8, d) * 30.0  # deliberately huge activations
    k = torch.randn(2, 4, 8, d) * 30.0
    qn, kn = norm(q), norm(k)
    scores = qn @ kn.transpose(-2, -1) / math.sqrt(d)
    # RMS-normalized rows have L2 norm ~sqrt(d), so |q.k|/sqrt(d) <= ~sqrt(d).
    assert scores.abs().max() <= math.sqrt(d) + 1e-3
    # Normalization removes input scale entirely.
    assert torch.allclose(norm(q * 1000), qn, atol=1e-4)


def test_rope_rotates_without_changing_norm():
    cos, sin = precompute_rope(16, 10)
    x = torch.randn(2, 3, 10, 16)
    y = apply_rope(x, cos, sin)
    # Position 0 has angle 0: identity rotation.
    assert torch.allclose(y[:, :, 0], x[:, :, 0], atol=1e-6)
    assert not torch.allclose(y[:, :, 5], x[:, :, 5], atol=1e-3)
    # Rotations are orthogonal maps: per-position vector norms are preserved.
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    # The kernels-module eager reference is the same function.
    assert torch.allclose(kernels.rope_reference(x, cos, sin), y, atol=1e-6)


def test_rmsnorm_matches_manual_formula():
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.mul_(1.5)
    x = torch.randn(4, 8, dtype=torch.float32)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * 1.5
    assert torch.allclose(norm(x), expected, atol=1e-5)
    assert torch.allclose(kernels.rmsnorm_reference(x, norm.weight), expected, atol=1e-5)


def test_swiglu_matches_manual_formula():
    mlp = SwiGLU(8, 16)
    x = torch.randn(3, 8)
    a, b = mlp.gate(x), mlp.up(x)
    assert torch.allclose(mlp(x), mlp.down(F.silu(a) * b), atol=1e-6)
    assert torch.allclose(kernels.swiglu_reference(a, b), F.silu(a) * b, atol=1e-6)


def test_embeddings_and_lm_head_are_tied():
    model = MiniGPT(tiny_config())
    assert model.lm_head.weight is model.embed.weight  # one tensor, two roles


@pytest.mark.parametrize("name", ["nano", "mini", "small"])
def test_param_count_formula_matches_built_model(name):
    cfg = get_config(name, use_triton=False, compile=False)
    if name == "small":
        assert cfg.context == 2_048  # the resume-scale context
    model = MiniGPT(cfg)
    assert model.num_parameters() == cfg.param_count().total


def test_2048_context_forward():
    # A thin model at the full 2,048-token context: construction + forward work.
    cfg = tiny_config(context=2_048, window=256, n_layers=1)
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 2_048))
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (1, 2_048, cfg.vocab_size)


def test_loss_mask_excludes_positions():
    cfg = tiny_config()
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    mask = torch.tensor([[0, 0, 1, 1, 0, 1, 0, 0]])
    with torch.no_grad():
        logits, masked_loss = model(x, x, loss_mask=mask)
        logp = torch.log_softmax(logits.float(), dim=-1)
    # Hand-compute CE over exactly the unmasked positions.
    picked = logp[0, mask[0] == 1].gather(-1, x[0, mask[0] == 1, None]).squeeze(-1)
    assert torch.allclose(masked_loss, -picked.mean(), atol=1e-5)


# =============================================================================
# Training
# =============================================================================

def test_forward_backward_gradients_are_finite():
    cfg = tiny_config()
    seed_everything(0)
    model = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_parameter_grouping_partitions_everything():
    model = MiniGPT(tiny_config())
    groups = classify_parameters(model)
    unique = {id(p) for p in model.parameters()}
    assert {id(p) for p in groups["hidden"]} | {id(p) for p in groups["misc"]} == unique
    assert all(p.ndim >= 2 for p in groups["hidden"])
    assert id(model.embed.weight) in {id(p) for p in groups["misc"]}  # tied weight -> misc, once


def test_newton_schulz_orthogonalizes():
    seed_everything(0)
    g = torch.randn(48, 16)
    sv = torch.linalg.svdvals(zeropower_via_newtonschulz5(g, steps=5))
    assert sv.min() > 0.5 and sv.max() < 1.35  # singular values pushed toward 1


@pytest.mark.parametrize("use_muon", [True, False])
def test_optimizers_update_parameters(use_muon):
    cfg = tiny_config(use_muon=use_muon)
    seed_everything(0)
    model = MiniGPT(cfg)
    opts = build_optimizers(model, cfg)
    assert [type(o).__name__ for o in opts] == (["Muon", "AdamW"] if use_muon else ["AdamW"])

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    for o in opts:
        o.step()
    moved = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert len(moved) == len(before)  # every parameter received an update


def test_lr_schedule_warms_up_then_decays_to_floor():
    assert lr_multiplier(0, warmup=10, max_steps=100, floor_frac=0.1) == pytest.approx(0.1)
    assert lr_multiplier(9, 10, 100, 0.1) == pytest.approx(1.0)
    assert lr_multiplier(55, 10, 100, 0.1) == pytest.approx(0.55, abs=0.01)
    assert lr_multiplier(100, 10, 100, 0.1) == pytest.approx(0.1)

    cfg = tiny_config(lr_adamw=1e-3, lr_muon=1e-2)
    model = MiniGPT(cfg)
    opts = build_optimizers(model, cfg)
    sched = WarmupCosine(opts, warmup=2, max_steps=10, floor_frac=0.1)
    for _ in range(5):
        sched.step()
    lrs = sched.get_last_lr()
    # The Muon/AdamW peak-LR ratio is preserved by the shared multiplier.
    assert lrs[0] / lrs[1] == pytest.approx(cfg.lr_muon / cfg.lr_adamw)


def test_tiny_batch_overfits():
    cfg = tiny_config()
    seed_everything(0)
    model = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.1 < first  # memorizes one fixed batch


def test_perplexity_of_uniform_model_is_vocab_size():
    V = 64

    class Uniform(nn.Module):
        def forward(self, idx, targets=None, **kw):
            return torch.zeros(*idx.shape, V), None

    windows = torch.randint(0, V, (6, 17))
    ppl = eval_mod.evaluate_perplexity(Uniform(), windows)
    assert ppl == pytest.approx(V, rel=1e-4)  # uniform prob 1/V -> ppl == V


def test_train_checkpoint_and_resume_match_uninterrupted_run(packed, tmp_path):
    # max_steps fixes the LR-schedule span for all three runs; `steps` only
    # bounds how far each run gets before stopping.
    cfg = tiny_config(vocab_size=512, warmup_steps=1, max_steps=6)

    # Uninterrupted: 6 steps.
    torch.use_deterministic_algorithms(True, warn_only=True)
    train(cfg, str(packed), str(tmp_path / "full"), steps=6, log_every=0,
          ckpt_every=0, eval_every=0)
    # Interrupted: 3 steps, then resume from the checkpoint for 3 more.
    train(cfg, str(packed), str(tmp_path / "half"), steps=3, log_every=0,
          ckpt_every=3, eval_every=0)
    train(cfg, str(packed), str(tmp_path / "resumed"), steps=6, log_every=0,
          ckpt_every=0, eval_every=0, resume=str(tmp_path / "half" / "ckpt_3.pt"))

    full = torch.load(tmp_path / "full" / "ckpt_final.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed" / "ckpt_final.pt", weights_only=False)
    assert full["step"] == resumed["step"] == 6
    for key, a in full["model"].items():
        assert torch.allclose(a, resumed["model"][key], atol=1e-6), key


def test_checkpoint_saves_and_loads_all_state(tmp_path):
    cfg = tiny_config()
    seed_everything(0)
    model = MiniGPT(cfg)
    opts = build_optimizers(model, cfg)
    sched = WarmupCosine(opts, 2, 10, 0.1)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    for o in opts:
        o.step()
    sched.step()
    save_checkpoint(tmp_path / "ck.pt", model=model, optimizers=opts, scheduler=sched,
                    step=1, sampler_state={"marker": 1})

    seed_everything(1)  # different seed -> different fresh weights
    model2 = MiniGPT(cfg)
    opts2 = build_optimizers(model2, cfg)
    sched2 = WarmupCosine(opts2, 2, 10, 0.1)
    payload = load_checkpoint(tmp_path / "ck.pt", model=model2, optimizers=opts2,
                              scheduler=sched2)
    assert payload["step"] == 1 and payload["sampler"] == {"marker": 1}
    assert sched2.last_step == 1
    for (n, a), (_, b) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(a, b), n


# =============================================================================
# Triton kernels: eager references on CPU, differential comparisons on CUDA
# =============================================================================

@pytest.mark.parametrize("chunk", [1, 7, 64, 100_000])
def test_chunked_ce_matches_cross_entropy_any_chunk(chunk):
    seed_everything(0)
    N, D, V = 20, 16, 97
    hidden = torch.randn(N, D, requires_grad=True)
    weight = torch.randn(V, D, requires_grad=True)
    targets = torch.randint(0, V, (N,))
    targets[::5] = IGNORE_INDEX  # some ignored positions

    loss = kernels.chunked_cross_entropy(hidden, weight, targets, chunk=chunk)
    loss.backward()

    h2 = hidden.detach().clone().requires_grad_(True)
    w2 = weight.detach().clone().requires_grad_(True)
    ref = F.cross_entropy(h2 @ w2.T, targets, ignore_index=IGNORE_INDEX)
    ref.backward()

    assert torch.allclose(loss, ref, atol=1e-5)                  # forward matches
    assert torch.allclose(hidden.grad, h2.grad, atol=1e-5)       # d/d hidden matches
    assert torch.allclose(weight.grad, w2.grad, atol=1e-5)       # d/d weight matches


def test_chunked_ce_all_ignored_is_finite():
    hidden = torch.randn(6, 8, requires_grad=True)
    weight = torch.randn(32, 8)
    targets = torch.full((6,), IGNORE_INDEX, dtype=torch.long)
    loss = kernels.chunked_cross_entropy(hidden, weight, targets, chunk=8)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


def test_kernels_fall_back_to_eager_on_cpu():
    # On CPU tensors the public dispatchers must run (via the references),
    # whether or not Triton is importable.
    x = torch.randn(3, 8)
    w = torch.ones(8)
    assert torch.allclose(kernels.rmsnorm(x, w), kernels.rmsnorm_reference(x, w), atol=1e-6)
    cos, sin = precompute_rope(8, 4)
    q = torch.randn(1, 2, 4, 8)
    assert torch.allclose(kernels.apply_rope(q, cos, sin),
                          kernels.rope_reference(q, cos, sin), atol=1e-6)
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    assert torch.allclose(kernels.swiglu(a, b), kernels.swiglu_reference(a, b), atol=1e-6)


def _compare_fwd_bwd(fn_kernel, fn_ref, inputs, atol):
    """Run kernel vs reference on cloned inputs; compare outputs and grads."""
    ins_k = [t.detach().clone().requires_grad_(t.requires_grad) for t in inputs]
    ins_r = [t.detach().clone().requires_grad_(t.requires_grad) for t in inputs]
    out_k = fn_kernel(*ins_k)
    out_r = fn_ref(*ins_r)
    assert torch.allclose(out_k, out_r, atol=atol), "forward mismatch"
    upstream = torch.randn_like(out_r)
    out_k.backward(upstream)
    out_r.backward(upstream.clone())
    for tk, tr in zip(ins_k, ins_r):
        if tk.requires_grad:
            assert torch.allclose(tk.grad, tr.grad, atol=atol), "gradient mismatch"


@needs_cuda_triton
@pytest.mark.parametrize("shape", [(4, 64), (2, 8, 64), (2, 4, 8, 16)])
def test_triton_rmsnorm_matches_reference_on_cuda(shape):
    seed_everything(0)
    x = torch.randn(*shape, device="cuda", requires_grad=True)
    w = torch.randn(shape[-1], device="cuda", requires_grad=True)
    _compare_fwd_bwd(
        lambda x, w: kernels.rmsnorm(x, w),
        lambda x, w: kernels.rmsnorm_reference(x, w),
        [x, w], atol=1e-4,
    )


@needs_cuda_triton
def test_triton_rmsnorm_bf16_forward_tolerance():
    seed_everything(0)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    y_k = kernels.rmsnorm(x, w)
    y_r = kernels.rmsnorm_reference(x, w)
    # bf16 keeps ~2-3 significant digits, and the two paths round at different
    # points, so the comparison needs a relative (not just absolute) tolerance.
    assert torch.allclose(y_k.float(), y_r.float(), atol=2e-2, rtol=3e-2)


@needs_cuda_triton
def test_triton_rope_matches_reference_on_cuda():
    seed_everything(0)
    cos, sin = precompute_rope(16, 8, device="cuda")
    x = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
    _compare_fwd_bwd(
        lambda x: kernels.apply_rope(x, cos, sin),
        lambda x: kernels.rope_reference(x, cos, sin),
        [x], atol=1e-4,
    )


@needs_cuda_triton
def test_triton_swiglu_matches_reference_on_cuda():
    seed_everything(0)
    a = torch.randn(4, 128, device="cuda", requires_grad=True)
    b = torch.randn(4, 128, device="cuda", requires_grad=True)
    _compare_fwd_bwd(kernels.swiglu, kernels.swiglu_reference, [a, b], atol=1e-4)


@needs_cuda_triton
def test_chunked_ce_peak_memory_scales_with_chunk():
    # The guaranteed property: peak memory follows the chunk size because the
    # full [N, V] logits tensor is never built.
    seed_everything(0)
    N, D, V = 2048, 64, 8192
    hidden = torch.randn(N, D, device="cuda")
    weight = torch.randn(V, D, device="cuda")
    targets = torch.randint(0, V, (N,), device="cuda")

    def peak(chunk: int) -> int:
        h = hidden.clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        kernels.chunked_cross_entropy(h, weight, targets, chunk=chunk).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    small, big = peak(chunk=256), peak(chunk=V)
    assert small < big  # smaller vocabulary tiles -> strictly lower peak memory


@needs_cuda_triton
def test_full_model_matches_eager_with_kernels_on_cuda():
    x = torch.randint(0, 256, (2, 16), device="cuda")
    outs = {}
    for use_triton in (False, True):
        seed_everything(0)  # identical init for both variants
        model = MiniGPT(tiny_config(use_triton=use_triton)).cuda()
        logits, loss = model(x, x)
        outs[use_triton] = (logits.detach(), loss.detach())
    assert torch.allclose(outs[False][0], outs[True][0], atol=1e-3)
    assert torch.allclose(outs[False][1], outs[True][1], atol=1e-4)


# =============================================================================
# Generation
# =============================================================================

def test_generation_is_deterministic_and_seeded(tok):
    cfg = tiny_config(vocab_size=512)
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    idx = torch.tensor([tok.encode("the quick", add_bos=True)])

    g1 = generate(model, idx, max_new_tokens=8)  # greedy
    g2 = generate(model, idx, max_new_tokens=8)
    assert torch.equal(g1, g2)

    s1 = generate(model, idx, max_new_tokens=8, temperature=1.0, top_k=20, seed=3)
    s2 = generate(model, idx, max_new_tokens=8, temperature=1.0, top_k=20, seed=3)
    s3 = generate(model, idx, max_new_tokens=8, temperature=1.0, top_k=20, seed=4)
    assert torch.equal(s1, s2) and not torch.equal(s1, s3)


def test_generation_enforces_context_window():
    class AssertsContext(nn.Module):
        """Fails the test if it is ever fed more than `context` tokens."""

        def __init__(self, context: int, vocab: int):
            super().__init__()
            self.cfg = type("C", (), {"context": context})()
            self.vocab = vocab
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, idx, targets=None, **kw):
            assert idx.shape[1] <= self.cfg.context, "context window exceeded"
            return torch.zeros(*idx.shape, self.vocab), None

    model = AssertsContext(context=8, vocab=32)
    idx = torch.randint(0, 32, (1, 6))
    out = generate(model, idx, max_new_tokens=10)  # 6 + 10 > 8 forces cropping
    assert out.shape[1] == 16


def test_generation_stops_on_eos(tok):
    class AlwaysEos(nn.Module):
        def __init__(self, eos: int, vocab: int):
            super().__init__()
            self.eos, self.vocab = eos, vocab

        def forward(self, idx, targets=None, **kw):
            logits = torch.zeros(*idx.shape, self.vocab)
            logits[:, -1, self.eos] = 100.0
            return logits, None

    model = AlwaysEos(tok.eos_id, 512)
    idx = torch.tensor([[5, 6, 7]])
    out = generate(model, idx, max_new_tokens=50, eos_id=tok.eos_id)
    assert out.shape[1] == 4 and out[0, -1].item() == tok.eos_id  # stopped immediately


# =============================================================================
# Post-training: chat template, SFT, rewards, GRPO
# =============================================================================

def test_loss_mask_covers_only_assistant_tokens(tok):
    conv = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "content": "result 42"},
        {"role": "assistant", "content": "done"},
    ]
    r = posttrain.render_chat(conv, tok)
    asst_id = tok.token_to_id("<|assistant|>")

    # Reconstruct the expected mask: content + eos of assistant turns only.
    expected = [0]  # bos
    for msg in conv:
        expected.append(0)  # role header is never predicted
        n = len(tok.encode(msg["content"]))
        is_asst = msg["role"] == "assistant"
        expected.extend([1 if is_asst else 0] * n)
        expected.append(1 if is_asst else 0)  # closing eos
    assert r.loss_mask == expected
    assert r.ids[0] == tok.bos_id
    # Role headers themselves are always unmasked.
    for i, t in enumerate(r.ids):
        if t == asst_id:
            assert r.loss_mask[i] == 0


def test_build_prompt_appends_generation_header(tok):
    ids = posttrain.build_prompt([{"role": "user", "content": "hi"}], tok)
    assert ids[-1] == tok.token_to_id("<|assistant|>")


def test_packed_conversations_are_isolated(tok):
    cfg = tiny_config(vocab_size=512, context=48)
    seed_everything(0)
    model = MiniGPT(cfg).eval()

    convs = [
        [{"role": "user", "content": "aa"}, {"role": "assistant", "content": "bb"}],
        [{"role": "user", "content": "cc"}, {"role": "assistant", "content": "dd"}],
    ]
    packed = posttrain.pack_conversations(convs, tok, seq_len=cfg.context)
    assert len(packed) == 1  # both conversations fit one packed row
    ids, seg = packed.input_ids, packed.segment_ids
    boundary = int((seg[0] == 0).sum())  # where conversation 1 begins

    with torch.no_grad():
        base, _ = model(ids, segment_ids=seg)
        edited_ids = ids.clone()
        edited_ids[0, boundary + 1] = (edited_ids[0, boundary + 1] + 1) % 512
        edited, _ = model(edited_ids, segment_ids=seg)
    # Editing conversation 1 must not change conversation 0's logits.
    assert torch.allclose(base[0, :boundary], edited[0, :boundary], atol=1e-5)


def test_sft_loss_is_masked_cross_entropy(tok):
    cfg = tiny_config(vocab_size=512, context=48)
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    packed = posttrain.pack_conversations(
        posttrain.synthetic_sft_conversations(3, seed=0), tok, seq_len=cfg.context
    )
    mb = next(packed.microbatches(len(packed)))
    ids, tgt, mask, seg = mb
    with torch.no_grad():
        loss = posttrain.sft_loss(model, mb)
        logits, _ = model(ids, segment_ids=seg)
        logp = torch.log_softmax(logits.float(), dim=-1)
    picked = logp[mask == 1].gather(-1, tgt[mask == 1, None]).squeeze(-1)
    assert torch.allclose(loss, -picked.mean(), atol=1e-4)
    assert mask.sum() > 0


def test_sft_step_reduces_loss(tok):
    cfg = tiny_config(vocab_size=512, context=48, grad_accum=1, micro_batch=2)
    seed_everything(0)
    model = MiniGPT(cfg)
    packed = posttrain.pack_conversations(
        posttrain.synthetic_sft_conversations(8, seed=0), tok, seq_len=cfg.context
    )
    losses = posttrain.train_sft(model, packed, cfg, steps=20)
    assert len(losses) == 20 and losses[-1] < losses[0]


def test_gsm8k_answer_extraction():
    ext = posttrain.extract_final_int
    assert ext("... so the answer is\n#### 72") == 72
    assert ext("#### -5") == -5
    assert ext("I think 12 plus 30 gives 42") == 42        # falls back to last int
    assert ext("The answer is #### 1200") == 1200          # comma-stripped upstream
    assert ext("no numbers here") is None


def test_gsm8k_reward_components():
    r = posttrain.gsm8k_reward("reasoning...\n#### 42", gold=42, terminated=True,
                               n_new_tokens=10, max_new_tokens=64)
    assert r.correct == 1.0 and r.format == 1.0 and r.total == pytest.approx(1.5)

    wrong = posttrain.gsm8k_reward("#### 41", gold=42, terminated=True,
                                   n_new_tokens=10, max_new_tokens=64)
    assert wrong.correct == 0.0 and wrong.total == pytest.approx(0.5)  # format only

    unparseable = posttrain.gsm8k_reward("no answer", gold=42, terminated=False,
                                         n_new_tokens=100, max_new_tokens=64)
    assert unparseable.total == 0.0


def test_group_advantages_normalize_within_groups():
    rewards = torch.tensor([1.0, 0.0, 2.0, 2.0])
    groups = torch.tensor([0, 0, 1, 1])
    adv = posttrain.group_advantages(rewards, groups)
    assert adv[0] > 0 > adv[1]                       # better completion -> positive
    assert adv[0] == pytest.approx(-adv[1].item())   # zero mean within the group
    assert torch.equal(adv[2:], torch.zeros(2))      # zero-variance group -> zero


def test_zero_variance_group_produces_zero_gradient(tok):
    cfg = tiny_config(vocab_size=512)
    seed_everything(0)
    model = MiniGPT(cfg)
    comps = posttrain.sample_groups(
        model, tok, [[{"role": "user", "content": "hi"}]],
        group_size=4, max_new_tokens=4, temperature=1.0, seed=0,
    )
    seqs, comp_mask, groups = posttrain.collate(comps, tok.pad_id)
    adv = posttrain.group_advantages(torch.ones(len(comps)), groups)  # all equal
    with torch.no_grad():
        old_logp, _ = posttrain.token_logprobs(model, seqs, comp_mask)
    loss = posttrain.grpo_loss(model, seqs, comp_mask, adv, old_logp)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-9)


def test_grpo_update_moves_logprobs_in_advantage_direction(tok):
    cfg = tiny_config(vocab_size=512)
    seed_everything(0)
    model = MiniGPT(cfg)
    comps = posttrain.sample_groups(
        model, tok, [[{"role": "user", "content": "count"}]],
        group_size=2, max_new_tokens=4, temperature=1.0, seed=1,
    )
    seqs, comp_mask, groups = posttrain.collate(comps, tok.pad_id)
    rewards = torch.tensor([1.0, 0.0])  # completion 0 good, completion 1 bad
    adv = posttrain.group_advantages(rewards, groups)
    assert adv[0] > 0 > adv[1]

    def summed_logp():
        with torch.no_grad():
            lp, m = posttrain.token_logprobs(model, seqs, comp_mask)
        return (lp * m.float()).sum(dim=1)

    before = summed_logp()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    posttrain.grpo_step(model, opt, seqs, comp_mask, adv)
    after = summed_logp()
    assert after[0] > before[0]  # positive advantage -> more likely
    assert after[1] < before[1]  # negative advantage -> less likely


def test_grpo_smoke_loop_runs_and_logs_mean_reward(tok):
    cfg = tiny_config(vocab_size=512)
    seed_everything(0)
    model = MiniGPT(cfg)
    bank = posttrain.build_arithmetic_bank(4, max_new_tokens=4, seed=0)
    rewards = posttrain.run_grpo(model, tok, bank, steps=2, group_size=2,
                                 prompts_per_step=2, max_new_tokens=4, lr=1e-4, seed=0)
    assert len(rewards) == 2 and all(math.isfinite(r) for r in rewards)


# =============================================================================
# Evaluation
# =============================================================================

class _PreferTokens(nn.Module):
    """A stub LM that assigns high logits to a fixed token set everywhere."""

    def __init__(self, preferred: set[int], vocab: int):
        super().__init__()
        self.preferred = sorted(preferred)
        self.vocab = vocab

    def forward(self, idx, targets=None, **kw):
        logits = torch.zeros(*idx.shape, self.vocab)
        logits[..., self.preferred] = 8.0
        return logits, None


def test_multiple_choice_scoring_picks_the_likelier_answer(tok):
    good, bad = " sunny sky", " qzxv"
    questions = [
        {"prompt": "The weather is", "choices": [good, bad], "answer": 0},
        {"prompt": "The weather is", "choices": [bad, good], "answer": 1},
    ]
    model = _PreferTokens(set(tok.encode(good)), tok.vocab_size)
    acc = eval_mod.evaluate_multiple_choice(model, tok, questions)
    assert acc == 1.0  # ARC/MMLU-format questions, scored by log-likelihood


def test_multiple_choice_accepts_arc_mmlu_format(tok):
    # The exact dict schema load_arc / load_mmlu produce.
    q = {
        "prompt": "Question: Which gas do plants absorb?\nAnswer:",
        "choices": [" carbon dioxide", " oxygen", " helium", " argon"],
        "answer": 0,
    }
    cfg = tiny_config(vocab_size=512)
    seed_everything(0)
    acc = eval_mod.evaluate_multiple_choice(MiniGPT(cfg).eval(), tok, [q])
    assert acc in (0.0, 1.0)


def test_humaneval_sandbox_pass_fail_and_timeout():
    assert eval_mod.run_code_sandbox("print('ok')\n") is True
    assert eval_mod.run_code_sandbox("raise AssertionError('bad')\n") is False
    assert eval_mod.run_code_sandbox("while True:\n    pass\n", timeout=1.0) is False


def test_humaneval_end_to_end_on_fixture(tok):
    cfg = tiny_config(vocab_size=512)
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    score = eval_mod.evaluate_humaneval(
        model, tok, eval_mod.SAMPLE_HUMANEVAL, max_new_tokens=8, timeout=5.0
    )
    assert 0.0 <= score <= 1.0  # a random model almost surely scores 0


def test_perplexity_on_validation_fixture(tok, packed):
    cfg = tiny_config(vocab_size=512)
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    ppl = eval_mod.perplexity_from_split(model, packed, context=16, n_windows=8)
    assert math.isfinite(ppl) and ppl > 1.0


def test_results_json_roundtrip_and_markdown_table(tmp_path):
    results = {
        "base": {"perplexity": 42.5, "arc_easy": 0.25, "humaneval": 0.0},
        "sft": {"perplexity": 40.1, "arc_easy": 0.30, "humaneval": 0.0},
    }
    path = tmp_path / "results.json"
    eval_mod.write_results(results, path)
    assert eval_mod.load_results(path) == results
    assert json.loads(path.read_text())["chance_baselines"]["arc_easy"] == 0.25

    table = eval_mod.format_tables(results)
    lines = table.strip().splitlines()
    assert "ARC-Easy (chance 25%)" in lines[0]
    assert lines[2].startswith("| base") and lines[3].startswith("| sft")
    assert "42.50" in table and "30.0%" in table


def test_generate_reply_is_wellformed(tok):
    cfg = tiny_config(vocab_size=512)
    seed_everything(0)
    model = MiniGPT(cfg).eval()
    reply = generate_reply(model, tok, [{"role": "user", "content": "What is 2 + 3?"}],
                           max_new_tokens=8)
    assert isinstance(reply, str)  # decodes cleanly, truncated at eos
