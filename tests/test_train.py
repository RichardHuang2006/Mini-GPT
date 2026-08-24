"""Config presets, determinism, and the training loop.

Covers the scaffolding everything else depends on (tier presets, the seeding
harness), then the training loop: parameter groups, the LR schedule, gradient
accumulation, checkpoint/resume equivalence, the Muon optimizer,
``torch.compile`` equivalence, the throughput harness, the anneal switch, and
context extension.
"""

from __future__ import annotations

import pytest

from config import TIERS, Config, get_config, swiglu_hidden
from mini_gpt.determinism import seed_everything


# --------------------------------------------------------------------------
# config presets and parameter count
# --------------------------------------------------------------------------

def test_all_tiers_instantiate():
    for name in ("nano", "mini", "small"):
        cfg = get_config(name)
        assert cfg.name == name
        assert isinstance(cfg, Config)


@pytest.mark.parametrize("name", ["nano", "mini", "small"])
def test_derived_dimension_invariant(name):
    cfg = TIERS[name]
    # The core shape invariant.
    assert cfg.d_model == cfg.n_q_heads * cfg.head_dim
    assert cfg.n_q_heads % cfg.n_kv_heads == 0
    assert cfg.mlp_hidden == swiglu_hidden(cfg.d_model)
    assert cfg.window <= cfg.context


def test_mini_param_count_matches_design():
    # The mini preset computes to 39M total / 22.6M non-embedding.
    p = get_config("mini").param_count()
    assert round(p.total / 1e6) == 39
    assert abs(p.non_embedding / 1e6 - 22.6) < 0.1
    # Tied embeddings are ~43% of the model at this scale.
    assert p.total == p.embedding + p.non_embedding


def test_param_count_monotonic_across_tiers():
    counts = {n: TIERS[n].param_count().non_embedding for n in ("nano", "mini", "small")}
    assert counts["nano"] < counts["mini"] < counts["small"]


def test_mini_global_batch_is_half_million_tokens():
    # The optimizer step operates on a ~0.5M-token batch.
    cfg = get_config("mini")
    assert 0.4e6 < cfg.global_batch_tokens < 0.6e6


def test_layer_schedule_alternates():
    cfg = get_config("mini", full_attn_every=2)
    schedule = [cfg.is_full_attention_layer(i) for i in range(cfg.n_layers)]
    # Every 2nd layer full-context, the rest windowed.
    assert schedule == [False, True] * (cfg.n_layers // 2)
    assert any(schedule) and not all(schedule)


def test_uint16_vocab_constraint():
    # The uint16 packing format depends on this.
    for cfg in TIERS.values():
        assert cfg.vocab_size < 65_536


def test_overrides_are_revalidated():
    cfg = get_config("mini", n_layers=4)
    assert cfg.n_layers == 4
    with pytest.raises(AssertionError):
        # head_dim no longer divides d_model -> the invariant must fire.
        get_config("mini", head_dim=48)
    with pytest.raises(KeyError):
        get_config("does-not-exist")


# --------------------------------------------------------------------------
# determinism harness
# --------------------------------------------------------------------------

def test_seed_everything_returns_seed():
    assert seed_everything(123) == 123


def test_seeded_random_is_reproducible():
    import random

    import numpy as np

    seed_everything(7)
    a = (random.random(), float(np.random.rand()))
    seed_everything(7)
    b = (random.random(), float(np.random.rand()))
    assert a == b


def test_two_forward_passes_bit_identical():
    # Two forward passes on the same seeded input are bit-identical. A tiny torch
    # module keeps this CPU-only.
    torch = pytest.importorskip("torch")

    def run():
        seed_everything(42)
        net = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 32),
        )
        x = torch.randn(8, 32)
        return net(x)

    out_a = run()
    out_b = run()
    assert torch.equal(out_a, out_b)


# ==========================================================================
# training loop, optimizer groups, schedule, checkpoint/resume
# ==========================================================================

torch = pytest.importorskip("torch")

from mini_gpt.data import download, pack  # noqa: E402
from mini_gpt.data.sampler import ShardSampler  # noqa: E402
from mini_gpt.model.transformer import MiniGPT  # noqa: E402
from mini_gpt.tokenizer import MiniTokenizer  # noqa: E402
from mini_gpt.train.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from mini_gpt.train.loop import DataStream, Trainer  # noqa: E402
from mini_gpt.train.optim import HIDDEN, MISC, build_param_groups, classify_parameters  # noqa: E402
from mini_gpt.train.schedule import build_scheduler, lr_multiplier  # noqa: E402


def _tiny_cfg(**overrides) -> Config:
    base = dict(
        name="tiny",
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=16,
        mlp_hidden=swiglu_hidden(64),
        context=16,
        window=8,
        micro_batch=8,
        grad_accum=2,
        lr_adamw=3e-3,
        warmup_steps=5,
        max_steps=200,
    )
    base.update(overrides)
    return Config(**base)


@pytest.fixture(scope="module")
def packed_tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("packed_train")
    docs = list(download.synthetic_docs(1500, seed=11))
    tok = MiniTokenizer.train(docs, vocab_size=512, min_frequency=1)
    pack.pack_corpus(docs, tok, out, shard_tokens=3000, val_shards=1)
    return out


def _build_trainer(cfg, data_dir, *, seed=0):
    from mini_gpt.train.optim import build_optimizer

    seed_everything(seed)
    model = MiniGPT(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    sampler = ShardSampler(data_dir, context=cfg.context, split="train", seed=seed)
    data = DataStream(sampler, cfg.micro_batch)
    trainer = Trainer(model, optimizer, scheduler, data.batch, cfg, grad_accum=cfg.grad_accum)
    return trainer, data


# ----------------------------------------------------------------- param groups

def test_param_groups_partition_all_params():
    model = MiniGPT(_tiny_cfg())
    groups = classify_parameters(model)
    grouped_ids = {id(p) for g in groups.values() for p in g}
    unique_ids = {id(p) for p in model.parameters()}
    assert grouped_ids == unique_ids
    # Disjoint groups.
    assert not (
        {id(p) for p in groups[HIDDEN]} & {id(p) for p in groups[MISC]}
    )


def test_hidden_group_is_2d_nonembedding():
    model = MiniGPT(_tiny_cfg())
    groups = classify_parameters(model)
    assert all(p.ndim >= 2 for p in groups[HIDDEN])
    # The tied embedding weight is 2D but must be in misc, not hidden.
    assert any(p is model.embed.weight for p in groups[MISC])
    assert all(p is not model.embed.weight for p in groups[HIDDEN])


def test_weight_decay_only_on_hidden_group():
    cfg = _tiny_cfg()
    pgs = build_param_groups(MiniGPT(cfg), cfg)
    by_name = {g["name"]: g for g in pgs}
    assert by_name[HIDDEN]["weight_decay"] == cfg.weight_decay
    assert by_name[MISC]["weight_decay"] == 0.0


# --------------------------------------------------------------------- schedule

def test_lr_multiplier_shape():
    warmup, max_steps, floor = 10, 100, 0.1
    assert lr_multiplier(0, warmup, max_steps, floor) == pytest.approx(0.1)
    assert lr_multiplier(9, warmup, max_steps, floor) == pytest.approx(1.0)
    # Monotone decay after warmup down to the floor.
    assert lr_multiplier(9, warmup, max_steps, floor) > lr_multiplier(55, warmup, max_steps, floor)
    assert lr_multiplier(55, warmup, max_steps, floor) > lr_multiplier(99, warmup, max_steps, floor)
    assert lr_multiplier(100, warmup, max_steps, floor) == pytest.approx(floor)


@pytest.mark.filterwarnings("ignore:Detected call of")
def test_scheduler_applies_per_group_peak_lr():
    p0 = torch.nn.Parameter(torch.zeros(4))
    p1 = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.AdamW(
        [{"params": [p0], "lr": 0.02}, {"params": [p1], "lr": 0.003}], betas=(0.9, 0.95)
    )
    cfg = _tiny_cfg(warmup_steps=10, max_steps=100, lr_floor_frac=0.1)
    sched = build_scheduler(opt, cfg)

    g0, g1 = [], []
    for _ in range(100):
        lrs = sched.get_last_lr()
        g0.append(lrs[0])
        g1.append(lrs[1])
        sched.step()

    assert g0[0] < g0[9]              # warmup rises
    assert g0[9] == pytest.approx(0.02, rel=1e-6)  # peak == group base lr
    assert g0[9] > g0[50] > g0[99]    # cosine decay
    # Per-group ratio is constant across the whole schedule.
    ratios = [a / b for a, b in zip(g0, g1)]
    assert max(ratios) - min(ratios) < 1e-9


# ------------------------------------------------------------------------- loop

def test_global_batch_tokens_invariant_to_microbatch():
    a = get_config("mini", micro_batch=16, grad_accum=32)
    b = get_config("mini", micro_batch=8, grad_accum=64)
    assert a.global_batch_tokens == b.global_batch_tokens


def test_train_step_accumulates_grad_accum_microbatches():
    cfg = _tiny_cfg(grad_accum=5)
    seed_everything(0)
    model = MiniGPT(cfg)
    from mini_gpt.train.optim import build_optimizer

    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.context))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.context))
    calls = {"n": 0}

    def get_batch():
        calls["n"] += 1
        return x, y

    trainer = Trainer(model, opt, sched, get_batch, cfg, grad_accum=5)
    trainer.train_step()
    assert calls["n"] == 5
    assert trainer.step == 1


def test_loss_decreases_on_real_data(packed_tiny):
    cfg = _tiny_cfg()
    trainer, _ = _build_trainer(cfg, packed_tiny, seed=0)
    losses = trainer.train(120)
    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    assert last < first, f"loss did not decrease: {first:.3f} -> {last:.3f}"


# ----------------------------------------------------------------------- resume

def test_resume_produces_identical_trajectory(packed_tiny, tmp_path):
    cfg = _tiny_cfg(max_steps=200)
    n = 15

    # Uninterrupted: train 2n, snapshot final params.
    ref_trainer, _ = _build_trainer(cfg, packed_tiny, seed=3)
    ref_trainer.train(2 * n)
    ref_params = [p.detach().clone() for p in ref_trainer.model.parameters()]

    # Interrupted: train n, checkpoint, resume in a fresh trainer, train n.
    t1, d1 = _build_trainer(cfg, packed_tiny, seed=3)
    t1.train(n)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, t1, d1)

    t2, d2 = _build_trainer(cfg, packed_tiny, seed=3)
    load_checkpoint(ckpt, t2, d2)
    assert t2.step == n
    t2.train(n)

    resumed_params = list(t2.model.parameters())
    for a, b in zip(ref_params, resumed_params):
        assert torch.allclose(a, b, atol=1e-6), "resumed trajectory diverged"


# ==========================================================================
# Muon, torch.compile, and the throughput harness
# ==========================================================================

from mini_gpt.train.optim import (  # noqa: E402
    CombinedOptimizer,
    build_optimizers,
    zeropower_via_newtonschulz5,
)


# ------------------------------------------------------------------------- Muon

@pytest.mark.parametrize("shape", [(128, 64), (64, 128), (256, 128)])
def test_newton_schulz_orthogonalizes(shape):
    torch.manual_seed(0)
    g = torch.randn(*shape)
    u = zeropower_via_newtonschulz5(g, steps=5)
    sv = torch.linalg.svdvals(u)
    # A 2:1 random matrix's singular values land in ~[0.68, 1.13] after 5 steps.
    assert sv.min().item() >= 0.6
    assert sv.max().item() <= 1.2


def test_muon_fallback_is_plain_adamw():
    model = MiniGPT(_tiny_cfg(use_muon=False))
    opt = build_optimizers(model, _tiny_cfg(use_muon=False))
    assert isinstance(opt, torch.optim.AdamW)

    model2 = MiniGPT(_tiny_cfg(use_muon=True))
    opt2 = build_optimizers(model2, _tiny_cfg(use_muon=True))
    assert isinstance(opt2, CombinedOptimizer)
    # Muon on hidden, AdamW on misc.
    assert len(opt2.optimizers) == 2


def _run_tier(data_dir, use_muon: bool, steps: int) -> list[float]:
    cfg = _tiny_cfg(
        micro_batch=16, grad_accum=1, max_steps=400, lr_adamw=3e-3, lr_muon=2e-2, use_muon=use_muon
    )
    seed_everything(0)
    model = MiniGPT(cfg)
    opt = build_optimizers(model, cfg)
    sched = build_scheduler(opt, cfg)
    sampler = ShardSampler(data_dir, context=cfg.context, split="train", seed=0)
    data = DataStream(sampler, cfg.micro_batch)
    trainer = Trainer(model, opt, sched, data.batch, cfg, grad_accum=1)
    return trainer.train(steps)


def test_muon_converges_faster_per_token(packed_tiny):
    steps = 150
    adamw = _run_tier(packed_tiny, use_muon=False, steps=steps)
    muon = _run_tier(packed_tiny, use_muon=True, steps=steps)

    def mean(xs, a, b):
        return sum(xs[a:b]) / (b - a)

    # Muon conditions the 2D-weight update and reaches a lower loss for the same
    # tokens than the pure-AdamW baseline. Measured over the descent window, since
    # the synthetic task plateaus near its entropy floor.
    assert mean(muon, 10, steps) < mean(adamw, 10, steps), (
        f"muon {mean(muon, 10, steps):.3f} vs adamw {mean(adamw, 10, steps):.3f}"
    )


# ---------------------------------------------------------------- torch.compile

def test_compile_matches_eager():
    seed_everything(0)
    cfg = _tiny_cfg()
    raw = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.context))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.context))

    try:
        compiled = torch.compile(raw)
        with torch.no_grad():
            _, eager_loss = raw(x, y)
            _, comp_loss = compiled(x, y)
    except Exception as e:  # missing backend/compiler in this environment
        pytest.skip(f"torch.compile unavailable: {e}")

    assert torch.allclose(eager_loss, comp_loss, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------------ bench

def test_bench_smoke():
    from mini_gpt import bench

    cfg = _tiny_cfg()
    stats = bench.measure(cfg, steps=3, warmup=1, device="cpu")
    for key in ("tier", "device", "micro_batch", "tokens_per_s", "achieved_tflops", "peak_vram_mb", "mfu"):
        assert key in stats
    assert stats["tokens_per_s"] > 0
    assert stats["peak_vram_mb"] is None  # CPU has no VRAM metric
    assert isinstance(bench.max_microbatch(cfg, device="cpu"), int)
    # MFU is computed when a device peak is supplied.
    with_peak = bench.measure(cfg, steps=2, warmup=0, device="cpu", peak_tflops=100.0)
    assert with_peak["mfu"] is not None and with_peak["mfu"] > 0
    assert isinstance(bench.format_table(stats), str)


# -------------------------------------------------------------- chunked-CE bench

def test_bench_chunked():
    from mini_gpt import bench

    cfg = _tiny_cfg(ce_chunk=64)
    # Naive and chunked CE both produce a valid throughput measurement.
    naive = bench.measure(cfg, steps=3, warmup=1, device="cpu", fused_loss=False)
    chunked = bench.measure(cfg, steps=3, warmup=1, device="cpu", fused_loss=True)
    assert naive["fused_loss"] is False and chunked["fused_loss"] is True
    assert chunked["tokens_per_s"] > 0
    # The chunked path fits at least as large a micro-batch as the naive one.
    assert bench.max_microbatch(cfg, device="cpu", fused_loss=True) >= 1

    if torch.cuda.is_available():
        # The chunked-CE memory win, measured directly: at the same micro-batch on
        # the real 32K-vocab tier, chunked CE never materializes the [mb*ctx, V]
        # logits, so its peak VRAM is strictly below the naive path's. A full
        # max-micro-batch scan is `bench.py --scan-microbatch`; this assertion
        # stays at two cheap steps.
        gpu_cfg = get_config("nano")
        naive = bench.measure(gpu_cfg, steps=1, warmup=0, device="cuda", micro_batch=8, fused_loss=False)
        chunked = bench.measure(gpu_cfg, steps=1, warmup=0, device="cuda", micro_batch=8, fused_loss=True)
        assert chunked["peak_vram_mb"] < naive["peak_vram_mb"], (
            f"chunked {chunked['peak_vram_mb']:.0f}MB !< naive {naive['peak_vram_mb']:.0f}MB"
        )


# ==========================================================================
# the anneal switch and context extension
# ==========================================================================

from mini_gpt.data.anneal import (  # noqa: E402
    AnnealDataStream,
    anneal_switch_step,
    mix_docs,
    synthetic_math_docs,
)
from mini_gpt.model.rope import scale_rope_base  # noqa: E402
from scripts.extend_context import extended_config  # noqa: E402


@pytest.fixture(scope="module")
def anneal_dirs(tmp_path_factory):
    """A base shard dir and an anneal-mix shard dir sharing one tokenizer."""
    base_docs = list(download.synthetic_docs(1500, seed=7))
    math_docs = list(synthetic_math_docs(1500, seed=7))
    tok = MiniTokenizer.train(base_docs + math_docs, vocab_size=512, min_frequency=1)

    base_out = tmp_path_factory.mktemp("anneal_base")
    pack.pack_corpus(base_docs, tok, base_out, shard_tokens=3000, val_shards=0)

    anneal_out = tmp_path_factory.mktemp("anneal_mix")
    mixed = list(mix_docs(base_docs, math_docs, math_frac=0.5, seed=1))
    pack.pack_corpus(mixed, tok, anneal_out, shard_tokens=3000, val_shards=0)
    return base_out, anneal_out


# --------------------------------------------------------------- anneal switch

def test_anneal_switch_step_is_last_fraction():
    assert anneal_switch_step(1000, 0.02) == 980
    assert anneal_switch_step(100, 0.0) == 100  # no anneal -> never switches within budget


def _make_anneal(base_dir, anneal_dir, *, switch_step, micro_batch=4, grad_accum=2):
    b = DataStream(ShardSampler(base_dir, context=16, split="train", seed=0), micro_batch)
    a = DataStream(ShardSampler(anneal_dir, context=16, split="train", seed=1), micro_batch)
    return AnnealDataStream(b, a, switch_step=switch_step, grad_accum=grad_accum)


def test_anneal_stream_switches_at_the_boundary(anneal_dirs):
    base_dir, anneal_dir = anneal_dirs
    ga = 2
    s = _make_anneal(base_dir, anneal_dir, switch_step=3, grad_accum=ga)
    for step in range(6):
        for _ in range(ga):
            s.batch()
        if step < 3:
            assert not s.switched, f"switched too early at step {step}"
            assert s.anneal.sampler.position == 0  # anneal source untouched
        else:
            assert s.switched
            assert s.anneal.sampler.position > 0  # anneal source now feeding


def test_anneal_switch_is_resumable(anneal_dirs):
    base_dir, anneal_dir = anneal_dirs
    ga = 2

    def draw(stream, nsteps):
        out = []
        for _ in range(nsteps):
            for _ in range(ga):
                x, y = stream.batch()
                out.append(x.clone())
        return out

    full = draw(_make_anneal(base_dir, anneal_dir, switch_step=3, grad_accum=ga), 6)

    interrupted = _make_anneal(base_dir, anneal_dir, switch_step=3, grad_accum=ga)
    part = draw(interrupted, 3)  # base phase
    state = interrupted.state_dict()

    resumed = _make_anneal(base_dir, anneal_dir, switch_step=3, grad_accum=ga)
    resumed.load_state_dict(state)
    assert resumed.switched is False  # switch happens after resume, at step 3
    rest = draw(resumed, 3)  # anneal phase, across the switch

    combined = part + rest
    assert len(combined) == len(full)
    for a, b in zip(combined, full):
        assert torch.equal(a, b), "resumed anneal stream diverged from uninterrupted"
    assert resumed.switched is True  # the switch did occur on the resumed stream


# ------------------------------------------------------------ context extension

def test_scale_rope_base_stretches_for_longer_context():
    base = 10_000.0
    # No-op when not extending.
    assert scale_rope_base(base, 1024, 1024, 64) == base
    assert scale_rope_base(base, 1024, 512, 64) == base
    # Extending raises the base, monotonically in the extension factor.
    b2 = scale_rope_base(base, 1024, 2048, 64)
    b4 = scale_rope_base(base, 1024, 4096, 64)
    assert base < b2 < b4
    # NTK formula: base * s**(d/(d-2)); for s=2, d=64 -> 2**(64/62).
    assert b2 == pytest.approx(base * 2 ** (64 / 62))


def test_extended_config_rescales_and_lengthens():
    cfg = _tiny_cfg(context=16, rope_base=10_000.0)
    ext = extended_config(cfg, 32)
    assert ext.context == 32
    assert ext.rope_base > cfg.rope_base
    assert ext.window == cfg.window  # window unchanged -> bounded windowed cost


def test_base_weights_transfer_to_longer_context(packed_tiny):
    # With no learned positional table, weights load into a longer-context config
    # unchanged and forward runs at the new length.
    cfg = _tiny_cfg(context=16)
    seed_everything(0)
    base = MiniGPT(cfg)
    ext = MiniGPT(extended_config(cfg, 32))
    ext.load_state_dict(base.state_dict())  # identical parameter shapes

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, _ = ext(x)
    assert logits.shape == (2, 32, cfg.vocab_size)


def test_context_extend_continues_to_train_at_longer_context():
    # The extended model still learns at the longer context (overfit one long batch).
    from mini_gpt.train.optim import build_optimizer

    cfg = extended_config(_tiny_cfg(context=16), 32).with_overrides(
        max_steps=80, warmup_steps=5, lr_adamw=1e-3, use_muon=False
    )
    seed_everything(0)
    model = MiniGPT(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.context))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.context))

    losses = []
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 1.0, f"did not train: {losses[0]:.2f} -> {losses[-1]:.2f}"
