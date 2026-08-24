"""Data pipeline tests: download idempotency, uint16 packing, and the sampler.

Everything runs offline: the download source is the deterministic synthetic
generator, and a small tokenizer is trained on the same text.
"""

from __future__ import annotations

import numpy as np
import pytest

from mini_gpt.data import download, pack, sampler
from mini_gpt.tokenizer import MiniTokenizer


@pytest.fixture(scope="module")
def tok() -> MiniTokenizer:
    corpus = list(download.synthetic_docs(2_000, seed=1))
    return MiniTokenizer.train(corpus, vocab_size=600, min_frequency=1)


# --------------------------------------------------------------------------
# download / parts / idempotency
# --------------------------------------------------------------------------

def test_download_lands_readable_utf8(tmp_path):
    out = tmp_path / "climbmix"
    written = download.download(
        out, parts=3, docs_per_part=50, source=lambda: download.synthetic_docs(1000, seed=0)
    )
    assert len(written) == 3
    assert all(p.exists() for p in written)
    docs = list(download.read_parts(out))
    assert len(docs) == 150  # 3 parts * 50 docs
    assert all(isinstance(d, str) and d for d in docs)


def test_download_is_idempotent(tmp_path):
    out = tmp_path / "climbmix"
    src = lambda: download.synthetic_docs(1000, seed=0)
    first = download.download(out, parts=2, docs_per_part=50, source=src)
    mtimes = {p: p.stat().st_mtime_ns for p in first}
    # Re-running keeps existing parts untouched (not re-written) and can extend.
    second = download.download(out, parts=4, docs_per_part=50, source=src)
    assert len(second) == 4
    for p in first:
        assert p.stat().st_mtime_ns == mtimes[p], "existing part was rewritten"
    assert download.existing_part_indices(out) == [0, 1, 2, 3]


# --------------------------------------------------------------------------
# pack to uint16 shards + manifest
# --------------------------------------------------------------------------

def test_pack_roundtrip_is_exact(tmp_path, tok):
    docs = list(download.synthetic_docs(400, seed=2))
    out = tmp_path / "packed"
    manifest = pack.pack_corpus(docs, tok, out, shard_tokens=5_000, val_shards=1)

    # Reconstruct the expected stream: each doc encoded, eos between docs.
    expected: list[int] = []
    for d in docs:
        expected.extend(tok.encode(d))
        expected.append(tok.eos_id)

    actual = pack.read_all_tokens(out)  # all splits, in shard order
    assert actual.dtype == np.uint16
    assert actual.tolist() == expected
    assert manifest.total_tokens == len(expected)


def test_pack_uint16_is_lossless(tmp_path, tok):
    docs = list(download.synthetic_docs(200, seed=3))
    out = tmp_path / "packed"
    pack.pack_corpus(docs, tok, out, shard_tokens=3_000)
    toks = pack.read_all_tokens(out)
    assert int(toks.max()) < 65_536
    assert int(toks.min()) >= 0


def test_manifest_pins_tokenizer(tmp_path, tok):
    docs = list(download.synthetic_docs(200, seed=4))
    out = tmp_path / "packed"
    m = pack.pack_corpus(docs, tok, out, shard_tokens=3_000)
    assert m.tokenizer_fingerprint == tok.fingerprint()
    assert m.vocab_size == tok.vocab_size
    assert m.dtype == "uint16"
    assert pack.verify_against_tokenizer(out, tok)

    other = MiniTokenizer.train(
        list(download.synthetic_docs(300, seed=99)), vocab_size=600, min_frequency=1
    )
    # A different tokenizer must not validate against these shards.
    assert not pack.verify_against_tokenizer(out, other)


def test_pack_splits_train_and_val(tmp_path, tok):
    docs = list(download.synthetic_docs(600, seed=5))
    out = tmp_path / "packed"
    m = pack.pack_corpus(docs, tok, out, shard_tokens=4_000, val_shards=1)
    assert len(m.shards) >= 2
    assert len(m.shards_for("val")) == 1
    assert len(m.shards_for("train")) == len(m.shards) - 1
    # The val shard is the last one.
    assert m.shards[-1].split == "val"


# --------------------------------------------------------------------------
# seeded, resumable sampler
# --------------------------------------------------------------------------

@pytest.fixture()
def packed(tmp_path_factory, tok):
    out = tmp_path_factory.mktemp("packed")
    docs = list(download.synthetic_docs(800, seed=6))
    pack.pack_corpus(docs, tok, out, shard_tokens=4_000, val_shards=1)
    return out


def test_sampler_windows_have_right_shape(packed):
    s = sampler.ShardSampler(packed, context=32, seed=0)
    batch = s.next_batch(4)
    assert batch.shape == (4, 33)  # context + 1
    assert batch.dtype == np.int64


def test_same_seed_same_stream(packed):
    a = sampler.ShardSampler(packed, context=16, seed=123)
    b = sampler.ShardSampler(packed, context=16, seed=123)
    xa = a.next_batch(20)
    xb = b.next_batch(20)
    assert np.array_equal(xa, xb)


def test_different_seed_different_stream(packed):
    a = sampler.ShardSampler(packed, context=16, seed=1)
    b = sampler.ShardSampler(packed, context=16, seed=2)
    assert not np.array_equal(a.next_batch(20), b.next_batch(20))


def test_save_restore_resumes_identical_stream(packed):
    # Full reference stream.
    ref = sampler.ShardSampler(packed, context=16, seed=7)
    _ = ref.next_batch(10)
    tail_ref = ref.next_batch(15)

    # Resume from a snapshot taken after the first 10 windows.
    a = sampler.ShardSampler(packed, context=16, seed=7)
    _ = a.next_batch(10)
    state = a.state_dict()

    resumed = sampler.ShardSampler(packed, context=16, seed=7)
    resumed.load_state_dict(state)
    tail_resumed = resumed.next_batch(15)

    assert np.array_equal(tail_ref, tail_resumed)
    assert resumed.position == 25


def test_train_and_val_use_disjoint_shards(packed):
    train = sampler.ShardSampler(packed, context=16, seed=0, split="train")
    val = sampler.ShardSampler(packed, context=16, seed=0, split="val")
    # The held-out val shard is never among the training sampler's shards.
    assert set(train.shard_names).isdisjoint(val.shard_names)


def test_val_only_split_is_available(packed):
    val = sampler.ShardSampler(packed, context=16, seed=0, split="val")
    assert len(val.shard_names) == 1
    assert val.next_window().shape == (17,)
