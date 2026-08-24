"""Tokenizer round-trip, special tokens, and the chat template.

A small tokenizer is trained once per session on the project's own README and
source: enough varied text to exercise merges without a network download. The
round-trip guarantee is a property of byte-level BPE, so it holds at any vocab
size.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_gpt.chat_template import Message, build_prompt, render_chat
from mini_gpt.tokenizer import DEFAULT_VOCAB_SIZE, SPECIAL_TOKENS, MiniTokenizer

ROOT = Path(__file__).resolve().parent.parent
TRAIN_VOCAB = 1_000


@pytest.fixture(scope="session")
def tok() -> MiniTokenizer:
    corpus_text = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "README.md",
            "config.py",
            "mini_gpt/model/transformer.py",
            "mini_gpt/posttrain/grpo.py",
            "mini_gpt/eval/harness.py",
        )
    )
    # Repeat so a small vocab has enough frequency to fill every merge slot.
    corpus = (corpus_text * 4).split("\n")
    return MiniTokenizer.train(corpus, vocab_size=TRAIN_VOCAB, min_frequency=1)


# --------------------------------------------------------------------------
# train / encode / decode / special tokens
# --------------------------------------------------------------------------

def test_vocab_size_is_exact(tok):
    assert tok.vocab_size == TRAIN_VOCAB


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "  leading and trailing spaces  ",
        "tabs\tand\nnewlines\r\n",
        "Café — naïve façade, Straße, ¡Hola!",
        "mixed unicode: \u2603 \u00e9 \u4f60\u597d",
        "code: x = n_q_heads * head_dim  # 512",
        "",
        "a",
        "   ",
    ],
)
def test_encode_decode_is_exact_identity(tok, text):
    # Byte-level BPE round-trips every input exactly.
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_have_stable_low_ids(tok):
    # Fixed order, first IDs, never reordered (see tokenizer.SPECIAL_TOKENS).
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(name) == i
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2


def test_add_bos_eos_wraps_the_sequence(tok):
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    # Stripping the wrappers and decoding recovers the text.
    assert tok.decode(ids[1:-1]) == "hi"


def test_save_and_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    reloaded = MiniTokenizer.load(path)
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.special_ids == tok.special_ids
    text = "reload me: façade 512"
    assert reloaded.encode(text) == tok.encode(text)


def test_loading_tokenizer_without_specials_raises(tmp_path):
    # A foreign tokenizer missing the special-token block must fail loudly rather
    # than silently produce None IDs.
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    backend = Tokenizer(models.BPE(unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["hello world"] * 50,
        trainer=trainers.BpeTrainer(vocab_size=300, show_progress=False),
    )
    path = tmp_path / "foreign.json"
    backend.save(str(path))
    with pytest.raises(ValueError):
        MiniTokenizer.load(path)


# --------------------------------------------------------------------------
# chat template and the assistant-only loss mask
# --------------------------------------------------------------------------

def test_template_masks_only_assistant_spans(tok):
    messages = [
        Message("system", "You are helpful."),
        Message("user", "What is 2+2?"),
        Message("assistant", "It is 4."),
    ]
    rendered = render_chat(messages, tok)
    assert len(rendered.ids) == len(rendered.loss_mask)

    # Every masked-in token must belong to the assistant content or its <|eos|>.
    assistant_ids = tok.encode("It is 4.") + [tok.eos_id]
    kept = [i for i, m in zip(rendered.ids, rendered.loss_mask) if m == 1]
    assert kept == assistant_ids

    # The <|assistant|> header itself is NOT in the loss.
    asst_header = tok.token_to_id("<|assistant|>")
    header_positions = [i for i, t in enumerate(rendered.ids) if t == asst_header]
    for pos in header_positions:
        assert rendered.loss_mask[pos] == 0


def test_template_masks_out_everything_before_first_assistant(tok):
    messages = [Message("user", "hello"), Message("assistant", "hi there")]
    rendered = render_chat(messages, tok)
    # Locate the assistant header; everything up to and including it is masked.
    asst_header = tok.token_to_id("<|assistant|>")
    idx = rendered.ids.index(asst_header)
    assert all(m == 0 for m in rendered.loss_mask[: idx + 1])
    assert any(m == 1 for m in rendered.loss_mask[idx + 1 :])


def test_multi_turn_unmasks_each_assistant_turn(tok):
    messages = [
        Message("user", "one"),
        Message("assistant", "first"),
        Message("user", "two"),
        Message("assistant", "second"),
    ]
    rendered = render_chat(messages, tok)
    kept = [i for i, m in zip(rendered.ids, rendered.loss_mask) if m == 1]
    expected = (
        tok.encode("first") + [tok.eos_id] + tok.encode("second") + [tok.eos_id]
    )
    assert kept == expected


def test_generation_prompt_appends_unmasked_assistant_header(tok):
    messages = [Message("user", "hello")]
    prompt = build_prompt(messages, tok)
    # Ends with an assistant header primed for generation.
    assert prompt[-1] == tok.token_to_id("<|assistant|>")
    # Consistency: build_prompt is render_chat with add_generation_prompt=True.
    assert prompt == render_chat(messages, tok, add_generation_prompt=True).ids


def test_template_starts_with_bos(tok):
    rendered = render_chat([Message("user", "x")], tok)
    assert rendered.ids[0] == tok.bos_id
    assert rendered.loss_mask[0] == 0


def test_unknown_role_raises(tok):
    with pytest.raises(ValueError):
        render_chat([Message("robot", "beep")], tok)  # type: ignore[arg-type]
