"""Byte-level BPE tokenizer.

A 32K byte-level BPE. Byte-level means every possible input round-trips: there
is no out-of-vocabulary token and no ``[UNK]``, which is the precondition for
the exact-round-trip test.

The conversation / tool special tokens are added to the trainer *before*
training so their IDs are the first, fixed entries in the vocabulary and are
stable across every tier and every retrain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Order is load-bearing: these occupy IDs 0..N-1 and must never be reordered,
# or every previously-packed shard and checkpoint would silently shift.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|tool_result|>",
)

DEFAULT_VOCAB_SIZE = 32_768


class MiniTokenizer:
    """Thin wrapper over a HF ``Tokenizer`` with stable special-token IDs.

    Training uses the fast Rust BPE trainer; encode/decode at train time is a
    plain byte-level pass with no special-token handling unless asked.
    """

    def __init__(self, backend: Tokenizer):
        self._tok = backend
        # Cache special-token IDs; assert they resolved (a load of a foreign
        # tokenizer without these would be a hard error, not a silent None).
        self.special_ids: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            tid = self._tok.token_to_id(tok)
            if tid is None:
                raise ValueError(f"tokenizer is missing required special token {tok!r}")
            self.special_ids[tok] = tid

    # ------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        *,
        min_frequency: int = 2,
    ) -> "MiniTokenizer":
        """Train a byte-level BPE on an iterable of text strings."""
        backend = Tokenizer(models.BPE(unk_token=None))
        # add_prefix_space=False keeps encode->decode an exact identity; the
        # ByteLevel alphabet guarantees all 256 bytes are representable.
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator(_as_iterator(corpus), trainer=trainer)
        return cls(backend)

    # --------------------------------------------------------- (de)serialize
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "MiniTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # -------------------------------------------------------- encode/decode
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode raw text to token IDs (no special tokens unless requested)."""
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    # ------------------------------------------------------------- accessors
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def fingerprint(self) -> str:
        """A stable content hash of the tokenizer.

        The manifest records this so a packed shard set can be checked against
        the exact tokenizer that produced it.
        """
        import hashlib

        return hashlib.sha256(self._tok.to_str().encode("utf-8")).hexdigest()

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    @property
    def pad_id(self) -> int:
        return self.special_ids["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_ids["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_ids["<|eos|>"]


def _as_iterator(corpus: Iterable[str]) -> Iterable[str]:
    # train_from_iterator accepts any iterable of str; keep it lazy.
    return iter(corpus)
