"""The 32K byte-level BPE tokenizer, with conversation and tool special tokens."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# IDs 0..7 in this order, registered before any merge so they survive a retrain.
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

DEFAULT_VOCAB_SIZE = 32_768  # == Config.vocab_size; must stay < 65,536 (uint16)


class MiniTokenizer:
    """Byte-level BPE with stable special IDs; encode->decode is exact."""

    def __init__(self, backend: Tokenizer):
        self._tok = backend
        self.special_ids: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            tid = self._tok.token_to_id(tok)
            if tid is None:
                raise ValueError(f"tokenizer is missing required special token {tok!r}")
            self.special_ids[tok] = tid

    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        *,
        min_frequency: int = 2,
    ) -> MiniTokenizer:
        """Learn merges over the 256-byte alphabet plus the special tokens."""
        backend = Tokenizer(models.BPE(unk_token=None))
        # add_prefix_space=False keeps encode->decode an exact identity.
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator(iter(corpus), trainer=trainer)
        return cls(backend)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> MiniTokenizer:
        return cls(Tokenizer.from_file(str(path)))

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

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def fingerprint(self) -> str:
        """sha256 of the serialized tokenizer; data.py records it in the manifest."""
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
