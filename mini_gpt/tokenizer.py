"""The 32K byte-level BPE tokenizer, with conversation and tool special tokens.

The vocabulary size lives in config.py and must fit in uint16.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# IDs 0..7, registered before any merge is learned so they survive a retrain.
# Reordering silently changes the meaning of every packed shard and checkpoint.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",          # 0: padding for batched sequences
    "<|bos|>",          # 1: beginning of sequence
    "<|eos|>",          # 2: end of sequence / end of a chat turn
    "<|system|>",       # 3: opens a system-role turn
    "<|user|>",         # 4: opens a user-role turn
    "<|assistant|>",    # 5: opens an assistant-role turn
    "<|tool_call|>",    # 6: marks an assistant tool invocation
    "<|tool_result|>",  # 7: opens a tool-result turn
)

DEFAULT_VOCAB_SIZE = 32_768  # == Config.vocab_size; must stay < 65,536 (uint16)


class MiniTokenizer:
    """A byte-level BPE (HuggingFace `tokenizers` backend) with stable special
    IDs. The 256-byte alphabet makes encode->decode an exact identity: no
    [UNK], no out-of-vocabulary failure mode.
    """

    def __init__(self, backend: Tokenizer):
        self._tok = backend
        # A foreign tokenizer missing these fails here rather than silently
        # producing None IDs downstream.
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
        """Train a byte-level BPE: start from the 256-byte alphabet plus the
        special tokens, then learn `vocab_size - 256 - 8` merges by greedily
        joining the most frequent adjacent pair."""
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
        """A content hash (sha256 of the serialized tokenizer).

        Packed shards are bare uint16 IDs, meaningless without the tokenizer
        that produced them; data.py stores this in the manifest so a mismatched
        pairing fails loudly instead of training on garbage.
        """
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
