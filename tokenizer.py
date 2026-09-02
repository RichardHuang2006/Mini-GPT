"""The 32K byte-level BPE tokenizer with conversation and tool special tokens.

What this file teaches
    How a byte-level BPE turns raw text into token IDs: every input string is
    first mapped to bytes (so nothing is ever out-of-vocabulary), then greedily
    merged into larger units learned from a training corpus. Special tokens for
    conversation roles and tool use are reserved at fixed IDs before training.

Read first
    config.py (for the 32,768 vocabulary size and why it must fit in uint16).

Inputs and outputs
    train():  iterable of text strings           -> a MiniTokenizer
    encode(): str                                -> list[int] token IDs
    decode(): list[int]                          -> str (exact round-trip)
    save()/load(): a single JSON file on disk.

Representative command (train a tokenizer as part of data preparation):
    python data.py --source synthetic --parts 2 --docs-per-part 2000 \
        --tokenizer data/tok.json --data data/packed --shard-tokens 100000
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Special tokens occupy IDs 0..7, registered with the trainer before any merge
# is learned so they are stable across every retrain. Reordering them would
# silently shift the meaning of every previously-packed shard and checkpoint,
# so the order is fixed. The conversation roles (<|system|>, <|user|>,
# <|assistant|>) and the tool tokens (<|tool_call|>, <|tool_result|>) are what
# posttrain.py's chat template is built from.
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
    """A byte-level BPE (HuggingFace `tokenizers` Rust backend) with stable
    special-token IDs.

    Byte-level means the base alphabet is all 256 bytes, so encode->decode is
    an exact identity for any input text: there is no [UNK] token and no
    out-of-vocabulary failure mode.
    """

    def __init__(self, backend: Tokenizer):
        self._tok = backend
        # Cache special-token IDs up front. Loading a foreign tokenizer that
        # lacks them is a hard error rather than a silent None, because the
        # chat template and packed data depend on these exact tokens.
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
        """Train a byte-level BPE on an iterable of text strings.

        The trainer starts from the 256-byte alphabet plus the special tokens
        and learns `vocab_size - 256 - 8` merges by greedily joining the most
        frequent adjacent pair.
        """
        backend = Tokenizer(models.BPE(unk_token=None))
        # add_prefix_space=False keeps encode->decode an exact identity, and
        # the ByteLevel alphabet makes all 256 bytes representable.
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

        Why this exists: packed token shards are just uint16 IDs, meaningless
        without the exact tokenizer that produced them. data.py records this
        fingerprint in the shard manifest, so mixing a shard directory with the
        wrong tokenizer is caught loudly instead of producing garbage training
        text.
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
