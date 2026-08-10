"""The decoder-only transformer.

Pre-norm blocks, tied input/output embeddings, GPT-2 residual-scaled init, and a
forward that returns cross-entropy loss. The parameter count computed here (via
``torch``) matches ``Config.param_count`` exactly, because both count the same
weights: tied embeddings, per-head QK-norm gains, two RMSNorm gains per layer
plus a final one, and no biases.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import Config
from mini_gpt.model.attention import Attention
from mini_gpt.model.mlp import SwiGLU
from mini_gpt.model.norm import RMSNorm
from mini_gpt.model.rope import precompute_rope

IGNORE_INDEX = -100


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        use_triton = getattr(cfg, "use_triton", False)
        self.attn_norm = RMSNorm(cfg.d_model, use_triton=use_triton)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, use_triton=use_triton)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden, use_triton=use_triton)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seg_equal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, seg_equal)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, use_triton=getattr(cfg, "use_triton", False))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Fused chunked-CE loss: opt-in. When on, forward computes the
        # loss without ever materializing the [B*T, V] logits, and returns
        # logits=None. Off by default so the eager `logits` API is preserved.
        self.fused_loss = False

        self.apply(self._init_weights)
        self._scale_residual_projections()
        # Tie AFTER init so the embedding's initialization is the one kept.
        self.lm_head.weight = self.embed.weight

    # ------------------------------------------------------------- init
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        # GPT-2 residual growth control: scale the projections that write into
        # the residual stream by 1/sqrt(2N) in the depth N.
        scale = (2 * self.cfg.n_layers) ** -0.5
        for block in self.layers:
            block.attn.o_proj.weight.data.mul_(scale)
            block.mlp.down.weight.data.mul_(scale)

    # ---------------------------------------------------------- forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        cos, sin = precompute_rope(
            self.cfg.head_dim, t, self.cfg.rope_base, device=idx.device, dtype=self.embed.weight.dtype
        )

        # Packed-conversation attention mask: two positions may attend
        # only if they carry the same segment id.
        seg_equal = None
        if segment_ids is not None:
            seg_equal = segment_ids[:, :, None] == segment_ids[:, None, :]

        x = self.embed(idx)
        for block in self.layers:
            x = block(x, cos, sin, seg_equal)
        x = self.final_norm(x)

        tgt = None
        if targets is not None:
            tgt = targets.clone()
            if loss_mask is not None:
                # Positions with mask==0 are excluded from the loss.
                tgt = tgt.masked_fill(loss_mask == 0, IGNORE_INDEX)

        # Memory-efficient path: stream the vocabulary instead of materializing
        # the full logits tensor (the pivotal batch-size unlock).
        if self.fused_loss and tgt is not None:
            from mini_gpt.kernels import chunked_cross_entropy

            loss = chunked_cross_entropy(
                x.reshape(-1, x.size(-1)),
                self.lm_head.weight,
                tgt.reshape(-1),
                ignore_index=IGNORE_INDEX,
                chunk=getattr(self.cfg, "ce_chunk", 8192),
            )
            return None, loss

        logits = self.lm_head(x)
        loss = None
        if tgt is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    # ------------------------------------------------------------ utils
    def num_parameters(self) -> int:
        """Unique parameter count (tied weight counted once)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    @classmethod
    def from_config(cls, cfg: Config) -> "MiniGPT":
        return cls(cfg)
