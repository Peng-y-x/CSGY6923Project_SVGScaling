from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mup import MuSharedReadout
except ImportError as exc:  # pragma: no cover - exercised in Colab/runtime env
    raise ImportError("Install mup>=1.0.0 to use MupDecoderOnlyTransformer.") from exc


@dataclass
class MupTransformerConfig:
    vocab_size: int
    block_size: int = 2048
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    pad_token_id: int = 0
    bias: bool = True
    base_d_model: int = 64


class MupCausalSelfAttention(nn.Module):
    def __init__(self, cfg: MupTransformerConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=2)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / float(self.head_dim)
        causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask.view(bsz, 1, 1, seq_len), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, width)
        return self.resid_drop(self.proj(y))


class MupMLP(nn.Module):
    def __init__(self, cfg: MupTransformerConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class MupTransformerBlock(nn.Module):
    def __init__(self, cfg: MupTransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MupCausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MupMLP(cfg)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MupDecoderOnlyTransformer(nn.Module):
    def __init__(self, cfg: MupTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MupTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = MuSharedReadout(self.token_emb.weight, bias=False)
        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.cfg.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.cfg.block_size}")

        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
        key_padding_mask = attention_mask.bool() if attention_mask is not None else None
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_emb.weight.numel()
        return total


def make_mup_base_config(cfg: MupTransformerConfig, d_model: int | None = None) -> MupTransformerConfig:
    base_width = int(d_model or cfg.base_d_model)
    if base_width <= 0:
        raise ValueError("base_d_model must be positive")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError("target d_model must be divisible by n_heads")

    d_ff_ratio = cfg.d_ff / cfg.d_model
    return MupTransformerConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        d_model=base_width,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=max(cfg.n_heads, int(round(base_width * d_ff_ratio))),
        dropout=cfg.dropout,
        pad_token_id=cfg.pad_token_id,
        bias=cfg.bias,
        base_d_model=base_width,
    )
