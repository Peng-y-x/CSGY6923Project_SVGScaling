from __future__ import annotations

import torch


def build_adamw(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not name.endswith("token_emb.weight"):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps)
