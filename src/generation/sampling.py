from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def filter_logits(logits: torch.Tensor, *, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    logits = logits.clone()
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    return logits


@torch.no_grad()
def generate_ids(
    model: torch.nn.Module,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: int | None = None,
    stop_text: str | None = None,
    decode_fn: Callable[[list[int]], str] | None = None,
) -> list[int]:
    cfg = model.cfg
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        context = ids[:, -int(cfg.block_size) :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = filter_logits(next_logits / temperature, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if eos_token_id is not None and int(next_id.item()) == eos_token_id:
            break
        if stop_text and decode_fn is not None:
            decoded = decode_fn([int(x) for x in ids[0].detach().cpu().tolist()])
            if stop_text in decoded:
                break
    return [int(x) for x in ids[0].detach().cpu().tolist()]
