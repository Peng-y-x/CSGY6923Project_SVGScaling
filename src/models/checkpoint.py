from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.transformer import DecoderOnlyTransformer, TransformerConfig


def build_model_from_checkpoint(ckpt: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("model")
    if not model_cfg:
        raise ValueError("Checkpoint is missing model_config/config.model")

    if "base_d_model" in model_cfg:
        from mup import set_base_shapes

        from src.models.mup_transformer import (
            MupDecoderOnlyTransformer,
            MupTransformerConfig,
            make_mup_base_config,
        )

        cfg = MupTransformerConfig(**model_cfg)
        model = MupDecoderOnlyTransformer(cfg)
        base_cfg = make_mup_base_config(cfg)
        delta_cfg = make_mup_base_config(cfg, d_model=base_cfg.d_model * 2)
        set_base_shapes(model, MupDecoderOnlyTransformer(base_cfg), delta=MupDecoderOnlyTransformer(delta_cfg))
    else:
        cfg = TransformerConfig(**model_cfg)
        model = DecoderOnlyTransformer(cfg)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def load_checkpoint_model(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(Path(path), map_location=device)
    return build_model_from_checkpoint(ckpt, device), ckpt
