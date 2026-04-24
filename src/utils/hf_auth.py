from __future__ import annotations

import os
from typing import Any


def ensure_hf_token_from_colab(config: dict[str, Any] | None = None) -> bool:
    """Load HF token from Colab userdata into env vars if available.

    Priority:
    1) Existing env vars (`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HF_HUB_TOKEN`)
    2) Colab userdata key (default: `HF_TOKEN`, configurable)
    """
    if (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    ):
        return True

    cfg = config or {}
    auth_cfg = cfg.get("hf_auth", {})
    if not bool(auth_cfg.get("auto_load_colab_key", True)):
        return False

    key_name = str(auth_cfg.get("colab_key_name", "HF_TOKEN"))
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return False

    try:
        token = userdata.get(key_name)
    except Exception:
        return False

    if not token:
        return False

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    os.environ["HF_HUB_TOKEN"] = token
    return True

