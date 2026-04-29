from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hf_auth import ensure_hf_token_from_colab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push tokenized train/validation/test splits to HF dataset repo."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Override tokenized dataset repo id.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _require_input_ids(rows: list[dict[str, Any]], split_name: str) -> None:
    if not rows:
        raise ValueError(f"Split '{split_name}' is empty. Cannot push empty tokenized split.")
    if "input_ids" not in rows[0]:
        raise ValueError(
            "Tokenized rows do not contain `input_ids`. "
            "Set tokenization.write_token_ids=true in configs/data.yaml, rerun run_tokenizer, then retry push."
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    loaded = ensure_hf_token_from_colab(cfg)
    if loaded:
        print("[auth] Loaded HF token from Colab key.")

    tok_cfg = cfg.get("tokenization", {})
    push_cfg = cfg.get("hf_tokenized_push", {})
    repo_id = args.repo_id or push_cfg.get("repo_id")
    private = bool(push_cfg.get("private", False))

    if not repo_id:
        raise ValueError("Tokenized dataset HF repo_id is missing. Set hf_tokenized_push.repo_id.")

    tokenizer_dir = Path(tok_cfg.get("output_dir", Path(cfg["output"]["dir"]) / "tokenizer"))
    encoded_dir = tokenizer_dir / "encoded"
    split_paths = {
        "train": encoded_dir / "train.jsonl",
        "validation": encoded_dir / "validation.jsonl",
        "test": encoded_dir / "test.jsonl",
    }
    for split, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing tokenized split file: {path}. Run `python scripts/run_tokenizer.py --config ...` first."
            )

    train_rows = _read_jsonl(split_paths["train"])
    val_rows = _read_jsonl(split_paths["validation"])
    test_rows = _read_jsonl(split_paths["test"])

    _require_input_ids(train_rows, "train")
    _require_input_ids(val_rows, "validation")
    _require_input_ids(test_rows, "test")

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
            "test": Dataset.from_list(test_rows),
        }
    )
    ds.push_to_hub(repo_id=repo_id, private=private)

    print(f"Pushed tokenized dataset to: {repo_id}")
    print(json.dumps({"repo_id": repo_id, "repo_type": "dataset"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
