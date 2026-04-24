from __future__ import annotations

import json
from pathlib import Path

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def push_splits_to_hf(
    *,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    repo_id: str,
    private: bool = False,
) -> None:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(_read_jsonl(train_path)),
            "validation": Dataset.from_list(_read_jsonl(val_path)),
            "test": Dataset.from_list(_read_jsonl(test_path)),
        }
    )
    dataset_dict.push_to_hub(repo_id=repo_id, private=private)
