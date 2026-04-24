from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def split_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not records:
        return {"train": [], "validation": [], "test": []}

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def split_records_by_group(
    records: list[dict[str, Any]],
    *,
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not records:
        return {"train": [], "validation": [], "test": []}

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = str(rec.get(group_key, ""))
        grouped.setdefault(key, []).append(rec)

    group_items = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    total_groups = len(group_items)
    n_train = int(total_groups * train_ratio)
    n_val = int(total_groups * val_ratio)
    n_test = total_groups - n_train - n_val

    train_groups = group_items[:n_train]
    val_groups = group_items[n_train : n_train + n_val]
    test_groups = group_items[n_train + n_val : n_train + n_val + n_test]

    def flatten(group_pairs: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _, items in group_pairs:
            out.extend(items)
        return out

    return {
        "train": flatten(train_groups),
        "validation": flatten(val_groups),
        "test": flatten(test_groups),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
