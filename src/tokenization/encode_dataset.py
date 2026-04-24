from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _histogram(values: list[int], bins: list[int]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for val in values:
        placed = False
        for i in range(len(bins) - 1):
            lo = bins[i]
            hi = bins[i + 1]
            if lo <= val < hi:
                out[f"{lo}-{hi - 1}"] += 1
                placed = True
                break
        if not placed:
            out[f"{bins[-1]}+"] += 1
    return dict(out)


def encode_splits_with_tokenizer(
    *,
    tokenizer_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    max_length: int | None = None,
    write_token_ids: bool = False,
) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": train_path,
        "validation": val_path,
        "test": test_path,
    }

    stats: dict[str, Any] = {
        "vocab_size": tokenizer.get_vocab_size(),
        "splits": {},
    }
    hist_bins = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for split_name, split_path in split_map.items():
        rows = _read_jsonl(split_path)
        encoded_rows: list[dict[str, Any]] = []
        lengths: list[int] = []
        dropped_too_long = 0

        for row in rows:
            enc = tokenizer.encode(row["svg"])
            ids = enc.ids
            seq_len = len(ids)
            if max_length is not None and seq_len > max_length:
                dropped_too_long += 1
                continue

            lengths.append(seq_len)
            encoded = {
                "id": row.get("id", ""),
                "source_dataset": row.get("source_dataset", ""),
                "source_filename": row.get("source_filename", ""),
                "seq_len": seq_len,
            }
            if write_token_ids:
                encoded["input_ids"] = ids
            encoded_rows.append(encoded)

        _write_jsonl(output_dir / f"{split_name}.jsonl", encoded_rows)

        stats["splits"][split_name] = {
            "num_sequences": len(encoded_rows),
            "dropped_too_long": dropped_too_long,
            "total_tokens": int(sum(lengths)),
            "seq_len_histogram": _histogram(lengths, hist_bins),
            "min_seq_len": int(min(lengths)) if lengths else 0,
            "max_seq_len": int(max(lengths)) if lengths else 0,
            "mean_seq_len": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        }

    return stats
