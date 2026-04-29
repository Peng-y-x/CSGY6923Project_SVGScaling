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

from src.tokenization.encode_dataset import encode_splits_with_tokenizer
from src.tokenization.train_tokenizer import train_bpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tokenizer and encode dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config yaml.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_train_text_file(input_jsonl: Path, output_txt: Path) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with input_jsonl.open("r", encoding="utf-8") as f_in, output_txt.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            row = json.loads(line)
            svg = row.get("svg", "").replace("\n", " ").strip()
            if svg:
                f_out.write(svg + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    processed_dir = Path(cfg["output"]["dir"])
    train_jsonl = processed_dir / "train.jsonl"
    val_jsonl = processed_dir / "validation.jsonl"
    test_jsonl = processed_dir / "test.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"Missing {train_jsonl}. Run preprocessing first: python scripts/run_preprocess.py --config {args.config}"
        )

    tok_cfg = cfg.get("tokenization", {})
    target_cfg = cfg.get("targets", {})
    tokenizer_out_dir = Path(tok_cfg.get("output_dir", processed_dir / "tokenizer"))
    vocab_size = int(tok_cfg.get("vocab_size", 4096))
    min_frequency = int(tok_cfg.get("min_frequency", 2))
    max_length = tok_cfg.get("max_length")
    max_length = int(max_length) if max_length is not None else None
    write_token_ids = bool(tok_cfg.get("write_token_ids", False))
    special_tokens = tok_cfg.get("special_tokens", ["<pad>", "<bos>", "<eos>", "<unk>"])

    print("[1/4] Preparing tokenizer training text...")
    train_text_path = tokenizer_out_dir / "train_text.txt"
    _write_train_text_file(train_jsonl, train_text_path)

    print("[2/4] Training BPE tokenizer...")
    tokenizer_path = train_bpe_tokenizer(
        text_files=[train_text_path],
        output_dir=tokenizer_out_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    print(f"  tokenizer: {tokenizer_path}")

    print("[3/4] Encoding train/validation/test splits...")
    tokenized_out_dir = tokenizer_out_dir / "encoded"
    stats = encode_splits_with_tokenizer(
        tokenizer_path=tokenizer_path,
        train_path=train_jsonl,
        val_path=val_jsonl,
        test_path=test_jsonl,
        output_dir=tokenized_out_dir,
        max_length=max_length,
        write_token_ids=write_token_ids,
    )

    print("[4/4] Writing tokenization stats...")
    stats["config"] = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "max_length": max_length,
        "write_token_ids": write_token_ids,
    }
    with (tokenizer_out_dir / "token_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    min_train_tokens = int(target_cfg.get("min_train_tokens_estimate", 0))
    if min_train_tokens > 0:
        train_tokens = int(stats["splits"]["train"]["total_tokens"])
        if train_tokens < min_train_tokens:
            raise RuntimeError(
                "Train token total below configured target after tokenizer encoding: "
                f"{train_tokens} < {min_train_tokens}. "
                "Increase source data (or max_samples), rerun preprocess+tokenizer."
            )

    print("Done.")
    print(f"Vocab size: {stats['vocab_size']}")
    print("Token totals:")
    for split in ("train", "validation", "test"):
        split_stats = stats["splits"][split]
        print(f"  {split}: {split_stats['total_tokens']}")


if __name__ == "__main__":
    main()
