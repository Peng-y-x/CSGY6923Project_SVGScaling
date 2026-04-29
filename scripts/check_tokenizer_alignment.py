from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that a tokenizer decodes tokenized SVG input_ids correctly.")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--dataset", default="Zala0429/svg-scaling-v2-tokenized")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to run tokenizer alignment checks.") from exc

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevelDecoder()

    ds = load_dataset(args.dataset, split=f"{args.split}[:{args.num_samples}]")
    failures = []
    for i, row in enumerate(ds):
        ids = [int(x) for x in row["input_ids"]]
        decoded = tokenizer.decode(ids).strip()
        ok = decoded.startswith("<svg") and ("</svg>" in decoded)
        print(
            {
                "index": i,
                "seq_len": len(ids),
                "starts_svg": decoded.startswith("<svg"),
                "has_close_svg": "</svg>" in decoded,
                "preview": decoded[:120],
            }
        )
        if not ok:
            failures.append(i)

    if failures:
        raise SystemExit(
            "Tokenizer alignment check failed. The tokenizer does not decode tokenized dataset rows "
            f"as complete SVGs. Failed sample indices: {failures}"
        )
    print(f"Tokenizer alignment check passed for {len(ds)} samples.")


if __name__ == "__main__":
    main()
