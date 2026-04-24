from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


def train_bpe_tokenizer(
    *,
    text_files: list[Path],
    output_dir: Path,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"],
    )

    tokenizer.train([str(x) for x in text_files], trainer=trainer)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "special_tokens": special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"],
        "train_files": [str(x) for x in text_files],
    }
    with (output_dir / "tokenizer_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return tokenizer_path
