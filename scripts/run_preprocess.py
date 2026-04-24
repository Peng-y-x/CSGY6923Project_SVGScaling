from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_splits import split_records, split_records_by_group, write_jsonl
from src.data.clean_svg import clean_svg_text
from src.data.download_hf import load_svg_records
from src.data.push_to_hf import push_splits_to_hf
from src.data.validate_svg import (
    estimate_token_count,
    has_svg_root,
    validate_render,
    validate_xml,
)

PIPELINE_VERSION = "preprocess-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SVG preprocessing pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to preprocessing config yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if manifest hash matches and outputs exist.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _count_stats(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not records:
        return {"count": 0, "min": 0, "max": 0, "mean": 0}

    values = [float(r[key]) for r in records]
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _histogram(values: list[int], bins: list[int]) -> dict[str, int]:
    if not values:
        return {}

    out: dict[str, int] = Counter()
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


def _write_complexity_examples(
    records: list[dict[str, Any]],
    output_dir: Path,
    enable_render_validation: bool,
) -> list[dict[str, Any]]:
    if not records:
        return []

    sorted_records = sorted(records, key=lambda x: x["token_estimate"])
    idx_low = max(0, int(len(sorted_records) * 0.10) - 1)
    idx_mid = max(0, int(len(sorted_records) * 0.50) - 1)
    idx_high = max(0, int(len(sorted_records) * 0.90) - 1)

    picks = [
        ("low", sorted_records[idx_low]),
        ("medium", sorted_records[idx_mid]),
        ("high", sorted_records[idx_high]),
    ]

    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []
    for label, rec in picks:
        svg_path = examples_dir / f"{label}.svg"
        svg_path.write_text(rec["svg"], encoding="utf-8")

        png_path = examples_dir / f"{label}.png"
        rendered = False
        render_error: str | None = None
        if enable_render_validation:
            try:
                import cairosvg

                cairosvg.svg2png(
                    bytestring=rec["svg"].encode("utf-8"),
                    write_to=str(png_path),
                )
                rendered = True
            except Exception as exc:  # pragma: no cover
                render_error = str(exc)

        manifests.append(
            {
                "label": label,
                "id": rec["id"],
                "source_dataset": rec["source_dataset"],
                "source_filename": rec.get("source_filename", ""),
                "token_estimate": rec["token_estimate"],
                "svg_path": str(svg_path),
                "png_path": str(png_path) if rendered else "",
                "rendered_png": rendered,
                "render_error": render_error,
            }
        )
    return manifests


def _build_manifest_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "seed": config.get("seed", 42),
        "cache_dir": config.get("cache_dir"),
        "sources": config.get("sources", []),
        "cleaning": config.get("cleaning", {}),
        "filters": config.get("filters", {}),
        "splits": config.get("splits", {}),
        "deduplication": config.get("deduplication", {}),
        "validation": config.get("validation", {}),
        "targets": config.get("targets", {}),
    }


def _manifest_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _outputs_complete(output_dir: Path) -> bool:
    required = [
        output_dir / "train.jsonl",
        output_dir / "validation.jsonl",
        output_dir / "test.jsonl",
        output_dir / "stats.json",
        output_dir / "examples_manifest.json",
        output_dir / "manifest.json",
    ]
    return all(path.exists() for path in required)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = config.get("preprocess", {})
    reuse_if_match = bool(preprocess_cfg.get("reuse_processed_if_match", True))
    force_rebuild = bool(preprocess_cfg.get("force_rebuild", False)) or args.force

    manifest_path = output_dir / "manifest.json"
    manifest_payload = _build_manifest_payload(config)
    expected_hash = _manifest_hash(manifest_payload)
    existing_manifest = _load_manifest(manifest_path)

    print("[1/8] Checking processed cache + manifest...")
    if (
        not force_rebuild
        and reuse_if_match
        and _outputs_complete(output_dir)
        and existing_manifest is not None
        and existing_manifest.get("config_hash") == expected_hash
    ):
        print("  Reuse hit: processed outputs are complete and config hash matches.")
        print(f"  Output directory: {output_dir}")
        return

    if force_rebuild:
        print("  Force rebuild enabled. Regenerating outputs.")
    elif existing_manifest is None:
        print("  No valid manifest found. Regenerating outputs.")
    else:
        print("  Manifest mismatch or incomplete outputs. Regenerating outputs.")

    cache_dir = config.get("cache_dir")
    sources = config["sources"]
    clean_cfg = config.get("cleaning", {})
    filter_cfg = config.get("filters", {})
    split_cfg = config.get("splits", {})
    target_cfg = config.get("targets", {})
    validation_cfg = config.get("validation", {})
    dedup_cfg = config.get("deduplication", {})

    min_chars = int(filter_cfg.get("min_chars", 50))
    max_chars = int(filter_cfg.get("max_chars", 10000))
    max_tokens_estimate = int(filter_cfg.get("max_tokens_estimate", 2048))
    seed = int(config.get("seed", 42))
    min_train_tokens_estimate = int(target_cfg.get("min_train_tokens_estimate", 0))

    enable_render_validation = bool(validation_cfg.get("render_check", True))
    split_by_file = bool(split_cfg.get("by_file", True))
    split_group_key = str(split_cfg.get("group_key", "file_key"))
    dedup_by_svg_hash = bool(dedup_cfg.get("by_svg_hash", True))

    print("[2/8] Loading source datasets from Hugging Face (uses cache_dir for reuse)...")
    raw_records = load_svg_records(sources=sources, cache_dir=cache_dir)
    print(f"  Loaded raw records: {len(raw_records)}")

    print("[3/8] Cleaning + filtering + validation...")
    cleaned_records: list[dict[str, Any]] = []
    dropped = {
        "empty_after_clean": 0,
        "too_short": 0,
        "too_long_chars": 0,
        "too_long_tokens_est": 0,
        "invalid_xml": 0,
        "non_svg_root": 0,
        "render_failed": 0,
        "dedup_svg_hash": 0,
    }
    seen_hashes: set[str] = set()

    for rec in raw_records:
        cleaned = clean_svg_text(
            rec["svg_raw"],
            strip_comments=bool(clean_cfg.get("strip_comments", True)),
            strip_metadata=bool(clean_cfg.get("strip_metadata", True)),
            normalize_precision=bool(clean_cfg.get("normalize_precision", True)),
            precision=int(clean_cfg.get("precision", 1)),
            canonicalize_attributes=bool(
                clean_cfg.get("canonicalize_attributes", False)
            ),
        )
        if not cleaned:
            dropped["empty_after_clean"] += 1
            continue

        char_len = len(cleaned)
        if char_len < min_chars:
            dropped["too_short"] += 1
            continue
        if char_len > max_chars:
            dropped["too_long_chars"] += 1
            continue

        token_est = estimate_token_count(cleaned)
        if token_est > max_tokens_estimate:
            dropped["too_long_tokens_est"] += 1
            continue

        ok_xml, _err = validate_xml(cleaned)
        if not ok_xml:
            dropped["invalid_xml"] += 1
            continue

        if not has_svg_root(cleaned):
            dropped["non_svg_root"] += 1
            continue

        if enable_render_validation:
            ok_render, _render_err = validate_render(cleaned)
            if not ok_render:
                dropped["render_failed"] += 1
                continue

        svg_hash = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
        if dedup_by_svg_hash and svg_hash in seen_hashes:
            dropped["dedup_svg_hash"] += 1
            continue
        seen_hashes.add(svg_hash)

        source_filename = str(rec.get("source_filename", ""))
        file_key = (
            f"{rec['source_dataset']}::{source_filename}"
            if source_filename
            else f"{rec['source_dataset']}::{rec['id']}"
        )

        cleaned_records.append(
            {
                "id": rec["id"],
                "source_dataset": rec["source_dataset"],
                "source_split": rec["source_split"],
                "source_subset": rec["source_subset"],
                "source_filename": source_filename,
                "file_key": file_key,
                "svg_hash": svg_hash,
                "svg": cleaned,
                "char_len": char_len,
                "token_estimate": token_est,
            }
        )

    print(f"  Records after cleaning + filtering: {len(cleaned_records)}")
    print(f"  Dropped summary: {json.dumps(dropped, ensure_ascii=False)}")

    print("[4/8] Building train/validation/test split...")
    if split_by_file:
        split_data = split_records_by_group(
            cleaned_records,
            group_key=split_group_key,
            train_ratio=float(split_cfg.get("train", 0.98)),
            val_ratio=float(split_cfg.get("validation", 0.01)),
            test_ratio=float(split_cfg.get("test", 0.01)),
            seed=seed,
        )
    else:
        split_data = split_records(
            cleaned_records,
            train_ratio=float(split_cfg.get("train", 0.98)),
            val_ratio=float(split_cfg.get("validation", 0.01)),
            test_ratio=float(split_cfg.get("test", 0.01)),
            seed=seed,
        )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"
    write_jsonl(train_path, split_data["train"])
    write_jsonl(val_path, split_data["validation"])
    write_jsonl(test_path, split_data["test"])

    print("[5/8] Writing stats + histograms...")
    token_hist_bins = [0, 32, 64, 128, 256, 512, 1024, 2048]
    summary = {
        "raw_records": len(raw_records),
        "cleaned_records": len(cleaned_records),
        "dropped": dropped,
        "split_mode": "by_file" if split_by_file else "random_record",
        "split_group_key": split_group_key if split_by_file else "",
        "split_sizes": {
            "train": len(split_data["train"]),
            "validation": len(split_data["validation"]),
            "test": len(split_data["test"]),
        },
        "char_len_stats": _count_stats(cleaned_records, "char_len"),
        "token_est_stats": _count_stats(cleaned_records, "token_estimate"),
        "token_est_histogram": _histogram(
            [int(x["token_estimate"]) for x in cleaned_records],
            token_hist_bins,
        ),
        "token_est_totals": {
            "train": int(sum(x["token_estimate"] for x in split_data["train"])),
            "validation": int(
                sum(x["token_estimate"] for x in split_data["validation"])
            ),
            "test": int(sum(x["token_estimate"] for x in split_data["test"])),
        },
        "seed": seed,
    }
    summary["train_token_est_total"] = summary["token_est_totals"]["train"]
    if min_train_tokens_estimate > 0:
        summary["target_min_train_tokens_estimate"] = min_train_tokens_estimate
        summary["target_reached"] = (
            summary["train_token_est_total"] >= min_train_tokens_estimate
        )

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[6/8] Exporting complexity examples (SVG + PNG)...")
    examples_manifest = _write_complexity_examples(
        records=cleaned_records,
        output_dir=output_dir,
        enable_render_validation=enable_render_validation,
    )
    with (output_dir / "examples_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(examples_manifest, f, ensure_ascii=False, indent=2)

    if min_train_tokens_estimate > 0 and summary["train_token_est_total"] < min_train_tokens_estimate:
        raise RuntimeError(
            "Train token estimate below configured target: "
            f"{summary['train_token_est_total']} < {min_train_tokens_estimate}. "
            "Increase source data (e.g., max_samples for large datasets) and rerun."
        )

    print("[7/8] Optional push to HF hub...")
    push_cfg = config.get("hf_push", {})
    if bool(push_cfg.get("enabled", False)):
        repo_id = push_cfg["repo_id"]
        private = bool(push_cfg.get("private", False))
        push_splits_to_hf(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            repo_id=repo_id,
            private=private,
        )
        print(f"  Pushed dataset to hub: {repo_id}")
    else:
        print("  Skipped (hf_push.enabled=false).")

    print("[8/8] Writing manifest...")
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": expected_hash,
        "manifest_payload": manifest_payload,
        "output_dir": str(output_dir),
        "stats_file": str(output_dir / "stats.json"),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

