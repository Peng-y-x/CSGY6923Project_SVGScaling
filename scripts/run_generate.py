from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.render import make_image_grid, render_svg_to_png
from src.generation.sampling import generate_ids
from src.models.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Part 4 SVG samples.")
    parser.add_argument("--config", default="configs/part4_generation.yaml")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--tokenizer-hf-repo-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--drive-output-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean_svg(text: str) -> str:
    start = text.find("<svg")
    if start >= 0:
        text = text[start:]
    end = text.find("</svg>")
    if end >= 0:
        text = text[: end + len("</svg>")]
    return text.strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generate_one(
    *,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    eos_id: int | None,
    text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    stop_text: str | None,
) -> str:
    input_ids = tokenizer.encode(text).ids
    ids = generate_ids(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_id,
        stop_text=stop_text,
        decode_fn=tokenizer.decode,
    )
    return _clean_svg(tokenizer.decode(ids))


def _write_sample(
    *,
    row: dict[str, Any],
    svg: str,
    svg_dir: Path,
    png_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    svg_path = svg_dir / f"{row['id']}.svg"
    png_path = png_dir / f"{row['id']}.png"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    render_ok = render_svg_to_png(svg, png_path)
    row.update(
        {
            "svg_path": str(svg_path),
            "png_path": str(png_path),
            "render_ok": render_ok,
            "svg": svg,
        }
    )
    return row, png_path if render_ok else None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint_path or cfg["checkpoint_path"])
    tokenizer_path = Path(args.tokenizer_path or cfg["tokenizer_path"])
    tokenizer_hf_repo_id = args.tokenizer_hf_repo_id or cfg.get("tokenizer_hf_repo_id")
    tokenizer_filename = str(cfg.get("tokenizer_filename", "tokenizer.json"))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/part4_samples"))
    drive_output_dir = args.drive_output_dir or cfg.get("drive_output_dir")
    max_new_tokens = int(args.max_new_tokens or cfg.get("max_new_tokens", 768))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not tokenizer_path.exists() and tokenizer_hf_repo_id:
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=tokenizer_hf_repo_id,
            filename=tokenizer_filename,
            repo_type="model",
        )
        shutil.copy2(downloaded, tokenizer_path)
        print(f"[tokenizer] Downloaded {tokenizer_filename} from {tokenizer_hf_repo_id} to {tokenizer_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Missing tokenizer: {tokenizer_path}. "
            "Pass --tokenizer-hf-repo-id or set tokenizer_hf_repo_id in the generation config."
        )

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    precision = str(cfg.get("precision", "bf16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.decoder is None:
        tokenizer.decoder = ByteLevelDecoder()
    eos_id = tokenizer.token_to_id("<eos>")
    model, ckpt = load_checkpoint_model(checkpoint_path, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = output_dir / "svg"
    png_dir = output_dir / "png"
    rows: list[dict[str, Any]] = []
    rendered_paths: list[Path] = []

    temperatures = [float(x) for x in cfg.get("temperatures", [0.8])]
    top_k = int(cfg.get("top_k", 0))
    top_p = float(cfg.get("top_p", 1.0))
    stop_text = cfg.get("stop_text", "</svg>")
    unconditional_count = int(cfg.get("unconditional_count", 10))
    prefix_count = int(cfg.get("prefix_count", 5))
    prefixes = cfg.get("prefixes", [])
    prefix_text = str(
        cfg.get(
            "unconditional_prefix",
            '<svg viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">',
        )
    )

    sample_index = 0
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
        temperature_groups = cfg.get("temperature_groups")
        if temperature_groups:
            group_prefixes = cfg.get("prefixes", [])
            for group in temperature_groups:
                group_name = str(group["name"])
                temp = float(group["temperature"])
                group_top_k = int(group.get("top_k", top_k))
                group_top_p = float(group.get("top_p", top_p))
                group_unconditional_count = int(group.get("unconditional_count", 20))
                group_prefix_count = int(group.get("prefix_count", 5))
                for i in range(group_unconditional_count):
                    svg = _generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        eos_id=eos_id,
                        text=prefix_text,
                        max_new_tokens=max_new_tokens,
                        temperature=temp,
                        top_k=group_top_k,
                        top_p=group_top_p,
                        stop_text=stop_text,
                    )
                    row, png_path = _write_sample(
                        row={
                            "id": f"{group_name}_unconditional_{i:02d}",
                            "group": group_name,
                            "kind": "unconditional",
                            "temperature": temp,
                            "top_k": group_top_k,
                            "top_p": group_top_p,
                            "prefix": prefix_text,
                            "stop_text": stop_text,
                        },
                        svg=svg,
                        svg_dir=svg_dir,
                        png_dir=png_dir,
                    )
                    rows.append(row)
                    if png_path is not None:
                        rendered_paths.append(png_path)
                    sample_index += 1

                for i, prefix in enumerate(group_prefixes[:group_prefix_count]):
                    text = str(prefix["text"])
                    svg = _generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        eos_id=eos_id,
                        text=text,
                        max_new_tokens=max_new_tokens,
                        temperature=temp,
                        top_k=group_top_k,
                        top_p=group_top_p,
                        stop_text=stop_text,
                    )
                    row, png_path = _write_sample(
                        row={
                            "id": f"{group_name}_prefix_{i:02d}_{prefix.get('name', 'sample')}",
                            "group": group_name,
                            "kind": "prefix",
                            "temperature": temp,
                            "top_k": group_top_k,
                            "top_p": group_top_p,
                            "prefix_name": prefix.get("name", ""),
                            "prefix": text,
                            "reference_svg": prefix.get("reference_svg", ""),
                            "reference_description": prefix.get("description", ""),
                            "stop_text": stop_text,
                        },
                        svg=svg,
                        svg_dir=svg_dir,
                        png_dir=png_dir,
                    )
                    rows.append(row)
                    if png_path is not None:
                        rendered_paths.append(png_path)
                    sample_index += 1
        else:
            for i in range(unconditional_count):
                temp = temperatures[i % len(temperatures)]
                svg = _generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    eos_id=eos_id,
                    text=prefix_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    stop_text=stop_text,
                )
                sample_id = f"unconditional_{i:02d}_t{temp:g}"
                row, png_path = _write_sample(
                    row={
                    "id": sample_id,
                    "kind": "unconditional",
                    "temperature": temp,
                    "top_k": top_k,
                    "top_p": top_p,
                    "prefix": prefix_text,
                    "stop_text": stop_text,
                    },
                    svg=svg,
                    svg_dir=svg_dir,
                    png_dir=png_dir,
                )
                rows.append(row)
                if png_path is not None:
                    rendered_paths.append(png_path)
                sample_index += 1

            for i, prefix in enumerate(prefixes[:prefix_count]):
                temp = temperatures[i % len(temperatures)]
                text = str(prefix["text"])
                svg = _generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    eos_id=eos_id,
                    text=text,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    stop_text=stop_text,
                )
                sample_id = f"prefix_{i:02d}_{prefix.get('name', 'sample')}_t{temp:g}"
                row, png_path = _write_sample(
                    row={
                    "id": sample_id,
                    "kind": "prefix",
                    "temperature": temp,
                    "top_k": top_k,
                    "top_p": top_p,
                    "prefix_name": prefix.get("name", ""),
                    "prefix": text,
                    "reference_svg": prefix.get("reference_svg", ""),
                    "stop_text": stop_text,
                    },
                    svg=svg,
                    svg_dir=svg_dir,
                    png_dir=png_dir,
                )
                rows.append(row)
                if png_path is not None:
                    rendered_paths.append(png_path)
                sample_index += 1

    _write_jsonl(output_dir / "samples.jsonl", rows)
    (output_dir / "generation_summary.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_hf_repo_id": tokenizer_hf_repo_id,
                "stop_text": stop_text,
                "num_samples": len(rows),
                "num_rendered": sum(1 for r in rows if r["render_ok"]),
                "temperature_groups": cfg.get("temperature_groups", []),
                "checkpoint_config": ckpt.get("config", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    make_image_grid([Path(r["png_path"]) for r in rows], output_dir / "generated_grid.png", cols=5)

    if drive_output_dir:
        drive_path = Path(os.path.expandvars(os.path.expanduser(drive_output_dir)))
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, drive_path, dirs_exist_ok=True)
        print(f"[generate] Synced results to Drive: {drive_path}")

    print(json.dumps({"output_dir": str(output_dir), "num_samples": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
