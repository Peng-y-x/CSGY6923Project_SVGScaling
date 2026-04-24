from __future__ import annotations

from typing import Any

DEFAULT_TEXT_FIELDS = (
    "svg",
    "Svg",
    "SVG",
    "svg_code",
    "code",
    "content",
    "text",
)

DEFAULT_FILENAME_FIELDS = (
    "Filename",
    "filename",
    "file_name",
    "name",
    "id",
)


def _pick_svg_field(example: dict[str, Any], preferred_field: str | None) -> str:
    if preferred_field:
        value = example.get(preferred_field)
        if isinstance(value, str):
            return value
        raise KeyError(f"Configured text_field '{preferred_field}' not found or not string.")

    # Fall back to case-insensitive key matching first.
    lowered = {k.lower(): k for k in example.keys()}
    for field in DEFAULT_TEXT_FIELDS:
        key = lowered.get(field.lower())
        if key is None:
            continue
        value = example.get(key)
        if isinstance(value, str):
            return value

    for field in DEFAULT_TEXT_FIELDS:
        value = example.get(field)
        if isinstance(value, str):
            return value

    raise KeyError(
        f"Could not find SVG text field. Available keys: {sorted(example.keys())}"
    )


def _pick_filename_field(example: dict[str, Any], preferred_field: str | None) -> str:
    if preferred_field:
        value = example.get(preferred_field)
        if value is not None:
            return str(value)

    lowered = {k.lower(): k for k in example.keys()}
    for field in DEFAULT_FILENAME_FIELDS:
        key = lowered.get(field.lower())
        if key is None:
            continue
        value = example.get(key)
        if value is not None:
            return str(value)

    return ""


def load_svg_records(
    sources: list[dict[str, Any]],
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load SVG strings from one or more HF dataset sources.

    Each source item supports:
    - name (required): HF dataset id
    - split (optional): dataset split, default "train"
    - subset (optional): config/subset name
    - text_field (optional): source field containing SVG text
    - filename_field (optional): source field containing original file id/name
    - max_samples (optional): max number of rows to keep from this source
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    all_records: list[dict[str, Any]] = []

    for source_idx, source in enumerate(sources):
        name = source["name"]
        split = source.get("split", "train")
        subset = source.get("subset")
        text_field = source.get("text_field")
        filename_field = source.get("filename_field")
        max_samples = source.get("max_samples")

        dataset = load_dataset(
            path=name,
            name=subset,
            split=split,
            cache_dir=cache_dir,
        )

        count = 0
        for row_idx, example in enumerate(dataset):
            try:
                svg_text = _pick_svg_field(example, text_field)
            except KeyError:
                continue

            all_records.append(
                {
                    "id": f"s{source_idx}-r{row_idx}",
                    "source_dataset": name,
                    "source_split": split,
                    "source_subset": subset or "",
                    "source_filename": _pick_filename_field(example, filename_field),
                    "svg_raw": svg_text,
                }
            )
            count += 1

            if max_samples is not None and count >= int(max_samples):
                break

    return all_records
