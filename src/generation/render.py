from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw


def render_svg_to_png(svg: str, png_path: Path, *, output_width: int = 256, output_height: int = 256) -> bool:
    try:
        import cairosvg

        png_path.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(png_path),
            output_width=output_width,
            output_height=output_height,
        )
        return True
    except Exception:
        return False


def make_image_grid(image_paths: list[Path], output_path: Path, *, cell_size: int = 256, cols: int = 5) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = max(1, math.ceil(len(image_paths) / cols))
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size), "white")
    draw = ImageDraw.Draw(grid)
    for i, path in enumerate(image_paths):
        x = (i % cols) * cell_size
        y = (i // cols) * cell_size
        try:
            img = Image.open(path).convert("RGB").resize((cell_size, cell_size))
            grid.paste(img, (x, y))
        except Exception:
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], outline="red", width=3)
            draw.text((x + 8, y + 8), "render failed", fill="red")
    grid.save(output_path)
