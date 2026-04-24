from __future__ import annotations

import re
import xml.etree.ElementTree as ET

try:
    from lxml import etree as LET  # type: ignore
except Exception:  # pragma: no cover
    LET = None


TOKEN_EST_RE = re.compile(r"<[^>]+>|[^<\s]+")


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def validate_xml(svg_text: str) -> tuple[bool, str | None]:
    """Return (is_valid, error_message)."""
    if LET is not None:
        try:
            LET.fromstring(svg_text.encode("utf-8"))
            return True, None
        except Exception as exc:
            return False, str(exc)

    try:
        ET.fromstring(svg_text)
        return True, None
    except Exception as exc:
        return False, str(exc)


def has_svg_root(svg_text: str) -> bool:
    try:
        root = ET.fromstring(svg_text)
    except Exception:
        return False
    return _local_tag(root.tag).lower() == "svg"


def estimate_token_count(svg_text: str) -> int:
    """Tokenizer-free rough token count for early filtering."""
    return len(TOKEN_EST_RE.findall(svg_text))


def validate_render(svg_text: str) -> tuple[bool, str | None]:
    """Try rendering SVG to PNG bytes using CairoSVG."""
    try:
        import cairosvg
    except ImportError as exc:  # pragma: no cover
        return False, f"cairosvg import failed: {exc}"

    try:
        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
        return True, None
    except Exception as exc:
        return False, str(exc)
