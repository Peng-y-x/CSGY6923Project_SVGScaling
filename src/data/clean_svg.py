from __future__ import annotations

import re
import xml.etree.ElementTree as ET


COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
XML_DECL_RE = re.compile(r"<\?xml.*?\?>", re.IGNORECASE | re.DOTALL)
DOCTYPE_RE = re.compile(r"<!DOCTYPE.*?>", re.IGNORECASE | re.DOTALL)
METADATA_TAG_RE = re.compile(
    r"<(?:metadata|desc|title)\b[^>]*>.*?</(?:metadata|desc|title)>",
    re.IGNORECASE | re.DOTALL,
)
WHITESPACE_BETWEEN_TAGS_RE = re.compile(r">\s+<")
MULTISPACE_RE = re.compile(r"\s+")

# Match standalone numeric values, avoiding replacements in ids / hex colors.
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_#-])[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?"
)


def _round_number_token(token: str, precision: int) -> str:
    if "." not in token and "e" not in token.lower():
        return token

    try:
        rounded = round(float(token), precision)
    except ValueError:
        return token

    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.{precision}f}".rstrip("0").rstrip(".")


def _canonicalize_attributes(svg_text: str) -> str:
    root = ET.fromstring(svg_text)

    for element in root.iter():
        if element.attrib:
            sorted_items = sorted(element.attrib.items(), key=lambda x: x[0])
            element.attrib.clear()
            element.attrib.update(sorted_items)

    return ET.tostring(root, encoding="unicode")


def clean_svg_text(
    svg_text: str,
    *,
    strip_comments: bool = True,
    strip_metadata: bool = True,
    normalize_precision: bool = True,
    precision: int = 1,
    canonicalize_attributes: bool = False,
) -> str:
    text = svg_text.strip()

    if strip_comments:
        text = COMMENT_RE.sub("", text)

    text = XML_DECL_RE.sub("", text)
    text = DOCTYPE_RE.sub("", text)

    if strip_metadata:
        text = METADATA_TAG_RE.sub("", text)

    # Normalize whitespace while preserving literal content where possible.
    text = WHITESPACE_BETWEEN_TAGS_RE.sub("><", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    if normalize_precision:
        text = NUMBER_RE.sub(lambda m: _round_number_token(m.group(0), precision), text)

    if canonicalize_attributes:
        text = _canonicalize_attributes(text)

    return text
