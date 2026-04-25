from __future__ import annotations

import re
from io import BytesIO
from typing import Any
import xml.etree.ElementTree as ET

try:
    from lxml import etree as LET  # type: ignore
except Exception:  # pragma: no cover
    LET = None


METADATA_TAGS = {"metadata", "desc", "title"}
NUMERIC_ATTRS = {
    "baseline-shift",
    "cx",
    "cy",
    "dx",
    "dy",
    "fill-opacity",
    "font-size",
    "height",
    "letter-spacing",
    "markerheight",
    "markerwidth",
    "offset",
    "opacity",
    "pathlength",
    "r",
    "rx",
    "ry",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-miterlimit",
    "stroke-opacity",
    "stroke-width",
    "viewbox",
    "width",
    "word-spacing",
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}
NUMERIC_STYLE_PROPS = NUMERIC_ATTRS | {
    "clip-path",
    "transform",
    "transform-origin",
}
NUMERIC_LIST_ATTRS = {
    "d",
    "points",
    "polygon",
    "polyline",
    "rotate",
    "style",
    "transform",
    "values",
}

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_#-])[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?"
)
PATH_NUMBER_RE = re.compile(
    r"(?:(?<=[AaCcHhLlMmQqSsTtVvZz])|(?<=[0-9.])(?=[-+])|(?<![A-Za-z0-9_#-]))"
    r"[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?"
)
WHITESPACE_BETWEEN_TAGS_RE = re.compile(r">\s+<")
MULTISPACE_RE = re.compile(r"\s+")


def _local_name(name: str) -> str:
    if "}" in name:
        name = name.rsplit("}", 1)[1]
    if ":" in name:
        name = name.rsplit(":", 1)[1]
    return name.lower()


def _strip_xml_decl_and_doctype(text: str) -> str:
    text = text.strip()
    if text.startswith("<?xml"):
        end = text.find("?>")
        if end != -1:
            text = text[end + 2 :].lstrip()

    lower = text.lower()
    start = lower.find("<!doctype")
    if start == -1:
        return text

    quote: str | None = None
    bracket_depth = 0
    i = start + len("<!doctype")
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]" and bracket_depth:
            bracket_depth -= 1
        elif ch == ">" and bracket_depth == 0:
            return (text[:start] + text[i + 1 :]).strip()
        i += 1
    return text[:start].strip()


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


def _normalize_numbers(value: str, precision: int) -> str:
    return NUMBER_RE.sub(lambda m: _round_number_token(m.group(0), precision), value)


def _normalize_path_data(value: str, precision: int) -> str:
    return PATH_NUMBER_RE.sub(lambda m: _round_number_token(m.group(0), precision), value)


def _normalize_style(value: str, precision: int) -> str:
    parts: list[str] = []
    for part in value.split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            parts.append(part.strip())
            continue
        prop, raw_val = part.split(":", 1)
        prop_name = prop.strip().lower()
        prop_val = raw_val.strip()
        if prop_name in NUMERIC_STYLE_PROPS and "url(" not in prop_val.lower():
            prop_val = _normalize_numbers(prop_val, precision)
        parts.append(f"{prop.strip()}:{prop_val}")
    return ";".join(parts)


def _normalize_attr_value(attr_name: str, value: str, precision: int) -> str:
    local = _local_name(attr_name)
    if local == "d":
        return _normalize_path_data(value, precision)
    if local == "style":
        return _normalize_style(value, precision)
    if local in NUMERIC_ATTRS or local in NUMERIC_LIST_ATTRS:
        return _normalize_numbers(value, precision)
    return value


def _serialize_lxml(root: Any, canonicalize_attributes: bool) -> str:
    if canonicalize_attributes:
        for element in root.iter():
            if not isinstance(element.tag, str) or not element.attrib:
                continue
            sorted_items = sorted(element.attrib.items(), key=lambda x: x[0])
            element.attrib.clear()
            element.attrib.update(sorted_items)

    return LET.tostring(  # type: ignore[union-attr]
        root,
        encoding="unicode",
        method="xml",
        xml_declaration=False,
        with_tail=False,
    )


def _clean_with_lxml(
    text: str,
    *,
    strip_comments: bool,
    strip_metadata: bool,
    normalize_precision: bool,
    precision: int,
    canonicalize_attributes: bool,
) -> str:
    parser = LET.XMLParser(  # type: ignore[union-attr]
        resolve_entities=False,
        no_network=True,
        remove_comments=strip_comments,
        remove_blank_text=True,
    )
    root = LET.parse(BytesIO(text.encode("utf-8")), parser).getroot()  # type: ignore[union-attr]

    if strip_metadata:
        for element in list(root.iter()):
            if isinstance(element.tag, str) and _local_name(element.tag) in METADATA_TAGS:
                parent = element.getparent()
                if parent is not None:
                    parent.remove(element)

    if normalize_precision:
        for element in root.iter():
            if not isinstance(element.tag, str):
                continue
            for attr_name, value in list(element.attrib.items()):
                element.attrib[attr_name] = _normalize_attr_value(
                    attr_name,
                    value,
                    precision,
                )

    return _serialize_lxml(root, canonicalize_attributes)


def _clean_with_elementtree(
    text: str,
    *,
    strip_comments: bool,
    strip_metadata: bool,
    normalize_precision: bool,
    precision: int,
    canonicalize_attributes: bool,
) -> str:
    if strip_comments:
        text = COMMENT_RE.sub("", text)
    root = ET.fromstring(text)

    if strip_metadata:
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for element in list(root.iter()):
            if _local_name(element.tag) in METADATA_TAGS:
                parent = parent_map.get(element)
                if parent is not None:
                    parent.remove(element)

    if normalize_precision:
        for element in root.iter():
            for attr_name, value in list(element.attrib.items()):
                element.attrib[attr_name] = _normalize_attr_value(
                    attr_name,
                    value,
                    precision,
                )

    if canonicalize_attributes:
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
    text = _strip_xml_decl_and_doctype(svg_text)
    if not text:
        return ""

    if LET is not None:
        cleaned = _clean_with_lxml(
            text,
            strip_comments=strip_comments,
            strip_metadata=strip_metadata,
            normalize_precision=normalize_precision,
            precision=precision,
            canonicalize_attributes=canonicalize_attributes,
        )
    else:
        cleaned = _clean_with_elementtree(
            text,
            strip_comments=strip_comments,
            strip_metadata=strip_metadata,
            normalize_precision=normalize_precision,
            precision=precision,
            canonicalize_attributes=canonicalize_attributes,
        )

    cleaned = WHITESPACE_BETWEEN_TAGS_RE.sub("><", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()
    return cleaned
