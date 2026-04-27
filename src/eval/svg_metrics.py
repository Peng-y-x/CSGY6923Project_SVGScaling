from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SvgValidity:
    xml_valid: bool
    structural_valid: bool
    root_is_svg: bool
    render_valid: bool
    error: str = ""


def check_xml_and_structure(svg: str) -> tuple[bool, bool, bool, str]:
    try:
        from lxml import etree

        root = etree.fromstring(svg.encode("utf-8"))
        tag = root.tag.split("}")[-1] if isinstance(root.tag, str) else ""
        root_is_svg = tag.lower() == "svg"
        structural = root_is_svg and len(root.attrib) > 0
        return True, structural, root_is_svg, ""
    except Exception as exc:
        return False, False, False, str(exc)


def evaluate_svg_string(svg: str, *, render_valid: bool) -> SvgValidity:
    xml_valid, structural_valid, root_is_svg, error = check_xml_and_structure(svg)
    return SvgValidity(
        xml_valid=xml_valid,
        structural_valid=structural_valid,
        root_is_svg=root_is_svg,
        render_valid=bool(render_valid),
        error=error,
    )
