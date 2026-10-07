"""Local patches fixing two confirmed pyDEXPI 1.2.0 SVG rendering bugs.

1. Dashed/dotted lines never render as dashed: SvgRenderer.convert_stroke
   (loaders/svg_loader.py) checks `stroke.dashStyle.value == "Dashed"` / `"Dotted"`, but
   the actual DashStyle enum members are `Dash`/`Dot` (confirmed by direct inspection:
   `list(DashStyle)` gives Solid/Dot/Dash/LongDash/... - there is no member whose value
   is literally "Dashed" or "Dotted"). So this condition can never be true for any real
   DashStyle value, and every stroke renders solid regardless of what's set.

2. Filled polygons never render filled: SvgRenderer.convert_polygon computes `fill` via
   `"none" if ... polygon.fillStyle.value == "Solid" else "none"` - both branches of
   that ternary return "none", so a Solid-filled polygon (needed here for arrowhead
   triangles) always renders as an unfilled outline instead.

Both confirmed by reading the source directly, not just observed behavior.

IMPLEMENTATION NOTE: subclassing SvgRenderer and swapping it into DrawDiagram is NOT
enough - confirmed by testing (stroke-dasharray never appeared in output despite the
subclass "working" in isolation). DrawDiagram._create_content_group hands off to nested
DrawRepresentationGroup/DrawRepresentationTypeGroup/DrawShapeUsage drawers, and EACH of
those independently constructs its own plain `SvgRenderer(padding, pretty)` inside
DrawSVG.__init__ (loaders/svg_loader.py line ~615) - there is no dependency-injection
point, so a subclass swapped in on the top-level DrawDiagram instance never reaches the
renderers actually used for nested content (which is where every real connector line and
arrowhead polygon lives). Patching every one of those drawer classes individually would
mean re-implementing most of DrawDiagram's internals. Instead, this monkeypatches the
two broken methods directly on the SvgRenderer class, so every renderer instance -
regardless of which drawer class constructs it - gets the fix. Applied once at import
time; idempotent (safe to import this module more than once).
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

from pydexpi.dexpi_classes.graphics import Polygon, Stroke
from pydexpi.loaders.svg_loader import SvgRenderer

_original_convert_stroke = SvgRenderer.convert_stroke
_original_convert_polygon = SvgRenderer.convert_polygon


def _patched_convert_stroke(self, stroke: Stroke) -> dict:
    attrs = _original_convert_stroke(self, stroke)
    if not stroke or not hasattr(stroke, "dashStyle"):
        return attrs

    stroke_width = stroke.width * self.line_width_multiplier
    dash_value = stroke.dashStyle.value

    if dash_value == "Dash":
        dash_length = max(4, stroke_width * 4)
        attrs["stroke-dasharray"] = f"{dash_length},{dash_length}"
    elif dash_value == "Dot":
        dot_length = max(1, stroke_width)
        gap_length = max(2, stroke_width * 2)
        attrs["stroke-dasharray"] = f"{dot_length},{gap_length}"
    elif dash_value != "Solid":
        # LongDash/ShortDash/etc: approximate with a Dash-length pattern rather than
        # leaving these solid too - a rough approximation beats silently ignoring the
        # style entirely.
        dash_length = max(4, stroke_width * 4)
        attrs["stroke-dasharray"] = f"{dash_length},{dash_length}"

    return attrs


def _patched_convert_polygon(self, polygon: Polygon) -> ET.Element:
    element = _original_convert_polygon(self, polygon)
    if hasattr(polygon, "fillStyle") and polygon.fillStyle.value == "Solid":
        element.set("fill", self.convert_color(polygon.stroke.color))
    return element


if SvgRenderer.convert_stroke is _original_convert_stroke:
    SvgRenderer.convert_stroke = _patched_convert_stroke
    SvgRenderer.convert_polygon = _patched_convert_polygon
