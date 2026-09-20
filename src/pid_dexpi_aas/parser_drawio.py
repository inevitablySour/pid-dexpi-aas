"""Parse decoded mxGraph (draw.io) XML into plain shape/edge records.

Every mxCell with vertex="1" is a shape (equipment, valve, instrument bubble, or a
floating text label with no `shape=` style token). Every mxCell with edge="1" is a
piping/signal connector between two shape ids.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class Shape:
    id: str
    shape_token: str | None  # e.g. "mxgraph.pid.vessels.reactor", or None for plain rects/text
    text: str  # HTML-stripped label/value text
    x: float
    y: float
    width: float
    height: float
    style: str

    @property
    def is_labeled_only(self) -> bool:
        """True for plain text/rect cells that carry a tag rather than an equipment symbol."""
        return self.shape_token is None


@dataclass
class Edge:
    id: str
    source: str | None
    target: str | None
    style: str
    # Raw (x, y) fallback endpoints from <mxPoint as="sourcePoint"/targetPoint">,
    # populated only when the corresponding source/target cell-id attribute is absent.
    # drawio stores an edge endpoint this way whenever it isn't snapped exactly onto a
    # shape (confirmed: roughly half of all edges across every team file lack a
    # source/target attribute and rely on this instead - not a rare edge case).
    source_point: tuple[float, float] | None = None
    target_point: tuple[float, float] | None = None


@dataclass
class ParsedDiagram:
    source_file: str
    shapes: list[Shape] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


_TAG_RE = re.compile(r"<[^>]+>")
_SHAPE_TOKEN_RE = re.compile(r"(?:^|;)shape=([^;]+)")


def _strip_html(value: str) -> str:
    """mxCell value attrs are HTML-escaped fragments like '&lt;span&gt;BPR&lt;/span&gt;'."""
    unescaped = html.unescape(value or "")
    no_tags = _TAG_RE.sub(" ", unescaped)
    return re.sub(r"\s+", " ", no_tags).strip()


def _extract_shape_token(style: str) -> str | None:
    match = _SHAPE_TOKEN_RE.search(style or "")
    if not match:
        return None
    token = match.group(1)
    # "text;html=1;..." styles use shape=text or no shape= at all; treat "text" as no shape.
    return None if token in ("text",) else token


def parse_drawio_xml(xml_text: str, source_file: str) -> ParsedDiagram:
    root = ET.fromstring(xml_text)
    diagram = ParsedDiagram(source_file=source_file)

    for cell in root.iter("mxCell"):
        cell_id = cell.get("id")
        if cell_id in (None, "0", "1"):
            continue  # the two mxGraph root cells carry no shape data

        style = cell.get("style", "")

        if cell.get("edge") == "1":
            source_point = target_point = None
            if cell.get("source") is None or cell.get("target") is None:
                geometry = cell.find("mxGeometry")
                if geometry is not None:
                    source_mxpoint = geometry.find("mxPoint[@as='sourcePoint']")
                    target_mxpoint = geometry.find("mxPoint[@as='targetPoint']")
                    if source_mxpoint is not None:
                        source_point = (
                            float(source_mxpoint.get("x", 0)),
                            float(source_mxpoint.get("y", 0)),
                        )
                    if target_mxpoint is not None:
                        target_point = (
                            float(target_mxpoint.get("x", 0)),
                            float(target_mxpoint.get("y", 0)),
                        )

            diagram.edges.append(
                Edge(
                    id=cell_id,
                    source=cell.get("source"),
                    target=cell.get("target"),
                    style=style,
                    source_point=source_point,
                    target_point=target_point,
                )
            )
            continue

        if cell.get("vertex") == "1":
            geometry = cell.find("mxGeometry")
            x = float(geometry.get("x", 0)) if geometry is not None else 0.0
            y = float(geometry.get("y", 0)) if geometry is not None else 0.0
            width = float(geometry.get("width", 0)) if geometry is not None else 0.0
            height = float(geometry.get("height", 0)) if geometry is not None else 0.0

            diagram.shapes.append(
                Shape(
                    id=cell_id,
                    shape_token=_extract_shape_token(style),
                    text=_strip_html(cell.get("value", "")),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    style=style,
                )
            )

    return diagram


def _distance_to_shape(shape: Shape, x: float, y: float) -> float:
    """Distance from (x, y) to the nearest point on shape's bounding box - 0 if the
    point is inside/on the box, matching how a drawio edge endpoint typically sits
    exactly on a symbol's edge rather than dead-center."""
    dx = max(shape.x - x, 0.0, x - (shape.x + shape.width))
    dy = max(shape.y - y, 0.0, y - (shape.y + shape.height))
    return (dx**2 + dy**2) ** 0.5


# Endpoints drawn "loosely" (not snapped exactly to a shape's connection points) can
# land a real distance away from the shape they visually touch - confirmed empirically
# on the gramflow reactor file, where a strict bounding-box-containment check (even with
# a small margin) missed a point that was genuinely still touching its target shape's
# edge from just outside it. A generous nearest-shape search, capped at this distance
# so an edge floating in genuinely empty space doesn't get wrongly attached to whatever
# happens to be closest, is more robust than tuning a fixed containment margin.
_MAX_RESOLUTION_DISTANCE = 40.0


def resolve_floating_edge_endpoints(diagram: ParsedDiagram) -> None:
    """Fill in Edge.source/target for edges that only have a raw mxPoint coordinate
    (see Edge.source_point/target_point), by finding the nearest shape to that point.
    Mutates diagram.edges in place. Real, non-label (equipment/valve/inst) shapes are
    checked first so a floating point lands on the actual symbol it's drawn touching,
    not a nearby text label that happens to sit closer.
    """
    real_shapes = [s for s in diagram.shapes if not s.is_labeled_only]
    label_shapes = [s for s in diagram.shapes if s.is_labeled_only]

    def resolve(point: tuple[float, float]) -> str | None:
        x, y = point
        for candidates in (real_shapes, label_shapes):
            if not candidates:
                continue
            nearest = min(candidates, key=lambda s: _distance_to_shape(s, x, y))
            if _distance_to_shape(nearest, x, y) <= _MAX_RESOLUTION_DISTANCE:
                return nearest.id
        return None

    for edge in diagram.edges:
        if edge.source is None and edge.source_point is not None:
            edge.source = resolve(edge.source_point)
        if edge.target is None and edge.target_point is not None:
            edge.target = resolve(edge.target_point)


def parse_drawio_xml_resolved(xml_text: str, source_file: str) -> ParsedDiagram:
    """parse_drawio_xml + resolve_floating_edge_endpoints in one call - the entry point
    build_model.py/run_pipeline.py/app.py should use, so floating (non-snapped) edges
    aren't silently dropped."""
    diagram = parse_drawio_xml(xml_text, source_file=source_file)
    resolve_floating_edge_endpoints(diagram)
    return diagram


def parse_drawio_file(path: str | Path) -> ParsedDiagram:
    path = Path(path)
    return parse_drawio_xml_resolved(path.read_text(encoding="utf-8"), source_file=path.name)
