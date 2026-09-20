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
    # exitX/exitY (source side) and entryX/entryY (target side) from the style string:
    # normalized (0-1) fractions of a shape's bounding box where the edge attaches.
    # ONLY meaningful when paired with source=/target= per the mxGraph convention - when
    # present WITHOUT the corresponding attribute, confirmed (by direct measurement
    # against real shape geometry) to be a roughly 50/50 mix of genuinely accurate
    # leftover hints and pure stale noise from a since-detached connection, so these are
    # used as a secondary, tighter-tolerance resolution signal, not trusted blindly.
    exit_fraction: tuple[float, float] | None = None
    entry_fraction: tuple[float, float] | None = None
    # How source/target actually got resolved - surfaced in reports so an inferred
    # connection is never presented with the same confidence as one the file states
    # explicitly. See mapping.py/build_model.py's BuildRecord for how this is reported.
    source_resolution: str = "explicit"  # "explicit" | "exit_fraction" | "nearest_shape" | "unresolved"
    target_resolution: str = "explicit"
    # Visual style read straight from the style string - dashed=1 for control/signal
    # lines drawn dashed in the original, endArrow=classic/startArrow=classic for a
    # drawn arrowhead. Previously never parsed at all, so every rendered connector came
    # out solid with no arrowhead regardless of what the source file actually specified.
    is_dashed: bool = False
    has_end_arrow: bool = False
    has_start_arrow: bool = False
    # True when this edge is very likely decorative linework belonging to a single
    # equipment symbol (e.g. a safety valve's hand-drawn spring/relief indicator), not a
    # real piping connection - see classify_decorative_edges(). Confirmed by direct
    # inspection: several team files draw this kind of symbol detail as multiple tiny,
    # criss-crossing mxCell edges clustered entirely around ONE valve, rather than as
    # part of the valve's own shape style - these were previously misclassified as
    # "self-loop" (both ends resolve to the same shape) alongside GENUINELY corrupt
    # short segments, conflating two different things under one misleading label.
    is_decorative: bool = False


@dataclass
class ParsedDiagram:
    source_file: str
    shapes: list[Shape] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


_TAG_RE = re.compile(r"<[^>]+>")
_SHAPE_TOKEN_RE = re.compile(r"(?:^|;)shape=([^;]+)")
_EXIT_RE = re.compile(r"(?:^|;)exitX=([\d.]+);exitY=([\d.]+)")
_ENTRY_RE = re.compile(r"(?:^|;)entryX=([\d.]+);entryY=([\d.]+)")
_DASHED_RE = re.compile(r"(?:^|;)dashed=1(?:;|$)")
_END_ARROW_RE = re.compile(r"(?:^|;)endArrow=([a-zA-Z]+)")
_START_ARROW_RE = re.compile(r"(?:^|;)startArrow=([a-zA-Z]+)")


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
            exit_fraction = entry_fraction = None
            has_source = cell.get("source") is not None
            has_target = cell.get("target") is not None

            if not has_source or not has_target:
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

                if not has_source:
                    exit_match = _EXIT_RE.search(style)
                    if exit_match:
                        exit_fraction = (float(exit_match.group(1)), float(exit_match.group(2)))
                if not has_target:
                    entry_match = _ENTRY_RE.search(style)
                    if entry_match:
                        entry_fraction = (float(entry_match.group(1)), float(entry_match.group(2)))

            end_arrow_match = _END_ARROW_RE.search(style)
            start_arrow_match = _START_ARROW_RE.search(style)

            diagram.edges.append(
                Edge(
                    id=cell_id,
                    source=cell.get("source"),
                    target=cell.get("target"),
                    style=style,
                    source_point=source_point,
                    target_point=target_point,
                    exit_fraction=exit_fraction,
                    entry_fraction=entry_fraction,
                    source_resolution="explicit" if has_source else "unresolved",
                    target_resolution="explicit" if has_target else "unresolved",
                    is_dashed=_DASHED_RE.search(style) is not None,
                    has_end_arrow=end_arrow_match is not None and end_arrow_match.group(1) != "none",
                    has_start_arrow=start_arrow_match is not None
                    and start_arrow_match.group(1) != "none",
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

# exitX/exitY (or entryX/entryY) without a paired source=/target= attribute is a real,
# tighter geometric signal - confirmed by direct measurement on the gramflow reactor
# file: reverse-computing "shape.x + shape.width*exitX, shape.y + shape.height*exitY"
# for every candidate shape and comparing against the edge's own sourcePoint/targetPoint
# found near-exact matches (0.4-27 units) for about half of these edges, and no good
# match at all for the other half (confirming this is a mix of genuine leftover
# attachment hints and pure stale noise from a since-detached connection - NOT reliably
# one or the other). This tight tolerance is what separates the two: a real hint
# reproduces the exact point almost exactly, stale noise does not land anywhere close.
_MAX_FRACTION_MATCH_DISTANCE = 10.0


def _predict_point_from_fraction(shape: Shape, fraction: tuple[float, float]) -> tuple[float, float]:
    fx, fy = fraction
    return (shape.x + shape.width * fx, shape.y + shape.height * fy)


# A cluster of at least this many tiny floating edges hugging one shape's boundary is
# decorative symbol linework, not piping - confirmed by direct inspection on the
# gramflow reactor file: three safety-relief valves (valveType=angle) each have 6-7 such
# edges with one endpoint exactly on the valve's boundary (distance 0) and the other a
# short distance (8-33 units) beyond it, all in the same direction - the visual signature
# of a hand-drawn spring/relief symbol, not a piping run (a real connection's far
# endpoint lands near a DIFFERENT, distant shape, e.g. PSV201's real vent line to
# "safe location" is 520+ units away, not 8-33). A single stray tiny edge could still be
# a genuine (if oddly short) connection, so this only fires on a cluster, not any one edge.
_DECORATIVE_CLUSTER_MIN_SIZE = 3
_DECORATIVE_MAX_DISTANCE = 40.0


def classify_decorative_edges(diagram: ParsedDiagram) -> None:
    """Mark Edge.is_decorative=True for edges whose both endpoints are near the same
    single shape AND that shape has a cluster of _DECORATIVE_CLUSTER_MIN_SIZE+ such
    edges - almost certainly hand-drawn symbol detail (e.g. a safety valve's spring
    indicator), not real piping. Must run BEFORE resolve_floating_edge_endpoints, since
    these edges should never be resolved into a (wrong) "self-loop" piping connection in
    the first place - they're not unresolved, they're identified as not-piping-at-all.
    """
    real_shapes = [s for s in diagram.shapes if not s.is_labeled_only]
    # Only edges with NEITHER end already resolved to a real cell-id are candidates - an
    # edge with an explicit source= or target= attribute is a real, file-stated
    # connection regardless of where its other floating end happens to land, never
    # decorative linework. Confirmed as a real bug during testing: two edges with a
    # genuine explicit target= were wrongly caught here before this check, because their
    # source_point (the floating end) happened to land near the same cluster.
    candidates = [
        e
        for e in diagram.edges
        if e.source_point is not None
        and e.target_point is not None
        and e.source is None
        and e.target is None
    ]

    # For each candidate edge, find its single nearest shape if BOTH endpoints are near it.
    edge_to_shape: dict[str, str] = {}
    for edge in candidates:
        sx, sy = edge.source_point
        tx, ty = edge.target_point
        for shape in real_shapes:
            source_distance = _distance_to_shape(shape, sx, sy)
            target_distance = _distance_to_shape(shape, tx, ty)
            if source_distance <= _DECORATIVE_MAX_DISTANCE and target_distance <= _DECORATIVE_MAX_DISTANCE:
                edge_to_shape[edge.id] = shape.id
                break

    cluster_sizes: dict[str, int] = {}
    for shape_id in edge_to_shape.values():
        cluster_sizes[shape_id] = cluster_sizes.get(shape_id, 0) + 1

    for edge in candidates:
        shape_id = edge_to_shape.get(edge.id)
        if shape_id is not None and cluster_sizes[shape_id] >= _DECORATIVE_CLUSTER_MIN_SIZE:
            edge.is_decorative = True


def resolve_floating_edge_endpoints(diagram: ParsedDiagram) -> None:
    """Fill in Edge.source/target for edges that only have a raw mxPoint coordinate
    (see Edge.source_point/target_point). Mutates diagram.edges in place, and records
    how each endpoint was actually resolved in source_resolution/target_resolution so
    downstream reporting never presents an inferred connection with the same confidence
    as one the file states explicitly (see Edge's docstring on *_resolution).

    Two tiers, tried in order:
    1. exit_fraction/entry_fraction match (see _MAX_FRACTION_MATCH_DISTANCE) - tighter,
       more trustworthy when it hits, but only available on ~1/3 of floating edges and
       only actually matches on about half of those.
    2. Nearest-shape-by-distance fallback (_MAX_RESOLUTION_DISTANCE) - the original,
       looser heuristic, for everything the fraction match didn't resolve.
    """
    real_shapes = [s for s in diagram.shapes if not s.is_labeled_only]
    label_shapes = [s for s in diagram.shapes if s.is_labeled_only]
    all_shapes = real_shapes + label_shapes

    def resolve_by_fraction(point: tuple[float, float], fraction: tuple[float, float]) -> str | None:
        best_shape_id = None
        best_distance = None
        for shape in all_shapes:
            if shape.width <= 0 or shape.height <= 0:
                continue
            predicted = _predict_point_from_fraction(shape, fraction)
            distance = ((predicted[0] - point[0]) ** 2 + (predicted[1] - point[1]) ** 2) ** 0.5
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_shape_id = shape.id
        if best_distance is not None and best_distance <= _MAX_FRACTION_MATCH_DISTANCE:
            return best_shape_id
        return None

    def resolve_by_nearest_shape(point: tuple[float, float]) -> str | None:
        x, y = point
        for candidates in (real_shapes, label_shapes):
            if not candidates:
                continue
            nearest = min(candidates, key=lambda s: _distance_to_shape(s, x, y))
            if _distance_to_shape(nearest, x, y) <= _MAX_RESOLUTION_DISTANCE:
                return nearest.id
        return None

    for edge in diagram.edges:
        if edge.is_decorative:
            # Classified by classify_decorative_edges as symbol linework, not piping -
            # deliberately never resolved into a (fake) connection, however tempting the
            # proximity match. Left unresolved and reported distinctly downstream.
            continue

        if edge.source is None and edge.source_point is not None:
            if edge.exit_fraction is not None:
                resolved = resolve_by_fraction(edge.source_point, edge.exit_fraction)
                if resolved is not None:
                    edge.source = resolved
                    edge.source_resolution = "exit_fraction"
            if edge.source is None:
                resolved = resolve_by_nearest_shape(edge.source_point)
                if resolved is not None:
                    edge.source = resolved
                    edge.source_resolution = "nearest_shape"

        if edge.target is None and edge.target_point is not None:
            if edge.entry_fraction is not None:
                resolved = resolve_by_fraction(edge.target_point, edge.entry_fraction)
                if resolved is not None:
                    edge.target = resolved
                    edge.target_resolution = "exit_fraction"
            if edge.target is None:
                resolved = resolve_by_nearest_shape(edge.target_point)
                if resolved is not None:
                    edge.target = resolved
                    edge.target_resolution = "nearest_shape"


def parse_drawio_xml_resolved(xml_text: str, source_file: str) -> ParsedDiagram:
    """parse_drawio_xml + classify_decorative_edges + resolve_floating_edge_endpoints in
    one call - the entry point build_model.py/run_pipeline.py/app.py should use, so
    floating (non-snapped) edges aren't silently dropped, and decorative symbol linework
    isn't wrongly resolved into a fake piping connection."""
    diagram = parse_drawio_xml(xml_text, source_file=source_file)
    classify_decorative_edges(diagram)
    resolve_floating_edge_endpoints(diagram)
    return diagram


def parse_drawio_file(path: str | Path) -> ParsedDiagram:
    path = Path(path)
    return parse_drawio_xml_resolved(path.read_text(encoding="utf-8"), source_file=path.name)
