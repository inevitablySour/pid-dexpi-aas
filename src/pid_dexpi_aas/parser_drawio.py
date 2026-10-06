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
    source_resolution: str = "explicit"  # "explicit" | "exit_fraction" | "nearest_shape" | "nearest_shape_by_margin" | "unresolved"
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
    # Set when this line's loose end sits on another line's drawn route (line_routes.py):
    # the id of that line. Such ends are junctions, never guessed onto a nearest shape.
    source_junction: str | None = None
    target_junction: str | None = None
    # Lines joined at junctions form one pipe network; its symbols are connected by
    # derived "net:" edges, and the member lines point here instead of being built alone.
    merged_into: str | None = None
    # Flow direction (flow_direction.infer_flow): +1 source -> target, -1 target -> source,
    # 0 unknown; basis "arrow" (drawn on this line) | "implied" (from neighbouring arrows).
    flow: int = 0
    flow_basis: str = ""
    # a closed frame drawn around a plant section (sections.py) - not a pipe
    is_section_frame: bool = False


@dataclass
class ParsedDiagram:
    source_file: str
    shapes: list[Shape] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    # shape id -> specification notes attached to it ("17 bar", "Volume = 5 L"), see
    # spec_labels.py; and notes that fitted no item (reported, never guessed)
    specs: dict = field(default_factory=dict)
    unassigned_specs: list = field(default_factory=list)
    # flow_direction.FlowSummary: how many connections got a direction, and how
    flow: object = None
    # {edge id: drawn route in draw.io coordinates} when the draw.io app was available
    # (line_routes.py) - the re-drawn "after" picture follows these instead of guessing
    routes: dict | None = None
    # the drawing's own lines, kept when a reviewed diagram is rebuilt from table rows
    # (review.to_dexpi), so the "after" picture can still draw each pipe as drawn
    drawn_edges: list | None = None
    # plant sections drawn as frames around groups of items (sections.py)
    sections: list = field(default_factory=list)


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
                    # no endArrow in the style = draw.io's default edge, which draws a classic arrow
                    has_end_arrow=end_arrow_match is None or end_arrow_match.group(1) != "none",
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
# edge from just outside it.
#
# A single fixed distance threshold is the wrong tool for this, not just imprecisely
# tuned: published research (Gadekar et al., "Grounded and Faithful P&ID Reasoning",
# arXiv:2609.05880 - see research/03-topology-extraction.md) shows a fixed 10px snap
# tolerance collapses connectivity edge F1 from 0.742 to 0.193 in an ablation against an
# adaptive approach - a ~4x degradation from exactly this class of heuristic. Their
# result and a US patent on flow-drawing repair (8,578,346) both point the same
# direction: use a MARGIN over the runner-up candidate, not a bare distance cutoff.
#
# Confirmed directly on our own data: a real V202 connection point sits 65 units away
# (beyond the old flat 40-unit cap) but the next-nearest shape is 130 units away - twice
# as far. That 2x gap is real signal a flat threshold discards entirely. Two tiers:
# - _MAX_RESOLUTION_DISTANCE: a "confidently close" cutoff, high enough alone.
# - _MARGIN_RESOLUTION_DISTANCE / _MARGIN_RATIO: a wider ceiling, but ONLY when the best
#   candidate beats the second-best by a healthy ratio - i.e. it is unambiguously the
#   right shape even though it's not close in absolute terms. Resolved this way is
#   reported as a distinct, lower-confidence tier ("nearest_shape_by_margin"), never
#   conflated with a confident short-distance match.
_MAX_RESOLUTION_DISTANCE = 40.0
_MARGIN_RESOLUTION_DISTANCE = 150.0
_MARGIN_RATIO = 1.5

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

    # Containers (a shape enclosing other real symbols - e.g. the reactor box around its
    # Mixing/Heating/Cooling compartments): a loose line end inside one belongs to the
    # container only if it sits on the container's BORDER. Counting "anywhere inside" as
    # distance 0 made every line end near a compartment wall resolve to the reactor.
    def _encloses(outer: Shape, inner: Shape) -> bool:
        cx, cy = inner.x + inner.width / 2, inner.y + inner.height / 2
        return (outer is not inner and outer.x <= cx <= outer.x + outer.width
                and outer.y <= cy <= outer.y + outer.height and outer.width * outer.height > inner.width * inner.height)

    # Only plain-shape containers (plain_shapes.py): a library symbol drawn from several
    # cells (pump P101 with its base trapezoid) must not count as one, or a line end on
    # the pump slides onto the decorative base.
    containers = {s.id for s in real_shapes if (s.shape_token or "").startswith("plain:")
                  and any(_encloses(s, o) for o in real_shapes)}

    def _resolution_distance(shape: Shape, x: float, y: float) -> float:
        d = _distance_to_shape(shape, x, y)
        if d == 0 and shape.id in containers:
            return min(x - shape.x, shape.x + shape.width - x, y - shape.y, shape.y + shape.height - y)
        return d

    def resolve_by_nearest_shape(point: tuple[float, float]) -> tuple[str, str] | tuple[None, None]:
        """Returns (shape_id, resolution_tier) or (None, None). Two tiers: a confident
        short-distance match, or a wider-but-unambiguous margin-based match (see
        _MARGIN_RESOLUTION_DISTANCE's docstring above for why margin matters more than
        absolute distance here)."""
        x, y = point
        for candidates in (real_shapes, label_shapes):
            if not candidates:
                continue
            ranked = sorted(candidates, key=lambda s: _resolution_distance(s, x, y))
            nearest = ranked[0]
            nearest_distance = _resolution_distance(nearest, x, y)

            if nearest_distance <= _MAX_RESOLUTION_DISTANCE:
                return nearest.id, "nearest_shape"

            if nearest_distance <= _MARGIN_RESOLUTION_DISTANCE and len(ranked) > 1:
                second_distance = _resolution_distance(ranked[1], x, y)
                if second_distance >= nearest_distance * _MARGIN_RATIO:
                    return nearest.id, "nearest_shape_by_margin"
        return None, None

    for edge in diagram.edges:
        if edge.is_decorative:
            # Classified by classify_decorative_edges as symbol linework, not piping -
            # deliberately never resolved into a (fake) connection, however tempting the
            # proximity match. Left unresolved and reported distinctly downstream.
            continue

        if edge.source is None and edge.source_point is not None and edge.source_junction is None:
            if edge.exit_fraction is not None:
                resolved = resolve_by_fraction(edge.source_point, edge.exit_fraction)
                if resolved is not None:
                    edge.source = resolved
                    edge.source_resolution = "exit_fraction"
            if edge.source is None:
                resolved_id, tier = resolve_by_nearest_shape(edge.source_point)
                if resolved_id is not None:
                    edge.source = resolved_id
                    edge.source_resolution = tier

        if edge.target is None and edge.target_point is not None and edge.target_junction is None:
            if edge.entry_fraction is not None:
                resolved = resolve_by_fraction(edge.target_point, edge.entry_fraction)
                if resolved is not None:
                    edge.target = resolved
                    edge.target_resolution = "exit_fraction"
            if edge.target is None:
                resolved_id, tier = resolve_by_nearest_shape(edge.target_point)
                if resolved_id is not None:
                    edge.target = resolved_id
                    edge.target_resolution = tier


def redirect_label_endpoints(diagram: ParsedDiagram) -> None:
    """A line end attached to a TEXT LABEL belongs to the symbol that label names.

    Measured on the team files: 15 line ends attach to a label lying INSIDE a symbol (the
    line from XV201 snaps onto the text "X101" inside the X101 column drawing, not the
    column), and others to a tag label next to its symbol. Redirect:
      1. label inside a symbol's box -> the smallest such symbol;
      2. label that is a real tag (GV204, PI 102 ...) associated with a symbol -> it.
    Anything else ("Fume hood [atm]", "Product") stays: a genuine off-page destination.
    """
    from pid_dexpi_aas.mapping import classify_tag
    from pid_dexpi_aas.tag_association import associate_tags

    shapes = {sh.id: sh for sh in diagram.shapes}
    symbols = [sh for sh in diagram.shapes if sh.shape_token]
    owner = {a.label_shape_id: a.target_shape_id for a in associate_tags(diagram) if a.target_shape_id}

    def target_for(label_id: str) -> str | None:
        lab = shapes.get(label_id)
        if lab is None or lab.shape_token:
            return None
        cx, cy = lab.x + lab.width / 2, lab.y + lab.height / 2
        holders = [sh for sh in symbols if sh.x <= cx <= sh.x + sh.width and sh.y <= cy <= sh.y + sh.height]
        if holders:
            return min(holders, key=lambda sh: sh.width * sh.height).id
        if lab.text.strip() and classify_tag(lab.text)[0] is not None:
            return owner.get(label_id)
        return None

    for e in diagram.edges:
        for attr in ("source", "target"):
            end = getattr(e, attr)
            new = target_for(end) if end else None
            if new is not None:
                setattr(e, attr, new)


def join_pipe_networks(diagram: ParsedDiagram, routes: dict) -> None:
    """Lines whose loose ends sit on other lines' routes form pipe networks. Every pair of
    symbols at a network's ends is connected (a tee joins all its legs); the member lines
    are marked merged_into and the connections added as derived "net:" edges."""
    from pid_dexpi_aas.line_routes import find_junctions

    joins = find_junctions(diagram.edges, routes)
    if not joins:
        return
    by_id = {e.id: e for e in diagram.edges}
    for (eid, end), other in joins.items():
        if end == 0:
            by_id[eid].source_junction = other
        else:
            by_id[eid].target_junction = other
    parent = {e.id: e.id for e in diagram.edges}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (eid, _), other in joins.items():
        parent[root(eid)] = root(other)
    networks: dict[str, list] = {}
    for e in diagram.edges:
        networks.setdefault(root(e.id), []).append(e)
    return_edges = []
    for k, members in enumerate(m for m in networks.values() if len(m) > 1):
        net_id = f"net{k}"
        ends: dict[str, str] = {}
        for e in members:
            e.merged_into = net_id
            for sid, res in ((e.source, e.source_resolution), (e.target, e.target_resolution)):
                if sid is not None:
                    ends.setdefault(sid, res)
        dashed = sum(e.is_dashed for e in members) > len(members) / 2
        ids = sorted(ends)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                guessed = any(ends[x] not in ("explicit", "adjacent") for x in (a, b))
                return_edges.append(Edge(
                    id=f"{net_id}:{a}:{b}", source=a, target=b, style="pipe network",
                    source_resolution="junction" if not guessed else "junction+guess",
                    target_resolution="junction" if not guessed else "junction+guess",
                    is_dashed=dashed,
                ))
    diagram.edges.extend(return_edges)


def parse_drawio_xml_resolved(xml_text: str, source_file: str, drawio_path: str | Path | None = None) -> ParsedDiagram:
    """parse_drawio_xml + classify_decorative_edges + resolve_floating_edge_endpoints in
    one call - the entry point build_model.py/run_pipeline.py/app.py should use, so
    floating (non-snapped) edges aren't silently dropped, and decorative symbol linework
    isn't wrongly resolved into a fake piping connection."""
    from pid_dexpi_aas.plain_shapes import mark_plain_symbols  # local: plain_shapes imports this module

    from pid_dexpi_aas.spec_labels import attach_specs

    diagram = parse_drawio_xml(xml_text, source_file=source_file)
    mark_plain_symbols(diagram)  # reactor/compartment boxes etc. become real symbols first
    # Spec notes ("17 bar") are data about an item, not tags or line ends: attach them,
    # then blank pure-note cells so tag association and line-end guessing ignore them.
    specs = attach_specs(diagram)
    diagram.specs, diagram.unassigned_specs = specs.by_shape, specs.unassigned
    for cell in diagram.shapes:
        if cell.id in specs.spec_cells:
            cell.text = ""
    classify_decorative_edges(diagram)
    routes = None
    if drawio_path is not None:
        from pid_dexpi_aas.line_routes import find_junctions, routes_for
        routes = routes_for(drawio_path, diagram.edges)
        if routes:
            # section frames first: closed lines around groups of items are not pipes
            from pid_dexpi_aas.sections import find_sections
            find_sections(diagram, routes)
            from pid_dexpi_aas.line_routes import find_dash_chains
            by_id = {e.id: e for e in diagram.edges}
            for chain in find_dash_chains(diagram.edges, routes):
                for eid in chain:  # a hand-drawn dashed line: signal, not decoration
                    by_id[eid].is_decorative, by_id[eid].is_dashed = False, True
            # Mark junction ends BEFORE guessing, so an end on a pipe is never guessed
            # onto the nearest symbol.
            for (eid, end), other in find_junctions(diagram.edges, routes).items():
                e = next(x for x in diagram.edges if x.id == eid)
                if end == 0:
                    e.source_junction = other
                else:
                    e.target_junction = other
    resolve_floating_edge_endpoints(diagram)
    redirect_label_endpoints(diagram)
    if routes:
        join_pipe_networks(diagram, routes)
    from pid_dexpi_aas.flow_direction import infer_flow
    diagram.flow = infer_flow(diagram)
    diagram.routes = routes or None
    return diagram


def parse_drawio_file(path: str | Path) -> ParsedDiagram:
    """A decoded mxGraph XML file, or a real .drawio file (decoded, with line routes when
    the draw.io app is available)."""
    path = Path(path)
    if path.suffix == ".drawio":
        from pid_dexpi_aas.decode_drawio import decode_drawio
        return parse_drawio_xml_resolved(decode_drawio(path)[0], source_file=path.name, drawio_path=path)
    return parse_drawio_xml_resolved(path.read_text(encoding="utf-8"), source_file=path.name)
