"""Export a parsed drawio diagram as a PID2Graph-schema .graphml file, so the team's own
7 P&ID files can be folded into PID2Graph as additional real, labeled training/eval data.

Schema confirmed by direct inspection of a real sample (PID2Graph/Complete/PID2Graph
OPEN100/3.graphml, downloaded from Zenodo record 14803338) - not guessed from the paper's
prose description, which only said "bounding box and label" and omitted the actual
GraphML key ids/types:

  <key id="d0" for="node" attr.name="label" attr.type="string" />
  <key id="d1" for="node" attr.name="xmin" attr.type="double" />
  <key id="d2" for="node" attr.name="xmax" attr.type="double" />
  <key id="d3" for="node" attr.name="ymin" attr.type="double" />
  <key id="d4" for="node" attr.name="ymax" attr.type="double" />
  <key id="d9" for="edge" attr.name="edge_label" attr.type="string" />

Real files also declare a second, long-typed xmin/xmax/ymin/ymax key set (d5-d8), but
that set is only ever populated on structural "connector"/"crossing" nodes, never on
real symbol nodes - see the module docstring's key ids below for exactly which set this
exporter writes for which node kind.

Node label vocabulary observed in the real OPEN100 subset: arrow, background, connector,
crossing, general, inlet/outlet, instrumentation, pump, tank, valve. Of these, only
{valve, pump, tank, instrumentation, inlet/outlet, general, arrow} are real symbol
classes (the paper's Table 1 taxonomy); background/connector/crossing are structural
node types the graph topology needs (a real edge never runs symbol-to-symbol directly -
every real edge in the dataset runs through at least one connector node - but this
exporter's own topology comes from Edge.source/target resolved directly onto shape ids,
so it emits connector nodes only where genuinely useful: at each edge's midpoint, to
keep the direct symbol-to-symbol connections this pipeline already has instead of
guessing at intermediate waypoints that were never drawn as separate cells).

Edge label vocabulary: "solid" is the only value observed in the real OPEN100 subset,
but the paper's Table 1 also defines "non-solid" for dashed/control lines - mapped from
Edge.is_dashed, since the taxonomy is documented even though this particular real subset
happens not to exercise it.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

from pid_dexpi_aas.dexpi.mapping import Category, classify_tag
from pid_dexpi_aas.drawio.parser import ParsedDiagram, Shape, parse_drawio_file
from pid_dexpi_aas.review import OFF_PAGE, SKIP, category_for_class, from_drawio

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

TANK_CLASSES = {"Tank", "PressureVessel", "CustomVessel", "Vessel", "ProcessColumn"}
PUMP_CLASSES = {"Pump", "CustomPump"}


def pid2graph_class(dexpi_class: str, shape_token: str | None) -> str | None:
    """Review row (DEXPI class + draw.io shape token) -> PID2Graph 7-class label, or None
    for things that are not symbols (text labels, decorative shapes)."""
    token = shape_token or ""
    if token.startswith("plain:"):
        # plain-shape symbol (plain_shapes.py) - its PID2Graph label is in the token,
        # and it is a symbol even when the reviewer/mapping skipped its DEXPI class
        return token.rsplit(":", 1)[1]
    if "arrows2.arrow" in token:
        return "inlet/outlet"  # e.g. "Fume hood [atm]": an off-page destination
    if "opto_electronics.lamp" in token:
        return "general"
    if dexpi_class in (SKIP, OFF_PAGE) or not shape_token:
        return None
    if dexpi_class == "SignalOffPageConnector":
        return "inlet/outlet"
    category = category_for_class(dexpi_class)
    if category == Category.PIPING:
        return "valve"
    if category == Category.INSTRUMENTATION:
        return "instrumentation"
    if dexpi_class in TANK_CLASSES:
        return "tank"
    if dexpi_class in PUMP_CLASSES:
        return "pump"
    return "general"


def rotated_box(shape: Shape) -> tuple[float, float, float, float]:
    """draw.io stores geometry BEFORE rotation; a +-90 deg shape occupies the box with
    width and height swapped around the same centre (off-page arrows, vertical valves,
    heat exchangers). 180 deg leaves the box unchanged."""
    m = re.search(r"(?:^|;)rotation=(-?[\d.]+)", shape.style or "")
    angle = abs(float(m.group(1))) % 180 if m else 0.0
    if 45 < angle < 135:
        cx, cy = shape.x + shape.width / 2, shape.y + shape.height / 2
        return (cx - shape.height / 2, cy - shape.width / 2, cx + shape.height / 2, cy + shape.width / 2)
    return (shape.x, shape.y, shape.x + shape.width, shape.y + shape.height)


def labelled_symbols(diagram: ParsedDiagram, arrows: bool = False, ink=None):
    """The symbols and connections of a drawing in PID2Graph terms - the ONE labelling
    used both for these exports and for the ML training data (make_team_data.py).

    Returns (symbols, edges): symbols = [(node_id, label, (x1, y1, x2, y2), drawio_id)] in
    draw.io coordinates; edges = [(node_id_a, node_id_b, "solid" | "non-solid")], one per
    connected pair. Only real symbols are labelled - the original exporter labelled EVERY
    draw.io cell, text labels included, as "general" (56 of LightUp's 113 nodes), which
    would teach a model to box text as symbols.

    Caveat: connections come from the draw.io lines via the parser; about half of them had
    one loose end resolved to the nearest shape (not verified individually).

    arrows=True also labels every drawn arrowhead as an "arrow" symbol and splits the
    connection it sits on (A - arrow - B), as OPEN100 does (_arrow_labels). Off: the
    labelling of the Team / Team v3 training data. ink(box) -> fraction of dark pixels in a
    draw.io-coordinate box of the rendered drawing; with it the arrow boxes are fitted to
    the drawn heads and arrows hidden under a shape are left out.
    """
    review = from_drawio(diagram)
    rows = {r.id: r for r in review.symbols}
    symbols = []
    for shape in diagram.shapes:
        row = rows.get(shape.id)
        label = pid2graph_class(row.dexpi_class, shape.shape_token) if (row and shape.shape_token) else None
        if label is not None:
            symbols.append((_sanitize_id(shape.id), label, rotated_box(shape), shape.id))
    id_map = {orig: sid for sid, _, _, orig in symbols}
    edges = []  # (a, b, line, group): group = the draw.io line or pipe network it comes from
    for c in review.connections:
        if c.include and c.from_id in id_map and c.to_id in id_map and c.from_id != c.to_id:
            a, b = sorted((id_map[c.from_id], id_map[c.to_id]))
            edges.append((a, b, "non-solid" if c.line_type == "signal" else "solid", c.id.split(":")[0]))
    if arrows:
        _arrow_labels(diagram, symbols, edges, id_map, ink)
    out, seen = [], set()
    for a, b, line, _ in edges:
        key = tuple(sorted((a, b)))
        if a != b and key not in seen:  # parallel lines between one pair -> one connection
            seen.add(key)
            out.append((key[0], key[1], line))
    return symbols, out


ARROW_LENGTH = 10.0  # a draw.io "classic" arrowhead (size 6, stroke 1), drawing units - for finding it
# The box an arrow is LABELLED with, as OPEN100 labels them: its long side 0.35 of the sheet's
# median symbol, its short side 0.73 of that (measured on the 12 OPEN100 sheets). A box the
# size of the drawn head (~0.12 of a symbol) is ~5 px for the model: Team v4 used that and the
# model found 7 of LightUp's 40 arrows (and its connections suffered on OPEN100 too).
ARROW_BOX = (0.35, 0.73)
HEAD_SPAN = (-2, 8)  # where a head lies along its line, past the route end (measured on LightUp)


def _wide_at(ink, x: float, y: float, dx: float, dy: float) -> bool:
    """Is the drawing wider than a plain line at (x, y), across the direction (dx, dy)?"""
    return ink([(x - dy * o, y + dx * o) for o in (-3, -2, 2, 3)]) >= 0.75


def _arrow_labels(diagram: ParsedDiagram, symbols: list, edges: list, id_map: dict, ink=None) -> None:
    """Every arrowhead drawn at a line end becomes an "arrow" symbol: a box over the head.
    The line's connections are split at it - OPEN100 labels arrows as nodes on the pipe (symbol - arrow - symbol), not as decoration. The arrow sits between the
    symbol at its end of the line (or, at a tee, the line's other end) and the rest; on a
    pipe network every connection of that symbol through the network is split.
    symbols / edges are extended in place."""
    import statistics
    sizes = [max(b[2] - b[0], b[3] - b[1]) for _, label, b, _ in symbols if label != "arrow"]
    box_long = ARROW_BOX[0] * (statistics.median(sizes) if sizes else 85.0)
    for e in diagram.edges:
        route = (diagram.routes or {}).get(e.id)
        if e.is_decorative or not route or len(route) < 2:
            continue
        group = e.merged_into or e.id
        for end, drawn in ((0, e.has_start_arrow), (1, e.has_end_arrow)):
            if not drawn:
                continue
            pts = route if end == 1 else route[::-1]
            (tx, ty) = pts[-1]
            prev = next(((x, y) for x, y in reversed(pts[:-1]) if (x, y) != (tx, ty)), None)
            if prev is None:
                continue
            dx, dy = tx - prev[0], ty - prev[1]
            n = (dx * dx + dy * dy) ** 0.5
            dx, dy = dx / n, dy / n
            # draw.io stops the line where an end head starts: the head mostly lies just past
            # the route end; some (start arrows) lie over the line's last stretch instead -
            # with the rendered drawing (ink) the side with the head wins, and a head that
            # is nowhere to be seen (under a shape) is not labelled
            h = ARROW_LENGTH / 2
            lo, hi = HEAD_SPAN
            spots = [(tx + dx * (lo + hi) / 2, ty + dy * (lo + hi) / 2),
                     (tx - dx * h, ty - dy * h)]  # head over the last stretch
            if ink is not None:
                ahead = sum(_wide_at(ink, tx + dx * t, ty + dy * t, dx, dy) for t in range(lo, hi + 1))
                behind = sum(_wide_at(ink, tx - dx * t, ty - dy * t, dx, dy) for t in range(-lo, hi + 3))
                if max(ahead, behind) < 2:
                    continue
                if behind > ahead:
                    spots.reverse()
            cx, cy = spots[0]
            sid = f"{_sanitize_id(e.id)}_arrow{end}"
            along, across = box_long / 2, box_long * ARROW_BOX[1] / 2  # half sizes; long side along the pipe
            hw, hh = (along, across) if abs(dx) >= abs(dy) else (across, along)
            symbols.append((sid, "arrow", (cx - hw, cy - hh, cx + hw, cy + hh), f"{e.id}:arrow{end}"))
            here, other = (e.target, e.source) if end == 1 else (e.source, e.target)
            near = id_map.get(here) or id_map.get(other)
            if near is None:
                continue
            if id_map.get(here) is None and e.merged_into is None:
                # a plain line whose arrow end is on no labelled symbol (a stream label):
                # the arrow is that end
                edges.append((near, sid, "non-solid" if e.is_dashed else "solid", group))
                continue
            for k, (a, b, line, g) in enumerate(list(edges)):
                if g == group and near in (a, b):
                    far = b if a == near else a
                    edges[k] = (sid, far, line, g)
                    edges.append((near, sid, line, g))


def _sanitize_id(shape_id: str) -> str:
    """GraphML node/edge ids must be valid XML names; drawio ids are already alnum/-/_
    but are prefixed here to guarantee a non-numeric first character regardless."""
    return f"n{shape_id}"


def diagram_to_pid2graph_xml(diagram: ParsedDiagram) -> ET.Element:
    ET.register_namespace("", GRAPHML_NS)
    root = ET.Element(f"{{{GRAPHML_NS}}}graphml")
    root.set(f"{{{XSI_NS}}}schemaLocation", f"{GRAPHML_NS} {GRAPHML_NS}/1.0/graphml.xsd")

    def add_key(key_id: str, for_: str, name: str, type_: str) -> None:
        key = ET.SubElement(root, f"{{{GRAPHML_NS}}}key")
        key.set("id", key_id)
        key.set("for", for_)
        key.set("attr.name", name)
        key.set("attr.type", type_)

    add_key("d0", "node", "label", "string")
    add_key("d1", "node", "xmin", "double")
    add_key("d2", "node", "xmax", "double")
    add_key("d3", "node", "ymin", "double")
    add_key("d4", "node", "ymax", "double")
    add_key("d9", "edge", "edge_label", "string")

    graph = ET.SubElement(root, f"{{{GRAPHML_NS}}}graph")
    graph.set("edgedefault", "undirected")

    def add_data(parent: ET.Element, key_id: str, value: str) -> None:
        data = ET.SubElement(parent, f"{{{GRAPHML_NS}}}data")
        data.set("key", key_id)
        data.text = value

    symbols, edges = labelled_symbols(diagram)
    for node_id, label, (x1, y1, x2, y2), _ in symbols:
        node = ET.SubElement(graph, f"{{{GRAPHML_NS}}}node")
        node.set("id", node_id)
        add_data(node, "d0", label)
        add_data(node, "d1", str(x1))
        add_data(node, "d2", str(x2))
        add_data(node, "d3", str(y1))
        add_data(node, "d4", str(y2))
    for a, b, line in edges:
        edge_el = ET.SubElement(graph, f"{{{GRAPHML_NS}}}edge")
        edge_el.set("source", a)
        edge_el.set("target", b)
        add_data(edge_el, "d9", line)

    return root


def export_pid2graph_graphml(diagram: ParsedDiagram) -> str:
    root = diagram_to_pid2graph_xml(diagram)
    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def export_drawio_file_to_pid2graph(drawio_path: str | Path, output_path: str | Path) -> None:
    """Accepts a real .drawio file (compressed diagram, decoded first) or already-decoded
    mxGraph XML. Reading a .drawio directly as XML finds no cells and writes an empty
    graph - which is exactly what happened when regenerating these exports."""
    from pid_dexpi_aas.drawio.decode import decode_drawio
    from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved

    path = Path(drawio_path)
    if path.suffix == ".drawio":
        diagram = parse_drawio_xml_resolved(decode_drawio(path)[0], source_file=path.name, drawio_path=path)
    else:
        diagram = parse_drawio_file(path)
    xml_text = export_pid2graph_graphml(diagram)
    Path(output_path).write_text(xml_text, encoding="utf-8")
