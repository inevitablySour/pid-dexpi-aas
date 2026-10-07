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


def labelled_symbols(diagram: ParsedDiagram):
    """The symbols and connections of a drawing in PID2Graph terms - the ONE labelling
    used both for these exports and for the ML training data (make_team_data.py).

    Returns (symbols, edges): symbols = [(node_id, label, (x1, y1, x2, y2), drawio_id)] in
    draw.io coordinates; edges = [(node_id_a, node_id_b, "solid" | "non-solid")], one per
    connected pair. Only real symbols are labelled - the original exporter labelled EVERY
    draw.io cell, text labels included, as "general" (56 of LightUp's 113 nodes), which
    would teach a model to box text as symbols.

    Caveat: connections come from the draw.io lines via the parser; about half of them had
    one loose end resolved to the nearest shape (not verified individually).
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
    edges, seen = [], set()
    for c in review.connections:
        if c.include and c.from_id in id_map and c.to_id in id_map and c.from_id != c.to_id:
            a, b = sorted((id_map[c.from_id], id_map[c.to_id]))
            if (a, b) not in seen:  # parallel lines between one pair -> one connection
                seen.add((a, b))
                edges.append((a, b, "non-solid" if c.line_type == "signal" else "solid"))
    return symbols, edges


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
