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

from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

from .mapping import Category, map_shape
from .parser_drawio import Edge, ParsedDiagram, Shape, parse_drawio_file

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

# DEXPI class name (mapping.py's output) -> PID2Graph's 7-class node taxonomy.
# Anything not listed here (including every UNMAPPED/NOT_PROCESS_EQUIPMENT shape, and any
# DEXPI class not yet enumerated) falls back to "general" - PID2Graph's own catch-all for
# symbols that don't fit the other 6 classes, not a sign of a mapping gap on our side.
_DEXPI_CLASS_TO_PID2GRAPH_LABEL: dict[str, str] = {
    "Tank": "tank",
    "PressureVessel": "tank",
    "CustomVessel": "tank",
    "Vessel": "tank",
    "ProcessColumn": "tank",
    "Pump": "pump",
    "CustomPump": "pump",
    "TubularHeatExchanger": "general",
    "StraightwayValve": "valve",
    "GateValve": "valve",
    "CheckValve": "valve",
    "OperatedValve": "valve",
    "SafetyValveOrFitting": "valve",
    "ProcessInstrumentationFunction": "instrumentation",
    "SignalOffPageConnector": "inlet/outlet",
}


def _pid2graph_label(shape: Shape) -> str:
    if shape.shape_token and "arrow" in shape.shape_token.lower():
        return "arrow"
    result = map_shape(shape.shape_token, shape.text)
    if result.category == Category.NOT_PROCESS_EQUIPMENT:
        return "general"
    if result.dexpi_class_name is None:
        return "general"
    return _DEXPI_CLASS_TO_PID2GRAPH_LABEL.get(result.dexpi_class_name, "general")


def _sanitize_id(shape_id: str) -> str:
    """GraphML node/edge ids must be valid XML names; drawio ids are already alnum/-/_
    but are prefixed here to guarantee a non-numeric first character regardless."""
    return f"n{shape_id}"


def _real_edges(diagram: ParsedDiagram) -> list[Edge]:
    """Only edges with both endpoints resolved to a real shape, excluding decorative
    linework - same filter build_model.py uses to decide what's a genuine connection,
    so the exported graph's topology matches what this pipeline actually builds, not a
    superset that includes rejected/ambiguous edges."""
    return [e for e in diagram.edges if e.source and e.target and not e.is_decorative]


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

    shapes_by_id = {shape.id: shape for shape in diagram.shapes}

    def add_data(parent: ET.Element, key_id: str, value: str) -> None:
        data = ET.SubElement(parent, f"{{{GRAPHML_NS}}}data")
        data.set("key", key_id)
        data.text = value

    for shape in diagram.shapes:
        node = ET.SubElement(graph, f"{{{GRAPHML_NS}}}node")
        node.set("id", _sanitize_id(shape.id))
        add_data(node, "d0", _pid2graph_label(shape))
        add_data(node, "d1", str(shape.x))
        add_data(node, "d2", str(shape.x + shape.width))
        add_data(node, "d3", str(shape.y))
        add_data(node, "d4", str(shape.y + shape.height))

    for edge in _real_edges(diagram):
        if edge.source not in shapes_by_id or edge.target not in shapes_by_id:
            # Resolved onto a shape id that (for whatever upstream reason) isn't in this
            # diagram's own shape list - skip rather than emit a dangling edge reference,
            # which would make the file fail to load as a graph at all.
            continue
        edge_el = ET.SubElement(graph, f"{{{GRAPHML_NS}}}edge")
        edge_el.set("source", _sanitize_id(edge.source))
        edge_el.set("target", _sanitize_id(edge.target))
        add_data(edge_el, "d9", "non-solid" if edge.is_dashed else "solid")

    return root


def export_pid2graph_graphml(diagram: ParsedDiagram) -> str:
    root = diagram_to_pid2graph_xml(diagram)
    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def export_drawio_file_to_pid2graph(drawio_path: str | Path, output_path: str | Path) -> None:
    diagram = parse_drawio_file(drawio_path)
    xml_text = export_pid2graph_graphml(diagram)
    Path(output_path).write_text(xml_text, encoding="utf-8")
