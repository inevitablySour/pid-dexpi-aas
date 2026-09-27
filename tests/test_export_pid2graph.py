"""Regression tests for the PID2Graph GraphML exporter: schema confirmed against a real
sample from Zenodo record 14803338 (PID2Graph/Complete/PID2Graph OPEN100/3.graphml), not
guessed from the paper's prose description - see export_pid2graph.py's module docstring
for the exact key ids this locks in.
"""
from pathlib import Path

import networkx as nx

from pid_dexpi_aas.export_pid2graph import export_pid2graph_graphml, _pid2graph_label
from pid_dexpi_aas.parser_drawio import parse_drawio_file

DECODED_DIR = Path(__file__).parent.parent / "handoff" / "decoded_xml"
PFD = DECODED_DIR / "PFD combined set up 301122__diagram0.xml"


def test_exported_graphml_uses_the_real_pid2graph_key_ids():
    """Locks in the exact key id -> attr.name/type mapping confirmed against a real
    downloaded sample, so a future refactor can't silently drift from the real schema."""
    diagram = parse_drawio_file(PFD)
    xml_text = export_pid2graph_graphml(diagram)
    assert '<key id="d0" for="node" attr.name="label" attr.type="string"' in xml_text
    assert '<key id="d1" for="node" attr.name="xmin" attr.type="double"' in xml_text
    assert '<key id="d4" for="node" attr.name="ymax" attr.type="double"' in xml_text
    assert '<key id="d9" for="edge" attr.name="edge_label" attr.type="string"' in xml_text


def test_exported_graphml_loads_via_networkx():
    """PID2Graph's own Zenodo description says to load these files with networkx - if
    that fails, the file is unusable for the dataset's intended purpose regardless of
    whether it's well-formed XML."""
    diagram = parse_drawio_file(PFD)
    xml_text = export_pid2graph_graphml(diagram)
    tmp_path = Path("/tmp") / "test_pid2graph_export.graphml"
    tmp_path.write_text(xml_text, encoding="utf-8")
    try:
        g = nx.read_graphml(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    assert g.number_of_nodes() == len(diagram.shapes)


def test_node_labels_use_only_pid2graph_taxonomy():
    """Every emitted node label must be one of the real vocabulary values observed in the
    dataset's real OPEN100 subset - a class outside this set would be silently unusable
    by any model trained on PID2Graph's taxonomy."""
    real_symbol_labels = {"arrow", "general", "inlet/outlet", "instrumentation", "pump", "tank", "valve"}
    diagram = parse_drawio_file(PFD)
    for shape in diagram.shapes:
        assert _pid2graph_label(shape) in real_symbol_labels


def test_decorative_and_unresolved_edges_are_excluded():
    """The exported graph's edge count must never exceed the count of edges this pipeline
    itself considers real (resolved on both ends, not decorative) - otherwise the folded-
    in file would claim connections build_model.py itself refuses to build."""
    diagram = parse_drawio_file(PFD)
    xml_text = export_pid2graph_graphml(diagram)
    tmp_path = Path("/tmp") / "test_pid2graph_export_edges.graphml"
    tmp_path.write_text(xml_text, encoding="utf-8")
    try:
        g = nx.read_graphml(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    real_edges = [e for e in diagram.edges if e.source and e.target and not e.is_decorative]
    assert g.number_of_edges() <= len(real_edges)
