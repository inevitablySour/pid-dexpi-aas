"""Regression tests for the PID2Graph GraphML exporter: schema confirmed against a real
sample from Zenodo record 14803338 (PID2Graph/Complete/PID2Graph OPEN100/3.graphml), not
guessed from the paper's prose description - see export_pid2graph.py's module docstring
for the exact key ids this locks in.
"""
from pathlib import Path

import networkx as nx

from pid_dexpi_aas.image.export_pid2graph import export_pid2graph_graphml, labelled_symbols
from pid_dexpi_aas.drawio.parser import parse_drawio_file

DECODED_DIR = Path(__file__).parent / "fixtures" / "decoded_xml"
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
    assert g.number_of_nodes() == len(labelled_symbols(diagram)[0])


def test_node_labels_use_only_pid2graph_taxonomy():
    """Every emitted node label must be one of the real vocabulary values observed in the
    dataset's real OPEN100 subset - a class outside this set would be silently unusable
    by any model trained on PID2Graph's taxonomy."""
    real_symbol_labels = {"arrow", "general", "inlet/outlet", "instrumentation", "pump", "tank", "valve"}
    diagram = parse_drawio_file(PFD)
    for _, label, _, _ in labelled_symbols(diagram)[0]:
        assert label in real_symbol_labels


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


LIGHTUP = DECODED_DIR / "P&ID gramflow reactor V3__diagram0.xml"


def test_text_labels_are_not_exported_as_symbols():
    """Regression: the first exporter labelled EVERY draw.io cell - tag labels like "GV204"
    and notes like "18 bar" - as a "general" symbol (56 of LightUp's 113 nodes), which
    would teach a model trained on these files to box text. Only real symbols count."""
    diagram = parse_drawio_file(LIGHTUP)
    exported = {orig for _, _, _, orig in labelled_symbols(diagram)[0]}
    text_cells = [s for s in diagram.shapes if (s.style or "").startswith("text;")]
    assert text_cells and not exported & {s.id for s in text_cells}
    labels = [label for _, label, _, _ in labelled_symbols(diagram)[0]]
    assert labels.count("valve") >= 20 and labels.count("general") < 10


def test_rotated_shape_box_swaps_width_and_height():
    from pid_dexpi_aas.image.export_pid2graph import rotated_box
    from pid_dexpi_aas.drawio.parser import Shape
    s = Shape(id="a", shape_token="x", text="", x=0, y=0, width=100, height=30, style="rotation=-90;")
    assert rotated_box(s) == (35.0, -35.0, 65.0, 65.0)


def test_compressed_drawio_file_exports_its_symbols(tmp_path):
    """Regression: exporting a real (compressed) .drawio file read it as plain XML, found
    no cells, and silently wrote an empty graph."""
    from pid_dexpi_aas.image.export_pid2graph import export_drawio_file_to_pid2graph
    src = Path(__file__).parent.parent / "subprojects" / "LightUp" / "P&ID gramflow reactor V3.drawio"
    out = tmp_path / "x.graphml"
    export_drawio_file_to_pid2graph(src, out)
    assert nx.read_graphml(out).number_of_nodes() > 40


def test_flow_arrows_are_labelled_and_split_their_line():
    """arrows=True (Team v4 on): every arrowhead is an "arrow" symbol on its line, as in
    OPEN100; the default labelling (Team v3) has none."""
    import pytest
    path = Path(__file__).resolve().parents[1] / "subprojects" / "LightUp" / "P&ID gramflow reactor V3.drawio"
    diagram = parse_drawio_file(path) if path.exists() else None
    if not diagram or not diagram.routes:
        pytest.skip("needs the team drawing and the draw.io app (line routes)")
    plain, plain_edges = labelled_symbols(diagram)
    syms, edges = labelled_symbols(diagram, arrows=True)
    arrows = {sid for sid, label, _, _ in syms if label == "arrow"}
    assert not any(label == "arrow" for _, label, _, _ in plain)
    assert len(arrows) >= 40 and len(syms) == len(plain) + len(arrows)
    on_lines = {x for a, b, _ in edges for x in (a, b)} & arrows
    assert len(on_lines) >= 0.9 * len(arrows)
    # boxed as OPEN100 boxes arrows: long side 0.35 of the median symbol (not the ~10-unit drawn head)
    import statistics
    med = statistics.median(max(b[2] - b[0], b[3] - b[1]) for sid, label, b, _ in syms if label != "arrow")
    assert all(abs(max(b[2] - b[0], b[3] - b[1]) - 0.35 * med) < 1e-6 for sid, _, b, _ in syms if sid in arrows)
