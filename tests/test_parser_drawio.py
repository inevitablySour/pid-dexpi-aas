from pathlib import Path

from pid_dexpi_aas.drawio.parser import parse_drawio_file

FIXTURE = (
    Path(__file__).parent
    / "fixtures" / "decoded_xml"
    / "PFD combined set up 301122__diagram0.xml"
)


def test_pfd_combined_smoke():
    diagram = parse_drawio_file(FIXTURE)
    assert len(diagram.shapes) == 16
    assert len(diagram.edges) == 13


def test_shape_tokens_parsed():
    diagram = parse_drawio_file(FIXTURE)
    tokens = {s.shape_token for s in diagram.shapes if s.shape_token}
    assert "mxgraph.pid.vessels.reactor" in tokens
    assert "mxgraph.pid2inst.discInst" in tokens


def test_labeled_only_shapes_have_no_token():
    diagram = parse_drawio_file(FIXTURE)
    autoclave = next(s for s in diagram.shapes if s.text == "Autoclave")
    assert autoclave.is_labeled_only


def test_html_stripped_from_disc_inst_value():
    diagram = parse_drawio_file(FIXTURE)
    bpr = next(s for s in diagram.shapes if s.shape_token == "mxgraph.pid2inst.discInst")
    assert bpr.text == "BPR"


def test_edge_source_target_present():
    diagram = parse_drawio_file(FIXTURE)
    connected = [e for e in diagram.edges if e.source and e.target]
    assert len(connected) > 0
