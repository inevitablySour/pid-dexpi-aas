"""Regression tests for the floating-edge-endpoint bug: drawio edges not snapped exactly
onto a shape store their endpoint as a raw <mxPoint>, not a source/target cell-id
attribute. Confirmed via direct XML inspection that roughly half of all edges across
every team file use this form - the parser originally ignored it entirely, silently
dropping ~half the piping topology (reported by the user as "the SVG is completely
incorrect" / "why did it skip so many" on the LightUp gramflow reactor file, which has
138 edges, the most of any team file, making the impact largest there).
"""
from pathlib import Path

from pid_dexpi_aas.build_model import build_model
from pid_dexpi_aas.parser_drawio import parse_drawio_file, parse_drawio_xml

DECODED_DIR = Path(__file__).parent.parent / "handoff" / "decoded_xml"
GRAMFLOW = DECODED_DIR / "P&ID gramflow reactor V3__diagram0.xml"


def test_edges_without_source_target_attrs_exist_in_the_fixture():
    """Sanity check that this fixture actually exercises the bug - if drawio's export
    format ever changes and stops producing bare mxPoint edges, this test should fail
    loudly rather than the real test below silently passing for the wrong reason.
    """
    xml_text = GRAMFLOW.read_text(encoding="utf-8")
    diagram = parse_drawio_xml(xml_text, source_file=GRAMFLOW.name)
    floating = [e for e in diagram.edges if e.source_point or e.target_point]
    assert len(floating) > 50


def test_floating_edges_resolve_to_real_shapes():
    """32 of the ~40+ originally-resolving edges are now correctly excluded as
    decorative symbol linework (see test_decorative_edges_are_classified_separately in
    test_decorative_edges.py) rather than resolved into fake connections, so the honest
    count of real resolved piping edges is lower than before that fix - 33, not 40+.
    """
    diagram = parse_drawio_file(GRAMFLOW)  # goes through parse_drawio_xml_resolved
    resolved_count = sum(
        1
        for e in diagram.edges
        if (e.source_point or e.target_point) and e.source and e.target and not e.is_decorative
    )
    assert resolved_count > 25


def test_resolving_floating_edges_meaningfully_increases_built_count():
    """Before the floating-edge fix, this file built only 69/251 elements. The nearest-
    shape resolver initially (wrongly) pushed this to 111 by also matching short dash-
    mark segments to whatever real shape happened to be within 40 units of BOTH their
    endpoints, producing self-loop connections (e.g. "TIC 203 -> TIC 203") that are
    invalid data, not just an ugly picture - see test_no_self_loop_connections_in_model
    below. After also rejecting self-loops, the honest count is lower (82) but the data
    is actually correct instead of inflated with garbage connections.
    """
    diagram = parse_drawio_file(GRAMFLOW)
    result = build_model(diagram)
    built = sum(1 for r in result.records if r.status == "built")
    assert built >= 75


def test_no_self_loop_connections_in_model():
    """Regression test: resolve_floating_edge_endpoints' nearest-shape search can match
    BOTH endpoints of a very short edge segment (e.g. one dash-mark of a hand-drawn
    dashed line) to the same nearby shape, producing a nonsense self-loop connection in
    the actual DEXPI model - not just a rendering issue. Confirmed on this file: 36 such
    edges (e.g. "TIC 203 -> TIC 203", "P101 -> P101") before build_model.py's wire_edges
    started rejecting edge.source == edge.target explicitly.
    """
    diagram = parse_drawio_file(GRAMFLOW)
    result = build_model(diagram)
    for system in result.model.conceptualModel.pipingNetworkSystems:
        for segment in system.segments:
            assert segment.sourceItem is not segment.targetItem or segment.sourceItem is None
