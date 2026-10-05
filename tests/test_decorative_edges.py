"""Tests for classify_decorative_edges: distinguishing hand-drawn symbol linework
(e.g. a safety valve's spring/relief indicator) from genuinely broken/corrupt piping
connections, both of which previously got lumped together under "self-loop, invalid".

Found via systematic verification after the user reported the rendered P&ID still
looked wrong on the left side and bottom: 36 edges were rejected as "self-loop", but
direct inspection showed 32 of them were tight clusters of 6-7 tiny edges all confined
to one valve's own bounding box (valveType=angle safety-relief valves specifically),
with one endpoint exactly on the valve boundary (distance 0) and the other a short
distance beyond it in a consistent direction - the visual signature of a hand-drawn
spring/relief symbol, not piping. A genuine connection's far endpoint lands near a
DIFFERENT, distant shape (confirmed: PSV201's real vent line to "safe location" is
520+ units away, not the 8-33 units seen in these clusters).
"""
from pathlib import Path

from pid_dexpi_aas.build_model import build_model
from pid_dexpi_aas.parser_drawio import parse_drawio_file, parse_drawio_xml

GRAMFLOW = (
    Path(__file__).parent.parent
    / "handoff"
    / "decoded_xml"
    / "P&ID gramflow reactor V3__diagram0.xml"
)


def test_decorative_edges_are_classified_on_this_fixture():
    diagram = parse_drawio_file(GRAMFLOW)
    decorative_count = sum(1 for e in diagram.edges if e.is_decorative)
    assert decorative_count >= 25


def test_decorative_edges_are_never_resolved_into_connections():
    diagram = parse_drawio_file(GRAMFLOW)
    for edge in diagram.edges:
        if edge.is_decorative:
            assert edge.source_resolution == "unresolved"
            assert edge.target_resolution == "unresolved"


def test_a_lone_short_edge_near_a_shape_is_not_classified_as_decorative():
    """A single tiny edge near a shape could still be a genuine (if short) connection -
    only a CLUSTER of several is a reliable decorative-symbol signal. Confirmed: after
    classification, the gramflow file still has a handful of unresolved single-instance
    edges near a shape that are NOT marked decorative (cluster size 1-2), left for
    individual review rather than assumed decorative.
    """
    diagram = parse_drawio_file(GRAMFLOW)
    non_decorative_unresolved = [
        e
        for e in diagram.edges
        if not e.is_decorative and e.source_point and e.target_point and not e.source
    ]
    assert len(non_decorative_unresolved) > 0


def test_decorative_edges_are_reported_distinctly_not_as_self_loops():
    diagram = parse_drawio_file(GRAMFLOW)
    result = build_model(diagram)
    decorative_records = [r for r in result.records if "decorative" in r.reason]
    self_loop_records = [r for r in result.records if "self-loop" in r.reason]
    assert len(decorative_records) >= 25
    # Self-loop count should have dropped sharply now that decorative clusters are
    # correctly excluded from that classification (was 36 before this fix). 11 since the
    # segment rewrite: dash fragments looping on an INSTRUMENT are now (correctly) called
    # self-loops instead of being filed as sensing edges. (Without line routes - this test
    # parses decoded XML - one is an 83-unit TT 202 line whose far end is really a junction
    # on the header; with routes that end resolves correctly.)
    assert len(self_loop_records) < 12
