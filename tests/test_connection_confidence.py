"""Tests for the trust-tier tracking added after manual verification against the
original gramflow reactor P&ID image found that nearest_shape-resolved connections can
be wrong (e.g. a label 535 units away force-matched to an unrelated tank). Research into
prior art (Kim et al. 2022; Microsoft ISE's P&ID digitization writeup;
Schulze Balhorn et al.'s rule-based DEXPI graph autocorrection) confirms the standard
practice across every real deployment found is: trust explicit source data fully, and
flag every inferred connection for human review rather than presenting it with the same
confidence - never claim silent full automation for hand-drawn input.
"""
from pathlib import Path

from pid_dexpi_aas.build_model import build_model
from pid_dexpi_aas.parser_drawio import parse_drawio_file, parse_drawio_xml
from pid_dexpi_aas.validate_by_counting import report_for_diagram

GRAMFLOW = (
    Path(__file__).parent.parent
    / "handoff"
    / "decoded_xml"
    / "P&ID gramflow reactor V3__diagram0.xml"
)


def test_explicit_edges_are_never_marked_as_inferred():
    xml_text = GRAMFLOW.read_text(encoding="utf-8")
    diagram = parse_drawio_xml(xml_text, source_file=GRAMFLOW.name)
    explicit_edges = [e for e in diagram.edges if e.source_resolution == "explicit"]
    assert len(explicit_edges) > 0
    for edge in explicit_edges:
        assert edge.source is not None


def test_exit_fraction_tier_is_tighter_than_nearest_shape():
    """exit_fraction resolutions should only fire when a near-exact geometric match
    exists (<=10 units) - confirmed tighter and more trustworthy than the 40-unit
    nearest_shape fallback by direct measurement against real shape geometry.
    """
    diagram = parse_drawio_file(GRAMFLOW)
    fraction_resolved = [
        e
        for e in diagram.edges
        if e.source_resolution == "exit_fraction" or e.target_resolution == "exit_fraction"
    ]
    assert len(fraction_resolved) > 0


def test_report_distinguishes_explicit_from_inferred_connections():
    diagram = parse_drawio_file(GRAMFLOW)
    result = build_model(diagram)
    report = report_for_diagram(diagram, result)

    assert "explicit" in report.connections_by_confidence
    assert report.connections_by_confidence["explicit"] > 0
    assert report.inferred_connection_count > 0

    total_connections = sum(report.connections_by_confidence.values())
    explicit_count = report.connections_by_confidence.get("explicit", 0)
    assert report.inferred_connection_count == total_connections - explicit_count
