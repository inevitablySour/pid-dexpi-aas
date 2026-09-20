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
    diagram = parse_drawio_file(GRAMFLOW)  # goes through parse_drawio_xml_resolved
    resolved_count = sum(
        1 for e in diagram.edges if (e.source_point or e.target_point) and e.source and e.target
    )
    assert resolved_count > 40


def test_resolving_floating_edges_meaningfully_increases_built_count():
    """Before the fix, this file built only 69/251 elements; after, 111/251 - pin a
    conservative floor so a future regression is caught, not just "still runs".
    """
    diagram = parse_drawio_file(GRAMFLOW)
    result = build_model(diagram)
    built = sum(1 for r in result.records if r.status == "built")
    assert built >= 100
