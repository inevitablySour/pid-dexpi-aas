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


def test_no_true_self_loop_connections_in_model():
    """Regression test, UPDATED after research/04-dexpi-semantics.md: a segment must
    never have the literal SAME connector object as both sourceItem and targetItem (that
    is always degenerate - a component piped to itself), but a same-EQUIPMENT connection
    via two DIFFERENT nozzles is legitimate (a real recirculation/kickback/reflux loop)
    and must be preserved, not rejected. See test_same_shape_edges_build_via_distinct_nozzles
    below for the positive case this test's sibling checks.
    """
    diagram = parse_drawio_file(GRAMFLOW)
    result = build_model(diagram)
    for system in result.model.conceptualModel.pipingNetworkSystems:
        for segment in system.segments:
            assert segment.sourceItem is not segment.targetItem or segment.sourceItem is None


def test_same_shape_edges_build_via_distinct_nozzles():
    """A real recirculation/kickback loop connects back to the SAME equipment shape -
    per research/04-dexpi-semantics.md, DEXPI's PipingSourceItem/PipingTargetItem type
    system excludes Equipment entirely, so this must be modelled as two distinct Nozzles
    on that equipment, never a literal equipment-level self-reference. Confirmed on this
    file: "safe location, V205" (a vessel) has two separate edges both resolving to it,
    and both must build as real segments with two DIFFERENT Nozzle objects - not be
    thrown away as a false self-loop (the original, overly broad rejection this session
    started with), and not collide on one shared nozzle (the original one-nozzle-per-
    equipment construction, which would make this structurally unrepresentable at all).
    """
    diagram = parse_drawio_file(GRAMFLOW)
    result = build_model(diagram)

    same_shape_edges = [
        e for e in diagram.edges if e.source and e.target and e.source == e.target
    ]
    assert len(same_shape_edges) > 0, "fixture no longer exercises this case"

    built_segments = []
    for system in result.model.conceptualModel.pipingNetworkSystems:
        built_segments.extend(system.segments)

    same_shape_ids = {e.source for e in same_shape_edges}
    matched_any = False
    for equipment in result.model.conceptualModel.taggedPlantItems:
        shape_id_for_equipment = next(
            (sid for sid, obj in result.shape_objects.items() if obj is equipment), None
        )
        if shape_id_for_equipment not in same_shape_ids:
            continue
        nozzles_used_by_this_equipment = [
            nozzle for nozzle in equipment.nozzles
            for segment in built_segments
            if segment.sourceItem is nozzle or segment.targetItem is nozzle
        ]
        if len(nozzles_used_by_this_equipment) >= 2:
            matched_any = True
            assert len(set(id(n) for n in nozzles_used_by_this_equipment)) >= 2

    assert matched_any, "expected at least one equipment to have 2+ distinct nozzles used by segments"


def test_zero_length_edges_are_rejected_not_built_as_recirculation():
    """A genuinely zero-length edge (identical source_point/target_point) is a degenerate
    artifact, not a recirculation loop - a real loop has actual pipe length between two
    distinct nozzle positions. Confirmed on this file: h17LKkp3DZ7R27TqlmPq-62 has
    source_point == target_point exactly.
    """
    diagram = parse_drawio_file(GRAMFLOW)
    result = build_model(diagram)
    zero_length_record = next(
        r for r in result.records if r.element_id == "h17LKkp3DZ7R27TqlmPq-62"
    )
    assert zero_length_record.status == "skipped"
    assert "zero-length" in zero_length_record.reason
