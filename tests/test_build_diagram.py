from pathlib import Path

from pid_dexpi_aas.render.build_diagram import _route_orthogonal, build_diagram
from pid_dexpi_aas.dexpi.build_model import build_model
from pid_dexpi_aas.drawio.parser import parse_drawio_file
from pid_dexpi_aas.render.render_svg import render_diagram_svg

DECODED_DIR = Path(__file__).parent / "fixtures" / "decoded_xml"
FIXTURE = DECODED_DIR / "PFD combined set up 301122__diagram0.xml"


def test_diagram_has_one_group_per_built_equipment_plus_piping_lines():
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    d = build_diagram(diagram, result, result.model)

    off_page_connector_count = sum(
        1
        for r in result.records
        if r.status == "built"
        and r.element_kind == "shape"
        and r.dexpi_class_name in ("FlowInPipeOffPageConnector", "FlowOutPipeOffPageConnector")
    )
    built_shape_count = sum(
        1 for r in result.records if r.status == "built" and r.element_kind == "shape"
    )
    # one RepresentationGroup per built shape EXCEPT off-page connectors (which get no
    # symbol of their own - see build_diagram.py, they're a floating text label already
    # drawn as its own separate shape, not equipment), plus (if any edges rendered) one
    # more group holding all the piping ConnectorLines.
    real_symbol_count = built_shape_count - off_page_connector_count
    assert len(d.groups) in (real_symbol_count, real_symbol_count + 1)


def test_diagram_bounds_are_sane_not_degenerate():
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    d = build_diagram(diagram, result, result.model)
    assert d.maxX - d.minX > 100
    assert d.maxY - d.minY > 100


def test_render_svg_produces_a_viewbox_matching_diagram_bounds(tmp_path):
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    d = build_diagram(diagram, result, result.model)
    svg_path = render_diagram_svg(d, tmp_path / "out.svg", padding=20.0)

    content = svg_path.read_text(encoding="utf-8")
    expected_width = d.maxX - d.minX + 40.0
    expected_height = d.maxY - d.minY + 40.0
    assert f'width="{expected_width}"' in content
    assert f'height="{expected_height}"' in content


def test_render_svg_contains_real_symbol_geometry(tmp_path):
    """Regression test for the confirmed pyDEXPI viewBox bug (see render_svg.py's
    docstring): without the fix, the SVG technically 'succeeds' but the viewBox collapses
    to a tiny box unrelated to content. This checks actual drawn primitives exist.
    """
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    d = build_diagram(diagram, result, result.model)
    svg_path = render_diagram_svg(d, tmp_path / "out.svg")

    content = svg_path.read_text(encoding="utf-8")
    assert "<polyline" in content or "<polygon" in content
    assert "<ellipse" in content


def test_all_seven_team_files_render_without_error(tmp_path):
    for xml_path in sorted(DECODED_DIR.glob("*.xml")):
        diagram = parse_drawio_file(xml_path)
        result = build_model(diagram)
        d = build_diagram(diagram, result, result.model)
        render_diagram_svg(d, tmp_path / f"{xml_path.stem}.svg")


def test_orthogonal_routing_produces_right_angle_bends_not_a_diagonal():
    """Regression test: connections used to be a single straight line from center to
    center (reported by the user as "not even close to being a real P&ID"). A routed
    path must have at least one intermediate bend point, and every consecutive segment
    must be purely horizontal or purely vertical (no diagonals).
    """
    source_box = (0.0, 0.0, 50.0, 50.0)
    target_box = (300.0, 200.0, 350.0, 250.0)
    points = _route_orthogonal(source_box, target_box)

    assert len(points) >= 3, "expected at least one bend, got a straight 2-point line"

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        is_horizontal = y1 == y2
        is_vertical = x1 == x2
        assert is_horizontal or is_vertical, f"diagonal segment: ({x1},{y1}) -> ({x2},{y2})"


def test_orthogonal_routing_side_by_side_boxes_routes_horizontally():
    """Two boxes separated mostly left-right should route via their facing vertical
    sides (exit right side of source, enter left side of target), not centers."""
    source_box = (0.0, 0.0, 50.0, 50.0)
    target_box = (200.0, 10.0, 250.0, 60.0)
    points = _route_orthogonal(source_box, target_box)

    start_x, _ = points[0]
    end_x, _ = points[-1]
    assert start_x == 50.0  # source's right edge (max_x), not its center
    assert end_x == 200.0  # target's left edge (min_x), not its center


def test_instrument_bubbles_get_their_own_text_as_a_label(tmp_path):
    """Regression test: instrument bubbles (e.g. drawio shape text "PI 201", "TT 203")
    carry their tag directly on the shape itself, not via a nearby floating label, so
    tag_lookup (built from tag_association.py, which only covers floating labels) is
    empty for them. build_diagram.py must fall back to shape.text directly, or every
    instrument bubble renders with no text at all - confirmed visually on the gramflow
    reactor file (19 instrument bubbles, all missing their tags) before this fix.
    """
    xml_path = DECODED_DIR / "P&ID gramflow reactor V3__diagram0.xml"
    diagram = parse_drawio_file(xml_path)
    result = build_model(diagram)
    d = build_diagram(diagram, result, result.model)
    svg_path = render_diagram_svg(d, tmp_path / "gramflow.svg")

    content = svg_path.read_text(encoding="utf-8")
    for expected_tag in ("PI 201", "TT 203", "TIC 203", "PC 201"):
        assert f">{expected_tag}<" in content, f"missing instrument tag: {expected_tag}"
