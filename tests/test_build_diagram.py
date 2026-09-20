from pathlib import Path

from pid_dexpi_aas.build_diagram import build_diagram
from pid_dexpi_aas.build_model import build_model
from pid_dexpi_aas.parser_drawio import parse_drawio_file
from pid_dexpi_aas.render_svg import render_diagram_svg

DECODED_DIR = Path(__file__).parent.parent / "handoff" / "decoded_xml"
FIXTURE = DECODED_DIR / "PFD combined set up 301122__diagram0.xml"


def test_diagram_has_one_group_per_built_equipment_plus_piping_lines():
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    d = build_diagram(diagram, result, result.model)

    built_shape_count = sum(
        1 for r in result.records if r.status == "built" and r.element_kind == "shape"
    )
    # one RepresentationGroup per built shape, plus (if any edges rendered) one more
    # group holding all the piping ConnectorLines.
    assert len(d.groups) in (built_shape_count, built_shape_count + 1)


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
