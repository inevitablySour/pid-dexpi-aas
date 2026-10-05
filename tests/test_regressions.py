"""Regression checks for the connection, symbol and DEXPI fixes made against the team's
real drawings. Each test names the bug it guards against.

Drawing-level tests need line routes, i.e. the draw.io desktop app (ml/tools/drawio.AppImage)
or a warm route cache; without either they are skipped, not failed.
"""
import collections
import re
from pathlib import Path

import pytest
from pydexpi.dexpi_classes import piping

from pid_dexpi_aas import line_routes
from pid_dexpi_aas.build_model import _segment_by_hand, build_model
from pid_dexpi_aas.parser_drawio import Edge, parse_drawio_file
from pid_dexpi_aas.review import from_drawio
from pid_dexpi_aas.serialize import save_and_verify
from pid_dexpi_aas.spec_labels import parse_specs

ROOT = Path(__file__).parent.parent
DRAWINGS = sorted((ROOT / "subprojects").glob("*/*.drawio"))
LIGHTUP = ROOT / "subprojects" / "LightUp" / "P&ID gramflow reactor V3.drawio"
COMBINED = ROOT / "subprojects" / "Lumen" / "P&ID Lumen combined set-up (1) (1) (1).drawio"
OFF_PAGE = (piping.FlowInPipeOffPageConnector, piping.FlowOutPipeOffPageConnector)

needs_routes = pytest.mark.skipif(not line_routes.available() or not DRAWINGS,
                                  reason="draw.io app or team drawings not available")


def _pairs(review):
    tag = {s.id: s.tag or s.dexpi_class for s in review.symbols}
    return {frozenset((tag.get(c.from_id, c.from_id), tag.get(c.to_id, c.to_id))) for c in review.connections}


# ------------------------------------------------------------- junction rules ---
def _edge(eid, src=None, tgt=None):
    return Edge(id=eid, source=src, target=tgt, style="")


def test_line_ending_on_another_line_is_a_tee():
    # A branch whose loose end lies on a pipe's route joins that pipe (was: left unconnected
    # or guessed onto the nearest symbol).
    edges = [_edge("main", "A", "B"), _edge("branch", "C", None)]
    routes = {"main": [(0, 0), (100, 0)], "branch": [(50, 60), (50, 3)]}
    assert line_routes.find_junctions(edges, routes) == {("branch", 1): "main"}


def test_two_loose_ends_meeting_are_one_pipe():
    edges = [_edge("a", "A", None), _edge("b", None, "B")]
    routes = {"a": [(0, 0), (50, 0)], "b": [(55, 0), (100, 0)]}
    assert line_routes.find_junctions(edges, routes) == {("a", 1): "b", ("b", 0): "a"}


def test_line_jump_gap_is_a_crossing_not_a_tee():
    # LightUp draws the GV drops into V101 with a gap where they cross the XV101-XV102
    # line; the two pieces are one pipe and the crossed line is NOT joined.
    edges = [_edge("crossed", "X", "Y"), _edge("top", "A", None), _edge("bottom", None, "B")]
    routes = {"crossed": [(0, 50), (100, 50)], "top": [(50, 0), (50, 40)], "bottom": [(50, 60), (50, 100)]}
    joins = line_routes.find_junctions(edges, routes)
    assert joins == {("top", 1): "bottom", ("bottom", 0): "top"}


def test_lines_that_merely_cross_are_not_joined():
    edges = [_edge("h", "A", "B"), _edge("v", "C", "D")]
    routes = {"h": [(0, 50), (100, 50)], "v": [(50, 0), (50, 100)]}
    assert line_routes.find_junctions(edges, routes) == {}


# ---------------------------------------------------------------- spec notes ---
@pytest.mark.parametrize("text, kind, value, unit, rest", [
    ("18 bar", "pressure", 18.0, "bar", ""),
    ("Volume = 5 L", "volume", 5.0, "L", ""),
    ("1 to 10ml/minute", "flow", 10.0, "ml/minute", ""),
    ("T201 [H2] 200 bar", "medium", None, "H2", "T201"),
])
def test_spec_notes_are_parsed(text, kind, value, unit, rest):
    specs, remaining = parse_specs(text)
    sp = next(s for s in specs if s.kind == kind)
    assert (sp.value, sp.unit, remaining) == (value, unit, rest)


def test_destination_in_brackets_is_not_a_medium():
    assert parse_specs("Fume hood [atm]")[0] == []


# ------------------------------------------------------- DEXPI segment builder ---
def test_off_page_first_segment_with_two_items_builds_and_round_trips(tmp_path):
    # pyDEXPI's piping toolkit rejects [inflow off-page, item, item] (Peristaltic: 8 build
    # errors); the hand-built fallback must give a segment that survives save/load.
    from pydexpi.dexpi_classes import equipment
    from test_proteus_patch import _bare_model

    off = piping.FlowInPipeOffPageConnector(nodes=[piping.PipingNode()])
    v1 = piping.GateValve(nodes=[piping.PipingNode(), piping.PipingNode()])
    v2 = piping.GateValve(nodes=[piping.PipingNode(), piping.PipingNode()])
    tank = equipment.Tank(nozzles=[equipment.Nozzle(nodes=[piping.PipingNode()])])
    seg = _segment_by_hand([off, v1, v2], off, tank.nozzles[0])
    assert [(type(c.sourceItem).__name__, type(c.targetItem).__name__) for c in seg.connections] == \
        [("FlowInPipeOffPageConnector", "GateValve"), ("GateValve", "GateValve"), ("GateValve", "Nozzle")]
    model = _bare_model()
    model.conceptualModel.taggedPlantItems = [tank]
    model.conceptualModel.pipingNetworkSystems = [piping.PipingNetworkSystem(segments=[seg])]
    back = save_and_verify(model, tmp_path, "x")
    assert len(back.conceptualModel.pipingNetworkSystems[0].segments[0].items) == 3


# ------------------------------------------------------------ team drawings ---
@pytest.fixture(scope="module")
def lightup():
    return from_drawio(parse_drawio_file(LIGHTUP))


@needs_routes
def test_lightup_tee_and_gap_connections(lightup):
    pairs = _pairs(lightup)
    for a, b in [("P101", "V101"), ("GV101", "V101"), ("GV102", "V101"), ("XV101", "XV102"),
                 ("PI 102", "XV101"), ("R201", "TT 203")]:
        assert frozenset((a, b)) in pairs, f"{a} - {b} missing"
    # the GV drops cross the XV101-XV102 line through a gap: not a junction with it
    assert frozenset(("GV101", "XV101")) not in pairs


@needs_routes
def test_lightup_spec_notes_become_specs_not_tags(lightup):
    specs = {s.tag: s.attributes.get("specs", "") for s in lightup.symbols}
    assert "18 bar" in specs["PSV201"] and "17 bar" in specs["PSV202"]
    assert "10ml/minute" in specs["P101"]
    assert "N2" in specs["T202"].replace(" ", "") and "5 bar" in specs["T202"]
    assert not any(re.fullmatch(r"\d+\s*bar", s.tag or "") for s in lightup.symbols)


@needs_routes
def test_plain_box_reactor_and_compartments_are_symbols():
    review = from_drawio(parse_drawio_file(COMBINED))
    plain = {s.tag: s.dexpi_class for s in review.symbols if s.detected_as.startswith("plain:")}
    assert plain.get("Reactor R301") == "Vessel"
    assert {plain.get("Mixing"), plain.get("Heating"), plain.get("Cooling")} == {"Mixer", "Heater", "HeatExchanger"}
    pairs = _pairs(review)
    assert frozenset(("Mixing", "Heating")) in pairs or frozenset(("Heating", "Cooling")) in pairs


@needs_routes
@pytest.mark.parametrize("drawing", DRAWINGS, ids=lambda p: p.stem[:30])
def test_dexpi_export_is_valid(drawing, tmp_path):
    result = build_model(parse_drawio_file(drawing))
    assert not [r for r in result.records if "piping_toolkit" in r.reason]
    back = save_and_verify(result.model, tmp_path, "x")
    xml = next(tmp_path.glob("x*")).read_text()
    ids = collections.Counter(re.findall(r'<\w+\b[^>]*\sID="([^"]+)"', xml))
    assert not [i for i, n in ids.items() if n > 1], "duplicate element IDs"
    built = [s for sy in result.model.conceptualModel.pipingNetworkSystems for s in sy.segments]
    loaded = [s for sy in back.conceptualModel.pipingNetworkSystems for s in sy.segments]
    assert [len(s.items) for s in built] == [len(s.items) for s in loaded]
    for o, b in zip(built, loaded):
        if any(isinstance(i, OFF_PAGE) for i in o.items):
            continue  # known pyDEXPI reader limitation: off-page segment ends are not re-linked
        # an end may be gained (the reader closes an open end on its last item), never lost
        assert not (o.sourceItem is not None and b.sourceItem is None)
        assert not (o.targetItem is not None and b.targetItem is None)


# ------------------------------------------------------------- flow direction ---
def test_flow_passes_through_inline_items_and_conserves_at_tees():
    from pid_dexpi_aas.flow_direction import Leg, propagate

    feed, valve, tee, out1, out2 = "feed", "valve", "tee", "out1", "out2"
    legs = [Leg(feed, valve, 1, "arrow"), Leg(valve, tee), Leg(tee, out1, 1, "arrow"), Leg(out2, tee)]
    propagate(legs, passes=lambda n: n == valve, junction=lambda n: n == tee)
    # valve passes feed -> tee; the tee then has one in, one out, one unknown: undecidable
    assert legs[1].dir == 1 and legs[1].basis == "implied"
    assert legs[3].dir == 0


def test_tee_with_all_known_legs_inflowing_sends_the_last_one_out():
    from pid_dexpi_aas.flow_direction import Leg, propagate

    legs = [Leg("a", "tee", 1, "arrow"), Leg("b", "tee", 1, "arrow"), Leg("tee", "c")]
    propagate(legs, passes=lambda n: False, junction=lambda n: n == "tee")
    assert legs[2].dir == 1


@needs_routes
def test_lightup_flow_directions(lightup):
    tag = {s.id: s.tag or s.dexpi_class for s in lightup.symbols}
    flows = {(tag[c.from_id], tag[c.to_id]): c.flow for c in lightup.connections if c.flow}
    assert flows[("V101", "P101")] == "drawn arrow"        # pump draws from the vessel
    assert flows[("GV202", "V202")] == "implied"           # arrow into V202 on the other leg
    assert flows[("V202", "StraightwayValve")] == "implied"  # V202 outlet to the PC 201 valve
    assert flows[("safe location", "Fume hood [atm]")] == "implied"


@needs_routes
def test_flow_orients_dexpi_segments(lightup):
    from pid_dexpi_aas.review import to_dexpi

    _, result = to_dexpi(lightup)
    segs = [s for sy in result.model.conceptualModel.pipingNetworkSystems for s in sy.segments]
    basis = collections.Counter(a.value for s in segs for a in s.customAttributes if a.attributeName == "FlowDirectionBasis")
    assert basis["arrow"] + basis["implied from neighbouring arrows"] >= 0.8 * len(segs)


# ------------------------------------------------------------------- flow map ---
@needs_routes
def test_flow_map_draws_routes_and_reviewer_edits(lightup):
    import copy

    from pid_dexpi_aas.flow_map import flow_map, review_overrides

    lightup = copy.deepcopy(lightup)  # the edits below must not leak into other tests
    diagram = lightup.drawio
    got = flow_map(diagram, LIGHTUP, overrides=review_overrides(lightup))
    assert got is not None
    _, stats = got
    assert stats["arrow"] > 30 and stats["signal"] > 0 and stats["reviewer"] == 0
    # reviewer reverses one directed pipe and excludes another
    directed = [c for c in lightup.connections
                if c.flow == "drawn arrow" and not c.id.startswith("net") and c.from_id != c.to_id]
    directed[0].from_id, directed[0].to_id, directed[0].flow = directed[0].to_id, directed[0].from_id, "reviewer"
    directed[1].include = False
    _, stats2 = flow_map(diagram, LIGHTUP, overrides=review_overrides(lightup))
    assert stats2["reviewer"] == 1 and stats2["excluded"] == 1 and stats2["arrow"] == stats["arrow"] - 2


@needs_routes
def test_map_markup_selects_by_row(lightup):
    from pid_dexpi_aas.flow_map import build_layers, map_markup

    layers = build_layers(lightup.drawio, LIGHTUP)
    net_line = next(ln for ln in layers.lines if any(r.startswith("net") for r in ln.rows))
    row = sorted(net_line.rows)[0]
    m = map_markup(layers, highlight_rows={row})
    # every line of that pipe network lights up, with the symbols at its ends, and the view zooms in
    lit = [ln for ln in layers.lines if row in ln.rows]
    assert len(lit) >= 1 and m["html"].count("rgb(230,0,200)") >= len(lit)
    assert m["view"] is not None and m["view"][2] < m["full"][2]
    assert f'data-rows="{"|".join(sorted(net_line.rows))}"' in m["html"]
    assert map_markup(layers)["view"] is None  # nothing selected: keep the user's zoom


# ----------------------------------------------------------------- picture input ---
def _sheet(ink, fill, bg_alpha=0):
    """A tiny transparent 'drawing': a filled box with a long line through it."""
    from PIL import Image, ImageDraw

    im = Image.new("RGBA", (200, 120), (0, 0, 0, bg_alpha))
    d = ImageDraw.Draw(im)
    d.rectangle((60, 30, 120, 90), fill=fill + (255,), outline=ink + (255,), width=2)
    d.line((0, 60, 199, 60), fill=ink + (255,), width=2)
    return im


def test_dark_theme_export_keeps_its_lines():
    # draw.io dark theme: white lines on transparent. Laid on white they vanished and the
    # model drew giant boxes around the remaining fills.
    import numpy as np
    from pid_dexpi_aas.sheet_image import to_dark_on_white

    out = np.asarray(to_dark_on_white(_sheet((255, 255, 255), (40, 40, 40))).convert("L"))
    assert out[60, 10] < 80 and out[5, 5] > 200  # the pipe is dark, the background white


def test_normal_export_with_white_fills_is_not_inverted():
    import numpy as np
    from pid_dexpi_aas.sheet_image import to_dark_on_white

    out = np.asarray(to_dark_on_white(_sheet((0, 0, 0), (255, 255, 255))).convert("L"))
    assert out[60, 10] < 80 and out[5, 5] > 200
