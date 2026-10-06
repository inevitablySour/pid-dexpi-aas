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
    assert flows[("V202", "GateValve")] == "implied"  # V202 outlet to the PC 201 valve (drawn as gate)
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


@needs_routes
def test_hidden_items_and_lines_leave_the_map(lightup):
    from pid_dexpi_aas.flow_map import build_layers, hide

    layers = build_layers(lightup.drawio, LIGHTUP)
    box = next(b for b in layers.boxes)
    direct = next(ln for ln in layers.lines if len(ln.rows) == 1)
    net = next(ln for ln in layers.lines if len(ln.rows) > 1)
    out = hide(layers, {box.id}, set(direct.rows) | {sorted(net.rows)[0]})
    assert box.id not in {b.id for b in out.boxes}
    assert direct not in out.lines
    assert net in out.lines  # a network piece stays while any of its pairs is still shown
    assert net not in hide(layers, set(), set(net.rows)).lines


# ------------------------------------------------------------ specification fields ---
def test_each_dexpi_class_gets_its_own_spec_fields():
    from pid_dexpi_aas.dexpi_fields import spec_fields

    names = lambda c: {f.name for f in spec_fields(c)}
    assert "nominalCapacityVolume" in names("Tank")
    assert {"designVolumeFlowRate", "differentialPressure"} <= names("Pump")
    assert "setPressureHigh" in names("SafetyValveOrFitting")
    assert "nominalCapacityVolume" not in names("Pump")
    assert next(f for f in spec_fields("Tank") if f.name == "nominalCapacityVolume").units[0] == "l"


@needs_routes
def test_reviewed_spec_values_reach_the_dexpi_file(lightup, tmp_path):
    import copy

    from pid_dexpi_aas.review import to_dexpi

    review = copy.deepcopy(lightup)
    v101 = next(s for s in review.symbols if s.tag == "V101")
    v101.attributes["dexpi"] = {"nominalCapacityVolume": {"value": 2.5, "unit": "l"}}
    v101.attributes["custom"] = {"Material": "316L"}
    _, result = to_dexpi(review)
    assert not result.spec_problems
    save_and_verify(result.model, tmp_path, "x")
    xml = next(tmp_path.glob("x*")).read_text()
    assert 'Value="2.5" Units="Litre"' in xml and 'Value="316L"' in xml


# ------------------------------------------------------------------- pipe tracer ---
def test_tracer_follows_a_pipe_and_reads_its_arrowhead():
    # Runs where scipy + scikit-image are installed (the ML venv; skipped otherwise).
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    import sys

    import numpy as np
    from PIL import Image, ImageDraw

    sys.path.insert(0, str(ROOT / "ml" / "pid2graph"))
    from trace_pipes import trace_connections

    im = Image.new("L", (600, 300), 255)
    d = ImageDraw.Draw(im)
    a, b = (40, 110, 120, 190), (480, 110, 560, 190)
    d.rectangle(a, outline=0, width=2)
    d.rectangle(b, outline=0, width=2)
    d.line((120, 150, 300, 150), fill=0, width=2)   # a pipe with a bend, into b
    d.line((300, 150, 300, 60), fill=0, width=2)
    d.line((300, 60, 440, 60), fill=0, width=2)
    d.line((440, 60, 440, 150), fill=0, width=2)
    d.line((440, 150, 480, 150), fill=0, width=2)
    d.polygon([(480, 150), (468, 144), (468, 156)], fill=0)  # arrowhead pointing into b
    out = trace_connections(np.asarray(im), [a, b])
    assert len(out) == 1 and {out[0]["a"], out[0]["b"]} == {0, 1}
    c = out[0]
    assert (c["flow"] == 1) == (c["b"] == 1)  # flows towards b
    assert len(c["path"]) > 20


def test_equipment_tag_goes_to_the_tank_not_the_instrument_tag():
    # "FI 102" sat closer to the tank box than "V101" and took it, leaving V101 unnamed
    from pid_dexpi_aas.ocr_tags import OcrTag, attach_tags

    boxes = {"tank": (100, 100, 300, 200), "bubble": (320, 60, 380, 120)}
    tags = [OcrTag("FI 102", (290, 95, 330, 110), 0.9, ["FI 102"]),   # on the tank's edge
            OcrTag("V101", (180, 140, 220, 160), 0.9, ["V101"])]      # inside the tank
    got = attach_tags(boxes, tags, classes={"tank": "tank", "bubble": "instrumentation"})
    assert got["tank"][0].text == "V101" and got["bubble"][0].text == "FI 102"


@needs_routes
def test_map_with_a_selection_it_does_not_draw(lightup):
    # selecting a picture-only item (or a line another tab draws) crashed the other maps
    from pid_dexpi_aas.flow_map import build_layers, map_markup

    layers = build_layers(lightup.drawio, LIGHTUP)
    m = map_markup(layers, highlight_rows={"not-on-this-map"}, highlight_symbols={"picture-only-item"})
    assert m["view"] is None and m["focus"] is None and "svg" in m["html"]


# --------------------------------------------------------------------- AAS export ---
@needs_routes
@pytest.mark.parametrize("version", ["3.0", "3.1"])
def test_aasx_follows_idta_02012_and_maps_every_tag(lightup, tmp_path, version):
    pytest.importorskip("aas_test_engines")
    import io
    import zipfile

    from basyx.aas.adapter import aasx

    from pid_dexpi_aas.aas_export import DexpiModelInput, aasx_bytes, check_aasx, mapping_entries
    from pid_dexpi_aas.review import to_dexpi

    _, result = to_dexpi(lightup)
    save_and_verify(result.model, tmp_path, "x")
    xml = next(tmp_path.glob("x*")).read_bytes()
    entries = mapping_entries(xml)
    tags = {e.tag for e in entries if e.kind == "TagMapping"}
    assert {"V101", "P101", "GV101", "PSV201", "PI 202"} <= tags  # equipment, valves, safety valves, instruments
    assert any(e.kind == "SubTagMapping" for e in entries)  # nozzles
    data = aasx_bytes([DexpiModelInput("P&ID gramflow reactor V3", xml)], {"ProcessPlantName": "LightUp"},
                      version=version)
    ok, problems = check_aasx(data)
    assert ok, problems
    if version == "3.1":
        return  # BaSyx 2.0.x reads 3.0 only; 3.1 is verified by aas-core3.1 in check_aasx
    # reads back: the DEXPI file is inside and the Submodel carries the template's structure
    from basyx.aas import model as aas_model

    objects, files = aas_model.DictObjectStore(), aasx.DictSupplementaryFileContainer()
    with aasx.AASXReader(io.BytesIO(data)) as reader:
        reader.read_into(objects, files)
    sm = next(o for o in objects if getattr(o, "id_short", "") == "DEXPI")
    model01 = sm.get_referable("Model01")
    assert {e.id_short for e in model01} >= {"ModelMetadata", "ModelFile", "MappingDirectory"}
    assert len(list(model01.get_referable("MappingDirectory"))) == len({e.local_id for e in entries})
    assert any(n.endswith(".xml") for n in zipfile.ZipFile(io.BytesIO(data)).namelist())


# ---------------------------------------------------------------- plant sections ---
@needs_routes
def test_section_frames_are_sections_not_pipes():
    d = parse_drawio_file(COMBINED)
    names = {s.name for s in d.sections}
    assert names == {"Autoclave", "Back pressure regulator with collecting vessel",
                     "N2 - H2 Containers and precautionary measures", "Peristaltic pump with precautionary measures",
                     "Piston, HPLC pump and injection point", "Reactor and lamp"}
    frames = [e for e in d.edges if e.is_section_frame]
    assert frames and not any(e.merged_into or e.source or e.target for e in frames)  # never pipes
    review = from_drawio(d)
    sec = {s.tag: s.section for s in review.symbols}
    assert sec["Reactor R301"] == sec["Mixing"] == sec["TIC 301"] == "Reactor and lamp"
    assert {s.dexpi_class for s in review.symbols if s.tag == "Reactor R301"} == {"Vessel"}  # pipes end on it


@needs_routes
def test_small_titled_frame_and_symbol_parts():
    # "Injection point" frames one item; the columns' inner boxes on LightUp are parts of the
    # column symbol, not items
    d = parse_drawio_file(ROOT / "subprojects" / "Lumen" / "P&ID Lumen New set-up nodes 081122 (1).drawio")
    assert "Injection point" in {s.name for s in d.sections}
    lu = parse_drawio_file(LIGHTUP)
    assert not [s for s in lu.shapes if (s.shape_token or "").startswith("plain:")] and not lu.sections


def test_router_goes_around_items_with_right_angles():
    from pid_dexpi_aas.routing import route

    a, b, wall = (0, 0, 40, 40), (300, 0, 340, 40), (140, -60, 200, 100)
    pts = route(a, b, [a, b, wall])
    assert all(abs(p[0] - q[0]) < 1e-6 or abs(p[1] - q[1]) < 1e-6 for p, q in zip(pts, pts[1:]))
    seg_hits = lambda p, q: (min(p[0], q[0]) < wall[2] and max(p[0], q[0]) > wall[0]
                             and min(p[1], q[1]) < wall[3] and max(p[1], q[1]) > wall[1])
    assert not any(seg_hits(p, q) for p, q in zip(pts, pts[1:]))


def test_box_drawn_by_reviewer_is_built_and_its_new_line_routed():
    """An item drawn on the map (Add item) goes into DEXPI with its tag, and a line added to
    it is drawn with right angles, not straight across the sheet."""
    from pid_dexpi_aas.review import ReviewConnection, ReviewSymbol, to_dexpi

    review = from_drawio(parse_drawio_file(LIGHTUP))
    other = next(s for s in review.symbols if s.box and s.include)
    review.symbols.append(ReviewSymbol(id="drawn1", source="manual", box=[0, 0, 40, 30], detected_as="drawn by you",
                                       tag="T-999", dexpi_class="Tank", identified_by="box drawn by you"))
    review.connections.append(ReviewConnection(id="manual_c1", from_id="drawn1", to_id=other.id, source="manual"))
    diagram, result = to_dexpi(review)
    assert result.shape_objects["drawn1"].tagName == "T-999"
    pts = diagram.routes["manual_c1"]
    assert len(pts) >= 2 and all(abs(p[0] - q[0]) < 1e-6 or abs(p[1] - q[1]) < 1e-6 for p, q in zip(pts, pts[1:]))


LUMEN_NODES = ROOT / "subprojects" / "Lumen" / "P&ID Lumen New set-up nodes 081122 (1).drawio"


@pytest.mark.skipif(line_routes._find_drawio() is None, reason="needs the draw.io app for line routes")
def test_line_ending_at_a_stream_name_is_a_stream_not_a_loop():
    """The Lumen product line starts on V301 and ends 20 from the text "Product" (72 from
    V301): a FlowOut stream, not a loop back onto V301; "Product" is no valve's tag."""
    diagram = parse_drawio_file(LUMEN_NODES)
    assert not [e.id for e in diagram.edges if e.source and e.source == e.target and e.source_point]
    result = build_model(diagram)
    product = next(s for s in diagram.shapes if s.text == "Product")
    assert isinstance(result.shape_objects[product.id], piping.FlowOutPipeOffPageConnector)
    tags = {s.tag for s in from_drawio(diagram).symbols if s.dexpi_class != "(off-page connector)"}
    assert "Product" not in tags and "N 2" not in tags


def _tracer():
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    import sys

    sys.path.insert(0, str(ROOT / "ml" / "pid2graph"))
    from trace_pipes import trace_connections
    return trace_connections


def test_tracer_joins_symbols_with_a_short_line_between_them():
    """Stacked instruments: the line is shorter than the erased margins around both."""
    import numpy as np
    from PIL import Image, ImageDraw

    trace = _tracer()
    im = Image.new("L", (400, 400), 255)
    d = ImageDraw.Draw(im)
    a, b = (150, 60, 230, 140), (150, 152, 230, 232)
    d.ellipse(a, outline=0, width=2)
    d.ellipse(b, outline=0, width=2)
    d.line((190, 140, 190, 152), fill=0, width=2)
    assert {(c["a"], c["b"]) for c in trace(np.asarray(im), [a, b])} == {(0, 1)}


def test_tracer_follows_a_pipe_through_a_label_written_on_it():
    import numpy as np
    from PIL import Image, ImageDraw

    trace = _tracer()
    im = Image.new("L", (600, 300), 255)
    d = ImageDraw.Draw(im)
    a, b = (40, 110, 120, 190), (480, 110, 560, 190)
    d.rectangle(a, outline=0, width=2)
    d.rectangle(b, outline=0, width=2)
    d.line((120, 150, 480, 150), fill=0, width=2)
    label = (250, 135, 350, 165)  # "17 bar" on the pipe, erased as text
    d.rectangle(label, fill=255)
    d.text((260, 140), "17 bar", fill=0)
    assert {(c["a"], c["b"]) for c in trace(np.asarray(im), [a, b], [label])} == {(0, 1)}


def test_tracer_does_not_attach_a_frame_running_past_a_symbol():
    """A section frame passing close alongside a symbol is not a pipe into it."""
    import numpy as np
    from PIL import Image, ImageDraw

    trace = _tracer()
    im = Image.new("L", (700, 400), 255)
    d = ImageDraw.Draw(im)
    a, b, c = (100, 200, 180, 280), (300, 200, 380, 280), (500, 200, 580, 280)
    for box in (a, b, c):
        d.rectangle(box, outline=0, width=2)
    d.line((180, 240, 300, 240), fill=0, width=2)  # the real pipe a -> b
    d.line((20, 195, 680, 195), fill=0, width=3)   # frame edge 5 px above all three
    pairs = {(x["a"], x["b"]) for x in trace(np.asarray(im), [a, b, c])}
    assert pairs == {(0, 1)}


# ------------------------------------------------------------------ symbol refinement ---
def _refine():
    pytest.importorskip("scipy")
    import sys

    sys.path.insert(0, str(ROOT / "ml" / "pid2graph"))
    sys.path.insert(0, str(ROOT / "src"))
    from refine_symbols import refine
    return refine


def _sym(i, box, cls, score=0.9):
    return {"id": f"m{i}", "box": list(box), "class": cls, "score": score, "class_votes": {cls: 1}, "n_tiles": 1}


def test_refine_finds_a_piped_container_and_its_compartments_but_not_a_frame():
    import numpy as np
    from PIL import Image, ImageDraw

    refine = _refine()
    im = Image.new("L", (1400, 900), 255)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((100, 100, 700, 400), radius=30, outline=0, width=2)  # reactor
    d.rectangle((150, 180, 300, 330), outline=0, width=2)                      # compartments
    d.rectangle((500, 180, 650, 330), outline=0, width=2)
    d.line((400, 400, 400, 520), fill=0, width=2)                              # pipe ending on it
    d.rectangle((360, 520, 440, 600), outline=0, width=2)                      # pump below
    d.rectangle((850, 100, 1350, 450), outline=0, width=2)                     # section frame
    v1, v2 = (900, 250, 980, 330), (1200, 250, 1280, 330)
    d.rectangle(v1, outline=0, width=2)
    d.rectangle(v2, outline=0, width=2)
    d.line((980, 290, 1200, 290), fill=0, width=2)
    d.line((1280, 290, 1390, 290), fill=0, width=2)                            # crosses the frame
    syms = [_sym(0, (360, 520, 440, 600), "pump"), _sym(1, v1, "valve"), _sym(2, v2, "valve")]
    out, _ = refine(np.asarray(im), syms, [])
    added = sorted((s["class"], tuple(round(v) for v in s["box"])) for s in out if s["id"].startswith("c"))
    assert [c for c, _ in added] == ["general", "general", "tank"]
    tank = next(b for c, b in added if c == "tank")
    assert abs(tank[0] - 100) <= 3 and abs(tank[2] - 700) <= 3  # the reactor, not the frame


def test_refine_fits_a_loose_tank_box_and_merges_its_fragments():
    import numpy as np
    from PIL import Image, ImageDraw

    refine = _refine()
    im = Image.new("L", (800, 800), 255)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((300, 150, 420, 600), radius=40, outline=0, width=2)   # a tall vessel
    d.line((360, 600, 360, 700), fill=0, width=2)
    syms = [_sym(0, (300, 150, 420, 380), "tank"), _sym(1, (300, 370, 420, 600), "tank"),
            _sym(2, (100, 100, 180, 180), "valve")]
    out, renamed = refine(np.asarray(im), syms, [])
    tanks = [s for s in out if s["class"] == "tank"]
    assert len(tanks) == 1 and renamed
    assert [round(v) for v in tanks[0]["box"]] == [299, 149, 421, 601] or \
        all(abs(a - b) <= 3 for a, b in zip(tanks[0]["box"], (300, 150, 420, 600)))


def test_refine_lets_a_tag_inside_a_box_set_its_class_but_not_an_off_page_arrow():
    import numpy as np

    refine = _refine()
    gray = np.full((400, 600), 255, np.uint8)
    syms = [_sym(0, (100, 100, 180, 180), "valve"), _sym(1, (300, 100, 420, 150), "inlet/outlet")]
    tokens = [{"text": "P101", "conf": 0.99, "box": [115, 130, 165, 150], "angle": 0},
              {"text": "V204", "conf": 0.99, "box": [320, 115, 380, 135], "angle": 0}]
    out, _ = refine(gray, syms, tokens, fixes=("tag",))
    assert [s["class"] for s in out] == ["pump", "inlet/outlet"]
