"""AutoCAD P&IDs (cad/to_drawio.py): a small DXF built here, in the style of a real plant P&ID
(symbols as blocks with their tags as attributes, equipment as an outline named by a label
inside it, pipes as separate line pieces). No company symbol table: general rules only."""
import json
import re

import ezdxf
import pytest

from pid_dexpi_aas.cad.to_drawio import dxf_to_drawio


def _block(doc, name, draw, tags=()):
    blk = doc.blocks.new(name)
    draw(blk)
    for k, t in enumerate(tags):
        blk.add_attdef(t, (0, -6 - 3 * k), dxfattribs={"height": 2})
    return blk


@pytest.fixture(scope="module")
def drawing():
    doc = ezdxf.new()
    doc.linetypes.add("HIDDEN", pattern=[5, 3, -2])
    doc.layers.add("SIGNAL", linetype="HIDDEN")
    msp = doc.modelspace()
    _block(doc, "VALVE", lambda b: (b.add_lwpolyline([(-4, -2), (4, 2), (4, -2), (-4, 2)], close=True)), ["TAG"])
    _block(doc, "PUMP", lambda b: b.add_circle((0, 0), 5), ["TAG"])
    _block(doc, "BUBBLE", lambda b: b.add_circle((0, 0), 4), ["LETTERS", "NUMBER"])
    _block(doc, "STREAM", lambda b: b.add_lwpolyline([(-10, -3), (8, -3), (10, 0), (8, 3), (-10, 3)], close=True), ["NAME"])
    _block(doc, "FLANGE", lambda b: (b.add_line((-1, -3), (-1, 3)), b.add_line((1, -3), (1, 3))))
    _block(doc, "LABEL", lambda b: None, ["NAME"])
    # a filled flow arrow built of three triangles (tip at +x), lying on the pipe
    _block(doc, "ARROW", lambda b: [b.add_lwpolyline(t, close=True) for t in (
        [(-1.5, -1.25), (-1.5, -0.4), (1.5, 0)], [(-1.5, -0.4), (-1.5, 0.4), (1.5, 0)], [(-1.5, 0.4), (-1.5, 1.25), (1.5, 0)])])

    def put(name, at, *values):
        ins = msp.add_blockref(name, at)
        if values:
            ins.add_auto_attribs(dict(zip([a.dxf.tag for a in doc.blocks.get(name).attdefs()], values)))
        return ins

    put("STREAM", (10, 50), "FEED")
    put("ARROW", (28, 50))   # FEED flows into the plant
    put("VALVE", (40, 50), "AFS001")
    put("FLANGE", (60, 50))
    put("PUMP", (80, 50), "%%UP101")
    put("BUBBLE", (50, 75), "PI", "101")
    msp.add_lwpolyline([(110, 30), (140, 30), (140, 70), (110, 70)], close=True)   # tank T001 as an outline
    put("LABEL", (115, 50), "%%UT001")
    for a, b in [((20, 50), (36, 50)), ((44, 50), (59, 50)), ((61, 50), (75, 50)), ((85, 50), (110, 50)),
                 ((50, 71), (50, 50)),              # the instrument's line ends on the pipe: a tap
                 ((140, 40), (170, 40))]:           # a line out of the tank into open space: an outlet
        msp.add_line(a, b)
    msp.add_line((50, 79), (50, 90), dxfattribs={"layer": "SIGNAL"})   # a dashed signal line
    msp.add_line((49, 84), (51, 86))                                      # its "//" decoration
    return dxf_to_drawio(doc, "test")


def test_symbols_by_general_rules(drawing):
    _, report = drawing
    assert report["symbols"] == {"stream": 1, "valve": 1, "equipment": 2, "instrument": 1}
    assert report["through"] == ["FLANGE"]


def test_pipes_attach_to_their_symbols(drawing):
    xml, _ = drawing
    ids = dict(re.findall(r'<(?:mxCell|object) id="(b\d+|q\d+)" (?:value|label)="([^"]*)"', xml))
    by_tag = {v: k for k, v in ids.items()}
    links = set(re.findall(r'edge="1" parent="1"(?: source="(\w+)")?(?: target="(\w+)")?', xml))
    want = {(by_tag["FEED"], by_tag["AFS001"]), (by_tag["AFS001"], by_tag["P101"]), (by_tag["P101"], by_tag["T001"])}
    assert want <= links          # the valve - pump pipe runs through the flange
    assert "PI&lt;br&gt;101" in xml   # the instrument bubble carries its letters and number
    assert 'dexpi_class="Pump"' in xml and 'dexpi_class="ProcessInstrumentationFunction"' in xml
    assert "shape=image;image=data:image/svg+xml," in xml   # drawn with its own CAD symbol


def test_decoration_and_signal_lines(drawing):
    xml, report = drawing
    assert report["signal_lines"] == 1 and "dashed=1" in xml
    assert xml.count('edge="1"') == 6   # 4 process pipes (one through the flange), the tap, the outlet... and the signal line


def test_flow_from_an_arrow_on_the_pipe(drawing):
    xml, report = drawing
    ids = dict(re.findall(r'<(?:mxCell|object) id="(b\d+|q\d+)" (?:value|label)="([^"]*)"', xml))
    by_tag = {v: k for k, v in ids.items()}
    edge = re.search(r'<mxCell id="l\d+" style="([^"]*)" edge="1" parent="1" source="%s" target="%s"' % (by_tag["FEED"], by_tag["AFS001"]), xml)
    assert edge and "endArrow=block" in edge.group(1)   # drawn FEED -> valve, arrowhead at the valve
    assert report["flow_from_arrows"] == 1


def test_units_planned_items_and_data_tables():
    """A frame that a pipe ends on is one unit; dashed grey symbols and pipes are planned (left
    out of DEXPI); an equipment data table fills the equipment's specifications."""
    doc = ezdxf.new()
    doc.linetypes.add("HIDDEN", pattern=[5, 3, -2])
    msp = doc.modelspace()
    _block(doc, "STREAM", lambda b: b.add_lwpolyline([(-10, -3), (8, -3), (10, 0), (8, 3), (-10, 3)], close=True), ["NAME"])
    _block(doc, "VALVE", lambda b: b.add_lwpolyline([(-4, -2), (4, 2), (4, -2), (-4, 2)], close=True), ["TAG"])
    _block(doc, "LABEL", lambda b: None, ["NAME"])
    table = doc.blocks.new("TABLE")
    for k, (label, unit) in enumerate([("Tag", ""), ("Name", ""), ("Volume:", "m"), ("Ontwerpdruk:", "bar"),
                                       ("Ontwerptemp.:", "°C"), ("Medium:", ""), ("Materiaal:", "")]):
        y = -6 * k
        table.add_text(label, dxfattribs={"insert": (0, y), "height": 2})
        table.add_attdef(f"F{k}", (40, y), dxfattribs={"height": 2})
        if unit:
            table.add_text(unit, dxfattribs={"insert": (50, y), "height": 2})
    table.add_text("3", dxfattribs={"insert": (52, 1), "height": 1})   # m + superscript 3
    msp.add_lwpolyline([(100, 20), (140, 20), (140, 60), (100, 60)], close=True)     # the vessel's outline
    lab = msp.add_blockref("LABEL", (110, 40)); lab.add_auto_attribs({"NAME": "%%UV010"})
    msp.add_lwpolyline([(0, 100), (60, 100), (60, 140), (0, 140)], close=True)        # a package unit's frame
    msp.add_text("SKID", dxfattribs={"insert": (20, 135), "height": 3})
    for k in range(2):                                                               # two things inside it
        msp.add_blockref("VALVE", (20 + 20 * k, 120)).add_auto_attribs({"TAG": f"AFS00{k + 1}"})
    msp.add_blockref("STREAM", (-40, 110)).add_auto_attribs({"NAME": "WATER"})
    msp.add_line((-30, 110), (0, 110))                                               # into the unit's frame
    msp.add_line((60, 120), (100, 40))                                               # unit -> vessel
    grey = {"linetype": "HIDDEN", "color": 252}
    pv = msp.add_blockref("VALVE", (120, 90), dxfattribs=grey); pv.add_auto_attribs({"TAG": "AFS099"})
    for ent in doc.blocks.get("VALVE"):
        ent.dxf.linetype, ent.dxf.color = "BYBLOCK", 0
    msp.add_line((120, 60), (120, 88), dxfattribs=grey)                              # a planned pipe to it
    t = msp.add_blockref("TABLE", (200, 50))
    t.add_auto_attribs({"F0": "V010", "F1": "VAT 10", "F2": "12", "F3": "8", "F4": "120", "F5": "WATER",
                        "F6": "XXXXXX"})
    xml, report = dxf_to_drawio(doc, "t")
    assert report["units"] == ["SKID"]
    assert report["tables"] == ["V010"]
    assert report["future"] >= 1 and 'in_dexpi="0"' in xml
    m = re.search(r"label=\"V010\"[^>]*specs='([^']*)'", xml)
    specs = json.loads(m.group(1))
    assert specs["dexpi"]["nominalCapacityVolume"] == {"unit": "m3", "value": 12.0}
    assert specs["custom"]["DesignPressure"] == {"unit": "bar", "value": 8.0}
    assert specs["custom"]["Medium"] == "WATER" and "Material" not in str(specs)   # XXXXXX: a placeholder


def test_data_box_names_a_symbol_and_line_numbers_go_on_pipes():
    """A data box against an untagged symbol names it ("PSV-001 / 5 BARG": a safety valve set at
    5 bar); a line number label beside a pipe goes onto that pipe, with its size."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    _block(doc, "STREAM", lambda b: b.add_lwpolyline([(-10, -3), (8, -3), (10, 0), (8, 3), (-10, 3)], close=True), ["NAME"])
    _block(doc, "SAFETY", lambda b: b.add_lwpolyline([(-2, -4), (2, -4), (0, 4)], close=True))
    box = _block(doc, "DATABOX", lambda b: b.add_lwpolyline([(-6, 0), (6, 0), (6, 9), (-6, 9)], close=True),
                 ["TAG", "NUMBER", "SET"])
    _block(doc, "LABEL", lambda b: None, ["TEXT"])
    msp.add_lwpolyline([(-50, -150), (250, -150), (250, 150), (-50, 150)], close=True)   # the sheet's frame
    msp.add_blockref("STREAM", (0, 0)).add_auto_attribs({"NAME": "FEED"})
    msp.add_blockref("SAFETY", (40, 0))
    msp.add_blockref("DATABOX", (40, 4)).add_auto_attribs({"TAG": "PSV-001", "NUMBER": "1-234", "SET": "5 BARG"})
    msp.add_line((10, 0), (38, 0))
    msp.add_blockref("LABEL", (20, 2)).add_auto_attribs({"TEXT": "LINE-001-D80"})
    xml, report = dxf_to_drawio(doc, "t")
    m = re.search(r"label=\"PSV-001\"[^>]*specs='([^']*)'", xml)
    assert m and json.loads(m.group(1))["dexpi"]["setPressureHigh"] == {"unit": "bar", "value": 5.0}
    assert report["line_numbers"] == 1 and "LINE-001-D80" in xml and "DN80" in xml


def test_a_symbol_legend_overrides_the_rules(drawing):
    """A person's legend sets a symbol type's DEXPI class for every symbol of that type; a type
    left out stays in the legend's list, to be put back."""
    from pid_dexpi_aas.cad.to_drawio import LEAVE_OUT
    xml, report = drawing
    rows = {r["block"]: r["dexpi_class"] for r in report["legend"]}
    assert rows["PUMP"] == "Pump" and rows["FLANGE"] == "(in-line: the pipe runs through)"
    assert all(r["svg"].startswith("<svg") for r in report["legend"])   # each type with its picture


def test_legend_choices_apply(tmp_path):
    import ezdxf as _ez
    doc = _ez.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(-50, -150), (250, -150), (250, 150), (-50, 150)], close=True)
    _block(doc, "THING", lambda b: b.add_circle((0, 0), 3))
    msp.add_blockref("THING", (20, 0)); msp.add_blockref("THING", (60, 0))
    msp.add_line((-30, 0), (17, 0)); msp.add_line((23, 0), (57, 0)); msp.add_line((63, 0), (100, 0))
    _, plain = dxf_to_drawio(doc, "t")
    assert {r["block"]: r["dexpi_class"] for r in plain["legend"]}["THING"] == "(in-line: the pipe runs through)"
    xml, rep = dxf_to_drawio(doc, "t", {"THING": "BallValve"})
    assert xml.count('dexpi_class="BallValve"') == 2
    _, out = dxf_to_drawio(doc, "t", {"THING": "(leave out)"})
    assert {r["block"]: r["dexpi_class"] for r in out["legend"]}["THING"] == "(leave out)"


def test_a_grey_line_with_ticks_is_a_planned_signal_line():
    """A dashed grey line carrying "//" marks between two instruments is a planned signal line
    (not a pipe), drawn dashed; its end a little short of a bubble (a dashed line's gap) is on it."""
    doc = ezdxf.new()
    doc.linetypes.add("HIDDEN", pattern=[5, 3, -2])
    doc.layers.add("PLANNED", linetype="HIDDEN", color=252)
    msp = doc.modelspace()
    _block(doc, "BUBBLE", lambda b: b.add_circle((0, 0), 4), ["LETTERS", "NUMBER"])
    _block(doc, "TICK", lambda b: (b.add_line((-1.2, -1), (0.2, 1)), b.add_line((-0.2, -1), (1.2, 1))))
    msp.add_lwpolyline([(-50, -150), (250, -150), (250, 150), (-50, 150)], close=True)   # the sheet's frame
    planned = {"layer": "PLANNED"}
    msp.add_blockref("BUBBLE", (0, 0), dxfattribs=planned).add_auto_attribs({"LETTERS": "FI", "NUMBER": "201"})
    msp.add_blockref("BUBBLE", (100, 0), dxfattribs=planned).add_auto_attribs({"LETTERS": "IC", "NUMBER": "104"})
    msp.add_line((6, 0), (96, 0), dxfattribs=planned)   # 2 mm short of FI's bubble
    for x in (30, 60):
        msp.add_blockref("TICK", (x, 0), dxfattribs=planned)
    xml, report = dxf_to_drawio(doc, "t")
    assert report["signal_lines"] == 1
    edge = re.search(r'<mxCell style="([^"]*)" edge="1" parent="1" source="(\w+)" target="(\w+)"', xml)
    assert edge and "dashed=1" in edge.group(1)
