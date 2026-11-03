"""AutoCAD P&IDs (cad/to_drawio.py): a small DXF built here, in the style of a real plant P&ID
(symbols as blocks with their tags as attributes, equipment as an outline named by a label
inside it, pipes as separate line pieces). No company symbol table: general rules only."""
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

    def put(name, at, *values):
        ins = msp.add_blockref(name, at)
        if values:
            ins.add_auto_attribs(dict(zip([a.dxf.tag for a in doc.blocks.get(name).attdefs()], values)))
        return ins

    put("STREAM", (10, 50), "FEED")
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
    ids = dict(re.findall(r'<mxCell id="(b\d+|q\d+)" value="([^"]*)"', xml))
    by_tag = {v: k for k, v in ids.items()}
    links = set(re.findall(r'edge="1" parent="1"(?: source="(\w+)")?(?: target="(\w+)")?', xml))
    want = {(by_tag["FEED"], by_tag["AFS001"]), (by_tag["AFS001"], by_tag["P101"]), (by_tag["P101"], by_tag["T001"])}
    assert want <= links          # the valve - pump pipe runs through the flange
    assert "PI&lt;br&gt;101" in xml   # the instrument bubble carries its letters and number


def test_decoration_and_signal_lines(drawing):
    xml, report = drawing
    assert report["signal_lines"] == 1 and "dashed=1" in xml
    assert xml.count('edge="1"') == 6   # 4 process pipes (one through the flange), the tap, the outlet... and the signal line
