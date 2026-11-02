"""The editor's guided check: what counts as an uncertain connection, and the checked marks
travelling with the drawing back into the review."""
from pid_dexpi_aas.drawio.editor_data import editor_patch
from pid_dexpi_aas.drawio.parser import parse_drawio_xml
from pid_dexpi_aas.review import ReviewConnection, from_drawio, low_confidence


def test_uncertain_connections():
    c = lambda **k: ReviewConnection(id="c", from_id="a", to_id="b", **k)
    assert low_confidence(c(source="image model", confidence="0.91"))          # predicted only
    assert not low_confidence(c(source="picture tracing", confidence="drawn line followed"))
    assert low_confidence(c(source="draw.io", confidence="nearest_shape"))     # an end guessed
    assert low_confidence(c(source="draw.io", confidence="junction+guess"))
    assert not low_confidence(c(source="draw.io", confidence="explicit"))
    assert not low_confidence(c(source="manual", confidence="drawn in the editor"))


XML = '''<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<UserObject id="v1" label="XV1" checked="1"><mxCell style="shape=mxgraph.pid.valves.gate;html=1;" vertex="1" parent="1">
  <mxGeometry x="0" y="0" width="40" height="20" as="geometry"/></mxCell></UserObject>
<mxCell id="v2" value="XV2" style="shape=mxgraph.pid.valves.gate;html=1;" vertex="1" parent="1">
  <mxGeometry x="200" y="0" width="40" height="20" as="geometry"/></mxCell>
<UserObject id="e1" label="" checked="1"><mxCell style="endArrow=none;html=1;" edge="1" parent="1" source="v1" target="v2">
  <mxGeometry relative="1" as="geometry"/></mxCell></UserObject>
</root></mxGraphModel>'''


def test_checked_marks_come_back_from_the_editor_and_go_out_again():
    review = from_drawio(parse_drawio_xml(XML, "g.drawio"))
    rows = {r.id: r for r in [*review.symbols, *review.connections]}
    assert rows["v1"].checked and not rows["v2"].checked and rows["e1"].checked
    patch = editor_patch(review)
    assert patch["v1"]["attrs"]["checked"] == "1" and patch["v2"]["attrs"]["checked"] == "0"
    assert patch["e1"]["attrs"]["checked"] == "1" and patch["e1"]["attrs"]["low"] == "0"


def test_flow_arrows_are_not_items():
    """A flow arrow the model found only sets its pipe's flow: hidden in the editor (and kept
    out of the check); once a person gives it a real type it is an item like any other."""
    from pid_dexpi_aas.review import SKIP, ReviewDiagram, ReviewSymbol, is_flow_arrow
    r = ReviewDiagram(source_name="p.png", image_path="p.png")
    r.symbols = [ReviewSymbol(id="a", source="image model", box=[0, 0, 10, 10], detected_as="arrow", dexpi_class=SKIP),
                 ReviewSymbol(id="b", source="image model", box=[0, 0, 10, 10], detected_as="arrow", dexpi_class="GateValve")]
    assert is_flow_arrow(r.symbols[0]) and not is_flow_arrow(r.symbols[1])
    patch = editor_patch(r)
    assert patch["a"]["attrs"]["arrow"] == "1" and patch["b"]["attrs"]["arrow"] == "0"
