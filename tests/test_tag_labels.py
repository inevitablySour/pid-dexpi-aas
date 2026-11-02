"""Tag labels made in the editor: a text linked to its item's tag (the item's tag_cell)."""
from pid_dexpi_aas.drawio.parser import parse_drawio_xml
from pid_dexpi_aas.review import from_drawio

XML = '''<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<UserObject id="a" label="" tag="P-101" tag_auto="0" tag_cell="t"><mxCell style="shape=mxgraph.pid.pumps.centrifugal_pump_1;html=1;" vertex="1" parent="1">
  <mxGeometry x="0" y="0" width="40" height="40" as="geometry"/></mxCell></UserObject>
<mxCell id="b" value="" style="shape=mxgraph.pid.valves.gate;html=1;" vertex="1" parent="1">
  <mxGeometry x="300" y="0" width="40" height="20" as="geometry"/></mxCell>
<mxCell id="t" value="P-102" style="text;html=1;" vertex="1" parent="1">
  <mxGeometry x="270" y="25" width="50" height="20" as="geometry"/></mxCell>
</root></mxGraphModel>'''


def test_a_tag_label_belongs_to_its_item_wherever_it_is_and_reads_as_it_is_now():
    # the label was dragged next to the valve and edited to P-102 on the drawing
    tags = {s.id: s.tag for s in from_drawio(parse_drawio_xml(XML, "t.drawio")).symbols}
    assert tags["a"] == "P-102" and not tags.get("b")


def test_a_tag_label_on_a_picture_is_not_an_item():
    from pid_dexpi_aas.drawio.picture_editor import ITEM_LAYER, PICTURE_LAYER, apply_editor_drawing
    from pid_dexpi_aas.review import ReviewDiagram, ReviewSymbol
    review = ReviewDiagram(source_name="p.png", image_path="p.png")
    review.symbols.append(ReviewSymbol(id="a", source="image model", box=[0, 0, 40, 40], detected_as="pump",
                                       tag="P101", dexpi_class="Pump"))
    xml = XML.replace('parent="1"', f'parent="{ITEM_LAYER}"').replace(
        '<mxCell id="1" parent="0"/>', f'<mxCell id="{PICTURE_LAYER}" parent="0"/><mxCell id="{ITEM_LAYER}" parent="0"/>')
    apply_editor_drawing(review, xml, known_ids={"a"})
    rows = {s.id: s for s in review.symbols}
    assert "t" not in rows and rows["a"].tag == "P-102"
