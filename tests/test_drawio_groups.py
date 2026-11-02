"""A draw.io group (Ctrl+G in the Editor tab) is one item: someone dresses up an equipment
symbol with extra shapes, and the whole thing is one DEXPI piece of equipment."""
from pid_dexpi_aas.drawio.parser import parse_drawio_xml
from pid_dexpi_aas.review import from_drawio

TABLE = ('&lt;table cellpadding="4" cellspacing="0" border="0" style="font-size:1em;width:100%;height:100%;"&gt;'
         '&lt;tr&gt;&lt;td&gt;TI&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;101&lt;/td&gt;&lt;/table&gt; ').replace('"', '&quot;')

XML = f"""<mxGraphModel><root>
<mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="g" value="" style="group" vertex="1" connectable="0" parent="1">
  <mxGeometry x="200" y="100" width="80" height="60" as="geometry"/></mxCell>
<mxCell id="circle" value="{TABLE}" style="html=1;shape=mxgraph.pid2inst.discInst;mounting=field;" vertex="1" parent="g">
  <mxGeometry x="0" y="0" width="50" height="50" as="geometry"/></mxCell>
<mxCell id="extra" value="" style="rounded=0;whiteSpace=wrap;html=1;" vertex="1" parent="g">
  <mxGeometry x="50" y="40" width="30" height="20" as="geometry"/></mxCell>
<mxCell id="deco" style="endArrow=none;html=1;" edge="1" parent="g">
  <mxGeometry relative="1" as="geometry"><mxPoint x="50" y="25" as="sourcePoint"/>
  <mxPoint x="65" y="40" as="targetPoint"/></mxGeometry></mxCell>
<mxCell id="valve" value="XV1" style="shape=mxgraph.pid.valves.gate;html=1;" vertex="1" parent="1">
  <mxGeometry x="400" y="105" width="40" height="20" as="geometry"/></mxCell>
<mxCell id="pipe" style="endArrow=none;html=1;" edge="1" parent="1" source="circle" target="valve">
  <mxGeometry relative="1" as="geometry"/></mxCell>
</root></mxGraphModel>"""


def test_a_group_is_one_item_at_its_page_position():
    d = parse_drawio_xml(XML, "g.drawio")
    shapes = {s.id: s for s in d.shapes}
    assert set(shapes) == {"g", "valve"}  # the circle and the extra box are parts of g
    g = shapes["g"]
    assert (g.x, g.y, g.width, g.height) == (200, 100, 80, 60)
    assert g.shape_token == "mxgraph.pid2inst.discInst" and g.text == "TI 101"
    assert g.parts == ("circle", "extra")
    assert [e.id for e in d.edges] == ["pipe"]  # the line drawn inside the group is not a pipe
    assert (d.edges[0].source, d.edges[0].target) == ("g", "valve")


def test_a_group_reads_as_the_equipment_its_main_shape_is():
    review = from_drawio(parse_drawio_xml(XML, "g.drawio"))
    row = next(s for s in review.symbols if s.id == "g")
    assert row.tag.replace(" ", "") == "TI101"
    assert any({c.from_id, c.to_id} == {"g", "valve"} for c in review.connections)


def test_shapes_inside_a_container_are_placed_on_the_page_and_stay_items():
    xml = XML.replace('style="group"', 'style="rounded=0;container=1;"')
    shapes = {s.id: s for s in parse_drawio_xml(xml, "c.drawio").shapes}
    assert {"g", "circle", "extra"} <= set(shapes)
    assert (shapes["extra"].x, shapes["extra"].y) == (250, 140)  # relative to the container


def test_a_group_on_a_picture_is_one_item():
    """The same in a picture's review (the model's result in the editor): a symbol dropped on
    the picture and dressed up with another shape, grouped, is one item; a line to a part
    joins the item."""
    from pid_dexpi_aas.drawio.picture_editor import ITEM_LAYER, PICTURE_LAYER, apply_editor_drawing
    from pid_dexpi_aas.review import ReviewDiagram, ReviewSymbol
    review = ReviewDiagram(source_name="p.png", image_path="p.png")
    review.symbols.append(ReviewSymbol(id="m1", source="image model", box=[400, 100, 440, 120],
                                       detected_as="valve", dexpi_class="GateValve"))
    xml = f"""<mxGraphModel><root><mxCell id="0"/>
    <mxCell id="{PICTURE_LAYER}" parent="0"/><mxCell id="{ITEM_LAYER}" parent="0"/>
    <mxCell id="m1" value="" style="rounded=0;" vertex="1" parent="{ITEM_LAYER}">
      <mxGeometry x="400" y="100" width="40" height="20" as="geometry"/></mxCell>
    <mxCell id="g" value="" style="group" vertex="1" connectable="0" parent="{ITEM_LAYER}">
      <mxGeometry x="200" y="100" width="80" height="60" as="geometry"/></mxCell>
    <mxCell id="pv" value="" style="html=1;shape=mxgraph.pid.valves.gate;" vertex="1" parent="g">
      <mxGeometry x="0" y="0" width="60" height="30" as="geometry"/></mxCell>
    <mxCell id="px" value="" style="rounded=0;html=1;" vertex="1" parent="g">
      <mxGeometry x="60" y="40" width="20" height="20" as="geometry"/></mxCell>
    <mxCell id="L" style="endArrow=none;" edge="1" parent="{ITEM_LAYER}" source="pv" target="m1">
      <mxGeometry relative="1" as="geometry"/></mxCell>
    </root></mxGraphModel>"""
    apply_editor_drawing(review, xml, known_ids={"m1"})
    rows = {s.id: s for s in review.symbols}
    assert set(rows) == {"m1", "g"}                       # the parts are not items
    assert rows["g"].box == [200, 100, 280, 160] and rows["g"].dexpi_class == "GateValve" and rows["g"].checked
    assert [(c.from_id, c.to_id) for c in review.connections] in ([("g", "m1")], [("m1", "g")])


def test_a_group_is_the_symbol_in_it_not_the_biggest_shape():
    """A plain circle drawn round (or next to) an instrument is decoration: the group is the
    instrument, with its own tag (TI over 22) - not a new item named after the circle."""
    xml = XML.replace('<mxCell id="extra" value="" style="rounded=0;whiteSpace=wrap;html=1;" vertex="1" parent="g">\n  <mxGeometry x="50" y="40" width="30" height="20" as="geometry"/></mxCell>',
                      '<mxCell id="extra" value="" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="g">\n  <mxGeometry x="0" y="0" width="80" height="60" as="geometry"/></mxCell>')
    g = next(s for s in parse_drawio_xml(xml, "g.drawio").shapes if s.id == "g")
    assert g.shape_token == "mxgraph.pid2inst.discInst" and g.text == "TI 101"


def test_a_group_on_a_picture_is_drawn_again_when_the_editor_reopens():
    """The picture's drawing is rebuilt from the review each time the editor opens: a group
    comes back as the group it was (its shapes as drawn), still one item."""
    from pid_dexpi_aas.drawio.picture_editor import ITEM_LAYER, PICTURE_LAYER, apply_editor_drawing, review_to_editor_xml
    from pid_dexpi_aas.review import ReviewDiagram, ReviewSymbol
    review = ReviewDiagram(source_name="p.png", image_path="p.png", image_size=(1000, 800))
    xml = f"""<mxGraphModel><root><mxCell id="0"/>
    <mxCell id="{PICTURE_LAYER}" parent="0"/><mxCell id="{ITEM_LAYER}" parent="0"/>
    <mxCell id="g" value="" style="group" vertex="1" connectable="0" parent="{ITEM_LAYER}">
      <mxGeometry x="200" y="100" width="80" height="60" as="geometry"/></mxCell>
    <mxCell id="pv" value="" style="html=1;shape=mxgraph.pid.valves.gate;" vertex="1" parent="g">
      <mxGeometry x="0" y="0" width="60" height="30" as="geometry"/></mxCell>
    <mxCell id="px" value="" style="ellipse;html=1;" vertex="1" parent="g">
      <mxGeometry x="60" y="40" width="20" height="20" as="geometry"/></mxCell>
    </root></mxGraphModel>"""
    apply_editor_drawing(review, xml, known_ids=set())
    again = review_to_editor_xml(review, "pic.png")
    assert 'id="g"' in again and 'style="group"' in again and 'id="pv"' in again and 'id="px"' in again
    apply_editor_drawing(review, again, known_ids={"g"})
    assert [s.id for s in review.symbols] == ["g"] and review.symbols[0].dexpi_class == "GateValve"
