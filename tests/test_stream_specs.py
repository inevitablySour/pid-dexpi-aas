"""A stream's conditions (feed pressure, temperature, flow ...): custom attributes on its DEXPI
off-page connector(s) - DEXPI 1.x has no typed fields for them - and its fluid and temperature
on its pipe's own typed fields."""
import io
import tempfile
from pathlib import Path

from openpyxl import load_workbook

from pid_dexpi_aas.dexpi.fields import spec_fields
from pid_dexpi_aas.dexpi.serialize import save_and_verify
from pid_dexpi_aas.dexpi.spec_sheet import template_xlsx
from pid_dexpi_aas.drawio.parser import parse_drawio_xml
from pid_dexpi_aas.review import OFF_PAGE, from_drawio, to_dexpi

XML = '''<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="n2" value="N2" style="text;html=1;" vertex="1" parent="1"><mxGeometry x="0" y="0" width="40" height="20" as="geometry"/></mxCell>
<mxCell id="v" value="V101" style="shape=mxgraph.pid.vessels.tank;html=1;" vertex="1" parent="1">
  <mxGeometry x="300" y="-20" width="60" height="80" as="geometry"/></mxCell>
<mxCell id="p" style="endArrow=classic;html=1;" edge="1" parent="1" source="n2" target="v"><mxGeometry relative="1" as="geometry"/></mxCell>
</root></mxGraphModel>'''


def test_stream_conditions_go_into_dexpi():
    review = from_drawio(parse_drawio_xml(XML, "s.drawio"))
    s = next(x for x in review.symbols if x.id == "n2")
    assert s.dexpi_class == OFF_PAGE and "OperatingPressure" in {f.name for f in spec_fields(OFF_PAGE)}
    s.attributes = {"custom": {"OperatingPressure": {"value": 3.0, "unit": "bar"}, "Fluid": "N2",
                               "OperatingTemperature": {"value": 20.0, "unit": "°C"}}}
    _, result = to_dexpi(review)
    with tempfile.TemporaryDirectory() as tmp:
        save_and_verify(result.model, Path(tmp), "t")
        xml = (Path(tmp) / "t.xml").read_text()
    assert 'Name="OperatingPressure"' in xml and 'Value="3 bar"' in xml          # on the FlowIn connector
    seg = result.model.conceptualModel.pipingNetworkSystems[0].segments[0]
    assert seg.fluidCode == "N2" and seg.operatingTemperature.value == 20.0       # and on its pipe


def test_the_spec_sheet_lists_streams_with_their_fields():
    from pid_dexpi_aas.dexpi.spec_sheet import STREAM_TYPE
    review = from_drawio(parse_drawio_xml(XML, "s.drawio"))
    wb = load_workbook(io.BytesIO(template_xlsx(review)))
    ws = wb["Items"]
    row = next(r for r in range(2, 6) if ws.cell(row=r, column=1).value == "N2")
    assert ws.cell(row=row, column=2).value == STREAM_TYPE
    stream = next(r for r in wb["Types"].iter_rows(min_row=2, values_only=True) if r[0] == STREAM_TYPE)
    assert any(str(x).startswith("Operating pressure") for x in stream[2:] if x)
