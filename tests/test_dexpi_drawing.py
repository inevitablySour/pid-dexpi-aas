"""The DEXPI file with its drawing (dexpi/graphics.py): Proteus 4.1 graphics as P&ID tools write
them (DEXPI TrainingTestCases) - added without changing the plant data."""
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pid_dexpi_aas.dexpi.graphics import add_connector_references, add_drawing
from pid_dexpi_aas.dexpi.proteus_patch import PatchedProteusSerializer
from pid_dexpi_aas.dexpi.serialize import save_and_verify
from pid_dexpi_aas.drawio import line_routes
from pid_dexpi_aas.drawio.decode import decode_drawio
from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved
from pid_dexpi_aas.review import from_drawio, to_dexpi

DRAWING = Path(__file__).parent / "data" / "reviewer_simple_pid.drawio"
pytestmark = pytest.mark.skipif(not line_routes.available(), reason="needs the draw.io app (line routes)")


@pytest.fixture(scope="module")
def files():
    review = from_drawio(parse_drawio_xml_resolved(decode_drawio(DRAWING)[0], source_file=DRAWING.name,
                                                   drawio_path=DRAWING))
    diagram, result = to_dexpi(review)
    with tempfile.TemporaryDirectory() as tmp:
        save_and_verify(result.model, Path(tmp), "plain")
        data = add_connector_references((Path(tmp) / "plain.xml").read_bytes(), diagram, result)
    return result, data, add_drawing(data, diagram, result, review.drawio)


def _load(xml: bytes):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "m.xml").write_bytes(xml)
        return PatchedProteusSerializer().load(Path(tmp), "m").conceptualModel


def test_the_drawing_leaves_the_data_as_it_is(files):
    result, data, drawn = files
    a, b = _load(data), _load(drawn)
    assert len(a.taggedPlantItems) == len(b.taggedPlantItems) == len(result.model.conceptualModel.taggedPlantItems)
    assert len(a.processInstrumentationFunctions) == len(b.processInstrumentationFunctions)
    assert sum(len(s.segments) for s in a.pipingNetworkSystems) == sum(len(s.segments) for s in b.pipingNetworkSystems)
    assert ET.fromstring(data).find("Drawing") is None   # data only: no drawing


def test_items_and_pipes_are_drawn_as_p_and_id_tools_write_them(files):
    _, _, drawn = files
    root = ET.fromstring(drawn)
    assert root.find("Extent") is not None and root.find("Drawing/Extent") is not None
    catalogue = {e.get("ComponentName"): e for e in root.find("ShapeCatalogue")}
    assert catalogue and all(len(e) for e in catalogue.values())   # each symbol has its strokes
    equipment = root.findall("Equipment")
    assert equipment
    for el in equipment:   # placed on the sheet, with a symbol from the catalogue
        assert el.get("ComponentName") in catalogue
        assert el.find("Position/Location") is not None and el.find("Extent/Min") is not None
    segments = list(root.iter("PipingNetworkSegment"))
    drawn_segments = [s for s in segments if s.find("CenterLine") is not None]
    assert len(drawn_segments) >= 0.8 * len(segments)
    for s in drawn_segments:   # centre lines come before the connection (DEXPI 1.3 reference files)
        tags = [c.tag for c in s]
        if "Connection" in tags:
            assert tags.index("CenterLine") < tags.index("Connection")
        assert int(s.find("CenterLine").get("NumPoints")) >= 2
    assert root.findall("Drawing/Label/Text")   # the tags, as labels naming their items
