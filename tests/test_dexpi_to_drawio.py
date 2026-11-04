"""DEXPI back to a P&ID (dexpi/to_drawio.py): a file with the plant data only, laid out as a
draw.io drawing - and that drawing through the app's draw.io route gives the same plant back."""
import collections
import re
import tempfile
from pathlib import Path

import pytest

from pid_dexpi_aas.dexpi.serialize import save_and_verify
from pid_dexpi_aas.dexpi.to_drawio import dexpi_to_drawio, read_plant
from pid_dexpi_aas.drawio import line_routes
from pid_dexpi_aas.drawio.decode import decode_drawio
from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved
from pid_dexpi_aas.review import from_drawio, to_dexpi

DRAWING = Path(__file__).parent / "data" / "reviewer_simple_pid.drawio"
pytestmark = pytest.mark.skipif(not line_routes.available(), reason="needs the draw.io app (line routes)")


def _dexpi(path: Path) -> bytes:
    review = from_drawio(parse_drawio_xml_resolved(decode_drawio(path)[0], source_file=path.name, drawio_path=path))
    _, result = to_dexpi(review)
    with tempfile.TemporaryDirectory() as tmp:
        save_and_verify(result.model, Path(tmp), "m")
        return (Path(tmp) / "m.xml").read_bytes()


def _plant(cm):
    return (sorted(e.tagName or "" for e in cm.taggedPlantItems), len(cm.processInstrumentationFunctions),
            collections.Counter(type(i).__name__ for s in cm.pipingNetworkSystems for g in s.segments for i in g.items))


@pytest.fixture(scope="module")
def round_trip(tmp_path_factory):
    xml = _dexpi(DRAWING)
    drawio, report = dexpi_to_drawio(xml, "back")
    path = tmp_path_factory.mktemp("rt") / "back.drawio"
    path.write_text(drawio)
    return xml, drawio, report, _dexpi(path)


def test_a_dexpi_file_is_drawn_with_classes_and_tags(round_trip):
    xml, drawio, report, _ = round_trip
    assert report["items"] >= 4 and report["pipes"] >= 3 and report["instruments"] == 1
    assert 'dexpi_class="ProcessInstrumentationFunction"' in drawio and "TI&lt;br&gt;" in drawio
    assert "edgeStyle=orthogonalEdgeStyle" in drawio   # pipes routed at right angles by draw.io
    assert re.search(r'dexpi_class="[A-Za-z]*Valve"', drawio)


def test_the_drawing_gives_the_same_plant_back(round_trip):
    xml, _, _, again = round_trip
    a, b = read_plant(xml)["model"].conceptualModel, read_plant(again)["model"].conceptualModel
    assert _plant(a) == _plant(b)
