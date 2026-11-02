"""A reviewer's simple P&ID (2026-11), drawn in the editor: a feed line with a loose end, a hand
valve, a TI whose line ends on the pipe, a packed column, and a product line with a loose end.
Before, both loose lines were dropped and the TI was a loose label in the re-drawing; now the
loose ends are an inlet and an outlet (FlowIn / FlowOut) and the TI measures on the pipe."""
from pathlib import Path

import pytest

from pid_dexpi_aas.drawio.decode import decode_drawio
from pid_dexpi_aas.drawio import line_routes
from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved
from pid_dexpi_aas.review import OFF_PAGE, from_drawio, to_dexpi

DRAWING = Path(__file__).parent / "data" / "reviewer_simple_pid.drawio"
pytestmark = pytest.mark.skipif(not line_routes.available(), reason="needs the draw.io app (line routes)")


@pytest.fixture(scope="module")
def built():
    review = from_drawio(parse_drawio_xml_resolved(decode_drawio(DRAWING)[0], source_file=DRAWING.name,
                                                   drawio_path=DRAWING))
    return review, *to_dexpi(review)


def test_loose_ends_become_an_inlet_and_an_outlet(built):
    review, _, result = built
    streams = sorted(s.tag for s in review.symbols if s.dexpi_class == OFF_PAGE)
    assert streams == ["Inlet 1", "Outlet 1"]
    items = [type(i).__name__ for sys in result.model.conceptualModel.pipingNetworkSystems
             for seg in sys.segments for i in seg.items]
    assert "FlowInPipeOffPageConnector" in items and "FlowOutPipeOffPageConnector" in items
    assert not [r for r in result.records if r.element_kind == "edge" and r.status != "built"]


def test_the_instrument_measures_on_the_pipe(built):
    _, _, result = built
    (pif,) = result.model.conceptualModel.processInstrumentationFunctions
    (sensor,) = pif.processSignalGeneratingFunctions
    assert type(sensor.sensingLocation).__name__ == "PipingNetworkSegment"


def test_no_stream_from_a_line_ending_on_a_pipe(built):
    review, _, _ = built
    assert not any(s.id.startswith("e2:") for s in review.symbols)   # the TI's line: a tap, not a stream
