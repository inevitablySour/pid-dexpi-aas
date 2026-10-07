import warnings
from pathlib import Path

from pid_dexpi_aas.dexpi.build_model import build_model
from pid_dexpi_aas.drawio.parser import parse_drawio_file

FIXTURE = (
    Path(__file__).parent
    / "fixtures" / "decoded_xml"
    / "PFD combined set up 301122__diagram0.xml"
)


def test_build_model_accounts_for_every_shape_and_edge():
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    assert len(result.records) == len(diagram.shapes) + len(diagram.edges)


def test_build_model_raises_no_warnings():
    diagram = parse_drawio_file(FIXTURE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_model(diagram)
    assert len(caught) == 0


def test_reactor_and_tank_and_pump_are_built():
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    built_classes = {r.dexpi_class_name for r in result.records if r.status == "built"}
    assert "Vessel" in built_classes
    assert "Tank" in built_classes
    assert "CustomPump" in built_classes


def test_custom_pump_gets_type_name_from_shape_token():
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    pump = next(
        obj
        for obj in result.model.conceptualModel.taggedPlantItems
        if type(obj).__name__ == "CustomPump"
    )
    assert pump.typeName == "mxgraph.pid.pumps.peristaltic"


def test_valid_piping_segments_are_built():
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    assert len(result.model.conceptualModel.pipingNetworkSystems) == 1
    assert len(result.model.conceptualModel.pipingNetworkSystems[0].segments) > 0


def test_instrument_edges_become_dexpi_instrumentation_wiring():
    # LightUp, not FIXTURE: the PFD's only instrument edge ran to "BPR", which is now
    # (correctly) a back-pressure regulator VALVE in the line, so it has none left.
    # Instrument lines used to be skipped ("sensing edge"); they are now measuring lines
    # (instrument on a vessel / valve / pipe), signal lines (instrument to instrument) or
    # actuating functions (instrument driving a valve over a dashed line).
    from pydexpi.dexpi_classes import instrumentation as inst

    diagram = parse_drawio_file(FIXTURE.parent / "P&ID gramflow reactor V3__diagram0.xml")
    result = build_model(diagram)
    assert not [r for r in result.records if r.element_kind == "edge" and "sensing" in r.reason]
    pifs = [p for p in result.model.conceptualModel.processInstrumentationFunctions
            if isinstance(p, inst.ProcessInstrumentationFunction)]
    lines = [f for p in pifs for f in p.signalConveyingFunctions]
    assert any(isinstance(f, inst.MeasuringLineFunction) for f in lines)
    for f in lines:
        assert f.source is not None and f.target is not None
    for p in pifs:
        for g in p.processSignalGeneratingFunctions:
            assert g.sensingLocation is not None
