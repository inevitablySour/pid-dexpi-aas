import warnings
from pathlib import Path

from pid_dexpi_aas.build_model import build_model
from pid_dexpi_aas.parser_drawio import parse_drawio_file

FIXTURE = (
    Path(__file__).parent.parent
    / "handoff"
    / "decoded_xml"
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


def test_equipment_to_instrument_edge_is_reported_not_silently_dropped():
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    sensing_edges = [
        r for r in result.records if r.element_kind == "edge" and "sensing" in r.reason
    ]
    assert len(sensing_edges) >= 1
