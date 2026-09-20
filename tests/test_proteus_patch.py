"""Round-trip tests for the local Proteus exporter patch (proteus_patch.py).

See proteus_patch.py's module docstring and ../README.md for why this patch exists:
pyDEXPI 1.2.0's stock ProteusExporter drops piping/instrumentation data on export.
"""
from datetime import datetime, timezone
from pathlib import Path

from pydexpi.dexpi_classes import equipment as eq
from pydexpi.dexpi_classes import piping
from pydexpi.dexpi_classes.dexpiModel import ConceptualModel, DexpiModel
from pydexpi.dexpi_classes.pydantic_classes import MetaData
from pydexpi.toolkits import piping_toolkit as pt

from pid_dexpi_aas.build_model import build_model
from pid_dexpi_aas.parser_drawio import parse_drawio_file
from pid_dexpi_aas.proteus_patch import PatchedProteusSerializer

FIXTURE = (
    Path(__file__).parent.parent
    / "handoff"
    / "decoded_xml"
    / "PFD combined set up 301122__diagram0.xml"
)


def _build_and_round_trip(tmp_path):
    diagram = parse_drawio_file(FIXTURE)
    result = build_model(diagram)
    serializer = PatchedProteusSerializer()
    serializer.save(result.model, tmp_path, "model.xml")
    reloaded = serializer.load(tmp_path, "model.xml")
    return result, reloaded, serializer


def test_equipment_survives_round_trip(tmp_path):
    result, reloaded, _ = _build_and_round_trip(tmp_path)
    assert len(reloaded.conceptualModel.taggedPlantItems) == len(
        result.model.conceptualModel.taggedPlantItems
    )


def test_piping_network_systems_survive_round_trip(tmp_path):
    result, reloaded, _ = _build_and_round_trip(tmp_path)
    original_systems = result.model.conceptualModel.pipingNetworkSystems
    reloaded_systems = reloaded.conceptualModel.pipingNetworkSystems
    assert len(reloaded_systems) == len(original_systems)
    assert len(reloaded_systems[0].segments) == len(original_systems[0].segments)


def test_piping_segment_source_and_target_items_resolve_to_nozzles(tmp_path):
    _, reloaded, _ = _build_and_round_trip(tmp_path)
    segments = reloaded.conceptualModel.pipingNetworkSystems[0].segments
    for segment in segments:
        assert type(segment.sourceItem).__name__ == "Nozzle"
        assert type(segment.targetItem).__name__ == "Nozzle"


def test_process_instrumentation_functions_survive_round_trip(tmp_path):
    result, reloaded, _ = _build_and_round_trip(tmp_path)
    assert len(reloaded.conceptualModel.processInstrumentationFunctions) == len(
        result.model.conceptualModel.processInstrumentationFunctions
    )


def test_known_parser_bug_produces_only_warnings_not_fatal_errors(tmp_path):
    """pyDEXPI 1.2.0's Proteus parser has a confirmed bug (dead-code reset in
    _extract_connections) where FromNode/ToNode are always parsed back as None, so
    sourceNode/targetNode-by-ID resolution can warn ("Source node with ID ... not found
    in registry") even though sourceItem/targetItem (the part we actually rely on)
    resolve correctly. This test pins that this stays a WARNING, not something that
    breaks the round-trip - if pyDEXPI fixes the bug, this test should still pass
    (errors list may just become empty).
    """
    _, _, serializer = _build_and_round_trip(tmp_path)
    errors = serializer.get_loading_errors().get_errors()
    for error in errors:
        assert error.level.value != "error", f"unexpected fatal parse error: {error.message}"


def _bare_model() -> DexpiModel:
    return DexpiModel(
        conceptualModel=ConceptualModel(metaData=MetaData(fileName="unit-test")),
        exportDateTime=datetime.now(timezone.utc),
        originatingSystemName="pid-dexpi-aas-test",
        originatingSystemVendorName="pid-dexpi-aas-test",
        originatingSystemVersion="0.1.0",
    )


def test_single_item_segment_round_trips(tmp_path):
    """build_model.py never produces a segment with a valve nested inside it (every
    mxGraph edge becomes a direct 0-item connection - see wire_edges()'s docstring), so
    this path is exercised directly here rather than through the real pipeline.
    """
    tank = eq.Tank(
        tagName="T1", nozzles=[eq.Nozzle(subTagName="N1", nodes=[piping.PipingNode()])]
    )
    pump = eq.Pump(
        tagName="P1", nozzles=[eq.Nozzle(subTagName="N1", nodes=[piping.PipingNode()])]
    )
    valve = piping.GateValve(nodes=[piping.PipingNode(), piping.PipingNode()])
    segment = pt.construct_new_segment(
        segment_items=[valve],
        segment_connections=[piping.PipingConnection(), piping.PipingConnection()],
        source_connector_item=tank.nozzles[0],
        target_connector_item=pump.nozzles[0],
        connectivity_convention=pt.PipingConnectionConvention.USE_ITEMS,
    )

    model = _bare_model()
    model.conceptualModel.taggedPlantItems.extend([tank, pump])
    model.conceptualModel.pipingNetworkSystems.append(
        piping.PipingNetworkSystem(segments=[segment])
    )

    serializer = PatchedProteusSerializer()
    serializer.save(model, tmp_path, "single_item.xml")
    reloaded = serializer.load(tmp_path, "single_item.xml")

    reloaded_segment = reloaded.conceptualModel.pipingNetworkSystems[0].segments[0]
    assert [type(item).__name__ for item in reloaded_segment.items] == ["GateValve"]
    assert type(reloaded_segment.sourceItem).__name__ == "Nozzle"
    assert type(reloaded_segment.targetItem).__name__ == "Nozzle"


def test_multi_item_segment_raises_not_implemented(tmp_path):
    """The exporter patch explicitly refuses (rather than silently mis-exports) segments
    with 2+ items, since that needs the full interleaved item/CenterLine ordering logic
    documented in proteus_patch.py's module docstring, not implemented here.
    """
    valve_a = piping.GateValve(nodes=[piping.PipingNode(), piping.PipingNode()])
    valve_b = piping.CheckValve(nodes=[piping.PipingNode(), piping.PipingNode()])
    segment = piping.PipingNetworkSegment(items=[valve_a, valve_b], connections=[])

    model = _bare_model()
    model.conceptualModel.pipingNetworkSystems.append(
        piping.PipingNetworkSystem(segments=[segment])
    )

    serializer = PatchedProteusSerializer()
    try:
        serializer.save(model, tmp_path, "multi_item.xml")
        assert False, "expected NotImplementedError for a multi-item segment"
    except NotImplementedError:
        pass
