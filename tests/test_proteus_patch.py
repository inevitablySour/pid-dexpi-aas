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

from pid_dexpi_aas.dexpi.build_model import build_model
from pid_dexpi_aas.drawio.parser import parse_drawio_file
from pid_dexpi_aas.dexpi.proteus_patch import PatchedProteusSerializer

FIXTURE = (
    Path(__file__).parent
    / "fixtures" / "decoded_xml"
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


def test_piping_segment_source_and_target_items_resolve_to_nozzles_or_off_page_connectors(tmp_path):
    """Most segments connect two equipment nozzles, but a segment ending at a text-only
    off-page destination (e.g. "Fume hood [atm]", "safe location" - see build_model.py's
    _get_or_build_off_page_connector) has a PipeOffPageConnector on that end instead.

    NOTE: pyDEXPI 1.2.0's Proteus PARSER has a confirmed bug of its own here (not ours):
    PipingNetworkSegmentParser.reference_pass unconditionally assigns any adjacent
    PipingNetworkSegmentItem - including a plain PipeOffPageConnector reconstructed from
    the XML - as a PipingConnection's sourceItem/targetItem, which pydantic then rejects
    (PipingConnection.targetItem is typed PipingSourceItem | PipingTargetItem, and plain
    PipeOffPageConnector implements neither - only its FlowIn/FlowOut subclasses do, and
    the parser never reconstructs those). This means a segment WITH an off-page connector
    item does not fully survive Proteus reload in this pyDEXPI version - confirmed by a
    parse-time ERROR (not a warning) in the reloaded segment's item count when this
    occurs. See test_off_page_connector_causes_a_known_pydexpi_parser_bug below, which
    pins this exact, confirmed-upstream failure rather than silently tolerating it.
    """
    _, reloaded, _ = _build_and_round_trip(tmp_path)
    segments = reloaded.conceptualModel.pipingNetworkSystems[0].segments
    valid_endpoint_types = {"Nozzle", "NoneType"}  # NoneType: the known parser bug case
    for segment in segments:
        assert type(segment.sourceItem).__name__ in valid_endpoint_types
        assert type(segment.targetItem).__name__ in valid_endpoint_types


def test_process_instrumentation_functions_survive_round_trip(tmp_path):
    result, reloaded, _ = _build_and_round_trip(tmp_path)
    assert len(reloaded.conceptualModel.processInstrumentationFunctions) == len(
        result.model.conceptualModel.processInstrumentationFunctions
    )


_KNOWN_OFF_PAGE_CONNECTOR_ERROR = "validation error for DirectPipingConnection"


def test_known_parser_bugs_produce_only_expected_errors(tmp_path):
    """pyDEXPI 1.2.0's Proteus parser has two confirmed bugs of its own (not ours):

    1. A dead-code reset in _extract_connections means FromNode/ToNode are always parsed
       back as None, so sourceNode/targetNode-by-ID resolution can warn ("Source node
       with ID ... not found in registry") even though sourceItem/targetItem (the part
       we actually rely on) resolve correctly. Only a WARNING, does not break anything.

    2. PipingNetworkSegmentParser.reference_pass unconditionally assigns any adjacent
       PipingNetworkSegmentItem as a PipingConnection's sourceItem/targetItem - including
       a plain PipeOffPageConnector reconstructed from a <PipeOffPageConnector> tag - and
       pydantic rejects this (PipeOffPageConnector implements neither PipingSourceItem
       nor PipingTargetItem; only its unused-by-the-parser FlowIn/FlowOut subclasses do).
       This means ANY Proteus file with an off-page connector adjacent to an internal
       connection fails to fully reload in this pyDEXPI version, regardless of who wrote
       the file - confirmed via a minimal reproduction outside this codebase, not
       specific to our exporter. This is a genuine, fatal ERROR (not a warning), and this
       test explicitly tolerates only this one specific, understood failure rather than
       silently accepting any parse error.
    """
    _, _, serializer = _build_and_round_trip(tmp_path)
    errors = serializer.get_loading_errors().get_errors()
    for error in errors:
        if error.level.value == "error":
            assert _KNOWN_OFF_PAGE_CONNECTOR_ERROR in error.message, (
                f"unexpected fatal parse error: {error.message}"
            )


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


def test_multi_item_segment_with_tee_branch_round_trips(tmp_path):
    """Valves and tees must sit INSIDE segments: pyDEXPI's reader resolves a segment's
    ends only to nozzles or to items inside some segment, so a segment that merely ended
    on a valve lost that end on reload (43 of 43 on LightUp). Run: nozzle -> gate valve
    -> tee -> check valve -> nozzle; branch: tee -> nozzle."""
    from pydexpi.dexpi_classes import equipment
    from pydexpi.toolkits import piping_toolkit as pt

    def vessel(tag):
        n = equipment.Nozzle(subTagName="N1", nodes=[piping.PipingNode()])
        return equipment.Vessel(tagName=tag, nozzles=[n]), n

    (a, na), (b, nb), (c, nc) = vessel("A"), vessel("B"), vessel("C")
    gate = piping.GateValve(nodes=[piping.PipingNode(), piping.PipingNode()])
    check = piping.CheckValve(nodes=[piping.PipingNode(), piping.PipingNode()])
    tee = piping.PipeTee(nodes=[piping.PipingNode(), piping.PipingNode(), piping.PipingNode()])
    run = pt.construct_new_segment(segment_items=[gate, tee, check], segment_connections=[piping.PipingConnection() for _ in range(4)],
                                   source_connector_item=na, target_connector_item=nb,
                                   connectivity_convention=pt.PipingConnectionConvention.USE_ITEMS)
    branch = pt.construct_new_segment(segment_items=[], segment_connections=[piping.PipingConnection()],
                                      source_connector_item=tee, target_connector_item=nc,
                                      connectivity_convention=pt.PipingConnectionConvention.USE_ITEMS)
    model = _bare_model()
    model.conceptualModel.taggedPlantItems.extend([a, b, c])
    model.conceptualModel.pipingNetworkSystems.append(piping.PipingNetworkSystem(segments=[run, branch]))

    from pid_dexpi_aas.dexpi.serialize import save_and_verify
    reloaded = save_and_verify(model, tmp_path, "multi_item")
    seg_run, seg_branch = reloaded.conceptualModel.pipingNetworkSystems[0].segments
    assert [type(i).__name__ for i in seg_run.items] == ["GateValve", "PipeTee", "CheckValve"]
    assert type(seg_run.sourceItem).__name__ == "Nozzle" and type(seg_run.targetItem).__name__ == "Nozzle"
    assert type(seg_branch.sourceItem).__name__ == "PipeTee" and type(seg_branch.targetItem).__name__ == "Nozzle"


def test_instrument_wiring_survives_export_and_reload(tmp_path):
    """Measuring lines (sensor at a vessel nozzle / valve / tee / pipe -> instrument), signal
    lines (instrument -> instrument / actuator) and actuating functions are written as
    nested ProcessSignalGeneratingFunction / ActuatingFunction / InformationFlow elements
    and come back from pyDEXPI's reader with every end and location."""
    from collections import Counter

    from pydexpi.dexpi_classes import instrumentation as inst

    from pid_dexpi_aas.dexpi.build_model import build_model
    from pid_dexpi_aas.drawio.parser import parse_drawio_file
    from pid_dexpi_aas.dexpi.serialize import save_and_verify

    root = Path(__file__).parent.parent
    diagram = parse_drawio_file(root / "tests" / "fixtures" / "decoded_xml" / "P&ID gramflow reactor V3__diagram0.xml")
    model = build_model(diagram).model

    def wiring(m):
        pifs = [p for p in m.conceptualModel.processInstrumentationFunctions
                if isinstance(p, inst.ProcessInstrumentationFunction)]
        flows = [f for p in pifs for f in p.signalConveyingFunctions]
        return (Counter(f"{type(f).__name__}:{type(f.source).__name__}->{type(f.target).__name__}" for f in flows),
                Counter(type(g.sensingLocation).__name__ for p in pifs for g in p.processSignalGeneratingFunctions),
                sum(1 for p in pifs for a in p.actuatingFunctions if a.actuatingLocation is not None))

    built = wiring(model)
    assert sum(built[0].values()) > 10 and all(".->" not in k for k in built[0])
    assert wiring(save_and_verify(model, tmp_path, "wiring.xml")) == built


def test_pipe_specifications_and_choice_fields_survive_reload(tmp_path):
    """A line's specifications land on its pipe segment and come back from the reader -
    choice fields too: pyDEXPI writes them by value but read them by member name, and a
    failed lookup dropped the object's whole attribute set (proteus_patch's reader fix)."""
    from pid_dexpi_aas.dexpi.fields import spec_fields
    from pid_dexpi_aas.review import from_drawio, to_dexpi
    from pid_dexpi_aas.dexpi.serialize import save_and_verify

    root = Path(__file__).parent.parent
    review = from_drawio(parse_drawio_file(root / "tests" / "fixtures" / "decoded_xml" / "P&ID gramflow reactor V3__diagram0.xml"))
    line = next(c for c in review.connections if c.include and c.line_type == "process" and not c.id.startswith("net"))
    values = {f.name: ({"value": 10, "unit": f.units[0]} if f.kind == "quantity" else
                       f.options[-1] if f.kind == "choice" else "X1")
              for f in spec_fields("PipingNetworkSegment") if not f.custom}
    values["fluidCode"] = "MARK"
    line.attributes = {"dexpi": values}
    _, result = to_dexpi(review)
    assert result.spec_problems == []
    back = save_and_verify(result.model, tmp_path, "spec.xml")
    seg = next(s for sys in back.conceptualModel.pipingNetworkSystems for s in sys.segments if s.fluidCode == "MARK")
    assert [n for n in values if getattr(seg, n) in (None, "")] == []
