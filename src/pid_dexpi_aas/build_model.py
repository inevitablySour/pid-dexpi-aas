"""Build a pyDEXPI DexpiModel from a parsed drawio diagram + shape mapping + tag lookup.

pyDEXPI has no create_dexpi_model()/add_equipment() convenience helpers (confirmed against
the installed 1.2.0 wheel) - you construct plain pydantic objects and append to the model's
list fields directly. This module encodes the exact working pattern:

  - equipment classes come from pydexpi.dexpi_classes.equipment, each given one Nozzle
    (one PipingNode) so it has something for piping_toolkit to connect to
  - valves/instruments are built per mapping.map_shape's result
  - edges become PipingNetworkSegments built via piping_toolkit.construct_new_segment
    with PipingConnectionConvention.USE_ITEMS - IN_NODE_0_OUT_NODE_1 (the default) warns
    on every connection here because our nozzles/valves only ever have one relevant node
    per side, not a distinct in/out pair
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydexpi.dexpi_classes import equipment as eq
from pydexpi.dexpi_classes import instrumentation as inst
from pydexpi.dexpi_classes import piping
from pydexpi.dexpi_classes.dexpiModel import ConceptualModel, DexpiModel
from pydexpi.dexpi_classes.pydantic_classes import MetaData
from pydexpi.toolkits import piping_toolkit as pt

from pid_dexpi_aas.mapping import Category, MappingResult, map_shape
from pid_dexpi_aas.parser_drawio import ParsedDiagram, Shape
from pid_dexpi_aas.tag_association import build_tag_lookup

_EQUIPMENT_CLASSES = {name: getattr(eq, name) for name in dir(eq) if not name.startswith("_")}
_PIPING_CLASSES = {name: getattr(piping, name) for name in dir(piping) if not name.startswith("_")}
_INSTRUMENTATION_CLASSES = {
    name: getattr(inst, name) for name in dir(inst) if not name.startswith("_")
}


@dataclass
class BuildRecord:
    """One entry per input shape/edge, tracking what happened - feeds validate_by_counting."""

    element_id: str
    element_kind: str  # "shape" or "edge"
    status: str  # "built", "skipped"
    dexpi_class_name: str | None
    reason: str = ""


@dataclass
class BuildResult:
    model: DexpiModel
    records: list[BuildRecord] = field(default_factory=list)
    # shape id -> constructed DEXPI object, for wiring edges in a second pass
    shape_objects: dict[str, object] = field(default_factory=dict)


def _tag_name(shape: Shape, tag_lookup: dict[str, list[str]]) -> str | None:
    tags = tag_lookup.get(shape.id) or []
    return tags[0] if tags else None


def _build_equipment(shape: Shape, mapping: MappingResult, tag_name: str | None):
    cls = _EQUIPMENT_CLASSES[mapping.dexpi_class_name]
    nozzle = eq.Nozzle(subTagName="N1", nodes=[piping.PipingNode()])
    kwargs = {"nozzles": [nozzle]}
    if tag_name:
        kwargs["tagName"] = tag_name
    # "Custom*" classes are DEXPI's fallback for symbols with no dedicated class (e.g.
    # peristaltic pump) and require a free-text typeName describing what it actually is.
    if "typeName" in cls.model_fields and cls.model_fields["typeName"].is_required():
        kwargs["typeName"] = shape.shape_token or "unknown"
    return cls(**kwargs)


def _build_piping_item(shape: Shape, mapping: MappingResult):
    cls = _PIPING_CLASSES[mapping.dexpi_class_name]
    return cls(nodes=[piping.PipingNode(), piping.PipingNode()])


def _build_instrumentation_item(shape: Shape, mapping: MappingResult, tag_name: str | None):
    cls = _INSTRUMENTATION_CLASSES[mapping.dexpi_class_name]
    kwargs = {}
    if tag_name:
        kwargs["tagName"] = tag_name
    return cls(**kwargs)


def build_shapes(diagram: ParsedDiagram) -> BuildResult:
    """Pass 1: map + construct one DEXPI object per shape. Does not wire edges yet."""
    # ProteusExporter crashes on several of DexpiModel/ConceptualModel's own documented
    # Optional[...] = None defaults (exportDateTime, conceptualModel.metaData,
    # originatingSystem*) rather than handling them, so all are set explicitly here -
    # confirmed against installed pydexpi 1.2.0, these are exporter bugs, not our data.
    model = DexpiModel(
        conceptualModel=ConceptualModel(
            metaData=MetaData(fileName=diagram.source_file)
        ),
        exportDateTime=datetime.now(timezone.utc),
        originatingSystemName="pid-dexpi-aas",
        originatingSystemVendorName="pid-dexpi-aas",
        originatingSystemVersion="0.1.0",
    )
    tag_lookup = build_tag_lookup(diagram)
    result = BuildResult(model=model)

    for shape in diagram.shapes:
        mapping = map_shape(shape.shape_token, shape.text)
        tag_name = _tag_name(shape, tag_lookup) or (shape.text or None)

        if mapping.category == Category.EQUIPMENT:
            obj = _build_equipment(shape, mapping, tag_name)
            model.conceptualModel.taggedPlantItems.append(obj)
            result.shape_objects[shape.id] = obj
            result.records.append(
                BuildRecord(shape.id, "shape", "built", mapping.dexpi_class_name)
            )

        elif mapping.category == Category.PIPING:
            obj = _build_piping_item(shape, mapping)
            result.shape_objects[shape.id] = obj
            result.records.append(
                BuildRecord(shape.id, "shape", "built", mapping.dexpi_class_name, mapping.reason)
            )

        elif mapping.category == Category.INSTRUMENTATION:
            obj = _build_instrumentation_item(shape, mapping, tag_name)
            model.conceptualModel.processInstrumentationFunctions.append(obj)
            result.shape_objects[shape.id] = obj
            result.records.append(
                BuildRecord(shape.id, "shape", "built", mapping.dexpi_class_name, mapping.reason)
            )

        else:
            result.records.append(
                BuildRecord(shape.id, "shape", "skipped", mapping.dexpi_class_name, mapping.reason)
            )

    return result


def _connector_item(obj):
    """Resolve the piping-network endpoint for a built DEXPI object.

    Equipment connects via its first nozzle; piping items (valves) connect directly.
    Instrumentation objects are not valid piping endpoints (a discInst bubble reading a
    process value is a sensing relationship, not a pipe) - return None so the caller
    reports it as an unsupported edge kind rather than attempting a doomed connection.
    """
    if isinstance(obj, inst.ProcessInstrumentationFunction):
        return None
    if isinstance(obj, eq.Equipment):
        return obj.nozzles[0] if obj.nozzles else None
    return obj


def wire_edges(diagram: ParsedDiagram, build_result: BuildResult) -> None:
    """Pass 2: for every edge whose endpoints both mapped to piping-capable objects,
    build a PipingNetworkSegment connecting them. Valve-shaped endpoints become segment
    items; equipment/instrument endpoints become source/target connector items.
    """
    system = piping.PipingNetworkSystem(segments=[])

    for edge in diagram.edges:
        source_obj = build_result.shape_objects.get(edge.source) if edge.source else None
        target_obj = build_result.shape_objects.get(edge.target) if edge.target else None

        if source_obj is None or target_obj is None:
            build_result.records.append(
                BuildRecord(
                    edge.id,
                    "edge",
                    "skipped",
                    None,
                    "source or target shape was not built (unmapped or missing)",
                )
            )
            continue

        source_connector = _connector_item(source_obj)
        target_connector = _connector_item(target_obj)

        if source_connector is None or target_connector is None:
            build_result.records.append(
                BuildRecord(
                    edge.id,
                    "edge",
                    "skipped",
                    None,
                    "equipment/instrument sensing edge - not a piping connection, "
                    "not yet handled (needs a SensingLocation/measuring-line mapping)",
                )
            )
            continue

        try:
            segment = pt.construct_new_segment(
                segment_items=[],
                segment_connections=[piping.PipingConnection()],
                source_connector_item=source_connector,
                target_connector_item=target_connector,
                connectivity_convention=pt.PipingConnectionConvention.USE_ITEMS,
            )
        except Exception as exc:  # pyDEXPI raises DexpiConnectionException & friends
            build_result.records.append(
                BuildRecord(edge.id, "edge", "skipped", None, f"piping_toolkit error: {exc}")
            )
            continue

        system.segments.append(segment)
        build_result.records.append(
            BuildRecord(edge.id, "edge", "built", "PipingNetworkSegment")
        )

    if system.segments:
        build_result.model.conceptualModel.pipingNetworkSystems.append(system)


def build_model(diagram: ParsedDiagram) -> BuildResult:
    result = build_shapes(diagram)
    wire_edges(diagram, result)
    return result
