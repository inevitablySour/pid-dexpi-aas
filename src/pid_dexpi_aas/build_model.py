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
    # For edges: how the connection's endpoints were actually determined - "explicit"
    # (the drawio file states it directly), "exit_fraction" (a tight, high-confidence
    # geometric match), "nearest_shape" (a looser proximity guess), or "mixed" (one
    # endpoint explicit, the other inferred). Confirmed via manual cross-checking against
    # the original P&ID image that nearest_shape resolutions can be wrong even when they
    # succeed - this field lets a reviewer immediately see which built connections are
    # ground truth and which are the tool's best guess, rather than treating both the
    # same way. Empty string for shapes (not applicable).
    confidence: str = ""


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
    # Equipment is built with ZERO nozzles up front - nozzles are allocated lazily, one
    # per edge that actually connects to this equipment, in wire_edges' pass (see
    # _allocate_nozzle). Building a single shared nozzle here (the original approach)
    # meant every edge touching the same vessel collided on that one nozzle, making a
    # real two-nozzle connection (e.g. a recirculation/kickback loop returning to the
    # same vessel through a different nozzle) structurally indistinguishable from two
    # edges both hitting the vessel's single generic attachment point - and, per DEXPI's
    # own type system (PipingSourceItem/PipingTargetItem exclude Equipment entirely -
    # see research/04-dexpi-semantics.md), the ONLY correct way to represent "this
    # equipment connects to itself" is via two distinct Nozzles, never a literal
    # equipment-level self-reference.
    kwargs = {"nozzles": []}
    if tag_name:
        kwargs["tagName"] = tag_name
    # "Custom*" classes are DEXPI's fallback for symbols with no dedicated class (e.g.
    # peristaltic pump) and require a free-text typeName describing what it actually is.
    if "typeName" in cls.model_fields and cls.model_fields["typeName"].is_required():
        kwargs["typeName"] = shape.shape_token or "unknown"
    return cls(**kwargs)


def _allocate_nozzle(equipment) -> "eq.Nozzle":
    """Add and return a brand-new Nozzle on this equipment - called once per edge
    endpoint that resolves to this equipment, so distinct connections always get
    distinct attachment points (see _build_equipment's docstring for why this matters
    for recirculation loops)."""
    nozzle = eq.Nozzle(subTagName=f"N{len(equipment.nozzles) + 1}", nodes=[piping.PipingNode()])
    equipment.nozzles.append(nozzle)
    return nozzle


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

    Equipment connects via a FRESH nozzle allocated for this specific edge (see
    _allocate_nozzle/_build_equipment) - never a shared/reused one, so two different
    edges touching the same equipment always get two distinct attachment points. This
    is what makes a real recirculation/kickback loop (two connections to the same
    vessel) representable at all: per DEXPI's own type system, Equipment can never be a
    PipingConnection's source/target directly (see research/04-dexpi-semantics.md), only
    a Nozzle can - so "this equipment connects to itself" MUST be two distinct nozzles,
    and allocating a fresh one per edge is what makes that the natural outcome instead of
    requiring special-case detection.

    Piping items (valves) connect directly. Instrumentation objects are not valid piping
    endpoints (a discInst bubble reading a process value is a sensing relationship, not a
    pipe) - return None so the caller reports it as an unsupported edge kind rather than
    attempting a doomed connection.
    """
    if isinstance(obj, inst.ProcessInstrumentationFunction):
        return None
    if isinstance(obj, eq.Equipment):
        return _allocate_nozzle(obj)
    return obj


def _get_or_build_off_page_connector(
    shape_id: str, build_result: BuildResult, shapes_by_id: dict, as_source: bool
):
    """Return the object already built for shape_id, or - if it's a floating text label
    with no DEXPI representation - lazily build an off-page connector for it and record
    that build, so an edge terminating at a text-only destination (e.g. "Fume hood
    [atm]", "safe location") isn't silently dropped just because it's not process
    equipment. Confirmed as a real, distinct gap (not a resolution failure): the shape
    and edge both resolve correctly, there's just nothing to connect the edge TO,
    because build_shapes() only builds equipment/piping/instrumentation shapes.
    Only fires for shapes with non-empty text - an empty/unlabeled floating rectangle
    has no meaningful identity to attach a connector to.

    Bare PipeOffPageConnector is NOT a valid PipingSourceItem/PipingTargetItem (confirmed
    by pydantic rejecting it at construction time) - only its FlowIn/FlowOut subclasses
    are, so which one to build depends on which side of the edge this shape is on.
    """
    existing = build_result.shape_objects.get(shape_id)
    if existing is not None:
        return existing

    shape = shapes_by_id.get(shape_id)
    if shape is None or not shape.is_labeled_only or not shape.text:
        return None

    connector_cls = piping.FlowInPipeOffPageConnector if as_source else piping.FlowOutPipeOffPageConnector
    connector = connector_cls(nodes=[piping.PipingNode()])
    build_result.shape_objects[shape_id] = connector

    # build_shapes() (pass 1) already appended a "skipped" record for this shape id
    # (labels with no shape token are skipped there) - update it in place rather than
    # appending a second record, or validate_by_counting's total-element accounting
    # would double-count this shape.
    for existing_record in build_result.records:
        if existing_record.element_id == shape_id and existing_record.element_kind == "shape":
            existing_record.status = "built"
            existing_record.dexpi_class_name = connector_cls.__name__
            existing_record.reason = (
                "text-only connection target (e.g. an off-page destination), not "
                "process equipment - built lazily because an edge terminates here"
            )
            break

    return connector


def wire_edges(diagram: ParsedDiagram, build_result: BuildResult) -> None:
    """Pass 2: for every edge whose endpoints both mapped to piping-capable objects,
    build a PipingNetworkSegment connecting them. Valve-shaped endpoints become segment
    items; equipment/instrument endpoints become source/target connector items.
    """
    system = piping.PipingNetworkSystem(segments=[])
    shapes_by_id = {shape.id: shape for shape in diagram.shapes}

    for edge in diagram.edges:
        if edge.is_decorative:
            # Classified by parser_drawio.classify_decorative_edges: a cluster of tiny
            # edges hugging one shape's boundary, almost certainly hand-drawn symbol
            # detail (e.g. a safety valve's spring/relief indicator) rather than piping -
            # confirmed by manual inspection against the source drawing. Reported
            # distinctly so it's not confused with a genuinely broken/corrupt connection.
            build_result.records.append(
                BuildRecord(
                    edge.id,
                    "edge",
                    "skipped",
                    None,
                    "decorative symbol linework (e.g. a valve's hand-drawn spring/relief "
                    "indicator), not a piping connection - part of a cluster of similar "
                    "tiny edges all confined to one shape's boundary",
                )
            )
            continue

        source_obj = (
            _get_or_build_off_page_connector(edge.source, build_result, shapes_by_id, as_source=True)
            if edge.source
            else None
        )
        target_obj = (
            _get_or_build_off_page_connector(edge.target, build_result, shapes_by_id, as_source=False)
            if edge.target
            else None
        )

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

        if (
            edge.source_point is not None
            and edge.target_point is not None
            and edge.source_point == edge.target_point
        ):
            # A genuinely zero-length edge (identical raw coordinates on both ends) -
            # not a recirculation loop (which is drawn with real pipe length between two
            # distinct nozzle positions), just a degenerate mxCell. Confirmed on the
            # gramflow reactor file: h17LKkp3DZ7R27TqlmPq-62 has source_point ==
            # target_point exactly. Reject regardless of which shape(s) it resolved to.
            build_result.records.append(
                BuildRecord(
                    edge.id,
                    "edge",
                    "skipped",
                    None,
                    "zero-length edge (identical source/target coordinates) - a "
                    "degenerate artifact, not a real connection",
                )
            )
            continue

        if isinstance(source_obj, (piping.FlowInPipeOffPageConnector, piping.FlowOutPipeOffPageConnector)) and isinstance(
            target_obj, (piping.FlowInPipeOffPageConnector, piping.FlowOutPipeOffPageConnector)
        ):
            # Both ends are text-only labels with no real equipment/valve/instrument
            # behind them - an off-page connector is meant to link real process flow to
            # one external destination, not one arbitrary floating label to another.
            # Confirmed as a real false-positive risk from the margin-based resolution
            # fix: one case connected the free-floating annotation "18 bar" to "PSV201"
            # (a valve's own tag label) purely because the margin test passed, not
            # because these labels represent a real connection. Reject rather than
            # build a plausible-looking but almost certainly wrong segment.
            build_result.records.append(
                BuildRecord(
                    edge.id,
                    "edge",
                    "skipped",
                    None,
                    "both endpoints are text-only labels with no real equipment behind "
                    "them - very likely two unrelated annotations, not a genuine "
                    "off-page connection; needs manual review",
                )
            )
            continue

        # NOTE: edge.source == edge.target (both ends resolving to the same SHAPE) is no
        # longer rejected here - per research/04-dexpi-semantics.md, a real recirculation/
        # kickback/reflux loop legitimately connects back to the same equipment, and
        # DEXPI requires that to be modelled as two distinct Nozzles on that equipment,
        # never a literal equipment-level self-reference. _connector_item allocates a
        # FRESH nozzle per call for Equipment (see _allocate_nozzle), so a same-shape
        # edge naturally produces two distinct connector objects here - exactly the
        # correct DEXPI representation - rather than needing to be specially detected.
        # The only thing that must still be rejected is the genuinely degenerate case:
        # both ends resolving to the literal SAME connector OBJECT (checked below, after
        # allocation) - which can only happen for non-equipment endpoints (e.g. the same
        # valve/off-page-connector object on both sides), since equipment always gets a
        # fresh nozzle and so can never collide with itself here.
        source_connector = _connector_item(source_obj)
        target_connector = _connector_item(target_obj)

        if source_connector is target_connector and source_connector is not None:
            # A valve or off-page connector object referenced as both ends of the same
            # edge - always wrong data (a single component can't be piped to itself),
            # unlike the same-EQUIPMENT case above, which is legitimate. Confirmed cause:
            # resolve_floating_edge_endpoints' nearest-shape search can match BOTH ends
            # of a very short line segment (e.g. one dash-mark of a hand-drawn dashed
            # line) to the same nearby valve.
            build_result.records.append(
                BuildRecord(
                    edge.id,
                    "edge",
                    "skipped",
                    None,
                    "both endpoints resolved to the same non-equipment object "
                    "(self-loop, invalid) - likely a fragment of a multi-segment "
                    "dashed line that couldn't be distinguished from a real short "
                    "connection",
                )
            )
            continue

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

        # An off-page connector must be an internal segment item, per the DEXPI/Proteus
        # convention the parser itself follows (PipingNetworkSegmentParser's docstring:
        # "In accordance with the DEXPI convention" - the first/last item IS the
        # source/target when it's an off-page connector, there is no external
        # <Connection> reference to it). A bare 0-item segment with the connector as
        # sourceItem/targetItem is legal at the pydantic level but has no schema-correct
        # export path (confirmed: <PlantModel>/<PipingNetworkSystem> both have closed,
        # enumerated child-tag lists with no slot for a standalone connector reference).
        segment_items = []
        if isinstance(source_obj, piping.FlowInPipeOffPageConnector):
            segment_items.append(source_connector)
        if isinstance(target_obj, piping.FlowOutPipeOffPageConnector):
            segment_items.append(target_connector)

        try:
            segment = pt.construct_new_segment(
                segment_items=segment_items,
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
        if edge.source_resolution == edge.target_resolution:
            confidence = edge.source_resolution
        else:
            confidence = "mixed"
        build_result.records.append(
            BuildRecord(edge.id, "edge", "built", "PipingNetworkSegment", confidence=confidence)
        )

    if system.segments:
        build_result.model.conceptualModel.pipingNetworkSystems.append(system)


def build_model(diagram: ParsedDiagram) -> BuildResult:
    result = build_shapes(diagram)
    wire_edges(diagram, result)
    return result
