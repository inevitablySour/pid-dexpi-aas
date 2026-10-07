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

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydexpi.dexpi_classes import equipment as eq
from pydexpi.dexpi_classes import instrumentation as inst
from pydexpi.dexpi_classes import piping
from pydexpi.dexpi_classes.dexpiModel import ConceptualModel, DexpiModel
from pydexpi.dexpi_classes.pydantic_classes import CustomStringAttribute, MetaData
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
    # edge id -> the PipingNetworkSegment(s) it became (a line's specifications go there)
    edge_segments: dict[str, list] = field(default_factory=dict)
    # edge id -> the instrumentation objects it became (measuring / signal line, ...)
    edge_signals: dict[str, list] = field(default_factory=dict)


def _tag_name(shape: Shape, tag_lookup: dict[str, list[str]]) -> str | None:
    # A plain box whose own text holds a tag ("V201") keeps it: a caption below it
    # ("Injection point") was winning the nearest-label lookup.
    if (shape.shape_token or "").startswith("plain:") and re.search(r"\b[A-Z]{1,4}-?\d{2,4}[A-Z]?\b", shape.text or ""):
        return shape.text
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
        plain = (shape.shape_token or "").startswith("plain:")
        kwargs["typeName"] = (shape.text or "unknown") if plain else (shape.shape_token or "unknown")
    return cls(**kwargs)


def _allocate_nozzle(equipment) -> "eq.Nozzle":
    """Add and return a brand-new Nozzle on this equipment - called once per edge
    endpoint that resolves to this equipment, so distinct connections always get
    distinct attachment points (see _build_equipment's docstring for why this matters
    for recirculation loops)."""
    nozzle = eq.Nozzle(subTagName=f"N{len(equipment.nozzles) + 1}", nodes=[piping.PipingNode()])
    equipment.nozzles.append(nozzle)
    return nozzle


_TAG_PARTS = re.compile(r"^\s*([A-Za-z]+)[\s\-_]*(.*?)\s*$")


def _tag_parts(tag: str) -> tuple[str, str]:
    """"PI 202" -> ("PI", "202"), "TI 201/2" -> ("TI", "201/2"), "GV101" -> ("GV", "101")."""
    m = _TAG_PARTS.match(tag or "")
    return (m.group(1), m.group(2)) if m else ("", tag or "")


def _tag_attribute(tag: str) -> CustomStringAttribute:
    return CustomStringAttribute(attributeName="TagName", attributeURI="urn:pid-dexpi-aas:TagName", value=tag)


def _build_piping_item(shape: Shape, mapping: MappingResult, tag_name: str | None = None):
    """A valve or other piping component, carrying its tag: DEXPI's piping component name
    ("GV101") and number ("101"); a class without those fields (SafetyValveOrFitting)
    gets a TagName custom attribute so the tag is never lost."""
    cls = _PIPING_CLASSES[mapping.dexpi_class_name]
    obj = cls(nodes=[piping.PipingNode(), piping.PipingNode()])
    if tag_name:
        fields = type(obj).model_fields
        if "pipingComponentName" in fields:
            obj.pipingComponentName = tag_name
            obj.pipingComponentNumber = _tag_parts(tag_name)[1] or None
        elif "customAttributes" in fields:
            obj.customAttributes = list(obj.customAttributes or []) + [_tag_attribute(tag_name)]
    return obj


def _build_instrumentation_item(shape: Shape, mapping: MappingResult, tag_name: str | None):
    """An instrument function: its tag split the DEXPI way, "PI 202" -> category "PI" +
    number "202" (a 'tagName' keyword, used before, is not a field of these classes and
    was silently dropped)."""
    cls = _INSTRUMENTATION_CLASSES[mapping.dexpi_class_name]
    obj = cls()
    if tag_name:
        fields = type(obj).model_fields
        if "processInstrumentationFunctionNumber" in fields:
            category, number = _tag_parts(tag_name)
            obj.processInstrumentationFunctionCategory = category or None
            obj.processInstrumentationFunctionNumber = number or tag_name
        elif "tagName" in fields:
            obj.tagName = tag_name
        elif "customAttributes" in fields:
            obj.customAttributes = list(obj.customAttributes or []) + [_tag_attribute(tag_name)]
    return obj


def build_shapes(
    diagram: ParsedDiagram,
    mapping_overrides: dict[str, MappingResult] | None = None,
    tag_overrides: dict[str, str | None] | None = None,
) -> BuildResult:
    """Pass 1: map + construct one DEXPI object per shape. Does not wire edges yet.

    mapping_overrides / tag_overrides (keyed by shape id) let a reviewed diagram (see
    review.py) dictate the DEXPI class and tagName instead of recomputing them from the
    draw.io shape token - that is how a reviewer's edits reach the DEXPI model.
    """
    mapping_overrides = mapping_overrides or {}
    tag_overrides = tag_overrides or {}
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
        if shape.id in tag_overrides:
            tag_name = tag_overrides[shape.id]
        else:
            tag_name = _tag_name(shape, tag_lookup) or (shape.text or None)
        # Valve/instrument subtypes come from the tag (GV -> GateValve, PSV -> safety
        # valve). In the team's drawings the tag is a SEPARATE text label next to the
        # symbol - the symbol's own text is empty on all 123 valves across the 7 files -
        # so classify from the associated tag, not shape.text. (Classifying from
        # shape.text silently made every valve a generic StraightwayValve.)
        mapping = mapping_overrides.get(shape.id) or map_shape(shape.shape_token, shape.text or tag_name or "",
                                                               shape.style)

        obj = None
        if mapping.category == Category.EQUIPMENT:
            obj = _build_equipment(shape, mapping, tag_name)
            model.conceptualModel.taggedPlantItems.append(obj)
            result.shape_objects[shape.id] = obj
            result.records.append(
                BuildRecord(shape.id, "shape", "built", mapping.dexpi_class_name)
            )

        elif mapping.category == Category.PIPING:
            obj = _build_piping_item(shape, mapping, tag_name)
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
        if obj is not None and diagram.specs.get(shape.id):
            from pid_dexpi_aas.spec_labels import apply_specs
            apply_specs(obj, diagram.specs[shape.id])

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


@dataclass
class _Link:
    """One validated process connection between two built objects (equipment, piping
    component, off-page connector or tee) - the edges it came from carry its record."""
    edge_ids: list
    a: object
    b: object
    edge: object = None
    reason: str = ""
    dir: int = 0  # flow: +1 a -> b, -1 b -> a, 0 unknown (flow_direction.py)
    basis: str = ""


def _is_instrument(obj) -> bool:
    return isinstance(obj, inst.ProcessInstrumentationFunction)


_OFF_PAGE = (piping.FlowInPipeOffPageConnector, piping.FlowOutPipeOffPageConnector)


_PIPE_TWIN: dict[tuple[int, bool, str], object] = {}


def _as_pipe_end(obj, as_source: bool, ctx: str = ""):
    """draw.io's offPageConnector shape maps to a SIGNAL off-page connector, but the team
    also uses it on process pipes (V102/V204 inlets). A pipe can only end on a PIPE
    off-page connector, so on a process line use one (one per drawn connector)."""
    if isinstance(obj, _OFF_PAGE) and ctx:
        # A text-label destination ("Fume hood [atm]") with several lines: same rule as below.
        key = (id(obj), as_source, ctx)
        twin = _PIPE_TWIN.get(key)
        if twin is None:
            twin = _PIPE_TWIN[key] = type(obj)(nodes=[piping.PipingNode()])
        return twin
    if isinstance(obj, inst.SignalOffPageConnector):
        # One per line (or per pipe network) and direction: an off-page connector has a
        # single connection point, so two pipes leaving the same drawn box are two
        # connectors in DEXPI (sharing one made pyDEXPI reject the second segment).
        key = (id(obj), as_source, ctx)
        twin = _PIPE_TWIN.get(key)
        if twin is None:
            cls = piping.FlowInPipeOffPageConnector if as_source else piping.FlowOutPipeOffPageConnector
            twin = _PIPE_TWIN[key] = cls(nodes=[piping.PipingNode()])
        return twin
    return obj




def _add_network_links(diagram, build_result, shapes_by_id, links, instrument_links=None) -> None:
    """Pipe networks (lines joined at junctions, parser_drawio.join_pipe_networks) arrive as
    pairwise "net" edges. Build each network ONCE: 2 items -> one link; k >= 3 items ->
    k - 2 PipeTee fittings in a chain, one link per leg - instead of k(k-1)/2 parallel
    segments, which wrote every item into several segments."""
    groups: dict[str, list] = {}
    for e in diagram.edges:
        if e.id.startswith("net") and e.style == "pipe network":
            groups.setdefault(e.id.split(":", 1)[0], []).append(e)
    member_sense = getattr(diagram.flow, "member_sense", {}) if diagram.flow else {}
    for net_id, edges in groups.items():
        members, sensing = [], []
        for e in edges:
            for end, as_src in ((e.source, True), (e.target, False)):
                obj = _get_or_build_off_page_connector(end, build_result, shapes_by_id, as_source=as_src)
                if obj is None:
                    continue
                if _is_instrument(obj):
                    sensing.append(end)
                    continue
                obj = _as_pipe_end(obj, as_src, net_id)
                if not any(m is obj for m, _ in members):
                    members.append((obj, member_sense.get((net_id, end), (0, ""))))
        # feeders first, fed last: the tee chain then carries flow from one end to the other
        members.sort(key=lambda m: -m[1][0])
        k = len(members)
        tees = [piping.PipeTee(nodes=[piping.PipingNode(), piping.PipingNode(), piping.PipingNode()])
                for _ in range(max(0, k - 2))]
        ids = [e.id for e in edges]
        if k == 2:
            (m0, (s0, b0)), (m1, (s1, b1)) = members
            d = 1 if (s0 == 1 or s1 == -1) else -1 if (s0 == -1 or s1 == 1) else 0
            links.append(_Link(ids, m0, m1, reason=f"pipe network {net_id}", dir=d, basis=b0 or b1))
        elif k >= 3:
            # chain of k-2 tees, three legs each: tee0 = m0, m1, tee1; tee_i = tee_i-1,
            # m_i+1, tee_i+1; last tee = previous tee, m_k-2, m_k-1
            reason = f"pipe network {net_id}: {k} items joined by {len(tees)} tee(s)"
            for j, (m, (sense, basis)) in enumerate(members):
                tee = tees[max(0, min(j - 1, len(tees) - 1))]
                links.append(_Link(ids if j == 0 else [], m, tee, reason=reason, dir=sense, basis=basis))
            for i in range(len(tees) - 1):
                links.append(_Link([], tees[i], tees[i + 1], reason=reason))
        # instruments tapped onto the network measure at it: at its first tee, else at the
        # segment its two pipe items make, else at its one pipe item
        if sensing and instrument_links is not None and members:
            where = tees[0] if tees else _NetworkSpot(net_id, [m for m, _ in members])
            inst_ids = {e.id for e in edges if any(_is_instrument(o) for o in (
                build_result.shape_objects.get(e.source), build_result.shape_objects.get(e.target)))}
            for sid in dict.fromkeys(sensing):
                instrument_links.append(_InstrumentLink(
                    sorted(i for i in inst_ids if sid in i), build_result.shape_objects[sid], where, False, 0))
        if k < 2:
            for e in edges:
                if not sensing or not members:
                    build_result.records.append(BuildRecord(
                        e.id, "edge", "skipped", None, "pipe network endpoints not built"))
        else:
            # every pairwise net edge is represented by the network's segments
            for e in edges:
                both_pipe = all(not _is_instrument(o) for o in (
                    _get_or_build_off_page_connector(e.source, build_result, shapes_by_id, True),
                    _get_or_build_off_page_connector(e.target, build_result, shapes_by_id, False)) if o is not None)
                if both_pipe:  # instrument edges are recorded by _wire_instruments
                    build_result.records.append(BuildRecord(
                        e.id, "edge", "built", "PipingNetworkSegment",
                        f"via pipe network {net_id} ({k} items, {len(tees)} tee(s))", confidence=e.source_resolution))


@dataclass
class _InstrumentLink:
    """A drawn connection with an instrument at one or both ends."""
    edge_ids: list
    a: object
    b: object  # an instrument, a built item, or a _NetworkSpot / PipeTee of a pipe network
    signal: bool  # drawn dashed
    dir: int = 0  # +1 a -> b, -1 b -> a, 0 unknown


@dataclass
class _NetworkSpot:
    """Where an instrument taps a pipe network of two pipe items: their segment."""
    net_id: str
    members: list


def _sensing_location(obj, system):
    """Where an instrument measures, as DEXPI allows it (a SensingLocation): a valve, fitting,
    tee or segment as it is; equipment through a nozzle of its own; a pipe network through
    the segment joining its items. None for what cannot be one (an off-page connector)."""
    from pydexpi.dexpi_classes.pydantic_classes import SensingLocation

    if isinstance(obj, _NetworkSpot):
        for seg in system.segments:
            ends = list(seg.items) + [seg.sourceItem, seg.targetItem]
            if all(any(m is x or (getattr(x, "id", None) and x in getattr(m, "nozzles", [])) for x in ends)
                   for m in obj.members):
                return seg
        m = obj.members[0]
        return _sensing_location(m, system)
    if isinstance(obj, eq.Equipment):
        return _allocate_nozzle(obj)
    return obj if isinstance(obj, SensingLocation) else None


def _segment_holding(obj, system):
    for seg in system.segments:
        if any(obj is x for x in seg.items) or seg.sourceItem is obj or seg.targetItem is obj:
            return seg
    return None


def _instrument_number(pif) -> str | None:
    return getattr(pif, "processInstrumentationFunctionNumber", None)


def _wire_instruments(build_result, instrument_links, system) -> None:
    """Instrument connections, the DEXPI way (an instrument owns its wiring):
      - instrument - instrument: a SignalLineFunction from one to the other;
      - instrument - valve over a dashed (signal) line: the instrument actuates it - an
        ActuatingFunction located on the valve's segment, reached by a SignalLineFunction;
      - instrument - anything else (vessel, valve, tee, pipe): it measures there - a
        ProcessSignalGeneratingFunction at that SensingLocation (a nozzle for equipment),
        and a MeasuringLineFunction from it to the instrument.
    Flow known on the line (+1 / -1) orders the signal line; otherwise as drawn."""
    for ln in instrument_links:
        a, b = (ln.b, ln.a) if ln.dir == -1 else (ln.a, ln.b)
        made, what = [], ""
        if _is_instrument(a) and _is_instrument(b):
            line = inst.SignalLineFunction(source=a, target=b)
            a.signalConveyingFunctions.append(line)
            made, what = [line], "SignalLineFunction"
        else:
            pif, other = (a, b) if _is_instrument(a) else (b, a)
            if ln.signal and isinstance(other, piping.PipingComponent) and not isinstance(other, piping.PipeTee):
                seg = _segment_holding(other, system)
                act = inst.ActuatingFunction(actuatingFunctionNumber=_instrument_number(pif), actuatingLocation=seg)
                tag = getattr(other, "pipingComponentName", None)
                if tag:
                    act.customAttributes = [CustomStringAttribute(
                        attributeName="ActuatedItem", attributeURI="urn:pid-dexpi-aas:ActuatedItem", value=tag)]
                line = inst.SignalLineFunction(source=pif, target=act)
                pif.actuatingFunctions.append(act)
                pif.signalConveyingFunctions.append(line)
                made, what = [act, line], "ActuatingFunction + SignalLineFunction"
            else:
                where = _sensing_location(other, system)
                if where is None:
                    for eid in ln.edge_ids:
                        build_result.records.append(BuildRecord(
                            eid, "edge", "skipped", None,
                            f"instrument line to a {type(other).__name__} - not a place DEXPI can measure at"))
                    continue
                sensor = inst.ProcessSignalGeneratingFunction(
                    processSignalGeneratingFunctionNumber=_instrument_number(pif), sensingLocation=where)
                line = inst.MeasuringLineFunction(source=sensor, target=pif)
                pif.processSignalGeneratingFunctions.append(sensor)
                pif.signalConveyingFunctions.append(line)
                made, what = [sensor, line], f"ProcessSignalGeneratingFunction at {type(where).__name__} + MeasuringLineFunction"
        for eid in ln.edge_ids:
            build_result.edge_signals.setdefault(eid, []).extend(made)
            build_result.records.append(BuildRecord(eid, "edge", "built", what))


def _propagate_link_flow(links) -> None:
    """Carry known flow directions across the built links (through in-line items and the
    tee fittings of pipe networks) - same rules as on the drawing, flow_direction.py."""
    from pid_dexpi_aas.flow_direction import Leg, propagate

    legs = [Leg(ln.a, ln.b, ln.dir, ln.basis) for ln in links]
    propagate(legs, passes=lambda o: not _is_instrument(o), junction=lambda o: isinstance(o, piping.PipeTee))
    for ln, leg in zip(links, legs):
        ln.dir, ln.basis = leg.dir, leg.basis


def _with_class(o, cls):
    """The same off-page connector as the other flow class (FlowIn <-> FlowOut)."""
    if isinstance(o, cls):
        return o
    return cls(**{k: getattr(o, k) for k in cls.model_fields if k not in ("id",) and hasattr(o, k)})


def _build_segments(build_result, links, system) -> None:
    """Chain links into DEXPI segments so every in-line piping component (valve, tee) sits
    INSIDE exactly one segment. Segments run between equipment nozzles, off-page
    connectors and branch points; a component with more than two legs (a tee) is inside
    the segment of its first two legs and is the end point of the others.

    Why: a segment that merely ENDS on a valve loses that end when the DEXPI file is read
    back (pyDEXPI resolves segment ends only to nozzles or to items inside segments) -
    measured: 43 of 43 valve ends lost on LightUp before this - so valves must be items.
    """
    adj: dict[int, list[int]] = {}
    for i, ln in enumerate(links):
        for o in (ln.a, ln.b):
            adj.setdefault(id(o), []).append(i)

    def is_component(o) -> bool:
        return isinstance(o, piping.PipingNetworkSegmentItem) and not isinstance(o, _OFF_PAGE)

    def passes_through(o, via: int) -> int | None:
        """The link to continue on if `o` is entered via link `via` on its run, else None."""
        legs = adj.get(id(o), [])
        if is_component(o) and len(legs) >= 2 and via in legs[:2]:
            return legs[1] if via == legs[0] else legs[0]
        return None

    used: set[int] = set()
    flow_score = [0]

    def walk(start, first: int):
        """Follow links from `start` through pass-through components; returns (items in
        order, end object, indices of the links walked)."""
        items, path, cur, prev = [], [], first, start
        while True:
            used.add(cur)
            path.append(cur)
            ln = links[cur]
            flow_score[0] += ln.dir if ln.a is prev else -ln.dir  # >0: flow runs start -> end
            nxt = ln.b if ln.a is prev else ln.a
            cont = passes_through(nxt, cur)
            if cont is None or cont in used:
                return items, nxt, path
            items.append(nxt)
            prev, cur = nxt, cont

    def starts():
        for i, ln in enumerate(links):
            for o in (ln.a, ln.b):
                if passes_through(o, i) is None:  # equipment, off-page, branch leg, dead end
                    yield o, i
        for i, ln in enumerate(links):  # leftover: closed loops of components only
            yield ln.a, i

    plans = []  # [items, start, end, link indices] - constructed after the loop fix-up below
    for start, first in starts():
        if first in used:
            continue
        flow_score[0] = 0
        items, end, seg_links = walk(start, first)
        # Orientation: source -> target in the flow direction (drawn arrows, carried along
        # by flow_direction.py); unknown -> an inflow off-page connector is the source, an
        # outflow one the target, else the drawing order.
        score = flow_score[0]
        if score < 0 or (score == 0 and (isinstance(end, piping.FlowInPipeOffPageConnector)
                                         or isinstance(start, piping.FlowOutPipeOffPageConnector))):
            start, end, items = end, start, items[::-1]
        # An off-page connector's class follows the flow: where it enters the sheet it is a
        # FlowIn (the source), where it leaves a FlowOut (the target).
        if isinstance(end, _OFF_PAGE):
            end = _with_class(end, piping.FlowOutPipeOffPageConnector)
        if isinstance(start, _OFF_PAGE):
            start = _with_class(start, piping.FlowInPipeOffPageConnector)
        bases = {links[i].basis for i in seg_links if links[i].dir}
        basis = "unknown - drawing order" if not score else "arrow" if bases == {"arrow"} else "implied from neighbouring arrows"
        plans.append([list(items), start, end, seg_links, basis])

    # A component that ended up inside no segment (a tee or valve on a closed loop of
    # pipes, reached only as an end point) is placed inside the first segment ending on
    # it - otherwise every segment ending there loses that end on reload.
    inside = {id(o) for p in plans for o in p[0]}
    for p in plans:
        for pos in (1, 2):
            o = p[pos]
            if is_component(o) and id(o) not in inside:
                p[0] = [o] + p[0] if pos == 1 else p[0] + [o]
                p[pos] = None
                inside.add(id(o))

    for items, start, end, seg_links, basis in plans:
        seg_items = list(items)
        src = _endpoint(start) if start is not None else None
        tgt = _endpoint(end) if end is not None else None
        if isinstance(start, _OFF_PAGE):
            seg_items.insert(0, start)
        if isinstance(end, _OFF_PAGE):
            seg_items.append(end)
        if start is not None and is_component(start) and len(adj.get(id(start), [])) == 1:  # dead end
            seg_items.insert(0, start); src = None
        if end is not None and is_component(end) and len(adj.get(id(end), [])) == 1:
            seg_items.append(end); tgt = None
        # A connector that is also an item must be the first (source) / last (target) item;
        # anything else (a loop back onto an item) is left unconnected on that end.
        if src is not None and any(src is i for i in seg_items) and seg_items[0] is not src:
            src = None
        if tgt is not None and any(tgt is i for i in seg_items) and seg_items[-1] is not tgt:
            tgt = None
        # pyDEXPI: one connection per gap - between items, plus one to each end that is
        # NOT itself an item (an off-page connector end is both connector and item).
        n_conn = len(seg_items) + 1 - (src is not None and any(src is i for i in seg_items)) \
            - (tgt is not None and any(tgt is i for i in seg_items))
        edge_ids = [e for i in seg_links for e in links[i].edge_ids]
        try:
            segment = pt.construct_new_segment(
                segment_items=seg_items,
                segment_connections=[piping.PipingConnection() for _ in range(max(n_conn, 1))],
                source_connector_item=src, target_connector_item=tgt,
                connectivity_convention=pt.PipingConnectionConvention.USE_ITEMS,
            )
        except Exception as exc:  # pyDEXPI raises DexpiConnectionException & friends
            try:
                segment = _segment_by_hand(seg_items, src, tgt)
            except Exception:
                for eid in edge_ids:
                    build_result.records.append(BuildRecord(eid, "edge", "skipped", None, f"piping_toolkit error: {exc}"))
                continue
        segment.customAttributes = list(segment.customAttributes or []) + [CustomStringAttribute(
            attributeName="FlowDirectionBasis", attributeURI="urn:pid-dexpi-aas:flow-direction-basis", value=basis)]
        system.segments.append(segment)
        for eid in edge_ids:
            build_result.edge_segments.setdefault(eid, []).append(segment)
        for i in seg_links:
            ln = links[i]
            if ln.edge is not None:
                e = ln.edge
                confidence = e.source_resolution if e.source_resolution == e.target_resolution else "mixed"
                build_result.records.append(BuildRecord(e.id, "edge", "built", "PipingNetworkSegment", confidence=confidence))


def _segment_by_hand(items, src, tgt):
    """Build a segment without pyDEXPI's toolkit, for the one shape it rejects: an inflow
    off-page connector as first item (and source) followed by two or more items - a
    toolkit bug (it then makes the connector the TARGET of a connection; with one item it
    works). Chain: [src ->] item0 -> item1 -> ... [-> tgt], each connection from the
    previous object's outlet node (its last node) to the next one's inlet node (node 0).
    """
    chain = list(items)
    if src is not None and not (chain and chain[0] is src):
        chain.insert(0, src)
    if tgt is not None and not (chain and chain[-1] is tgt):
        chain.append(tgt)
    connections = [
        piping.PipingConnection(sourceItem=a, sourceNode=a.nodes[-1], targetItem=b, targetNode=b.nodes[0])
        for a, b in zip(chain, chain[1:])
    ]
    return piping.PipingNetworkSegment(
        items=list(items), connections=connections,
        sourceItem=src, sourceNode=src.nodes[-1] if src is not None else None,
        targetItem=tgt, targetNode=tgt.nodes[0] if tgt is not None else None,
    )


def _endpoint(o):
    """Segment connector for an end object: a fresh nozzle for equipment, else itself."""
    if isinstance(o, eq.Equipment):
        return _allocate_nozzle(o)
    return o


def wire_edges(diagram: ParsedDiagram, build_result: BuildResult) -> None:
    """Pass 2: for every edge whose endpoints both mapped to piping-capable objects,
    build a PipingNetworkSegment connecting them. Valve-shaped endpoints become segment
    items; equipment/instrument endpoints become source/target connector items.
    """
    system = piping.PipingNetworkSystem(segments=[])
    shapes_by_id = {shape.id: shape for shape in diagram.shapes}
    _PIPE_TWIN.clear()
    links: list[_Link] = []  # validated process connections, chained into segments below
    instrument_links: list[_InstrumentLink] = []  # wired after the segments (_wire_instruments)

    for edge in diagram.edges:
        if edge.id.startswith("net") and edge.style == "pipe network":
            continue  # a pipe network's derived connections - handled by _add_network_links
        if edge.merged_into is not None:
            build_result.records.append(
                BuildRecord(edge.id, "edge", "skipped", None,
                            f"part of pipe network {edge.merged_into} (lines joined at a junction) - "
                            "built as that network's derived connections")
            )
            continue
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
        if source_obj is target_obj and not isinstance(source_obj, eq.Equipment):
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

        if _is_instrument(source_obj) or _is_instrument(target_obj):
            # an instrument's connection: wired as measuring / signal / actuating once the
            # pipe segments exist (an actuator sits on its valve's segment)
            instrument_links.append(_InstrumentLink([edge.id], source_obj, target_obj, edge.is_dashed, edge.flow))
            continue

        # An off-page connector must be an internal segment item, per the DEXPI/Proteus
        # convention the parser itself follows (PipingNetworkSegmentParser's docstring:
        # "In accordance with the DEXPI convention" - the first/last item IS the
        # source/target when it's an off-page connector, there is no external
        # <Connection> reference to it). A bare 0-item segment with the connector as
        # sourceItem/targetItem is legal at the pydantic level but has no schema-correct
        # export path (confirmed: <PlantModel>/<PipingNetworkSystem> both have closed,
        # enumerated child-tag lists with no slot for a standalone connector reference).
        links.append(_Link([edge.id], _as_pipe_end(source_obj, True, edge.id), _as_pipe_end(target_obj, False, edge.id), edge,
                           dir=edge.flow, basis=edge.flow_basis))

    _add_network_links(diagram, build_result, shapes_by_id, links, instrument_links)
    _propagate_link_flow(links)
    _build_segments(build_result, links, system)
    _wire_instruments(build_result, instrument_links, system)

    if system.segments:
        build_result.model.conceptualModel.pipingNetworkSystems.append(system)


def build_model(
    diagram: ParsedDiagram,
    mapping_overrides: dict[str, MappingResult] | None = None,
    tag_overrides: dict[str, str | None] | None = None,
) -> BuildResult:
    result = build_shapes(diagram, mapping_overrides, tag_overrides)
    wire_edges(diagram, result)
    return result
