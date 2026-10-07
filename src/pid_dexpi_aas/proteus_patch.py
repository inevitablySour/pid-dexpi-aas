"""Local patch adding PipingNetworkSystem/ProcessInstrumentationFunction export to
pyDEXPI 1.2.0's Proteus exporter.

WHY THIS EXISTS: pyDEXPI 1.2.0's ProteusExporter/PlantModelExporter only implements
export for MetaData/Equipment/Nozzle - confirmed by reading exporter_modules.py directly
and cross-checked against the process-intelligence-research/pyDEXPI GitHub `master`
branch (same gap, not yet fixed upstream, framed as future work in their README, no
open issue as of 2026-09-20). See ../../README.md for the full writeup and the decision
record. **pydexpi is pinned to ==1.2.0 in pyproject.toml because of this file** - it reads
private internals (ModuleContext, ExporterModule, ElementRegistry) that are not a
documented public API. Before bumping pydexpi's version: re-run
tests/test_proteus_patch.py, and check whether upstream has shipped real
piping/instrumentation export (if so, delete this file and go back to stock
ProteusSerializer).

SCOPE: this patch handles the shapes build_model.py actually produces - PipingNetworkSegments
built via piping_toolkit.construct_new_segment with an empty or single-valve item list and
PipingConnectionConvention.USE_ITEMS (source/target are always equipment Nozzles or valve
PipingComponents, never multi-item segments with internal DirectPipingConnections), and the
instrument wiring build_model._wire_instruments makes. It does NOT attempt instrumentation
loop functions, actuating systems or process signal generating systems (not produced).

XML SHAPE (reverse-engineered from pydexpi's Proteus PARSER, which does understand these
classes, since no exporter or example fixture exists to copy from - see parser_modules.py's
PipingComponentParser/PipingNetworkSegmentParser/PipingNetworkSystemParser and
ProcessInstrumentationFunctionParser/InformationFlowParser):

- <PipingNetworkSystem ID="..."><PipingNetworkSegment ID="...">...</PipingNetworkSegment></PipingNetworkSystem>
  as a direct child of <PlantModel>, sibling of <Equipment>.
- Each <PipingNetworkSegment> contains, in order: its item elements (<PipingComponent
  ComponentClass="...">) each with a nested <ConnectionPoints><Node ID="..." Type="process"/>
  ...</ConnectionPoints>, and exactly one <Connection FromID="..." ToID="..." FromNode="0"
  ToNode="0"/> declaring the segment's external source/target items (resolved via the
  Proteus ID already written on that item's element, fetched from the exporter's element
  registry). NOTE: pyDEXPI 1.2.0's parser has a confirmed bug where FromNode/ToNode are
  always parsed back as None regardless of what's written (dead-code reset in
  _extract_connections) - we still write real indices for Proteus-format compliance (other
  tools may read them correctly) but our own round-trip test only asserts on
  sourceItem/targetItem identity via ID, not sourceNode/targetNode.
- <ProcessInstrumentationFunction ID="..."> as a direct child of <PlantModel>, with its
  sensors, actuating functions and measuring / signal lines nested inside it (see
  ProcessInstrumentationFunctionExporter).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO

from pydexpi.dexpi_classes import instrumentation as inst
from pydexpi.dexpi_classes import piping
from pydexpi.dexpi_classes.pydantic_classes import DexpiModel
from pydexpi.loaders import ProteusSerializer
from pydexpi.loaders.proteus_serializer.proteus_exporter.core import (
    ExporterModule,
    ModuleContext,
)
from pydexpi.loaders.proteus_serializer.proteus_exporter.exporter_modules import (
    GenericAttributeExporter,
    PlantModelExporter,
)
from pydexpi.loaders.proteus_serializer.proteus_exporter.exporter_utils import (
    make_default_component_class_uri,
)
from pydexpi.loaders.proteus_serializer.proteus_exporter.proteus_exporter import (
    ProteusExporter,
)


class PipingNodeExporter(ExporterModule):
    """Exports a single PipingNode as a <Node Type="process"> child of <ConnectionPoints>."""

    def __init__(self, context: ModuleContext, node: piping.PipingNode) -> None:
        super().__init__(context)
        self.node = node

    def compositional_pass(self) -> ET.Element:
        node_elem = ET.Element("Node")
        node_elem.set("ID", self.make_id_attribute(self.node))
        node_elem.set("Type", "process")
        self.register_object(self.node.id, node_elem)
        return node_elem


# Off-page connectors get their own dedicated XML tag per the parser (confirmed:
# PipeOffPageConnectorParser is registered against the "PipeOffPageConnector" tag,
# separate from "PipingComponent" - parser_factory.py's item-dispatch loop checks
# subelement.tag == "PipeOffPageConnector" as a distinct branch, not a ComponentClass
# value under PipingComponent). FlowIn/FlowOutPipeOffPageConnector (the only classes
# that satisfy PipingSourceItem/PipingTargetItem, needed for build_model.py's off-page
# connector fallback - see _get_or_build_off_page_connector) are subclasses of
# PipeOffPageConnector and round-trip through this same tag; the FlowIn/FlowOut
# distinction itself is not preserved by the parser (it always reconstructs plain
# PipeOffPageConnector), which is an acceptable minor fidelity loss for a text-only
# off-page destination label - the connection topology itself is preserved exactly.
_OFF_PAGE_CONNECTOR_CLASSES = (piping.PipeOffPageConnector,)


class PipingComponentExporter(ExporterModule):
    """Exports a PipingNetworkSegmentItem (e.g. a valve, or an off-page connector) as
    <PipingComponent> or <PipeOffPageConnector> depending on its class.

    Mirrors NozzleExporter/EquipmentExporter's pattern exactly (same ExporterModule
    contract), adding a <ConnectionPoints> block since piping connections need to
    reference this item's nodes by Proteus ID - a step the existing NozzleExporter
    skips because nothing in the stock exporter ever needed a nozzle's node identity.
    """

    def __init__(self, context: ModuleContext, item: piping.PipingNetworkSegmentItem) -> None:
        super().__init__(context)
        self.item = item
        self.item_element: ET.Element | None = None
        self.generic_attribute_exporter = GenericAttributeExporter(context, item)
        self.node_exporters = [
            PipingNodeExporter(context, node) for node in getattr(item, "nodes", []) or []
        ]
        self.register_submodule_list(self.node_exporters)

    def compositional_pass(self) -> ET.Element:
        class_name = type(self.item).__name__
        is_off_page_connector = isinstance(self.item, _OFF_PAGE_CONNECTOR_CLASSES)
        tag = "PipeOffPageConnector" if is_off_page_connector else "PipingComponent"
        elem = ET.Element(tag)
        elem.set("ID", self.make_id_attribute(self.item))
        # PipeOffPageConnectorParser reuses PipingComponentParser.compositional_pass
        # unconditionally (PipeOffPageConnectorParser.compositional_pass calls
        # super().compositional_pass() first thing), which always reads "ComponentClass"
        # regardless of tag - confirmed by a crash without this
        # ("attribute name must be string, not 'NoneType'" from getattr(piping, None)).
        # For a real off-page connector this class name is always "PipeOffPageConnector"
        # itself here (FlowIn/FlowOut subtypes aren't preserved through this tag, see
        # this module's _OFF_PAGE_CONNECTOR_CLASSES comment) since getattr(piping,
        # "PipeOffPageConnector") is what the parser needs to resolve back to the base
        # class successfully.
        elem.set("ComponentClass", "PipeOffPageConnector" if is_off_page_connector else class_name)
        if not is_off_page_connector:
            elem.set(*make_default_component_class_uri(class_name))

        self.generic_attribute_exporter.compose_and_add_attributes(elem)

        if self.node_exporters:
            connection_points = ET.SubElement(elem, "ConnectionPoints")
            connection_points.set("NumPoints", str(len(self.node_exporters)))
            for node_exporter in self.node_exporters:
                connection_points.append(node_exporter.compositional_pass())

        self.item_element = elem
        self.register_object(self.item.id, elem)
        return elem


def _resolve_connector_proteus_id(context: ModuleContext, obj) -> str | None:
    """Look up the already-emitted Proteus ID string for a connector object (an
    equipment Nozzle or a piping item) via the element registry, keyed by pydantic id.
    """
    if obj is None:
        return None
    element = context.element_registry.get_element(obj.id)
    return element.get("ID") if element is not None else None


class PipingNetworkSegmentExporter(ExporterModule):
    """Exports one PipingNetworkSegment.

    SCOPE LIMIT: only handles segments with zero or one internal item. In practice
    build_model.py's wire_edges() always produces zero-item segments (every mxGraph edge
    becomes a direct connection between two connector endpoints - equipment nozzles or
    valve objects directly - never a valve nested inside a longer segment), so the
    single-item path here is defensive, not currently exercised by the pipeline; confirmed
    by inspecting segment.items across all 7 team drawio files (always empty). A segment
    with 2+ items and internal DirectPipingConnections would need the full interleaved
    item/connection ordering logic described in the parser
    (see PipingNetworkSegmentParser.compositional_pass) - not implemented here.
    """

    def __init__(self, context: ModuleContext, segment: piping.PipingNetworkSegment) -> None:
        super().__init__(context)
        self.segment = segment
        self.segment_element: ET.Element | None = None
        # the segment's own attributes (build_model's FlowDirectionBasis note)
        self.generic_attribute_exporter = GenericAttributeExporter(context, segment)

        # Items are written in segment order with no CenterLine between them; the
        # Proteus reader infers a DirectPipingConnection between consecutive items
        # (PipingNetworkSegmentParser), which is exactly what build_model produces:
        # nozzle -> valve -> tee -> valve -> nozzle. Round-trip tested (items, order and
        # both ends - including a branch segment ending on a tee in another segment).
        self.item_exporters = [PipingComponentExporter(context, item) for item in segment.items]

        # A segment's sourceItem/targetItem is usually an equipment Nozzle (registered
        # via EquipmentExporter's tree) or one of segment.items (registered above) - but
        # build_model.py's off-page-connector fallback (see _get_or_build_off_page_
        # connector) produces a standalone FlowIn/FlowOutPipeOffPageConnector that is
        # NEITHER: it's not nested under any equipment, and it's not one of this
        # segment's own items (it typically IS the segment's connector item, not
        # something inside it). Nothing else ever builds an exporter for it, so without
        # this it's simply never registered and _resolve_connector_proteus_id silently
        # returns None - confirmed via a failing round-trip test (targetItem came back
        # None after reload) before this fix.
        self.endpoint_exporters = []
        # Export each standalone end item ONCE per model. An in-line valve ends two
        # segments (one per side) and a tee ends three; exporting it per segment wrote the
        # same element - same ID - several times (LightUp: 12 duplicated IDs, up to 4x),
        # which is invalid DEXPI. Later segments just reference it by ID.
        exported = context.__dict__.setdefault("_pid_dexpi_exported_endpoints", set())
        for endpoint in (segment.sourceItem, segment.targetItem):
            if (isinstance(endpoint, piping.PipingNetworkSegmentItem) and endpoint not in segment.items
                    and endpoint.id not in exported):
                exported.add(endpoint.id)
                self.endpoint_exporters.append(PipingComponentExporter(context, endpoint))

        self.register_submodule_list(self.item_exporters)
        self.register_submodule_list(self.endpoint_exporters)

    def compositional_pass(self) -> ET.Element:
        elem = ET.Element("PipingNetworkSegment")
        elem.set("ID", self.make_id_attribute(self.segment))
        self.generic_attribute_exporter.compose_and_add_attributes(elem)

        for item_exporter in self.item_exporters:
            elem.append(item_exporter.compositional_pass())

        # Build (and register) endpoint exporters now so _resolve_connector_proteus_id
        # can find them in reference_pass, but do NOT nest their elements inside this
        # <PipingNetworkSegment> - the parser interleaves item/CenterLine children by
        # document order to build the segment's internal items/connections (see
        # PipingNetworkSegmentParser.compositional_pass), so appending an unrelated
        # standalone connector here would corrupt that structure on reload. Elements are
        # returned via get_orphan_elements() for the caller to place as top-level
        # PlantModel siblings instead - the same flat, registry-resolved placement a
        # Nozzle gets (nested under its owning Equipment, referenced purely by ID).
        self.orphan_elements = [exporter.compositional_pass() for exporter in self.endpoint_exporters]

        self.segment_element = elem
        self.register_object(self.segment.id, elem)
        return elem

    def get_orphan_elements(self) -> list[ET.Element]:
        return getattr(self, "orphan_elements", [])

    def reference_pass(self) -> None:
        super().reference_pass()

        # NOTE (measured 2026-10-05): a segment ending in a pipe off-page connector item
        # does not get its ends back from pyDEXPI's reader whether or not the connector's
        # ID is written here - a reader limitation, present since the first version. The
        # XML itself carries the information; kept as originally written.
        source_id = _resolve_connector_proteus_id(self.module_context, self.segment.sourceItem)
        target_id = _resolve_connector_proteus_id(self.module_context, self.segment.targetItem)

        if source_id is None and target_id is None:
            return

        connection_elem = ET.SubElement(self.segment_element, "Connection")
        if source_id is not None:
            connection_elem.set("FromID", source_id)
            connection_elem.set("FromNode", "0")
        if target_id is not None:
            connection_elem.set("ToID", target_id)
            connection_elem.set("ToNode", "0")


class PipingNetworkSystemExporter(ExporterModule):
    def __init__(self, context: ModuleContext, system: piping.PipingNetworkSystem) -> None:
        super().__init__(context)
        self.system = system
        # Items that sit INSIDE some segment are exported there; a branch segment that
        # starts or ends on one of them (a tee) must only reference it, not export it a
        # second time as a standalone end item.
        exported = context.__dict__.setdefault("_pid_dexpi_exported_endpoints", set())
        exported.update(item.id for segment in system.segments for item in segment.items)
        self.segment_exporters = [
            PipingNetworkSegmentExporter(context, segment) for segment in system.segments
        ]
        self.register_submodule_list(self.segment_exporters)

    def compositional_pass(self) -> ET.Element:
        elem = ET.Element("PipingNetworkSystem")
        elem.set("ID", self.make_id_attribute(self.system))
        for segment_exporter in self.segment_exporters:
            elem.append(segment_exporter.compositional_pass())
        self.register_object(self.system.id, elem)
        return elem

    def get_orphan_elements(self) -> list[ET.Element]:
        orphans = []
        for segment_exporter in self.segment_exporters:
            orphans.extend(segment_exporter.get_orphan_elements())
        return orphans


class ProcessInstrumentationFunctionExporter(ExporterModule):
    """Exports a ProcessInstrumentationFunction with its wiring (build_model._wire_instruments):

        <ProcessInstrumentationFunction ID=.. ComponentClass=..>   (+ its GenericAttributes)
          <ProcessSignalGeneratingFunction ID=..>  <Association Type="is located in" ItemID=sensing location/>
          <ActuatingFunction ID=..>                <Association Type="is located in" ItemID=segment/>
          <InformationFlow ID=.. ComponentClass="MeasuringLineFunction" | "SignalLineFunction">
             <Association Type="has logical start" ItemID=../> <Association Type="has logical end" ItemID=../>
          <Association Type="is logical start of" | "is logical end of" ItemID=information flow/>

    - the shape pyDEXPI's reader takes apart (ProcessInstrumentationFunctionParser,
    ProcessSignalGeneratingFunctionParser, ActuatingFunctionParser, InformationFlowParser):
    children by tag, links by Association type, referenced items by Proteus ID. A signal
    line to another instrument is written inside its source instrument; the far end gets
    the matching "is logical end of". Round-trip tested (tests/test_proteus_patch.py)."""

    def __init__(self, context: ModuleContext, pif: inst.ProcessInstrumentationFunction) -> None:
        super().__init__(context)
        self.pif = pif
        self.generic_attribute_exporter = GenericAttributeExporter(context, pif)
        # (signal off-page connectors share this list in pyDEXPI's model and have no wiring)
        self.parts = (list(getattr(pif, "processSignalGeneratingFunctions", None) or [])
                      + list(getattr(pif, "actuatingFunctions", None) or []))
        self.flows = list(getattr(pif, "signalConveyingFunctions", None) or [])
        self.part_attrs = [GenericAttributeExporter(context, o) for o in self.parts + self.flows]
        self.elements: dict[str, ET.Element] = {}
        # every flow in the model, so an instrument can name the flows that end on it
        context.__dict__.setdefault("_pid_dexpi_flows", []).extend(self.flows)

    def compositional_pass(self) -> ET.Element:
        elem = ET.Element("ProcessInstrumentationFunction")
        elem.set("ID", self.make_id_attribute(self.pif))
        elem.set(*make_default_component_class_uri("ProcessInstrumentationFunction"))
        self.generic_attribute_exporter.compose_and_add_attributes(elem)
        for obj, attrs in zip(self.parts + self.flows, self.part_attrs):
            tag = ("InformationFlow" if isinstance(obj, inst.SignalConveyingFunction) else type(obj).__name__
                   if isinstance(obj, (inst.ProcessSignalGeneratingFunction, inst.ActuatingFunction)) else None)
            if tag is None:
                continue
            child = ET.SubElement(elem, tag)
            child.set("ID", self.make_id_attribute(obj))
            # the reader picks the class from ComponentClass (an InformationFlow without it is
            # dropped); ComponentClassURI is the DEXPI reference-data link
            child.set("ComponentClass", type(obj).__name__)
            child.set(*make_default_component_class_uri(type(obj).__name__))
            attrs.compose_and_add_attributes(child)
            self.register_object(obj.id, child)
            self.elements[obj.id] = child
        self.register_object(self.pif.id, elem)
        self.elements[self.pif.id] = elem
        return elem

    def reference_pass(self) -> None:
        super().reference_pass()
        ref = lambda obj: _resolve_connector_proteus_id(self.module_context, obj)

        def associate(owner, kind: str, obj) -> None:
            target_id = ref(obj)
            if owner is not None and target_id is not None:
                a = ET.SubElement(owner, "Association")
                a.set("Type", kind)
                a.set("ItemID", target_id)

        for part in self.parts:
            where = getattr(part, "sensingLocation", None) or getattr(part, "actuatingLocation", None)
            associate(self.elements.get(part.id), "is located in", where)
        for flow in self.flows:
            associate(self.elements.get(flow.id), "has logical start", flow.source)
            associate(self.elements.get(flow.id), "has logical end", flow.target)
        # the reverse links on this instrument and its parts, for every flow in the model
        mine = {self.pif.id} | {p.id for p in self.parts}
        for flow in self.module_context.__dict__.get("_pid_dexpi_flows", []):
            for end, kind in ((flow.source, "is logical start of"), (flow.target, "is logical end of")):
                if end is not None and end.id in mine:
                    associate(self.elements.get(end.id), kind, flow)


class PatchedPlantModelExporter(PlantModelExporter):
    """PlantModelExporter extended with piping/instrumentation export.

    Overrides rather than duplicates the base __init__/compositional_pass logic where
    possible, since the base class's equipment/metadata handling is correct as-is.
    """

    def __init__(self, context: ModuleContext, dexpi_model: DexpiModel, override_export_info: bool = False) -> None:
        super().__init__(context, dexpi_model, override_export_info=override_export_info)

        conceptual_model = dexpi_model.conceptualModel
        self.piping_system_exporters = [
            PipingNetworkSystemExporter(context, system)
            for system in conceptual_model.pipingNetworkSystems
        ]
        self.register_submodule_list(self.piping_system_exporters)

        self.pif_exporters = [
            ProcessInstrumentationFunctionExporter(context, pif)
            for pif in conceptual_model.processInstrumentationFunctions
        ]
        self.register_submodule_list(self.pif_exporters)

    def compositional_pass(self) -> ET.Element:
        plant_model = super().compositional_pass()

        for exporter in self.piping_system_exporters:
            plant_model.append(exporter.compositional_pass())

        for exporter in self.pif_exporters:
            plant_model.append(exporter.compositional_pass())

        # Standalone off-page-connector items referenced by a segment's sourceItem/
        # targetItem but not nested under any equipment or segment - see
        # PipingNetworkSegmentExporter's endpoint_exporters. Placed as flat PlantModel
        # siblings, resolved purely by Proteus ID via <Connection FromID=.../ToID=...>,
        # matching how the parser's registry-based ID resolution works for any object.
        for exporter in self.piping_system_exporters:
            for orphan_element in exporter.get_orphan_elements():
                plant_model.append(orphan_element)

        return plant_model


class PatchedProteusExporter(ProteusExporter):
    """ProteusExporter using PatchedPlantModelExporter instead of the stock one."""

    def export_xml_element(self, dexpi_model: DexpiModel) -> ET.Element:
        from pydexpi.loaders.proteus_serializer.proteus_exporter.core import ElementRegistry

        element_registry = ElementRegistry()
        self.context = ModuleContext(
            element_registry=element_registry, override_proteus_id=self.override_proteus_id
        )
        plant_model_exporter = PatchedPlantModelExporter(
            self.context, dexpi_model, override_export_info=self.override_export_info
        )
        plant_model_xml = plant_model_exporter.compositional_pass()
        plant_model_exporter.reference_pass()
        return plant_model_xml


class PatchedProteusSerializer(ProteusSerializer):
    """ProteusSerializer using PatchedProteusExporter for save(); load() is unchanged
    since the stock Proteus PARSER already supports piping/instrumentation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proteus_exporter = PatchedProteusExporter(
            override_export_info=self.proteus_exporter.override_export_info,
            override_proteus_id=self.proteus_exporter.override_proteus_id,
        )
