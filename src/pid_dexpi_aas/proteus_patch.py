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
PipingComponents, never multi-item segments with internal DirectPipingConnections). It does
NOT attempt to handle instrumentation loop functions, actuating systems, or process signal
generating systems, since build_model.py doesn't produce those yet either.

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
- <ProcessInstrumentationFunction ID="..."> as a direct child of <PlantModel>, no nested
  signal wiring (build_model.py doesn't produce signalConveyingFunctions yet).
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

        if len(segment.items) > 1:
            raise NotImplementedError(
                f"PipingNetworkSegmentExporter only supports 0 or 1 items per segment "
                f"(got {len(segment.items)}); multi-item segments need interleaved "
                f"item/CenterLine export not yet implemented in this local patch."
            )

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
        for endpoint in (segment.sourceItem, segment.targetItem):
            if isinstance(endpoint, piping.PipingNetworkSegmentItem) and endpoint not in segment.items:
                self.endpoint_exporters.append(PipingComponentExporter(context, endpoint))

        self.register_submodule_list(self.item_exporters)
        self.register_submodule_list(self.endpoint_exporters)

    def compositional_pass(self) -> ET.Element:
        elem = ET.Element("PipingNetworkSegment")
        elem.set("ID", self.make_id_attribute(self.segment))

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
    """Exports a ProcessInstrumentationFunction with no nested signal wiring.

    SCOPE LIMIT: build_model.py does not populate signalConveyingFunctions/
    processSignalGeneratingFunctions/actuatingFunctions, so this exporter does not
    attempt the <InformationFlow>/<Association> wiring documented in the parser for
    those fields - only the bare tagged function itself is exported.
    """

    def __init__(self, context: ModuleContext, pif: inst.ProcessInstrumentationFunction) -> None:
        super().__init__(context)
        self.pif = pif
        self.generic_attribute_exporter = GenericAttributeExporter(context, pif)

    def compositional_pass(self) -> ET.Element:
        elem = ET.Element("ProcessInstrumentationFunction")
        elem.set("ID", self.make_id_attribute(self.pif))
        elem.set(*make_default_component_class_uri("ProcessInstrumentationFunction"))
        self.generic_attribute_exporter.compose_and_add_attributes(elem)
        self.register_object(self.pif.id, elem)
        return elem


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
