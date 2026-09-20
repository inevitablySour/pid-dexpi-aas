"""Build a pyDEXPI Diagram (real symbols, real positions) from a parsed drawio diagram.

pyDEXPI 1.2.0's own Proteus/loaders code never populates DexpiModel.diagram - confirmed
by direct inspection (proteus_serializer.py never references Diagram/Shape, and there is
no ConceptualModel -> Diagram builder anywhere in the installed package). This module is
that missing builder, using symbol_library.py's hand-drawn ISO-10628-style shapes and the
original drawio x/y/width/height coordinates (already parsed by parser_drawio.py) for
layout, so DrawDiagram (pydexpi.loaders.svg_loader) can render a real, recognizable P&ID
picture rather than an empty canvas.

Coordinate systems: drawio/mxGraph uses screen convention (Y increases downward). pyDEXPI's
Diagram/Point positions use DEXPI's own math convention (Y increases upward) - confirmed by
reading svg_loader.py's SvgRenderer.to_svg_y docstring verbatim: "DEXPI uses mathematical
coordinate system (Y increases upward), SVG uses screen coordinate system (Y increases
downward)", and it negates Y when converting. So every drawio Y coordinate is negated here
before being placed in a Point/Diagram field, or shapes render upside-down/off-canvas
(confirmed empirically: without the flip, everything came out with large negative Y and an
uninitialized-looking viewBox).
"""
from __future__ import annotations

from pydexpi.dexpi_classes.graphics import (
    Color,
    ConnectorLine,
    DashStyle,
    Diagram,
    Point,
    RepresentationGroup,
    Shape,
    ShapeUsage,
    Static,
    Stroke,
    Text,
    TextAlignment,
)
from pydexpi.dexpi_classes.pydantic_classes import DexpiBaseModel, DexpiModel

from pid_dexpi_aas.build_model import BuildResult
from pid_dexpi_aas.parser_drawio import ParsedDiagram
from pid_dexpi_aas.symbol_library import build_primitives_for_class
from pid_dexpi_aas.tag_association import build_tag_lookup

_BACKGROUND = Color(r=255, g=255, b=255)
_PADDING = 20.0
_LINE_STROKE = Stroke(color=Color(r=0, g=0, b=0), dashStyle=DashStyle.Solid, width=1.5)
_LABEL_COLOR = Color(r=0, g=0, b=0)


def build_diagram(diagram: ParsedDiagram, result: BuildResult, model: DexpiModel) -> Diagram:
    """Construct and return a Diagram; also sets model.diagram."""
    representation_groups: list[RepresentationGroup] = []
    tag_lookup = build_tag_lookup(diagram)

    # drawio Y grows downward from each shape's top edge; DEXPI Y grows upward, so a
    # shape's DEXPI-space top-left corner is at (x, -y) and its bottom edge (visually
    # lower) is at DEXPI Y = -(y + height), i.e. numerically smaller.
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    # shape.id -> DEXPI-space center point, needed to draw piping edges as lines below.
    shape_centers: dict[str, Point] = {}

    for shape in diagram.shapes:
        dexpi_object = result.shape_objects.get(shape.id)
        if dexpi_object is None or not isinstance(dexpi_object, DexpiBaseModel):
            continue  # unmapped/skipped shapes (see BuildRecord) have no DEXPI object

        class_name = type(dexpi_object).__name__
        primitives = build_primitives_for_class(class_name)
        symbol_shape = Shape(name=class_name, primitives=primitives)

        # symbol_library draws every shape in a normalized 0-100 box (Y increases
        # downward within that local box, matching how the box was authored); scale it
        # to this shape's actual drawio width/height, then place its origin at the
        # shape's DEXPI-space top-left corner (x, -y).
        width = shape.width or 100.0
        height = shape.height or 100.0
        shape_usage = ShapeUsage(
            isMirrored=False,
            position=Point(x=shape.x, y=-shape.y),
            rotation=0.0,
            scaleX=width / 100.0,
            scaleY=-(height / 100.0),  # flip the local box's downward Y into DEXPI's upward Y
            shape=symbol_shape,
        )

        elements: list = [shape_usage]

        tags = tag_lookup.get(shape.id) or []
        if tags:
            label_text = Text(
                position=Point(x=shape.x, y=-(shape.y + height) - 15.0),
                rotation=0.0,
                alignment=TextAlignment.CenterTop,
                color=_LABEL_COLOR,
                font="Arial",
                size=14.0,
                text=", ".join(tags),
            )
            elements.append(label_text)

        representation_groups.append(
            RepresentationGroup(represents=dexpi_object, groups=[Static(elements=elements)])
        )

        center_x = shape.x + width / 2.0
        center_y = -(shape.y + height / 2.0)
        shape_centers[shape.id] = Point(x=center_x, y=center_y)

        min_x = min(min_x, shape.x)
        max_x = max(max_x, shape.x + width)
        min_y = min(min_y, -(shape.y + height) - 15.0)  # extra room for the tag label
        max_y = max(max_y, -shape.y)

    edge_elements = []
    for edge in diagram.edges:
        source_center = shape_centers.get(edge.source) if edge.source else None
        target_center = shape_centers.get(edge.target) if edge.target else None
        if source_center is None or target_center is None:
            continue  # matches build_model.wire_edges' own "not built" skip condition
        edge_elements.append(
            ConnectorLine(innerPoints=[source_center, target_center], stroke=_LINE_STROKE)
        )

    if edge_elements:
        # `represents=model` is a placeholder: RepresentationGroup.represents is only
        # used for the SVG element's id/metadata string (confirmed in svg_loader.py, not
        # validated against the actual DEXPI relationship), and build_diagram.py doesn't
        # have access to individual PipingNetworkSegment objects per edge (only
        # build_model.wire_edges does) - one group for all piping lines is good enough
        # for rendering purposes.
        representation_groups.append(
            RepresentationGroup(represents=model, groups=[Static(elements=edge_elements)])
        )

    if min_x == float("inf"):  # no shapes were built at all
        min_x = min_y = 0.0
        max_x = max_y = 100.0

    diagram_obj = Diagram(
        name=diagram.source_file,
        backgroundColor=_BACKGROUND,
        represents=model,
        minX=min_x - _PADDING,
        minY=min_y - _PADDING,
        maxX=max_x + _PADDING,
        maxY=max_y + _PADDING,
        groups=representation_groups,
    )

    model.diagram = diagram_obj
    return diagram_obj
