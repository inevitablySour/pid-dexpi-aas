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


def _route_orthogonal(
    source_box: tuple[float, float, float, float],
    target_box: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Route a connector between two axis-aligned boxes (min_x, min_y, max_x, max_y in
    DEXPI space) as a right-angle ("elbow") path, standard P&ID/schematic drawing
    convention - straight diagonal lines between shape centers (the previous behavior)
    don't read as a real P&ID no matter how correct the underlying connection is.

    Picks an exit/entry side on each box (whichever pair of sides face each other most
    directly) and connects them with a single mid-point bend, matching how draw.io's own
    default "orthogonal" edge style routes an unconstrained connector.
    """
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    s_cx, s_cy = (sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0
    t_cx, t_cy = (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0

    horizontal_gap = tx0 - sx1 if tx0 > sx1 else (sx0 - tx1 if sx0 > tx1 else 0.0)
    vertical_gap = ty0 - sy1 if ty0 > sy1 else (sy0 - ty1 if sy0 > ty1 else 0.0)

    if horizontal_gap >= vertical_gap:
        # Boxes are more separated left-right: exit/enter through the facing vertical
        # sides, bend at the horizontal midpoint between them.
        start_x = sx1 if tx0 >= sx1 else (sx0 if sx0 >= tx1 else s_cx)
        end_x = tx0 if tx0 >= sx1 else (tx1 if sx0 >= tx1 else t_cx)
        mid_x = (start_x + end_x) / 2.0
        return [(start_x, s_cy), (mid_x, s_cy), (mid_x, t_cy), (end_x, t_cy)]
    else:
        # Boxes are more separated top-bottom: exit/enter through the facing horizontal
        # sides, bend at the vertical midpoint between them.
        start_y = sy1 if ty0 >= sy1 else (sy0 if sy0 >= ty1 else s_cy)
        end_y = ty0 if ty0 >= sy1 else (ty1 if sy0 >= ty1 else t_cy)
        mid_y = (start_y + end_y) / 2.0
        return [(s_cx, start_y), (s_cx, mid_y), (t_cx, mid_y), (t_cx, end_y)]


def build_diagram(diagram: ParsedDiagram, result: BuildResult, model: DexpiModel) -> Diagram:
    """Construct and return a Diagram; also sets model.diagram."""
    representation_groups: list[RepresentationGroup] = []
    tag_lookup = build_tag_lookup(diagram)

    # drawio Y grows downward from each shape's top edge; DEXPI Y grows upward, so a
    # shape's DEXPI-space top-left corner is at (x, -y) and its bottom edge (visually
    # lower) is at DEXPI Y = -(y + height), i.e. numerically smaller.
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    # shape.id -> DEXPI-space bounding box (min_x, min_y, max_x, max_y), needed to
    # route piping edges as orthogonal lines below.
    shape_boxes: dict[str, tuple[float, float, float, float]] = {}

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

        # tag_lookup only covers floating labels associated to a nearby shape
        # (tag_association.py); shapes that carry their own text directly - e.g. every
        # instrument bubble ("PI 201", "TT 203") - never appear there and were silently
        # rendered with no label at all until this fallback was added (confirmed against
        # the gramflow reactor file: 19 instrument bubbles, all missing their tags).
        tags = tag_lookup.get(shape.id) or ([shape.text] if shape.text else [])
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

        # DEXPI-space box: top-left is (x, -y), extending +width in x and -height in y
        # (matching the same Y-flip used for the ShapeUsage position above).
        box_min_x = shape.x
        box_max_x = shape.x + width
        box_min_y = -(shape.y + height)
        box_max_y = -shape.y
        shape_boxes[shape.id] = (box_min_x, box_min_y, box_max_x, box_max_y)

        min_x = min(min_x, shape.x)
        max_x = max(max_x, shape.x + width)
        min_y = min(min_y, -(shape.y + height) - 15.0)  # extra room for the tag label
        max_y = max(max_y, -shape.y)

    edge_elements = []
    for edge in diagram.edges:
        source_box = shape_boxes.get(edge.source) if edge.source else None
        target_box = shape_boxes.get(edge.target) if edge.target else None
        if source_box is None or target_box is None:
            continue  # matches build_model.wire_edges' own "not built" skip condition
        routed_points = _route_orthogonal(source_box, target_box)
        edge_elements.append(
            ConnectorLine(
                innerPoints=[Point(x=x, y=y) for x, y in routed_points], stroke=_LINE_STROKE
            )
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
