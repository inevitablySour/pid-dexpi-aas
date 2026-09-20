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
    FillStyle,
    Point,
    Polygon,
    RepresentationGroup,
    Shape,
    ShapeUsage,
    Static,
    Stroke,
    Text,
    TextAlignment,
)
from pydexpi.dexpi_classes import piping
from pydexpi.dexpi_classes.pydantic_classes import DexpiBaseModel, DexpiModel

from pid_dexpi_aas.build_model import BuildResult
from pid_dexpi_aas.parser_drawio import Edge, ParsedDiagram
from pid_dexpi_aas.symbol_library import build_primitives_for_class, compute_local_bounding_box
from pid_dexpi_aas.tag_association import build_tag_lookup

_BACKGROUND = Color(r=255, g=255, b=255)
_PADDING = 20.0
_LINE_COLOR = Color(r=0, g=0, b=0)
_SOLID_LINE_STROKE = Stroke(color=_LINE_COLOR, dashStyle=DashStyle.Solid, width=1.5)
_DASHED_LINE_STROKE = Stroke(color=_LINE_COLOR, dashStyle=DashStyle.Dash, width=1.5)
_LABEL_COLOR = Color(r=0, g=0, b=0)
_ARROW_LENGTH = 12.0
_ARROW_HALF_WIDTH = 5.0


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


def _arrowhead_polygon(tip: tuple[float, float], from_point: tuple[float, float]) -> Polygon:
    """A small filled triangle pointing from from_point toward tip, for drawing an
    endArrow/startArrow at a connector's end - pyDEXPI's SVG renderer draws no
    arrowheads at all (confirmed: no marker/arrow code anywhere in svg_loader.py), so
    flow direction shown in the original drawing (endArrow=classic/startArrow=classic)
    was previously lost entirely.

    Only handles axis-aligned segments (horizontal or vertical), which is all
    _route_orthogonal ever produces.
    """
    tip_x, tip_y = tip
    from_x, from_y = from_point

    if tip_y == from_y:  # horizontal segment
        direction = 1.0 if tip_x > from_x else -1.0
        base_x = tip_x - direction * _ARROW_LENGTH
        return _polygon_from_points(
            (tip_x, tip_y), (base_x, tip_y - _ARROW_HALF_WIDTH), (base_x, tip_y + _ARROW_HALF_WIDTH)
        )
    else:  # vertical segment
        direction = 1.0 if tip_y > from_y else -1.0
        base_y = tip_y - direction * _ARROW_LENGTH
        return _polygon_from_points(
            (tip_x, tip_y), (tip_x - _ARROW_HALF_WIDTH, base_y), (tip_x + _ARROW_HALF_WIDTH, base_y)
        )


def _polygon_from_points(*points: tuple[float, float]) -> Polygon:
    return Polygon(
        points=[Point(x=x, y=y) for x, y in points],
        stroke=_SOLID_LINE_STROKE,
        fillStyle=FillStyle.Solid,
    )


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

        if isinstance(dexpi_object, (piping.FlowInPipeOffPageConnector, piping.FlowOutPipeOffPageConnector)):
            # This shape is a floating TEXT LABEL (e.g. "Fume hood [atm]") that
            # build_model.py's off-page-connector fallback turned into a DEXPI object
            # purely so an edge could terminate somewhere real - it's not equipment with
            # its own symbol. Drawing the generic-box fallback symbol here produced a
            # visibly wrong second box overlapping the label/pentagon that's already
            # drawn as its own separate shape right next to it. Skip the symbol
            # entirely; still register a (tiny, point-sized) box below so connector
            # routing still terminates at the right place.
            width = shape.width or 100.0
            height = shape.height or 100.0
            box_center_x = shape.x + width / 2.0
            box_center_y = -(shape.y + height / 2.0)
            shape_boxes[shape.id] = (box_center_x, box_center_y, box_center_x, box_center_y)
            min_x = min(min_x, shape.x)
            max_x = max(max_x, shape.x + width)
            min_y = min(min_y, -(shape.y + height))
            max_y = max(max_y, -shape.y)
            continue

        primitives = build_primitives_for_class(class_name)
        symbol_shape = Shape(name=class_name, primitives=primitives)
        local_box = compute_local_bounding_box(primitives)

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

        # Transform the symbol's REAL local bounding box (compute_local_bounding_box,
        # which reflects the actual drawn geometry - e.g. a pump's inscribed circle, not
        # its full declared 0-100 box) into DEXPI space with the same scale/flip used for
        # the ShapeUsage above. Using the full declared box here (the original approach)
        # left a visible gap between a routed connector's endpoint and the symbol's
        # actual drawn edge for every inset symbol (pumps, valves, etc) - reported by the
        # user as "they still don't look connected".
        local_min_x, local_min_y, local_max_x, local_max_y = local_box
        scale_x = width / 100.0
        scale_y = height / 100.0
        box_min_x = shape.x + local_min_x * scale_x
        box_max_x = shape.x + local_max_x * scale_x
        # local Y grows downward; DEXPI Y grows upward, and the shape's origin is at
        # DEXPI Y = -shape.y, so a larger local Y maps to a SMALLER DEXPI Y.
        box_max_y = -shape.y - local_min_y * scale_y
        box_min_y = -shape.y - local_max_y * scale_y
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
        stroke = _DASHED_LINE_STROKE if edge.is_dashed else _SOLID_LINE_STROKE
        edge_elements.append(
            ConnectorLine(
                innerPoints=[Point(x=x, y=y) for x, y in routed_points], stroke=stroke
            )
        )

        if edge.has_end_arrow:
            edge_elements.append(_arrowhead_polygon(routed_points[-1], routed_points[-2]))
        if edge.has_start_arrow:
            edge_elements.append(_arrowhead_polygon(routed_points[0], routed_points[1]))

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
