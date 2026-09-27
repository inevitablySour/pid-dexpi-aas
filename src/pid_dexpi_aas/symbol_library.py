"""A small, hand-drawn ISO-10628-style P&ID symbol library.

pyDEXPI 1.2.0 ships no symbol geometry at all (confirmed by direct inspection of the
installed package: svg_loader.py is a pure renderer, dispatching only on primitive type,
never on DEXPI class name - see README.md/HANDOFF.md for the full writeup). Real symbol
proportions here are drawn from ISO 10628 / ISA-5.1 simplified line-drawing convention
(bowtie valves, circular pumps/instruments, capsule vessels, pentagon off-page
connectors), the same visual vocabulary draw.io's own mxgraph.pid.* stencils use -
independently hand-encoded here as simple point lists, not copied from any source, so no
licensing concerns.

Each symbol is defined in a normalized 0-100 x 0-100 box (Y increases downward, matching
SVG/mxGraph convention) as a list of GraphicalPrimitive objects. build_diagram.py scales
and positions these using each shape's actual drawio width/height/x/y.

Coverage: every DEXPI class mapping.py can produce (see EQUIPMENT_SHAPE_MAP,
VALVE_TAG_PREFIX_MAP) has an entry here. Anything unmapped falls back to draw_generic_box.
"""
from __future__ import annotations

from pydexpi.dexpi_classes.graphics import (
    Color,
    DashStyle,
    Ellipse,
    FillStyle,
    Point,
    PolyLine,
    Polygon,
    Stroke,
    GraphicalPrimitive,
)

_BLACK = Color(r=0, g=0, b=0)
_STROKE = Stroke(color=_BLACK, dashStyle=DashStyle.Solid, width=2.0)
_NO_FILL = FillStyle.Transparent


def _p(x: float, y: float) -> Point:
    return Point(x=x, y=y)


def _line(*coords: tuple[float, float]) -> PolyLine:
    return PolyLine(points=[_p(x, y) for x, y in coords], stroke=_STROKE)


def _polygon(*coords: tuple[float, float]) -> Polygon:
    return Polygon(points=[_p(x, y) for x, y in coords], stroke=_STROKE, fillStyle=_NO_FILL)


def _circle(cx: float, cy: float, r: float) -> Ellipse:
    return Ellipse(
        center=_p(cx, cy),
        horizontalSemiAxis=r,
        verticalSemiAxis=r,
        rotation=0.0,
        stroke=_STROKE,
        fillStyle=_NO_FILL,
    )


def draw_generic_box() -> list[GraphicalPrimitive]:
    """Fallback for any DEXPI class with no dedicated symbol below."""
    return [_polygon((5, 5), (95, 5), (95, 95), (5, 95))]


def draw_tank() -> list[GraphicalPrimitive]:
    return [_polygon((10, 15), (90, 15), (90, 90), (10, 90))]


def draw_custom_vessel() -> list[GraphicalPrimitive]:
    """Generic cylindrical container: covers CustomVessel's two current uses (a gas
    bottle, a piston) - both are pressure/volume-holding chambers with no dedicated
    DEXPI class, so a plain rounded-top cylinder is a fair, honest generic symbol
    rather than picking either shape's specific look."""
    return [
        _polygon((15, 20), (85, 20), (85, 90), (15, 90)),
        Ellipse(center=_p(50, 20), horizontalSemiAxis=35, verticalSemiAxis=8,
                rotation=0.0, stroke=_STROKE, fillStyle=_NO_FILL),
    ]


def draw_pressure_vessel() -> list[GraphicalPrimitive]:
    return [
        _line((25, 10), (75, 10)),
        Ellipse(center=_p(50, 15), horizontalSemiAxis=25, verticalSemiAxis=8,
                rotation=0.0, stroke=_STROKE, fillStyle=_NO_FILL),
        _line((25, 15), (25, 85)),
        _line((75, 15), (75, 85)),
        Ellipse(center=_p(50, 85), horizontalSemiAxis=25, verticalSemiAxis=8,
                rotation=0.0, stroke=_STROKE, fillStyle=_NO_FILL),
    ]


def draw_vessel() -> list[GraphicalPrimitive]:
    """Generic vessel / reactor: pressure-vessel outline plus a centered agitator shaft."""
    return draw_pressure_vessel() + [
        _line((50, 5), (50, 60)),
        _line((40, 60), (60, 60), (50, 72), (40, 60)),
    ]


def draw_jacketed_vessel() -> list[GraphicalPrimitive]:
    return draw_pressure_vessel() + [
        _line((15, 35), (15, 85)),
        _line((85, 35), (85, 85)),
        _line((15, 85), (85, 85)),
    ]


def draw_tubular_heat_exchanger() -> list[GraphicalPrimitive]:
    return [
        _polygon((5, 30), (95, 30), (95, 70), (5, 70)),
        _line(
            (5, 50), (20, 35), (35, 65), (50, 35), (65, 65), (80, 35), (95, 50)
        ),
    ]


def draw_process_column() -> list[GraphicalPrimitive]:
    primitives: list[GraphicalPrimitive] = [
        _line((35, 5), (65, 5)),
        Ellipse(center=_p(50, 8), horizontalSemiAxis=15, verticalSemiAxis=3,
                rotation=0.0, stroke=_STROKE, fillStyle=_NO_FILL),
        _line((35, 8), (35, 95)),
        _line((65, 8), (65, 95)),
        _line((35, 95), (65, 95)),
    ]
    for y in (25, 45, 65, 85):
        primitives.append(_line((35, y), (42, y)))
        primitives.append(_line((58, y), (65, y)))
    return primitives


def draw_pump() -> list[GraphicalPrimitive]:
    """Centrifugal pump: circle casing with a discharge-direction triangle."""
    return [
        _circle(50, 50, 40),
        _polygon((85, 40), (85, 60), (98, 50)),
    ]


def draw_custom_pump() -> list[GraphicalPrimitive]:
    """Generic/custom pump: bare circle, no direction indicator - matches the
    peristaltic-pump-style shapes that have no exact DEXPI class (see mapping.py)."""
    return [_circle(50, 50, 40)]


def _bowtie() -> list[GraphicalPrimitive]:
    return [_polygon((10, 20), (10, 80), (50, 50)), _polygon((90, 20), (90, 80), (50, 50))]


def draw_straightway_valve() -> list[GraphicalPrimitive]:
    return _bowtie()


def draw_gate_valve() -> list[GraphicalPrimitive]:
    return _bowtie() + [_line((50, 50), (50, 15)), _polygon((35, 5), (65, 5), (65, 15), (35, 15))]


def draw_check_valve() -> list[GraphicalPrimitive]:
    return _bowtie() + [_line((50, 30), (50, 70)), _polygon((45, 35), (55, 35), (50, 50))]


def draw_operated_valve() -> list[GraphicalPrimitive]:
    return _bowtie() + [_line((50, 50), (50, 10)), _circle(50, 5, 10)]


def draw_safety_valve_or_fitting() -> list[GraphicalPrimitive]:
    return _bowtie() + [
        _polygon((35, 5), (65, 5), (65, 25), (35, 25)),
        _line((50, 25), (50, 50)),
        _line((50, 5), (35, -10)),
    ]


def draw_process_instrumentation_function() -> list[GraphicalPrimitive]:
    """Instrument bubble: plain circle with a horizontal divider line (ISA-5.1 style)."""
    return [_circle(50, 50, 45), _line((5, 50), (95, 50))]


def draw_signal_off_page_connector() -> list[GraphicalPrimitive]:
    """Off-page/signal connector: pentagon 'home plate' pointing right."""
    return [_polygon((5, 15), (70, 15), (95, 50), (70, 85), (5, 85))]


# Dispatch table: DEXPI class name -> geometry builder. Covers every class that
# mapping.py's EQUIPMENT_SHAPE_MAP / VALVE_TAG_PREFIX_MAP can produce.
SYMBOL_BUILDERS = {
    "Tank": draw_tank,
    "PressureVessel": draw_pressure_vessel,
    "CustomVessel": draw_custom_vessel,
    "Vessel": draw_vessel,
    # jacketed vessels map to plain "Vessel" in mapping.py today; kept as a distinct
    # builder here in case that mapping is refined later to a dedicated class.
    "TubularHeatExchanger": draw_tubular_heat_exchanger,
    "Pump": draw_pump,
    "CustomPump": draw_custom_pump,
    "ProcessColumn": draw_process_column,
    "StraightwayValve": draw_straightway_valve,
    "GateValve": draw_gate_valve,
    "CheckValve": draw_check_valve,
    "OperatedValve": draw_operated_valve,
    "SafetyValveOrFitting": draw_safety_valve_or_fitting,
    "ProcessInstrumentationFunction": draw_process_instrumentation_function,
    "SignalOffPageConnector": draw_signal_off_page_connector,
}


def build_primitives_for_class(dexpi_class_name: str) -> list[GraphicalPrimitive]:
    builder = SYMBOL_BUILDERS.get(dexpi_class_name, draw_generic_box)
    return builder()


def compute_local_bounding_box(
    primitives: list[GraphicalPrimitive],
) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) of a symbol's actual drawn geometry, in its
    own local 0-100 coordinate space.

    Every symbol_library shape is inset from its own declared 0-100 box (e.g. draw_pump's
    circle has radius 40 centered at (50,50), so it only spans 10-90, not 0-100; _bowtie's
    valve triangles span 10-90 too). Routing connector lines to the full 0-100 box left a
    visible gap between the line's endpoint and the symbol's actual visible edge -
    reported by the user as "they still don't look connected". Computing the real
    bounding box from the primitives themselves (rather than guessing a fixed margin)
    means routing stays correct even as symbols are added or adjusted later.
    """
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for primitive in primitives:
        if isinstance(primitive, (PolyLine, Polygon)):
            for point in primitive.points:
                min_x, max_x = min(min_x, point.x), max(max_x, point.x)
                min_y, max_y = min(min_y, point.y), max(max_y, point.y)
        elif isinstance(primitive, Ellipse):
            cx, cy = primitive.center.x, primitive.center.y
            rx, ry = primitive.horizontalSemiAxis, primitive.verticalSemiAxis
            min_x, max_x = min(min_x, cx - rx), max(max_x, cx + rx)
            min_y, max_y = min(min_y, cy - ry), max(max_y, cy + ry)

    if min_x == float("inf"):  # no primitives with point/center geometry (shouldn't happen)
        return (0.0, 0.0, 100.0, 100.0)

    return (min_x, min_y, max_x, max_y)
