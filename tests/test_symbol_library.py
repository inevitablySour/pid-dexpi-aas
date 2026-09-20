from pid_dexpi_aas.mapping import EQUIPMENT_SHAPE_MAP, VALVE_TAG_PREFIX_MAP
from pid_dexpi_aas.symbol_library import (
    SYMBOL_BUILDERS,
    build_primitives_for_class,
    compute_local_bounding_box,
)


def test_every_reachable_dexpi_class_has_a_symbol():
    """Every non-None class name mapping.py can actually produce must render as
    something other than the generic-box fallback, so the re-render is meaningfully
    more informative than an undifferentiated box per shape.
    """
    reachable_classes = {v for v in EQUIPMENT_SHAPE_MAP.values() if v} | {
        v for _, v, _ in VALVE_TAG_PREFIX_MAP if v
    }
    reachable_classes |= {"ProcessInstrumentationFunction", "SignalOffPageConnector"}
    for class_name in reachable_classes:
        assert class_name in SYMBOL_BUILDERS, f"{class_name} has no dedicated symbol"


def test_unknown_class_falls_back_to_generic_box():
    primitives = build_primitives_for_class("SomeClassNotInTheLibrary")
    assert len(primitives) == 1


def test_all_symbols_produce_at_least_one_primitive():
    for class_name, builder in SYMBOL_BUILDERS.items():
        primitives = builder()
        assert len(primitives) >= 1, f"{class_name} symbol has no primitives"


def test_pump_bounding_box_matches_its_inscribed_circle():
    """Regression test: draw_pump's circle has radius 40 centered at (50,50), so it
    only spans local x=10-90 (plus the discharge triangle reaching x=98) - NOT the full
    declared 0-100 box. Routing connector lines to the full box left a visible gap
    between the line and the symbol (reported by the user as "they still don't look
    connected"). This checks the computed box reflects the real, inset geometry.
    """
    primitives = build_primitives_for_class("Pump")
    min_x, min_y, max_x, max_y = compute_local_bounding_box(primitives)
    assert min_x == 10.0
    assert min_y == 10.0
    assert max_x == 98.0  # extended by the discharge-direction triangle
    assert max_y == 90.0


def test_bowtie_valve_bounding_box_is_inset_from_declared_box():
    primitives = build_primitives_for_class("StraightwayValve")
    min_x, min_y, max_x, max_y = compute_local_bounding_box(primitives)
    assert (min_x, max_x) == (10.0, 90.0)
    assert not (min_x == 0.0 and max_x == 100.0)


def test_bounding_box_falls_back_to_full_box_when_no_geometry():
    assert compute_local_bounding_box([]) == (0.0, 0.0, 100.0, 100.0)
