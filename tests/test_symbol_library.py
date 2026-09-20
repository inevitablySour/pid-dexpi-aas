from pid_dexpi_aas.mapping import EQUIPMENT_SHAPE_MAP, VALVE_TAG_PREFIX_MAP
from pid_dexpi_aas.symbol_library import SYMBOL_BUILDERS, build_primitives_for_class


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
