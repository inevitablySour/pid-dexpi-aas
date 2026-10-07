from pydexpi.dexpi_classes import equipment, instrumentation, piping

from pid_dexpi_aas.dexpi.mapping import EQUIPMENT_SHAPE_MAP, VALVE_TAG_PREFIX_MAP

_MODULES = [equipment, piping, instrumentation]


def _resolves(class_name: str) -> bool:
    return any(hasattr(module, class_name) for module in _MODULES)


def test_all_mapped_equipment_classes_exist_in_installed_pydexpi():
    for class_name in EQUIPMENT_SHAPE_MAP.values():
        if class_name is not None:
            assert _resolves(class_name), f"{class_name} not found in installed pydexpi"


def test_all_mapped_valve_classes_exist_in_installed_pydexpi():
    for _, class_name, _ in VALVE_TAG_PREFIX_MAP:
        if class_name is not None:
            assert _resolves(class_name), f"{class_name} not found in installed pydexpi"
