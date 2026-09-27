from pid_dexpi_aas.mapping import Category, classify_valve_tag, map_shape


def test_reactor_maps_to_vessel():
    result = map_shape("mxgraph.pid.vessels.reactor", "")
    assert result.category == Category.EQUIPMENT
    assert result.dexpi_class_name == "Vessel"


def test_gas_bottle_maps_via_custom_vessel_fallback():
    """DEXPI has no dedicated gas-bottle class, but its own answer for "no exact class"
    is the Custom* family (CustomVessel etc, each requiring only a free-text typeName),
    not silently dropping the shape - the user's own question ("aren't we supposed to
    make classifications of them for ourselves?") is exactly what DEXPI's Custom*
    fallback classes exist for.
    """
    result = map_shape("mxgraph.pid.vessels.gas_bottle", "")
    assert result.category == Category.EQUIPMENT
    assert result.dexpi_class_name == "CustomVessel"


def test_generic_valve_dispatches_on_tag_prefix():
    assert classify_valve_tag("PSV201")[0] == "SafetyValveOrFitting"
    assert classify_valve_tag("GV101")[0] == "GateValve"
    assert classify_valve_tag("CV301")[0] == "CheckValve"
    assert classify_valve_tag("V101")[0] == "StraightwayValve"


def test_valve_shape_uses_tag_for_subtype():
    result = map_shape("mxgraph.pid2valves.valve", "PSV201")
    assert result.category == Category.PIPING
    assert result.dexpi_class_name == "SafetyValveOrFitting"


def test_disc_inst_maps_to_process_instrumentation_function():
    result = map_shape("mxgraph.pid2inst.discInst", "PC 202 H2")
    assert result.category == Category.INSTRUMENTATION
    assert result.dexpi_class_name == "ProcessInstrumentationFunction"


def test_no_shape_token_is_unmapped():
    result = map_shape(None, "Reactor")
    assert result.category == Category.UNMAPPED


def test_off_page_connector():
    result = map_shape("offPageConnector", "")
    assert result.dexpi_class_name == "SignalOffPageConnector"


def test_decorative_lamp_is_not_process_equipment():
    result = map_shape("mxgraph.electrical.opto_electronics.lamp", "")
    assert result.category == Category.NOT_PROCESS_EQUIPMENT
