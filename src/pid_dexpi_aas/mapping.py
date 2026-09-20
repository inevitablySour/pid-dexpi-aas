"""Map mxgraph shape tokens (and tag-prefix conventions) to pyDEXPI classes.

Two independent signals drive the mapping, per HANDOFF.md's findings:
  - the `shape=` token identifies the drawn symbol (vessel, pump, generic valve, ...)
  - the associated tag text (e.g. "PSV201", "GV101") identifies the *subtype* for
    valves and instruments, since draw.io's P&ID library uses one generic valve
    shape and one generic instrument-bubble shape for everything.

Anything with no clean pyDEXPI equivalent maps to `None` and must be reported as a
discrepancy (validate_by_counting.py), not silently forced into the nearest class.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class Category(Enum):
    EQUIPMENT = auto()
    PIPING = auto()
    INSTRUMENTATION = auto()
    NOT_PROCESS_EQUIPMENT = auto()  # arrows, decorative shapes, off-page connectors
    UNMAPPED = auto()  # no shape token (a floating tag) or no known mapping


@dataclass
class MappingResult:
    category: Category
    dexpi_class_name: str | None
    reason: str  # non-empty when dexpi_class_name is None, explains the gap


# --- shape token -> DEXPI equipment class ------------------------------------------------
# Values are pyDEXPI class *names* (looked up dynamically against dexpi_classes.equipment)
# rather than imported classes directly, so this table stays readable as data.
EQUIPMENT_SHAPE_MAP: dict[str, str | None] = {
    "mxgraph.pid.vessels.container,_tank,_cistern": "Tank",
    "mxgraph.pid.vessels.pressurized_vessel": "PressureVessel",
    "mxgraph.pid.vessels.gas_bottle": None,  # no DEXPI equivalent; flag as discrepancy
    "mxgraph.pid.vessels.jacketed_mixing_vessel": "Vessel",  # + Agitator composition, see build_model
    "mxgraph.pid.vessels.reactor": "Vessel",  # DEXPI has no dedicated Reactor class
    "mxgraph.pid.heat_exchangers.shell_and_tube_heat_exchanger_2": "TubularHeatExchanger",
    "mxgraph.pid.pumps.peristaltic": "CustomPump",  # no exact DEXPI equivalent
    "mxgraph.pid.pumps_-_iso.pump_(liquid)": "Pump",
    "mxgraph.pid2misc.column": "ProcessColumn",
}

# Shapes that are drawn as generic geometry, not equipment; how to treat them:
NOT_PROCESS_EQUIPMENT_SHAPES: set[str] = {
    "mxgraph.electrical.opto_electronics.lamp",
    "mxgraph.arrows2.arrow",
    "trapezoid",
}

# Cells with these shape tokens need a second-pass classifier (valve/instrument subtype).
GENERIC_VALVE_SHAPE = "mxgraph.pid2valves.valve"
DISC_INST_SHAPE = "mxgraph.pid2inst.discInst"
CYLINDER_SHAPE = "cylinder3"  # observed usage: piston (LightUp reactor), not a DEXPI class
OFF_PAGE_CONNECTOR_SHAPE = "offPageConnector"

# --- tag-prefix -> valve subtype ---------------------------------------------------------
# Order matters: check longer/more specific prefixes before short ones (e.g. "PSV" before "P").
VALVE_TAG_PREFIX_MAP: list[tuple[str, str | None, str]] = [
    ("PSV", "SafetyValveOrFitting", ""),
    ("GV", "GateValve", "inferred from usage, not a legend; confirm with team"),
    ("CV", "CheckValve", ""),
    ("XV", "OperatedValve", "inferred as actuated on/off valve; confirm with team"),
    ("FV", "OperatedValve", "inferred as flow control valve; confirm with team"),
    ("V", "StraightwayValve", "generic manual valve default, no prefix letter beyond 'V'"),
]

# --- ISA-5.1 loop-tag first letter -> measured variable (for instrumentation) ------------
ISA_MEASURED_VARIABLE = {
    "P": "pressure",
    "T": "temperature",
    "F": "flow",
    "L": "level",
    "A": "analysis",
}
# ISA-5.1 second letter -> function; DEXPI's ProcessInstrumentationFunction doesn't encode
# this directly as a class split (unlike valves), so it's captured as metadata/tagName only.
ISA_FUNCTION_LETTER = {
    "C": "control",
    "I": "indicate",
    "T": "transmit",
    "R": "record",
    "S": "switch",
    "Q": "totalize",
}

_TAG_TOKEN_RE = re.compile(r"^([A-Za-z]+)")


def classify_valve_tag(tag_text: str) -> tuple[str | None, str]:
    """Given adjacent tag text like 'GV201' or 'PSV-201', return (dexpi_class_name, note)."""
    match = _TAG_TOKEN_RE.match(tag_text.strip())
    prefix = match.group(1).upper() if match else ""
    for candidate_prefix, dexpi_class, note in VALVE_TAG_PREFIX_MAP:
        if prefix.startswith(candidate_prefix):
            return dexpi_class, note
    return "StraightwayValve", f"unrecognized tag prefix {prefix!r}, defaulted to generic valve"


def classify_instrument_tag(tag_text: str) -> tuple[str, dict]:
    """Given a discInst label like 'PC 202 H2' or 'PI 201 N2', extract ISA-5.1 loop info.

    Returns (dexpi_class_name, metadata) — dexpi_class_name is always
    'ProcessInstrumentationFunction' (the only generic PIF class in the 34-class
    instrumentation vocabulary); metadata captures the parsed ISA letters for the tagName
    and for the discrepancy report, since DEXPI does not subtype PIFs by loop-tag letters.
    """
    match = _TAG_TOKEN_RE.match(tag_text.strip())
    letters = match.group(1).upper() if match else ""
    measured = ISA_MEASURED_VARIABLE.get(letters[0]) if letters else None
    function = ISA_FUNCTION_LETTER.get(letters[1]) if len(letters) > 1 else None
    return "ProcessInstrumentationFunction", {
        "loop_letters": letters,
        "measured_variable": measured,
        "function": function,
    }


def map_shape(shape_token: str | None, tag_text: str) -> MappingResult:
    """Top-level dispatch: shape token (+ tag text where needed) -> MappingResult."""
    if shape_token is None:
        return MappingResult(Category.UNMAPPED, None, "no shape token; likely a floating tag/label")

    if shape_token == GENERIC_VALVE_SHAPE:
        dexpi_class, note = classify_valve_tag(tag_text)
        return MappingResult(Category.PIPING, dexpi_class, note)

    if shape_token == DISC_INST_SHAPE:
        dexpi_class, metadata = classify_instrument_tag(tag_text)
        note = f"loop={metadata['loop_letters']!r} measured={metadata['measured_variable']} function={metadata['function']}"
        return MappingResult(Category.INSTRUMENTATION, dexpi_class, note)

    if shape_token == OFF_PAGE_CONNECTOR_SHAPE:
        return MappingResult(Category.INSTRUMENTATION, "SignalOffPageConnector", "")

    if shape_token in NOT_PROCESS_EQUIPMENT_SHAPES:
        return MappingResult(Category.NOT_PROCESS_EQUIPMENT, None, "decorative/non-process shape")

    if shape_token == CYLINDER_SHAPE:
        return MappingResult(Category.UNMAPPED, None, "cylinder3 (e.g. piston) has no DEXPI equivalent")

    if shape_token in EQUIPMENT_SHAPE_MAP:
        dexpi_class = EQUIPMENT_SHAPE_MAP[shape_token]
        if dexpi_class is None:
            return MappingResult(Category.UNMAPPED, None, f"{shape_token} has no DEXPI equivalent")
        return MappingResult(Category.EQUIPMENT, dexpi_class, "")

    return MappingResult(Category.UNMAPPED, None, f"unrecognized shape token {shape_token!r}")
