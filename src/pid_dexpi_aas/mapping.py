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
    # No dedicated DEXPI class for a gas bottle; DEXPI's own answer for "equipment with
    # no matching concrete class" is the Custom* family (see CustomPump below), not a
    # None/discrepancy - a gas bottle is a pressure vessel in function, so CustomVessel
    # (which requires only a free-text typeName) is the correct DEXPI-native fallback,
    # not an unmapped gap. Confirmed against pydexpi.dexpi_classes.equipment: 25
    # Custom* subclasses exist precisely for this "no exact class" case.
    "mxgraph.pid.vessels.gas_bottle": "CustomVessel",
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
    # Back-pressure regulator: a self-acting pressure-control valve in the process line,
    # even though the team draws it as an instrument bubble ("BPR 401").
    ("BPR", "OperatedValve", "back-pressure regulator (self-acting pressure control valve)"),
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


# draw.io's own valve body type (style "valveType=..." on mxgraph.pid2valves.valve) ->
# DEXPI class. Before this, an untagged check valve (LightUp, between P201 and GV205)
# came out as a generic StraightwayValve. Three-way has no DEXPI class.
VALVE_TYPE_MAP = {
    "gate": "GateValve", "check": "CheckValve", "needle": "NeedleValve", "angle": "AngleValve",
    "ball": "BallValve", "globe": "GlobeValve", "butterfly": "ButterflyValve", "plug": "PlugValve",
}
_VALVE_TYPE_RE = re.compile(r"(?:^|;)valveType=([A-Za-z]+)")
# tag classes that say more than the body shape: function, not geometry
_FUNCTION_VALVES = {"SafetyValveOrFitting", "OperatedValve"}


def _classify_valve(tag_text: str, style: str) -> tuple[str, str]:
    """Valve subtype from the tag and draw.io's drawn valve type: a function tag (PSV,
    XV, FV, BPR) wins; otherwise the drawn body type; otherwise the tag prefix."""
    tag_class, tag_note = classify_valve_tag(tag_text)
    tag_known = not tag_note.startswith("unrecognized")
    m = _VALVE_TYPE_RE.search(style or "")
    drawn = m.group(1) if m else None
    if tag_known and tag_class in _FUNCTION_VALVES:
        return tag_class, tag_note
    if drawn in VALVE_TYPE_MAP:
        cls = VALVE_TYPE_MAP[drawn]
        note = f"drawn as a {drawn} valve"
        if tag_known and tag_class != cls:
            note += f" (tag suggested {tag_class})"
        return cls, note
    if drawn == "threeWay":
        return tag_class, f"three-way valve (no DEXPI three-way class); {tag_note}".rstrip("; ")
    return tag_class, tag_note


def map_shape(shape_token: str | None, tag_text: str, style: str = "") -> MappingResult:
    """Top-level dispatch: shape token (+ tag text, + draw.io style for valve bodies) -> MappingResult."""
    if shape_token is None:
        return MappingResult(Category.UNMAPPED, None, "no shape token; likely a floating tag/label")

    if shape_token.startswith("plain:"):
        # A symbol drawn with a plain draw.io shape (see plain_shapes.py).
        _, cat, cls, _label = shape_token.split(":", 3)
        reason = ("plain draw.io shape - type unknown, please review" if cls == "CustomEquipment"
                  else f"plain draw.io shape named {tag_text!r}")
        return MappingResult(Category[cat], cls, reason)

    if shape_token == GENERIC_VALVE_SHAPE:
        dexpi_class, note = _classify_valve(tag_text, style)
        return MappingResult(Category.PIPING, dexpi_class, note)

    if shape_token == DISC_INST_SHAPE:
        if classify_tag(tag_text)[0] == "valve":
            # An instrument bubble carrying a valve tag (e.g. "BPR 401") is a line valve.
            dexpi_class, note = classify_valve_tag(tag_text)
            return MappingResult(Category.PIPING, dexpi_class, f"valve tag in an instrument bubble: {note}".strip(": "))
        dexpi_class, metadata = classify_instrument_tag(tag_text)
        note = f"loop={metadata['loop_letters']!r} measured={metadata['measured_variable']} function={metadata['function']}"
        return MappingResult(Category.INSTRUMENTATION, dexpi_class, note)

    if shape_token == OFF_PAGE_CONNECTOR_SHAPE:
        return MappingResult(Category.INSTRUMENTATION, "SignalOffPageConnector", "")

    if shape_token in NOT_PROCESS_EQUIPMENT_SHAPES:
        return MappingResult(Category.NOT_PROCESS_EQUIPMENT, None, "decorative/non-process shape")

    if shape_token == CYLINDER_SHAPE:
        # A piston has no dedicated DEXPI class; CustomVessel (a chamber owner, matching
        # a piston's function as a pressure/volume-displacing chamber) is the correct
        # DEXPI-native fallback, same reasoning as gas_bottle above.
        return MappingResult(Category.EQUIPMENT, "CustomVessel", "cylinder3 (e.g. piston), mapped via CustomVessel fallback")

    if shape_token in EQUIPMENT_SHAPE_MAP:
        dexpi_class = EQUIPMENT_SHAPE_MAP[shape_token]
        if dexpi_class is None:
            return MappingResult(Category.UNMAPPED, None, f"{shape_token} has no DEXPI equivalent")
        return MappingResult(Category.EQUIPMENT, dexpi_class, "")

    return MappingResult(Category.UNMAPPED, None, f"unrecognized shape token {shape_token!r}")


# --- tag-prefix -> equipment class (image path: OCR tags, no draw.io shape token) -----------
# Team convention as seen on the LightUp/Lumen drawings (V201 vessel, P101 pump, T201 gas
# bottle/tank, R201 reactor, H201 heater, X101 column, LED 201 light source). Single-letter
# prefixes only ever match EXACTLY ("P" not "PI"), so ISA instrument tags never land here.
EQUIPMENT_TAG_PREFIX_MAP: dict[str, tuple[str, str]] = {
    "P": ("Pump", ""),
    "V": ("Vessel", ""),
    "T": ("Tank", ""),
    "R": ("Vessel", "reactor - DEXPI has no dedicated Reactor class"),
    "H": ("Heater", "inferred heater/heat exchanger; confirm with team"),
    "E": ("HeatExchanger", "ISA-style 'E' prefix; confirm with team"),
    "X": ("ProcessColumn", "inferred from X101/X102 column drawings; confirm with team"),
    "LED": ("CustomEquipment", "light source (photoreactor LED); no DEXPI class"),
}
_ISA_FOLLOW_LETTERS = set(ISA_FUNCTION_LETTER) | {"V", "Y", "Z", "E"}


def tag_letters(tag_text: str) -> str:
    match = _TAG_TOKEN_RE.match(tag_text.strip())
    return match.group(1).upper() if match else ""


def classify_tag(tag_text: str) -> tuple[str | None, str | None, str]:
    """What a tag alone says: (kind, dexpi_class_name, note), kind in
    "valve" | "instrument" | "equipment" | None. Valve prefixes first (PSV before the
    ISA check, since P-S-V would otherwise read as pressure+switch+valve), then ISA
    instrument letters, then exact equipment prefixes."""
    letters = tag_letters(tag_text)
    if not letters:
        return None, None, ""
    for prefix, dexpi_class, note in VALVE_TAG_PREFIX_MAP:
        if prefix != "V" and letters == prefix:
            return "valve", dexpi_class, note
    if (
        2 <= len(letters) <= 4
        and letters[0] in ISA_MEASURED_VARIABLE
        and all(c in _ISA_FOLLOW_LETTERS for c in letters[1:])
    ):
        return "instrument", "ProcessInstrumentationFunction", ""
    if letters in EQUIPMENT_TAG_PREFIX_MAP:
        dexpi_class, note = EQUIPMENT_TAG_PREFIX_MAP[letters]
        return "equipment", dexpi_class, note
    return None, None, f"tag prefix {letters!r} not in any lookup table"


# Model (PID2Graph) broad class -> fallback when there is no usable tag.
MODEL_CLASS_FALLBACK: dict[str, tuple[Category, str | None, str]] = {
    "valve": (Category.PIPING, "StraightwayValve", "generic valve - no tag to pick a subtype"),
    "instrumentation": (Category.INSTRUMENTATION, "ProcessInstrumentationFunction", ""),
    "pump": (Category.EQUIPMENT, "Pump", ""),
    "tank": (Category.EQUIPMENT, "Tank", ""),
    "inlet/outlet": (Category.NOT_PROCESS_EQUIPMENT, None, "off-page inlet/outlet - becomes a connector if piped"),
    "arrow": (Category.NOT_PROCESS_EQUIPMENT, None, "flow arrow - direction marker, not a DEXPI object"),
    "general": (Category.UNMAPPED, None, "generic symbol with no tag - needs manual classification"),
}
_KIND_CATEGORY = {"valve": Category.PIPING, "instrument": Category.INSTRUMENTATION, "equipment": Category.EQUIPMENT}


def classify_detection(model_class: str, tag_text: str | None) -> tuple[MappingResult, str]:
    """Combine the image model's broad class with the OCR tag.

    Returns (MappingResult, identified_by). The tag wins when it is unambiguous, because
    P&ID tag conventions are more reliable than the model on real drawings (it calls
    real pumps "general" and some instruments "tank"); disagreements are kept in the
    reason so the reviewer sees them. "V" is the one genuinely ambiguous prefix
    (vessel V201 vs manual valve), so the model's class decides that case.
    """
    kind, dexpi_class, note = classify_tag(tag_text) if tag_text else (None, None, "")
    letters = tag_letters(tag_text) if tag_text else ""

    if kind == "equipment" and letters == "V" and model_class == "valve":
        cls, vnote = classify_valve_tag(tag_text or "")
        return MappingResult(Category.PIPING, cls, vnote), "tag 'V' + model: valve"

    if kind is not None:
        if kind == "valve":
            dexpi_class, note = classify_valve_tag(tag_text or "")
        expected = {"valve": "valve", "instrument": "instrumentation"}.get(kind)
        agrees = model_class == expected or (kind == "equipment" and model_class in ("pump", "tank", "general"))
        reason = note if agrees else f"model said {model_class!r}, tag {letters!r} overrides. {note}".strip()
        return MappingResult(_KIND_CATEGORY[kind], dexpi_class, reason), f"tag prefix {letters!r}"

    category, dexpi_class, reason = MODEL_CLASS_FALLBACK.get(
        model_class, (Category.UNMAPPED, None, f"unknown model class {model_class!r}")
    )
    if tag_text and note:
        reason = f"{reason}; {note}"
    return MappingResult(category, dexpi_class, reason), f"model class {model_class!r}"
