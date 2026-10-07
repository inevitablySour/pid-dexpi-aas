"""Specification notes on a drawing ("17 bar", "Volume = 5 L", "1 to 10ml/minute",
"[N2]") -> equipment/valve specifications in DEXPI.

The team's drawings carry operating data as free text next to (or inside) items. Before
this module the parser treated such a note as a TAG: the nearest valve got the tagName
"17 bar". Now a note is recognised by its content, kept out of tag association, attached
to the item it describes, and stored:

  - as a typed DEXPI attribute where pyDEXPI 1.2 has one: a vessel's nominal capacity
    volume, a safety valve's set pressure, a pump's design volume flow rate;
  - otherwise as a DEXPI custom attribute (name + value + unit text), so nothing is lost.

A tag cell that also carries specs ("T201 [H2] 200 bar") keeps its tag and gives up the
spec parts. Attachment: the note's centre inside an item -> that item; else the nearest
item (centre to box edge) within 1.5x that item's size. Notes that fit nothing are kept
as unassigned (reported, never guessed onto something far away).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from pid_dexpi_aas.drawio.parser import ParsedDiagram, Shape

_NUM = r"(\d+(?:[.,]\d+)?)"
_PATTERNS = [
    # (kind, regex) - order matters: flow before volume ("ml/minute" contains "ml")
    ("flow", re.compile(rf"(?:{_NUM}\s*(?:to|-|–)\s*)?{_NUM}\s*(ml|l|L|m3|m³)\s*/\s*(min(?:ute)?|h(?:our)?|s(?:ec)?)\b", re.I)),
    ("pressure", re.compile(rf"{_NUM}\s*(barg|bara|bar|mbar|kPa|MPa|Pa|psi)\b", re.I)),
    ("volume", re.compile(rf"(?:volume\s*=?\s*)?{_NUM}\s*(ml|mL|l|L|m3|m³)\b(?!\s*/)", re.I)),
    ("temperature", re.compile(rf"(-?{_NUM})\s*°?\s*(°C|degC|K)\b", re.I)),
    # a chemical formula in brackets ("[N2]", "[H2]", "[CO2]") - not "[atm]", which is a
    # destination ("Fume hood [atm]"), not a medium
    ("medium", re.compile(r"\[\s*((?:[A-Z][a-z]?\s?\d*){1,4})\s*\]")),
]


@dataclass
class Spec:
    kind: str  # pressure | volume | flow | temperature | medium
    value: float | None  # numeric value (upper value of a range) in `unit`
    unit: str
    text: str  # exactly as written
    low: float | None = None  # lower value of a range ("1 to 10 ml/minute")


@dataclass
class SpecResult:
    by_shape: dict[str, list[Spec]] = field(default_factory=dict)
    spec_cells: set[str] = field(default_factory=set)  # text cells that are pure spec notes
    unassigned: list[tuple[str, list[Spec]]] = field(default_factory=list)


def _f(s: str | None) -> float | None:
    return None if s is None else float(s.replace(",", "."))


def parse_specs(text: str) -> tuple[list[Spec], str]:
    """All specs in a text, and the text with them removed (what remains may be a tag)."""
    t = " ".join(text.split())
    t = re.sub(r"\b([HN])\s+2\b", r"\g<1>2", t)  # "N 2" (subscript split) -> "N2"
    specs, rest = [], t
    for kind, rx in _PATTERNS:
        for m in rx.finditer(rest):
            if kind == "flow":
                low, high, vol, per = m.group(1), m.group(2), m.group(3), m.group(4)
                specs.append(Spec("flow", _f(high), f"{vol}/{per}", m.group(0), _f(low)))
            elif kind == "medium":
                specs.append(Spec("medium", None, "", m.group(0), None))
                specs[-1].unit = m.group(1).replace(" ", "")
            elif kind == "temperature":
                specs.append(Spec(kind, _f(m.group(1)), m.group(3), m.group(0)))
            else:
                specs.append(Spec(kind, _f(m.group(1)), m.group(2), m.group(0)))
        rest = rx.sub(" ", rest)
    rest = re.sub(r"\bvolume\s*=?\s*", " ", rest, flags=re.I)
    return specs, " ".join(rest.split())


def _box(s: Shape):
    return (s.x, s.y, s.x + s.width, s.y + s.height)


def _dist(p, b) -> float:
    return math.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3]))


def attach_specs(diagram: ParsedDiagram) -> SpecResult:
    """Find spec notes, attach them to items, strip them from mixed tag cells (in place)."""
    from pid_dexpi_aas.dexpi.mapping import Category, map_shape

    result = SpecResult()
    # Only real items: never a decorative part of a symbol (P101's base trapezoid took the
    # pump's "1 to 10ml/minute" before this filter).
    items = [s for s in diagram.shapes if s.shape_token
             and map_shape(s.shape_token, s.text).category in (Category.EQUIPMENT, Category.PIPING, Category.INSTRUMENTATION)]
    for cell in diagram.shapes:
        if cell.shape_token or not cell.text.strip():
            continue
        specs, rest = parse_specs(cell.text)
        if not specs:
            continue
        if not rest.strip():
            result.spec_cells.add(cell.id)  # a pure note: never a tag, never a connection end
        cx, cy = cell.x + cell.width / 2, cell.y + cell.height / 2
        inside = [s for s in items if s.x <= cx <= s.x + s.width and s.y <= cy <= s.y + s.height]
        if inside:
            target = min(inside, key=lambda s: s.width * s.height)
        else:
            cands = [(d, s) for s in items for d in [_dist((cx, cy), _box(s))]
                     if d <= 1.5 * max(s.width, s.height, 40)]
            target = min(cands, key=lambda t: t[0])[1] if cands else None
        if target is None:
            result.unassigned.append((cell.text.strip(), specs))
            continue
        result.by_shape.setdefault(target.id, []).extend(specs)
        if rest.strip() and rest != " ".join(cell.text.split()):
            cell.text = rest  # "T201 [H2] 200 bar" -> tag "T201"; specs moved to the item
    for s in items:  # specs written in an item's own text ("PI 201 N2" has none; "T202 5 bar" would)
        specs, rest = parse_specs(s.text)
        if specs and rest:
            result.by_shape.setdefault(s.id, []).extend(specs)
            s.text = rest
    return result


# ------------------------------------------------------------------------- DEXPI ---
_PRESSURE_UNITS = {"bar": "Bar", "barg": "Bar", "bara": "Bar", "mbar": "Millibar", "kpa": "Kilopascal",
                   "mpa": "Megapascal", "pa": "Pascal", "psi": "PoundForcePerInchSquared"}
_VOLUME_TO_L = {"ml": 1e-3, "l": 1.0, "m3": 1e3, "m³": 1e3}
_PER_TO_S = {"min": 60, "minute": 60, "h": 3600, "hour": 3600, "s": 1, "sec": 1}


def apply_specs(obj, specs: list[Spec]) -> list[str]:
    """Set specs on a built pyDEXPI object; returns human-readable notes of what was set."""
    import pydexpi.dexpi_classes.physicalQuantities as pq
    from pydexpi.dexpi_classes import pydantic_classes as pc

    notes = []
    custom = list(getattr(obj, "customAttributes", None) or [])
    fields = getattr(type(obj), "model_fields", {})
    for sp in specs:
        typed = False
        if sp.kind == "volume" and "nominalCapacityVolume" in fields and sp.value is not None:
            litres = sp.value * _VOLUME_TO_L.get(sp.unit.lower(), 1.0)
            obj.nominalCapacityVolume = pq.Volume(value=litres, unit=pq.VolumeUnit.Litre)
            typed = True
        elif sp.kind == "pressure" and "setPressureHigh" in fields and sp.value is not None:
            unit = getattr(pq.PressureGaugeUnit, _PRESSURE_UNITS.get(sp.unit.lower(), "Bar"))
            obj.setPressureHigh = pq.PressureGauge(value=sp.value, unit=unit)
            typed = True
        elif sp.kind == "flow" and "designVolumeFlowRate" in fields and sp.value is not None:
            vol, per = sp.unit.split("/")
            lps = sp.value * _VOLUME_TO_L.get(vol.lower(), 1.0) / _PER_TO_S.get(per.lower(), 60)
            obj.designVolumeFlowRate = pq.VolumeFlowRate(value=lps, unit=pq.VolumeFlowRateUnit.LitrePerSecond)
            typed = True
        name = {"pressure": "Pressure", "volume": "Volume", "flow": "FlowRate", "temperature": "Temperature",
                "medium": "Medium"}[sp.kind]
        # Always keep the text as written too: typed fields hold one number, the drawing
        # may say more ("1 to 10 ml/minute"). The exporter needs a URI on custom attributes.
        custom.append(pc.CustomStringAttribute(
            attributeName=f"{name}AsDrawn", attributeURI=f"urn:pid-dexpi-aas:drawing-note:{name}",
            value=sp.text if sp.kind != "medium" else sp.unit))
        notes.append(f"{name}: {sp.text if sp.kind != 'medium' else sp.unit}{' (typed)' if typed else ''}")
    if "customAttributes" in fields:
        obj.customAttributes = custom
    return notes
