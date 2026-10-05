"""The specification fields a DEXPI class has, for the review app's edit panel - read from
pyDEXPI's own class definitions, so every class gets exactly its own fields:

  - quantities with a unit (a tank's nominal capacity volume, a pump's design flow rate,
    a safety valve's set pressure, ...): number + unit picker;
  - fixed choices (a valve's operation, number of ports, ...): a picker;
  - text and numbers (fluid code, piping class code, ...).

Plus a few common properties pyDEXPI 1.2 has no typed field for (design pressure and
temperature, material), written as DEXPI custom attributes.

Values live in ReviewSymbol.attributes["dexpi"] (typed) and ["custom"] (extra), and are
written onto the built objects by apply_fields after build_model - over anything the
drawing's spec notes filled in (spec_labels.apply_specs), since the reviewer has the last word.
"""
from __future__ import annotations

import enum
import re
import typing
from dataclasses import dataclass, field

import pydexpi.dexpi_classes.physicalQuantities as pq
from pydexpi.dexpi_classes import equipment, instrumentation, piping
from pydexpi.dexpi_classes import pydantic_classes as pc


@dataclass
class SpecField:
    name: str  # attribute name on the DEXPI class
    label: str  # human-readable
    kind: str  # quantity | choice | text | number
    units: list = field(default_factory=list)  # quantity: unit symbols, first = default
    options: list = field(default_factory=list)  # choice: allowed values
    custom: bool = False  # not a typed DEXPI field: stored as a custom attribute


# Not specifications: identity, structure, graphics and bookkeeping.
_SKIP = re.compile(r"^(id|uri|proteusId|customAttributes|nozzles|chambers|nodes|tagName.*|typeName.*|"
                   r"subTagName|.*AssignmentClass|.*Representation|.*Specialization|plant(Train|System|Area)|"
                   r"parentStructure|sourceItem|sourceNode|targetItem|targetNode|items|connections|"
                   r"agitator|deviceInformation|typicalInformation|processInstrumentationFunctions|"
                   r"pipingClassArtefact|equipmentDescription.*|positionNumber|locationRegistrationNumber)$")

# Common properties with no typed pyDEXPI 1.2 field on equipment.
EXTRA_EQUIPMENT_FIELDS = [
    SpecField("DesignPressure", "Design pressure", "quantity", ["bar", "barg", "mbar", "kPa", "MPa", "psi"], custom=True),
    SpecField("DesignTemperature", "Design temperature", "quantity", ["°C", "K"], custom=True),
    SpecField("Material", "Material", "text", custom=True),
]


# Default (first) unit when a quantity has several: lab-scale, metric.
_PREFERRED_UNITS = ["l", "bar", "barg", "°C", "l/s", "m3/h", "mm", "m", "m2", "kW", "kW/(m2·K)", "kg", "kg/h", "rpm",
                    "%", "V", "Hz", "N"]


def _label(name: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()
    return words[:1].upper() + words[1:]


def dexpi_class(name: str):
    for module in (equipment, piping, instrumentation):
        cls = getattr(module, name, None)
        if isinstance(cls, type):
            return cls
    return None


def _quantity_class(annotation):
    """pq.Volume for pc.NullableVolume (pq.PressureGauge for NullablePressureGauge, ...)."""
    name = getattr(annotation, "__name__", "")
    if name.startswith("Nullable"):
        return getattr(pq, name[len("Nullable"):], None)
    return None


def _unwrap(annotation):
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    return args[0] if len(args) == 1 else annotation


def spec_fields(class_name: str) -> list[SpecField]:
    cls = dexpi_class(class_name)
    if cls is None:
        return []
    out = []
    for name, f in cls.model_fields.items():
        if _SKIP.match(name):
            continue
        ann = _unwrap(f.annotation)
        qcls = _quantity_class(ann)
        if qcls is not None:
            unit_enum = qcls.model_fields["unit"].annotation
            units = sorted((u.value for u in unit_enum),
                           key=lambda u: _PREFERRED_UNITS.index(u) if u in _PREFERRED_UNITS else 99)
            out.append(SpecField(name, _label(name), "quantity", units))
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            out.append(SpecField(name, _label(name), "choice", options=[e.value for e in ann if e.name != "NULL"]))
        elif ann is str:
            out.append(SpecField(name, _label(name), "text"))
        elif ann in (int, float):
            out.append(SpecField(name, _label(name), "number"))
    order = {"quantity": 0, "choice": 1, "number": 2, "text": 3}
    out.sort(key=lambda s: order[s.kind])
    if issubclass(cls, equipment.Equipment):
        out += EXTRA_EQUIPMENT_FIELDS
    return out


def apply_fields(obj, values: dict, custom: dict) -> list[str]:
    """Write reviewed values onto a built pyDEXPI object; returns notes on what failed."""
    problems = []
    fields = {f.name: f for f in spec_fields(type(obj).__name__)}
    for name, v in (values or {}).items():
        f = fields.get(name)
        if f is None or v in (None, "", {}):
            continue
        try:
            ann = _unwrap(type(obj).model_fields[name].annotation)
            if f.kind == "quantity":
                qcls = _quantity_class(ann)
                unit_enum = qcls.model_fields["unit"].annotation
                setattr(obj, name, qcls(value=float(v["value"]), unit=unit_enum(v["unit"])))
            elif f.kind == "choice":
                setattr(obj, name, ann(v))
            elif f.kind == "number":
                setattr(obj, name, ann(v))
            else:
                setattr(obj, name, str(v))
        except Exception as exc:  # a value pyDEXPI rejects: report, don't crash the build
            problems.append(f"{f.label}: {exc}")
    if custom and "customAttributes" in type(obj).model_fields:
        attrs = list(getattr(obj, "customAttributes", None) or [])
        for name, v in custom.items():
            if v in (None, "", {}):
                continue
            text = f"{v['value']:g} {v['unit']}" if isinstance(v, dict) else str(v)
            attrs.append(pc.CustomStringAttribute(attributeName=name, attributeURI=f"urn:pid-dexpi-aas:spec:{name}",
                                                  value=text))
        obj.customAttributes = attrs
    return problems


def prefill(class_name: str, specs_text: str) -> dict:
    """Field values the drawing's spec notes give ("Volume = 5 L", "18 bar" on a PSV,
    "1 to 10 ml/minute" on a pump) - the same typed fields spec_labels.apply_specs fills,
    shown in the panel so the reviewer sees and can correct them."""
    from pid_dexpi_aas.spec_labels import _PER_TO_S, _VOLUME_TO_L, parse_specs

    names = {f.name for f in spec_fields(class_name)}
    out = {}
    for part in (specs_text or "").split(";"):
        for sp in parse_specs(part)[0]:
            if sp.value is None:
                continue
            if sp.kind == "volume" and "nominalCapacityVolume" in names:
                out["nominalCapacityVolume"] = {"value": sp.value * _VOLUME_TO_L.get(sp.unit.lower(), 1.0), "unit": "l"}
            elif sp.kind == "pressure" and "setPressureHigh" in names and sp.unit.lower() in ("bar", "barg"):
                out["setPressureHigh"] = {"value": sp.value, "unit": "bar"}
            elif sp.kind == "flow" and "designVolumeFlowRate" in names:
                vol, per = sp.unit.split("/")
                out["designVolumeFlowRate"] = {
                    "value": sp.value * _VOLUME_TO_L.get(vol.lower(), 1.0) / _PER_TO_S.get(per.lower(), 60), "unit": "l/s"}
    return out
