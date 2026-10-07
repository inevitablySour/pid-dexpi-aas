"""Symbols drawn with plain draw.io shapes instead of P&ID library symbols.

Not every P&ID uses a symbol library. The team's drawings draw the reactor R301 as a
rounded box holding three plain boxes (Mixing, Heating, Cooling), a vessel as a
rectangle reading "V201", an instrument as a circle reading "PC 201 N2", and a mass
flow controller as a circle reading "MFC". All of these are real plant items. Before
this module they were treated as text: missing from the DEXPI model, the exports and
the training labels, along with every pipe connected to them.

Rule: every CLOSED, BORDERED shape that is not a library symbol is a symbol - unless it
is a grouping frame. Its name is its own text, or else the text cell sitting inside it
(the reactor's "Mixing" label is a separate cell on top of an empty box). The name only
refines the type; an unrecognised name still yields a symbol (generic equipment,
flagged for review), never nothing.

Grouping frames (the coloured section boxes "N2 - H2 containers", "Injection point",
...) are not symbols: a frame encloses at least one library symbol, or has a coloured
(non-black) border. The reactor container encloses only plain boxes and is drawn in
black, so it stays a symbol.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pid_dexpi_aas.mapping import Category, classify_tag
from pid_dexpi_aas.parser_drawio import ParsedDiagram, Shape

_CLOSED_STYLES = ("rounded=", "whiteSpace=", "ellipse", "rhombus", "shape=process", "triangle", "hexagon")
_STROKE_RE = re.compile(r"(?:^|;)strokeColor=([^;]+)")
_BLACK = {"#000000", "#000", "black", "default", "#1a1a1a", "#333333"}

# Name keyword -> (DEXPI class, PID2Graph label). Matched against the whole name.
_NAME_RULES: list[tuple[str, str, str]] = [
    (r"(reactor|autoclave)(\s+[a-z]{0,3}-?\d+[a-z]?)?", "Vessel", "tank"),
    (r"(vessel|tank|drum|separator|column)(\s+.*)?", "Vessel", "tank"),
    (r"mix(ing|er)(\s+.*)?", "Mixer", "general"),
    (r"heat(ing|er)(\s+.*)?", "Heater", "general"),
    (r"cool(ing|er)(\s+.*)?", "HeatExchanger", "general"),
    (r"(heat\s*exchanger|condenser)(\s+.*)?", "HeatExchanger", "general"),
    (r"pump(\s+.*)?", "Pump", "pump"),
    (r"filter(\s+.*)?", "Filter", "general"),
]
_PUMP, _TANK = {"Pump", "CustomPump"}, {"Tank", "PressureVessel", "CustomVessel", "Vessel", "ProcessColumn"}


@dataclass
class PlainSymbol:
    shape_id: str
    name: str  # what the drawing calls it ("" if unnamed)
    category: Category
    dexpi_class: str
    pid2graph_label: str
    reason: str  # how the type was decided - shown to the reviewer
    confident: bool  # False -> generic fallback, needs review


def _is_closed_bordered(shape: Shape) -> bool:
    style = shape.style or ""
    if shape.shape_token or style.startswith("text;") or "strokeColor=none" in style:
        return False
    if shape.width < 8 or shape.height < 8:
        return False
    return style.startswith(_CLOSED_STYLES) or style == ""


def _contains(outer: Shape, inner: Shape) -> bool:
    cx, cy = inner.x + inner.width / 2, inner.y + inner.height / 2
    return outer.x <= cx <= outer.x + outer.width and outer.y <= cy <= outer.y + outer.height and outer is not inner


def _near_border(p, s: Shape, tol: float = 10.0) -> bool:
    """A point on (or right next to) a box's outline."""
    x, y = p
    x1, y1, x2, y2 = s.x, s.y, s.x + s.width, s.y + s.height
    if not (x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol):
        return False
    return min(abs(x - x1), abs(x - x2), abs(y - y1), abs(y - y2)) <= tol


def _is_frame(shape: Shape, diagram: ParsedDiagram) -> bool:
    """A box drawn around a group (a plant section), not an item. Decided by what it holds
    and what connects to it - never by colour or line weight:
      - it contains at least two items, or one item with lines or a title;
      - nothing connects to the box itself: no line is attached to it, and no loose line
        end lands on its outline (pipes go to the items inside instead).
    The reactor container holds its compartments but has pipes ending on its outline, so it
    stays equipment."""
    items = [o for o in diagram.shapes if o is not shape and (o.shape_token or _is_closed_bordered(o))
             and _contains(shape, o) and o.width * o.height < shape.width * shape.height]
    if not items:
        return False
    item_ids = {o.id for o in items}
    lines_inside = any(e.source in item_ids or e.target in item_ids for e in diagram.edges)
    if len(items) < 2 and not lines_inside:
        from pid_dexpi_aas.sections import _is_title  # local: sections imports this module
        titled = any(not t.shape_token and (t.style or "").startswith("text;") and _is_title(t.text)
                     and _contains(shape, t) for t in diagram.shapes)
        if not titled:
            return False
    for e in diagram.edges:
        if shape.id in (e.source, e.target):
            return False
        for pt in (e.source_point, e.target_point):
            if pt and _near_border(pt, shape):
                return False
    return True


def classify_name(name: str, ellipse: bool) -> tuple[Category, str, str, str, bool]:
    """(category, DEXPI class, PID2Graph label, reason, confident) for a plain shape's name."""
    text = " ".join(name.split())
    kind, dexpi_class, note = classify_tag(text) if text else (None, None, "")
    if kind == "instrument":
        return Category.INSTRUMENTATION, "ProcessInstrumentationFunction", "instrumentation", \
            f"plain {'circle' if ellipse else 'box'} with instrument letters", True
    if kind == "valve":
        return Category.PIPING, dexpi_class, "valve", "plain shape with a valve tag", True
    if kind == "equipment":
        label = "pump" if dexpi_class in _PUMP else "tank" if dexpi_class in _TANK else "general"
        return Category.EQUIPMENT, dexpi_class, label, f"plain shape with equipment tag {text!r}", True
    for pattern, cls, label in _NAME_RULES:
        if re.fullmatch(pattern, text, re.I):
            return Category.EQUIPMENT, cls, label, f"plain shape named {text!r}", True
    return Category.EQUIPMENT, "CustomEquipment", "general", \
        f"plain shape{' named ' + repr(text) if text else ' without a name'} - type unknown, please review", False


def find_plain_symbols(diagram: ParsedDiagram) -> dict[str, PlainSymbol]:
    """shape id -> PlainSymbol for every plain-shape symbol in the diagram."""
    library = [s for s in diagram.shapes if s.shape_token]
    # a box inside a library symbol is part of that symbol's drawing (LightUp's columns X101/X102
    # carry their body, separator and liquid level as plain boxes), not an item of its own
    part_of_symbol = lambda s: any(_contains(lib, s) and s.width * s.height < lib.width * lib.height
                                   for lib in library)
    candidates = [s for s in diagram.shapes if _is_closed_bordered(s) and not part_of_symbol(s)
                  and not _is_frame(s, diagram)]
    texts = [s for s in diagram.shapes if not s.shape_token and s.text.strip() and (s.style or "").startswith("text;")]
    # Each text cell names the SMALLEST candidate it sits in (the "Mixing" label is inside
    # both the Mixing box and the reactor container; it belongs to the Mixing box).
    names: dict[str, list[str]] = {}
    for t in texts:
        holders = [c for c in candidates if _contains(c, t)]
        if holders:
            smallest = min(holders, key=lambda c: c.width * c.height)
            names.setdefault(smallest.id, []).append(t.text.strip())
    out = {}
    for s in candidates:
        name = s.text.strip() or " ".join(names.get(s.id, []))
        category, cls, label, reason, confident = classify_name(name, (s.style or "").startswith("ellipse"))
        out[s.id] = PlainSymbol(s.id, name, category, cls, label, reason, confident)
    return out


def tag_from_name(name: str) -> str | None:
    """The tag inside a name, if any: "Reactor R301" -> "R301"; else the name itself."""
    m = re.search(r"\b([A-Z]{1,4}-?\d{2,4}[A-Z]?)\b", name)
    return m.group(1) if m else (name or None)


def adjacent_compartments(diagram: ParsedDiagram, plain: dict[str, PlainSymbol], tol: float = 3.0) -> list[tuple[str, str]]:
    """Pairs of plain boxes that share a wall inside the same container (Mixing|Heating in
    R301): drawn with no line between them, but connected as compartments of one unit."""
    shapes = {s.id: s for s in diagram.shapes if s.id in plain}
    pairs = []
    ids = list(shapes)
    for i, a_id in enumerate(ids):
        a = shapes[a_id]
        for b_id in ids[i + 1:]:
            b = shapes[b_id]
            if _contains(a, b) or _contains(b, a):
                continue
            share_x = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
            share_y = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
            touch_v = abs((a.x + a.width) - b.x) <= tol or abs((b.x + b.width) - a.x) <= tol
            touch_h = abs((a.y + a.height) - b.y) <= tol or abs((b.y + b.height) - a.y) <= tol
            same_parent = [c for c in shapes.values() if _contains(c, a)] == [c for c in shapes.values() if _contains(c, b)]
            if same_parent and ((touch_v and share_y > 0.5 * min(a.height, b.height))
                                or (touch_h and share_x > 0.5 * min(a.width, b.width))):
                pairs.append((a_id, b_id))
    return pairs


PLAIN_PREFIX = "plain:"


def mark_plain_symbols(diagram: ParsedDiagram) -> None:
    """Make plain-shape symbols first-class shapes, in place (called by
    parse_drawio_xml_resolved, before loose line ends are resolved - so a loose end can
    land on the reactor box instead of being guessed onto the nearest library symbol).

      - shape_token := "plain:<Category>:<DEXPI class>:<PID2Graph label>" (everything that
        branches on "has a shape token" now treats it as a real symbol; map_shape decodes it)
      - text := its name, when the name came from a separate text cell; that cell's text
        is cleared, so tag association can't also hang it on a neighbouring symbol
      - boxes sharing a wall inside one container get an "adjacent" connection edge
    """
    from pid_dexpi_aas.parser_drawio import Edge

    plain = find_plain_symbols(diagram)
    if not plain:
        return
    by_id = {s.id: s for s in diagram.shapes}
    consumed = set()
    candidates = [by_id[i] for i in plain]
    for t in diagram.shapes:
        if not t.shape_token and t.text.strip() and (t.style or "").startswith("text;"):
            holders = [c for c in candidates if _contains(c, t)]
            if holders and not min(holders, key=lambda c: c.width * c.height).text.strip():
                consumed.add(t.id)
    for sid, p in plain.items():
        s = by_id[sid]
        s.shape_token = f"{PLAIN_PREFIX}{p.category.name}:{p.dexpi_class}:{p.pid2graph_label}"
        if not s.text.strip():
            s.text = p.name
    for t in diagram.shapes:
        if t.id in consumed:
            t.text = ""
    for a, b in adjacent_compartments(diagram, plain):
        diagram.edges.append(Edge(id=f"adjacent:{a}:{b}", source=a, target=b, style="adjacent",
                                  source_resolution="adjacent", target_resolution="adjacent"))


def decode_plain_token(token: str) -> tuple[Category, str, str] | None:
    if not token or not token.startswith(PLAIN_PREFIX):
        return None
    cat, cls, label = token[len(PLAIN_PREFIX):].split(":", 2)
    return Category[cat], cls, label


def find_frames(diagram: ParsedDiagram) -> list[Shape]:
    """Boxes drawn around groups of items (see _is_frame) - plant sections."""
    return [s for s in diagram.shapes if _is_closed_bordered(s) and _is_frame(s, diagram)]
