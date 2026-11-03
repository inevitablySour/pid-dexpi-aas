"""A CAD P&ID (an ezdxf document from cad/dwg_read.py) as a draw.io drawing, so it goes through
the app's draw.io route unchanged: editor, guided check, DEXPI, AAS.

No company symbol table: what each symbol is comes from general rules (2026-11, first test: a
plant P&ID from AutoCAD 2023, symbols as blocks with their tags as attributes):

  - a block's tag split in two fields ("PT" / "0101") is an instrument bubble (ISA-5.1 style);
  - otherwise its tag prefix: ISA letters, P/T/V/... equipment, valve prefixes
    (mapping.classify_tag), plus a few common Dutch valve abbreviations (VALVE_WORDS);
  - a block that only holds text is a label: an underlined name inside an outline names that
    equipment (equipment drawn as an outline, not a block);
  - a closed outline holding one equipment name is that equipment; one around other symbols is
    a frame (package unit), left out;
  - unknown text at the end of a pipe is a stream entering or leaving the drawing;
  - a symbol without text, or a size change ("80/50"), with a pipe on each side is in-line
    (flange, flow arrow, reducer): the pipe runs through it;
  - a dashed line type is a signal line; short diagonal strokes on a line ("//") are its
    decoration; lines inside an equipment outline are its internals.

Line pieces are joined where exactly two meet; a pipe end on a symbol (within ATTACH mm of its
outline) is attached to it; a pipe end on another pipe is left on it, which the draw.io route
reads as a tee.
"""
from __future__ import annotations

import base64
import html
import math
import re
from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

from pid_dexpi_aas.dexpi.mapping import classify_tag

ATTACH = 1.5   # mm: a pipe end this close to a symbol's outline is on it
SNAP = 0.3     # mm: two line ends this close are one point
SCALE = 4.0    # draw.io units per mm (an A0 sheet ~ 4800 x 3400)
DASHED = re.compile(r"HIDDEN|DASH|DOT|CENTER|PHANTOM", re.I)
TEXT_TYPES = {"TEXT", "MTEXT", "ATTDEF", "ATTRIB"}
# Dutch valve abbreviations seen on P&IDs: afsluiter, kogelafsluiter, terugslagklep
VALVE_WORDS = {"AFS": "", "KAF": "ball", "TSK": "check", "TK": "check", "KA": "ball", "VA": ""}
GREY = {8, 9, 250, 251, 252, 253, 254}   # AutoCAD colour numbers drawn grey
_REDUCER = re.compile(r"^\d+\s*/\s*\d+$")
_LETTERS = re.compile(r"^[A-Z]{1,5}$")
_NUMBER = re.compile(r"^[0-9]{1,6}[A-Z]?$")


@dataclass
class Item:
    id: str
    kind: str   # valve | instrument | equipment | stream | inline | unknown
    box: tuple  # mm: x0, y0, x1, y1
    tag: str = ""
    extra: list = field(default_factory=list)
    style: str = ""
    block: str = ""
    why: str = ""
    cls: str = ""   # a DEXPI class set here (written as editor data, so the review keeps it)
    rot: int = 0    # the block's rotation in the CAD drawing, degrees counter-clockwise
    ents: list = field(default_factory=list)   # its drawn geometry, in drawing coordinates
    future: bool = False   # drawn dashed and grey: planned or out of service, not in DEXPI by default
    section: str = ""      # the unit it is drawn inside (DEXPI plant section)
    specs: dict = field(default_factory=dict)   # {"dexpi": {...}, "custom": {...}} from a data table


def _look(doc, e) -> tuple[bool, bool]:
    """(dashed, grey) as an entity is drawn, its layer's line type and colour where it has none."""
    layer = doc.layers.get(e.dxf.layer) if doc.layers.has_entry(e.dxf.layer) else None
    lt = e.dxf.get("linetype", "BYLAYER") or "BYLAYER"
    if lt.upper() in ("BYLAYER", "BYBLOCK"):
        lt = layer.dxf.linetype if layer is not None else ""
    col = e.dxf.get("color", 256)
    if col in (0, 256):
        col = abs(layer.dxf.color) if layer is not None else 7
    return bool(DASHED.search(lt or "")), col in GREY


def _is_future(doc, ents) -> bool:
    """Drawn dashed and grey (most of it): planned, or out of service."""
    looks = [_look(doc, e) for e in ents]
    return bool(looks) and sum(d and g for d, g in looks) >= 0.7 * len(looks)


def _fields(ins) -> list[str]:
    out = []
    for a in ins.attribs:
        t = (a.dxf.text or "").replace("%%U", "").replace("%%u", "").strip()
        if t:
            out.append(t)
    return out


def _parts(ins, depth=0) -> list:
    """A block reference's drawn geometry in drawing coordinates (nested blocks opened up),
    without its texts."""
    out = []
    try:
        for e in ins.virtual_entities():
            # AutoCAD's rules for what a block draws with: BYBLOCK takes the reference's line type
            # and colour, and things on layer 0 take the reference's layer
            if (e.dxf.get("linetype", "") or "").upper() == "BYBLOCK":
                e.dxf.linetype = ins.dxf.get("linetype", "BYLAYER")
            if e.dxf.get("color", 256) == 0:
                e.dxf.color = ins.dxf.get("color", 256)
            if e.dxf.get("layer", "0") == "0":
                e.dxf.layer = ins.dxf.get("layer", "0")
            if e.dxftype() == "INSERT" and depth < 4:
                out += _parts(e, depth + 1)
            elif e.dxftype() not in TEXT_TYPES:
                out.append(e)
    except Exception:
        pass
    return out


def _geometry_box(ins):
    """(The symbol's outline box, its parts): its drawn geometry, not its attribute texts."""
    from ezdxf import bbox
    parts = _parts(ins)
    if not parts:
        return None, []
    ext = bbox.extents(parts)
    if not ext.has_data:
        return None, []
    return (ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y), parts


def _text_only(doc, ins) -> bool:
    blk = doc.blocks.get(ins.dxf.name)
    return blk is not None and all(e.dxftype() in TEXT_TYPES for e in blk)


def _classify(fields: list[str]) -> tuple[str, str, str, list, str]:
    """(kind, tag, valve body type, extra fields, why) from a block's text fields."""
    if len(fields) >= 2 and _LETTERS.match(fields[0]) and _NUMBER.match(fields[1]):
        return "instrument", f"{fields[0]} {fields[1]}", "", fields[2:], "tag in two fields: an instrument bubble"
    tag = fields[0]
    if _REDUCER.match(tag):
        return "inline", tag, "", fields[1:], "size change (reducer)"
    m = re.match(r"^([A-Z]+)-?\s*\d", tag)
    prefix = m.group(1) if m else ""
    if prefix in VALVE_WORDS:
        return "valve", tag, VALVE_WORDS[prefix], fields[1:], f"valve abbreviation {prefix}"
    kind, cls, _ = classify_tag(tag)
    if kind:
        return kind, tag, "", fields[1:], f"tag prefix ({cls})"
    return "unknown", tag, "", fields[1:], "tag not recognised"


def _within(e, box) -> bool:
    from ezdxf import bbox
    ext = bbox.extents([e])
    return ext.has_data and _inside(ext.extmin, box, 0.01) and _inside(ext.extmax, box, 0.01)


def symbol_svg(ents, box, faded: bool = False) -> str:
    """The symbol as drawn in the CAD file, as an SVG image of exactly its box (y up -> down)."""
    from ezdxf import path as dpath
    w, h = max(box[2] - box[0], 0.5), max(box[3] - box[1], 0.5)
    pad = 0.3
    shapes = []
    for e in ents:
        try:
            paths = (list(dpath.from_hatch(e)) if e.dxftype() == "HATCH" else [dpath.make_path(e)])
        except Exception:
            continue
        filled = e.dxftype() in ("HATCH", "SOLID", "TRACE")
        for pth in paths:
            pts = list(pth.flattening(0.05))
            if len(pts) < 2:
                continue
            d = "M" + " L".join(f"{p.x - box[0] + pad:.2f},{box[3] - p.y + pad:.2f}" for p in pts)
            shapes.append(f'<path d="{d}{" Z" if filled else ""}" fill="{"#000" if filled else "none"}"/>')
    vw, vh = w + 2 * pad, h + 2 * pad
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:.2f} {vh:.2f}" width="{vw:.2f}mm" '
            f'height="{vh:.2f}mm"><g stroke="{"#8a8a8a" if faded else "#000"}" stroke-width="0.25" '
            f'{"stroke-dasharray=\"1.2,0.8\" " if faded else ""}stroke-linecap="round" '
            f'stroke-linejoin="round">{"".join(shapes)}</g></svg>')


def _inside(p, box, margin=0.0) -> bool:
    return box[0] - margin <= p[0] <= box[2] + margin and box[1] - margin <= p[1] <= box[3] + margin


def _dist_to_box(p, box) -> float:
    dx = max(box[0] - p[0], 0, p[0] - box[2])
    dy = max(box[1] - p[1], 0, p[1] - box[3])
    return math.hypot(dx, dy)


def read_symbols(doc) -> tuple[list[Item], list, list]:
    """Symbols (blocks), equipment (outlines named by a label inside) and frames."""
    from ezdxf import bbox
    msp = doc.modelspace()
    sheet = bbox.extents(msp)
    W, H = sheet.size.x, sheet.size.y
    items, labels = [], []
    for k, ins in enumerate(msp.query("INSERT")):
        fields = _fields(ins)
        if len(fields) > 6:
            continue  # title block, equipment data table
        if _text_only(doc, ins):
            if fields:
                labels.append((ins.dxf.insert.x, ins.dxf.insert.y, fields))
            continue
        box, parts = _geometry_box(ins)
        if box is None or (box[2] - box[0]) > 0.25 * W or (box[3] - box[1]) > 0.25 * H:
            continue  # the drawing frame
        iid = f"b{k}"
        if not fields:
            items.append(Item(iid, "inline", box, block=ins.dxf.name, why="symbol without text",
                              rot=round(ins.dxf.rotation) % 360, ents=parts, future=_is_future(doc, parts)))
            continue
        if len(fields) >= 3 and not (_LETTERS.match(fields[0]) and _NUMBER.match(fields[1])):
            continue  # a data box beside a symbol (a safety valve's set pressure), not a symbol
        kind, tag, body, extra, why = _classify(fields)
        items.append(Item(iid, kind, box, tag=tag, extra=extra, block=ins.dxf.name, why=why, rot=round(ins.dxf.rotation) % 360, ents=parts,
                          future=_is_future(doc, parts),
                          style=f"valveType={body};" if body else ""))
    # equipment drawn as an outline, named by a label inside it; outlines around symbols: frames
    frames = []
    for k, pl in enumerate(msp.query("LWPOLYLINE")):
        if not pl.closed:
            continue
        pts = [(p[0], p[1]) for p in pl.get_points()]
        box = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
        inside_syms = [it for it in items if _inside(((it.box[0] + it.box[2]) / 2, (it.box[1] + it.box[3]) / 2), box)]
        names = [f for (x, y, f) in labels if _inside((x, y), box)]
        if len(inside_syms) >= 2:
            frames.append(box)
            name = next((t.dxf.text.strip() for t in msp.query("TEXT MTEXT")
                         if t.dxftype() == "TEXT" and _inside((t.dxf.insert.x, t.dxf.insert.y), box) and t.dxf.text.strip()), "")
            items.append(Item(f"u{k}", "unit", box, tag=name or "Package unit", ents=[pl], cls="CustomEquipment",
                              why="a frame that pipes end on: a package unit, as one item", future=_is_future(doc, [pl])))
            continue
        for f in names:
            kind, cls, _ = classify_tag(f[0])
            if kind == "equipment":
                inner = [e for e in msp.query("LINE LWPOLYLINE ARC CIRCLE ELLIPSE")
                         if e is not pl and _within(e, box)]
                items.append(Item(f"q{k}", "equipment", box, tag=f[0], extra=f[1:], why=f"outline named {f[0]}",
                                  ents=[pl] + inner, future=_is_future(doc, [pl])))
                break
    return items, labels, frames


# words in a data table's labels -> (specification field, default unit); Dutch and English
TABLE_FIELDS = [
    (re.compile(r"volume|inhoud", re.I), "nominalCapacityVolume", "m3"),
    (re.compile(r"ontwerpdruk|design\s*pressure|ontwerp-?\s*druk", re.I), "DesignPressure", "bar"),
    (re.compile(r"ontwerptemp|design\s*temp", re.I), "DesignTemperature", "°C"),
    (re.compile(r"materia", re.I), "Material", ""),
    (re.compile(r"^medium", re.I), "Medium", ""),
]


def read_tables(doc) -> dict:
    """Equipment data tables (a block of labelled fields, the first the equipment's tag) as
    {tag: {"dexpi": {...}, "custom": {...}}}: each value paired with the label on its row to its
    left, and the unit to its right. Placeholders (####, XXXX) are skipped."""
    from pid_dexpi_aas.dexpi.fields import spec_fields
    from pid_dexpi_aas.dexpi.mapping import classify_tag
    from pid_dexpi_aas.dexpi.spec_sheet import to_value
    out = {}
    for ins in doc.modelspace().query("INSERT"):
        atts = [(a.dxf.insert.x, a.dxf.insert.y, (a.dxf.text or "").replace("%%U", "").strip()) for a in ins.attribs]
        if len(atts) <= 6:
            continue
        texts = []
        for e in ins.virtual_entities():
            if e.dxftype() == "TEXT" and (e.dxf.text or "").strip():
                texts.append((e.dxf.insert.x, e.dxf.insert.y, e.dxf.text.strip()))
        if not texts:
            continue
        tag = max(atts, key=lambda a: a[1])[2]   # the top field
        kind, cls, _ = classify_tag(tag)
        if kind != "equipment":
            continue
        fields = {f.name: f for f in spec_fields(cls)}
        spec = {"dexpi": {}, "custom": {}}
        for x, y, value in atts:
            if not value or re.fullmatch(r"[#X\-.]+", value):
                continue
            row = [t for t in texts if abs(t[1] - y) < 1.5]
            left = max((t for t in row if t[0] < x), key=lambda t: t[0], default=None)
            # the unit: every short text right of the value, together ("m" + superscript "3" = "m3")
            drawn_unit = "".join(t[2] for t in sorted(row, key=lambda t: t[0]) if t[0] > x and len(t[2]) <= 3)
            drawn_unit = drawn_unit.replace("³", "3").replace("²", "2").replace(" ", "")
            if left is None:
                continue
            for rx, name, unit in TABLE_FIELDS:
                if rx.search(left[2]):
                    f = fields.get(name)
                    if f is not None and f.kind == "quantity":
                        num = value.replace(",", ".")
                        use = next((u for u in (drawn_unit, unit) if u in f.units), f.units[0])
                        try:
                            v = {"value": float(num), "unit": use}
                        except ValueError:
                            break
                        spec["custom" if f.custom else "dexpi"][name] = v
                    else:
                        v, problem = to_value(f, value)
                        if not problem:
                            spec["custom" if f is None or f.custom else "dexpi"][name] = v
                    break
        spec = {k: v for k, v in spec.items() if v}
        if spec:
            out[tag] = spec
    return out


def read_lines(doc, items: list[Item]) -> list[dict]:
    """Line pieces: [{"pts": [(x, y), ...], "dashed": bool}], without decoration and internals."""
    msp = doc.modelspace()
    layer_lt = {l.dxf.name: (l.dxf.linetype or "") for l in doc.layers}
    equipment = [it.box for it in items if it.kind == "equipment"]
    out = []
    for e in msp.query("LINE LWPOLYLINE"):
        if e.dxftype() == "LINE":
            pts = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
        else:
            if e.closed:
                continue  # outlines: equipment and frames (read_symbols)
            pts = [(p[0], p[1]) for p in e.get_points()]
        length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        if length < 0.8:
            continue
        a, b = pts[0], pts[-1]
        diagonal = abs(a[0] - b[0]) > 0.2 and abs(a[1] - b[1]) > 0.2
        if diagonal and length < 4.0 and len(pts) == 2:
            continue  # a "//" stroke on a signal line
        if any(all(_inside(p, box, -0.5) for p in pts) for box in equipment):
            continue  # inside an equipment outline: its internals
        dashed, grey = _look(doc, e)
        # dashed and grey: a planned / out-of-service pipe; dashed in colour: a signal line
        out.append({"pts": pts, "dashed": dashed and not grey, "future": dashed and grey})
    return out


def _key(p):
    return (round(p[0] / SNAP), round(p[1] / SNAP))


def join_pieces(pieces: list[dict]) -> list[dict]:
    """Pieces into pipes: joined where exactly two ends meet (and nothing else runs there)."""
    ends: dict = {}
    for i, pc in enumerate(pieces):
        for end in (0, 1):
            ends.setdefault(_key(pc["pts"][0 if end == 0 else -1]), []).append((i, end))
    used, chains = set(), []
    for i in range(len(pieces)):
        if i in used:
            continue
        used.add(i)
        pts = list(pieces[i]["pts"])
        dashed = pieces[i]["dashed"]
        for direction in (1, 0):   # grow at the end, then at the start
            while True:
                tip = pts[-1] if direction == 1 else pts[0]
                here = ends.get(_key(tip), [])
                nxt = [(j, e) for j, e in here if j not in used and pieces[j]["dashed"] == dashed
                       and pieces[j].get("future") == pieces[i].get("future")]
                if len(here) != 2 or len(nxt) != 1:
                    break
                j, e = nxt[0]
                used.add(j)
                seg = pieces[j]["pts"] if e == 0 else pieces[j]["pts"][::-1]
                if direction == 1:
                    pts.extend(seg[1:])
                else:
                    pts[:0] = seg[::-1][:-1]
        chains.append({"pts": pts, "dashed": dashed, "future": pieces[i].get("future", False), "a": None, "b": None})
    return chains


def attach(chains: list[dict], items: list[Item]) -> None:
    """A chain end within ATTACH mm of a symbol's outline is on that symbol (the nearest)."""
    def dist(p, it):
        if it.kind != "unit":
            return _dist_to_box(p, it.box)
        b = it.box   # a unit is a frame: only its border counts (things drawn inside are its own)
        if _inside(p, b):
            return min(p[0] - b[0], b[2] - p[0], p[1] - b[1], b[3] - p[1])
        return _dist_to_box(p, b)
    for c in chains:
        for end, p in (("a", c["pts"][0]), ("b", c["pts"][-1])):
            best = min(((dist(p, it), it.id) for it in items), default=(None, None))
            if best[0] is not None and best[0] <= ATTACH:
                c[end] = best[1]


def pass_through(chains: list[dict], items: list[Item]) -> list[str]:
    """An in-line symbol with exactly two pipe ends on it: the two pipes become one, through it."""
    gone = []
    by_id = {it.id: it for it in items}
    changed = True
    while changed:
        changed = False
        for it in list(by_id.values()):
            if it.kind != "inline":
                continue
            on = [(c, end) for c in chains for end in ("a", "b") if c[end] == it.id]
            if len(on) != 2 or on[0][0] is on[1][0] or on[0][0]["dashed"] != on[1][0]["dashed"]:
                continue
            (c1, e1), (c2, e2) = on
            p1 = c1["pts"] if e1 == "b" else c1["pts"][::-1]   # runs into the symbol
            p2 = c2["pts"] if e2 == "a" else c2["pts"][::-1]   # runs out of it
            far1 = c1["a"] if e1 == "b" else c1["b"]
            far2 = c2["b"] if e2 == "a" else c2["a"]
            votes = ({v if e1 == "b" else -v for v in c1.get("votes", set())}
                     | {v if e2 == "a" else -v for v in c2.get("votes", set())})
            d = _pointing(it.ents) if not it.tag else None
            if d is not None:   # a flow arrow the pipe was cut for: its way along the joined pipe
                run = (p2[0][0] - p1[-1][0], p2[0][1] - p1[-1][1])
                if math.hypot(*run) < 1e-6 and len(p2) > 1:
                    run = (p2[1][0] - p2[0][0], p2[1][1] - p2[0][1])
                dot = d[0] * run[0] + d[1] * run[1]
                if abs(dot) > 1e-9:
                    votes.add(1 if dot > 0 else -1)
            c1.update(pts=p1 + p2, a=far1, b=far2, votes=votes)
            chains.remove(c2)
            del by_id[it.id]
            gone.append(it.tag or it.block)
            changed = True
    items[:] = list(by_id.values())
    return gone


def bridge_ends(chains: list[dict], items: list[Item]) -> list[str]:
    """An in-line symbol (no text) with ONE pipe end on it sits right against another symbol (a
    flange on a vessel nozzle or a valve): the pipe goes on to that neighbour, the flange drops out."""
    gone = []
    for it in list(items):
        if it.kind != "inline" or it.tag:
            continue
        on = [(c, end) for c in chains for end in ("a", "b") if c[end] == it.id]
        if len(on) != 1:
            continue
        others = [(_box_gap(it.box, o.box), o) for o in items if o is not it and o.kind != "inline"]
        gap, near = min(others, key=lambda t: t[0], default=(None, None))
        c, end = on[0]
        d = _pointing(it.ents)
        if d is not None and len(c["pts"]) > 1:   # an arrow at the pipe's end: flow to or from it
            q, r = (c["pts"][0], c["pts"][1]) if end == "a" else (c["pts"][-1], c["pts"][-2])
            inward = (r[0] - q[0], r[1] - q[1])
            dot = d[0] * inward[0] + d[1] * inward[1]
            if abs(dot) > 1e-9:
                out = dot < 0   # points out through this end
                c.setdefault("votes", set()).add((1 if out else -1) if end == "b" else (-1 if out else 1))
        c[end] = near.id if near is not None and gap <= 2 * ATTACH else None
        items.remove(it)
        gone.append(it.block)
    return gone


def _seg_dist(p, a, b) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    L = ax * ax + ay * ay
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L))
    return math.dist(p, (a[0] + t * ax, a[1] + t * ay))


def _ends_on_pipes(chains: list[dict]) -> int:
    """A loose pipe end lying on another pipe is a branch: attached to that pipe (c["on_a"/"on_b"]
    = its id), as a line dropped onto a line in draw.io - the draw.io route reads a tee there."""
    n = 0
    for c in chains:
        for end, p in (("a", c["pts"][0]), ("b", c["pts"][-1])):
            if c[end]:
                continue
            for o in chains:
                if o is c or o["dashed"] != c["dashed"]:
                    continue
                if any(_seg_dist(p, q, r) <= ATTACH for q, r in zip(o["pts"], o["pts"][1:])):
                    c["on_" + end] = o["id"]
                    n += 1
                    break
    return n


def _fraction(p, frame) -> tuple[float, float]:
    """A point as draw.io's fraction of a (turned) shape's own box, 0-1 each way."""
    cx, cy, w, h, turn = frame
    dx, dy = p[0] - cx, p[1] - cy
    t = math.radians(-turn)
    ux, uy = dx * math.cos(t) - dy * math.sin(t), dx * math.sin(t) + dy * math.cos(t)
    clamp = lambda v: round(min(1.0, max(0.0, v)), 3)
    return clamp(0.5 + ux / w), clamp(0.5 + uy / h)


def _pointing(ents):
    """Which way a small arrow symbol points (a unit vector), or None: an open chevron (two
    strokes meeting at the tip) or a small closed triangle (the apex opposite its shortest side)."""
    segs, tris = [], []
    for e in ents:
        t = e.dxftype()
        if t == "LINE":
            segs.append(((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)))
        elif t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            if e.closed and len(pts) == 3:
                tris.append(pts)
            elif not e.closed:
                segs += list(zip(pts, pts[1:]))
        elif t in ("SOLID", "HATCH", "POINT"):
            continue
        else:
            return None
    if tris and segs and len(tris) > 1:
        segs = []   # the shaft of a filled arrow
    tip = base = None
    if len(segs) == 2 and not tris:
        ends = [q for sg in segs for q in sg]
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4) if i // 2 != j // 2 and math.dist(ends[i], ends[j]) < 0.1]
        if len(pairs) != 1:
            return None
        i, j = pairs[0]
        tip = ends[i]
        others = [ends[k] for k in range(4) if k not in (i, j)]
        base = ((others[0][0] + others[1][0]) / 2, (others[0][1] + others[1][1]) / 2)
    elif len(tris) > 1:
        # a filled arrowhead built of several triangles (plus a shaft): along its length, the end
        # where it narrows to a point is the tip, the wide end its base
        pts = [q for t in tris for q in t]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))
        along = (lambda q: q[0]) if horizontal else (lambda q: q[1])
        across = (lambda q: q[1]) if horizontal else (lambda q: q[0])
        lo, hi = min(map(along, pts)), max(map(along, pts))
        if hi - lo < 0.5:
            return None
        spread = lambda end: (lambda c: max(c) - min(c))([across(q) for q in pts if abs(along(q) - end) < 0.15 * (hi - lo)])
        s_lo, s_hi = spread(lo), spread(hi)
        if max(s_lo, s_hi) < 0.3 or abs(s_lo - s_hi) < 0.3:
            return None
        sign = 1 if s_hi < s_lo else -1   # the narrow end is the tip
        return (sign, 0.0) if horizontal else (0.0, sign)
    elif len(tris) == 1 and not segs:
        v = tris[0]
        sides = [math.dist(v[(k + 1) % 3], v[(k + 2) % 3]) for k in range(3)]   # side opposite vertex k
        k = min(range(3), key=lambda n: sides[n])
        if sorted(sides)[1] < 1.15 * sides[k]:
            return None   # about equilateral: no clear apex
        tip, a, b = v[k], v[(k + 1) % 3], v[(k + 2) % 3]
        base = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    else:
        return None
    dx, dy = tip[0] - base[0], tip[1] - base[1]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-6 else None


def flow_from_arrows(chains: list[dict], items: list[Item]) -> int:
    """Flow arrows lying on a pipe set its direction: c["flow"] = +1 (as drawn, a -> b) or -1.
    Arrows on one pipe that disagree leave it unknown."""
    votes: dict = {}
    for it in items:
        w, h = it.box[2] - it.box[0], it.box[3] - it.box[1]
        if it.kind != "inline" or it.tag or max(w, h) > 6.0:
            continue
        d = _pointing(it.ents)
        if d is None:
            continue
        centre = ((it.box[0] + it.box[2]) / 2, (it.box[1] + it.box[3]) / 2)
        best = None
        for k, c in enumerate(chains):
            for q, r in zip(c["pts"], c["pts"][1:]):
                dist = _seg_dist(centre, q, r)
                if dist <= max(w, h) / 2 + 0.3 and (best is None or dist < best[0]):
                    best = (dist, k, (r[0] - q[0], r[1] - q[1]))
        if best is None:
            continue
        _, k, along = best
        dot = d[0] * along[0] + d[1] * along[1]
        if abs(dot) > 1e-6:
            chains[k].setdefault("votes", set()).add(1 if dot > 0 else -1)
    n = 0
    for c in chains:
        v = c.get("votes") or set()
        if len(v) == 1:
            c["flow"] = next(iter(v))
            n += 1
    return n


def _box_gap(a, b) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def _dexpi_class(it: Item) -> str:
    """The DEXPI class an item is written with (as editor data): its drawing is the CAD symbol."""
    from pid_dexpi_aas.dexpi.mapping import DISC_INST_SHAPE, GENERIC_VALVE_SHAPE, map_shape
    if it.cls:
        return it.cls
    if it.kind == "valve":
        return map_shape(GENERIC_VALVE_SHAPE, it.tag, it.style).dexpi_class_name or "CustomPipingComponent"
    if it.kind == "instrument":
        return map_shape(DISC_INST_SHAPE, it.tag).dexpi_class_name or "ProcessInstrumentationFunction"
    if it.kind == "equipment":
        return classify_tag(it.tag)[1] or "CustomEquipment"
    if it.kind == "stream":
        return "(off-page connector)"
    return "CustomPipeFitting"


def _style(it: Item) -> tuple[str, str]:
    """(draw.io style, label) for an item - draw.io P&ID library shapes the app maps to DEXPI."""
    base = "html=1;verticalLabelPosition=bottom;verticalAlign=top;fontSize=11;"
    if it.kind == "valve":
        return "shape=mxgraph.pid2valves.valve;" + it.style + base, it.tag
    if it.kind == "instrument":
        letters, _, number = it.tag.partition(" ")
        return ("shape=mxgraph.pid2inst.discInst;mounting=field;html=1;fontSize=10;whiteSpace=wrap;",
                f"{html.escape(letters)}<br>{html.escape(number)}")
    if it.kind == "equipment":
        kind, cls, _ = classify_tag(it.tag)
        token = {"Pump": "mxgraph.pid.pumps_-_iso.pump_(liquid)", "Tank": "mxgraph.pid.vessels.container,_tank,_cistern",
                 "ProcessColumn": "mxgraph.pid2misc.column"}.get(cls, "mxgraph.pid.vessels.pressurized_vessel")
        return f"shape={token};" + base, it.tag
    if it.kind == "stream":
        return "text;html=1;align=center;verticalAlign=middle;fontSize=11;", it.tag
    # unknown or in-line with a tag: a plain box, flagged in the review
    return "rounded=0;whiteSpace=wrap;html=1;dashed=1;strokeColor=#d9480f;fontSize=10;", it.tag


def dxf_to_drawio(doc, title: str = "drawing") -> tuple[str, dict]:
    """(draw.io XML, report) for an ezdxf document."""
    from ezdxf import bbox
    items, labels, frames = read_symbols(doc)
    pieces = read_lines(doc, items)
    chains = join_pieces(pieces)
    attach(chains, items)
    # unknown text where a pipe ends: a stream in or out of the drawing; else a note
    ends_on = {c[e] for c in chains for e in ("a", "b") if c[e]}
    for it in items:
        if it.kind == "unknown" and it.id in ends_on and not any(c["a"] == it.id and c["b"] == it.id for c in chains):
            n_on = sum(1 for c in chains for e in ("a", "b") if c[e] == it.id)
            if n_on == 1:
                it.kind, it.why = "stream", "text where a pipe ends: a stream"
    ends_on = {c[e] for c in chains for e in ("a", "b") if c[e]}
    units = [it for it in items if it.kind == "unit"]
    for u in units:
        if u.id not in ends_on:
            items.remove(u)   # a frame only (nothing piped to it)
            continue
        for it in items:   # what is drawn inside it is part of it
            if it is not u and _inside(((it.box[0] + it.box[2]) / 2, (it.box[1] + it.box[3]) / 2), u.box):
                it.section = u.tag
    tables = read_tables(doc)
    for it in items:
        if it.kind == "equipment" and it.tag in tables:
            it.specs = tables[it.tag]
    through = pass_through(chains, items)
    bridged = bridge_ends(chains, items)
    with_flow = flow_from_arrows(chains, items)
    for c in chains:   # drawn the way it flows: draw.io's arrowhead at the end the flow reaches
        if c.get("flow") == -1:
            c["pts"].reverse()
            c["a"], c["b"] = c["b"], c["a"]
            c["flow"] = 1
    for it in items:   # what is left unknown keeps its place, typed as generically as DEXPI allows
        n_on = sum(1 for c in chains for e in ("a", "b") if c[e] == it.id)
        if it.kind == "inline" and n_on:
            it.cls, it.why = "CustomPipeFitting", f"symbol without text, {n_on} pipe ends"
        elif it.kind == "unknown" and n_on:
            it.cls, it.why = "CustomPipingComponent", f"tag {it.tag} not recognised"
    ends_on = {c[e] for c in chains for e in ("a", "b") if c[e]}
    items = [it for it in items if it.kind not in ("unknown", "inline") or it.id in ends_on]
    items.sort(key=lambda it: it.kind != "unit")   # units first: drawn under what is inside them
    ids = {it.id for it in items}
    for c in chains:
        c["a"] = c["a"] if c["a"] in ids else None
        c["b"] = c["b"] if c["b"] in ids else None

    ext = bbox.extents(doc.modelspace())
    x0, ytop = ext.extmin.x, ext.extmax.y
    X = lambda x: round((x - x0) * SCALE, 1)
    Y = lambda y: round((ytop - y) * SCALE, 1)
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    frame = {}   # item id -> (centre x, centre y, unrotated w, h, clockwise rotation) in draw.io units
    for it in items:
        _, label = _style(it)
        svg = base64.b64encode(symbol_svg(it.ents, it.box, it.future or it.kind == "unit").encode()).decode()
        style = (f"shape=image;image=data:image/svg+xml,{svg};imageAspect=0;html=1;fontSize=10;"
                 + ("verticalLabelPosition=middle;verticalAlign=middle;" if it.kind == "instrument"
                    else "verticalLabelPosition=middle;verticalAlign=top;align=center;" if it.kind == "unit"
                    else "verticalLabelPosition=bottom;verticalAlign=top;")
                 + ("fontColor=#8a8a8a;" if it.future else ""))
        if not it.ents:   # a stream drawn as text only: stays a text label
            style = "text;html=1;align=center;verticalAlign=middle;fontSize=11;"
        it.cls = _dexpi_class(it)
        x, y = X(it.box[0]), Y(it.box[3])
        w, h = max(4.0, X(it.box[2]) - x), max(4.0, Y(it.box[1]) - y)
        cx, cy = x + w / 2, y + h / 2
        turn = 0
        if False:   # drawn as placed in the CAD file: no turning needed
            # draw.io turns a shape clockwise about its centre; CAD counter-clockwise. The box read
            # from CAD is the turned one: a quarter turn swaps its sides back.
            turn = (360 - it.rot) % 360
            if turn in (90, 270):
                w, h = h, w
            x, y = cx - w / 2, cy - h / 2
            style += f"rotation={turn};"
        frame[it.id] = (cx, cy, w, h, turn)
        geo = f'<mxGeometry x="{round(x, 1)}" y="{round(y, 1)}" width="{round(w, 1)}" height="{round(h, 1)}" as="geometry"/>'
        if it.cls and it.ents:   # its class, kept as the editor keeps a person's choice (review._apply_editor_data)
            data = {"label": label, "dexpi_class": it.cls, "dexpi_auto": "0", "notes": it.why}
            if it.future:
                data.update(in_dexpi="0", notes=it.why + "; drawn dashed and grey: planned or out of service, "
                                                        "left out of DEXPI (tick In DEXPI to include it)")
            if it.section:
                data["section"] = it.section
            if it.specs:
                import json
                data["specs"] = json.dumps(it.specs, sort_keys=True)
            if it.tag:
                data.update(tag=it.tag, tag_auto="0")
            attrs = " ".join(f"{k}={quoteattr(v)}" for k, v in data.items())
            cells.append(f'<object id="{it.id}" {attrs}><mxCell style={quoteattr(style)} vertex="1" parent="1">{geo}</mxCell></object>')
        else:
            cells.append(f'<mxCell id="{it.id}" value={quoteattr(label)} style={quoteattr(style)} vertex="1" parent="1">{geo}</mxCell>')
    kept = [c for c in chains if len(c["pts"]) >= 2 and not (
        c["a"] is None and c["b"] is None and sum(math.dist(a, b) for a, b in zip(c["pts"], c["pts"][1:])) < 3)]
    for k, c in enumerate(kept):
        c["id"] = f"l{k}"
    tees = _ends_on_pipes(kept)
    for c in kept:
        pts = c["pts"]
        style = ("endArrow=block;endFill=1;endSize=4;" if c.get("flow") else "endArrow=none;") + \
            "html=1;rounded=0;edgeStyle=none;" + ("dashed=1;" if c["dashed"] else "") + \
            ("strokeColor=#9a9a9a;dashPattern=6 4;" if c.get("future") else "")
        # each end exactly where it touches its symbol in the CAD drawing - not draw.io's own
        # choice (towards the shape's centre), which drew slanted pipes
        for end, side, q in (("a", "exit", pts[0]), ("b", "entry", pts[-1])):
            f = frame.get(c[end])
            if f:
                fx, fy = _fraction((X(q[0]), Y(q[1])), f)
                style += f"{side}X={fx};{side}Y={fy};{side}Dx=0;{side}Dy=0;{side}Perimeter=0;"
        # a branch end on another pipe stays a point ON that pipe (exact), not attached to the line:
        # draw.io would draw it to a point of its own choosing; the draw.io route reads the tee
        src, tgt = c["a"], c["b"]
        terms = (f' source="{src}"' if src else "") + (f' target="{tgt}"' if tgt else "")
        inner = "".join(f'<mxPoint x="{X(p[0])}" y="{Y(p[1])}"/>' for p in pts[1:-1])
        if c.get("future"):   # a planned / out-of-service pipe: in the drawing, left out of DEXPI
            note = "drawn dashed and grey: planned or out of service, left out of DEXPI"
            cells.append(f'<object id="{c["id"]}" in_dexpi="0" notes={quoteattr(note)}><mxCell style="{style}" edge="1" parent="1"{terms}>'
                         f'<mxGeometry relative="1" as="geometry">'
                         f'<mxPoint x="{X(pts[0][0])}" y="{Y(pts[0][1])}" as="sourcePoint"/>'
                         f'<mxPoint x="{X(pts[-1][0])}" y="{Y(pts[-1][1])}" as="targetPoint"/>'
                         + (f'<Array as="points">{inner}</Array>' if inner else "") + "</mxGeometry></mxCell></object>")
            continue
        cells.append(f'<mxCell id="{c["id"]}" style="{style}" edge="1" parent="1"{terms}><mxGeometry relative="1" as="geometry">'
                     f'<mxPoint x="{X(pts[0][0])}" y="{Y(pts[0][1])}" as="sourcePoint"/>'
                     f'<mxPoint x="{X(pts[-1][0])}" y="{Y(pts[-1][1])}" as="targetPoint"/>'
                     + (f'<Array as="points">{inner}</Array>' if inner else "") + "</mxGeometry></mxCell>")
    W, H = X(ext.extmax.x), Y(ext.extmin.y)
    xml = (f'<mxfile><diagram name={quoteattr(title)} id="cad"><mxGraphModel dx="0" dy="0" grid="1" gridSize="10" '
           f'page="1" pageWidth="{W}" pageHeight="{H}"><root>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
    kinds: dict = {}
    for it in items:
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    report = {"symbols": kinds, "pipes": sum(1 for c in chains if not c["dashed"]),
              "signal_lines": sum(1 for c in chains if c["dashed"]), "through": through, "bridged": bridged, "tees": tees, "flow_from_arrows": with_flow,
              "units": [it.tag for it in items if it.kind == "unit"], "future": sum(it.future for it in items),
              "tables": sorted(tables),
              "frames": len(frames), "unrecognised": sorted({it.tag or it.block for it in items if it.cls})}
    return xml, report
