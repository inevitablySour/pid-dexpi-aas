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
    decoration, and make it a signal line also when it is drawn grey (a planned signal line);
    lines inside an equipment outline are its internals.

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
AIM = 3.0      # mm: ... or this close when the line runs straight at it (a dashed line's gap)
SNAP = 0.3     # mm: two line ends this close are one point
SCALE = 4.0    # draw.io units per mm (an A0 sheet ~ 4800 x 3400)
DASHED = re.compile(r"HIDDEN|DASH|DOT|CENTER|PHANTOM", re.I)
TEXT_TYPES = {"TEXT", "MTEXT", "ATTDEF", "ATTRIB"}
# Dutch valve abbreviations seen on P&IDs: afsluiter, kogelafsluiter, terugslagklep
VALVE_WORDS = {"AFS": "", "KAF": "ball", "TSK": "check", "TK": "check", "KA": "ball", "VA": ""}
GREY = {8, 9, 250, 251, 252, 253, 254}
# a symbol legend's choices beside the DEXPI types (cad_legend: one choice per symbol type)
IN_LINE = "(in-line: the pipe runs through)"
FLOW_ARROW = "(flow arrow)"
LEAVE_OUT = "(leave out)"
LEGEND_SPECIAL = [IN_LINE, FLOW_ARROW, LEAVE_OUT]   # AutoCAD colour numbers drawn grey
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
    lines: list = field(default_factory=list)   # piping drawn inside the block (an assembly's lines)
    prompts: list = field(default_factory=list)  # [(prompt, value)]: what the CAD block asks for each text
    outline: list = field(default_factory=list)  # an open outline's points (a drip tray): pipes end on it


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


def _prompted(doc, ins) -> list:
    """[(prompt, value)] of a block reference's texts: the question the CAD block asks for each
    ("Pompcode?", "Leidingcode?", "Alarm High", "PID nummer?") - what that text is."""
    blk = doc.blocks.get(ins.dxf.name)
    asks = {a.dxf.tag: (a.dxf.get("prompt", "") or "") for a in blk.query("ATTDEF")} if blk is not None else {}
    return [(asks.get(a.dxf.tag, ""), (a.dxf.text or "").replace("%%U", "").replace("%%u", "").strip())
            for a in ins.attribs]


# a title block's prompts -> DEXPI MetaData fields (Dutch and English CAD templates)
TITLE_FIELDS = [
    (re.compile(r"tek(ening)?\.?\s*nummer|drawing\s*(no|num)", re.I), "drawingNumber"),
    (re.compile(r"titel|title", re.I), "drawingName"),
    (re.compile(r"wijzigingsletter|revision", re.I), "revisionNumber"),
    (re.compile(r"tekenaar|drawn|drafter", re.I), "drafterName"),
    (re.compile(r"controleur|checke", re.I), "checkerName"),
    (re.compile(r"fabriek|plant", re.I), "processPlantName"),
    (re.compile(r"bedrijf|company", re.I), "enterpriseName"),
    (re.compile(r"afdeling|department", re.I), "responsibleDepartmentName"),
    (re.compile(r"project\s*n", re.I), "projectNumber"),
]
_PLACEHOLDER = re.compile(r"^[-#X.\s]*$|^\.\.-\.\.$|^1:X+$", re.I)


def read_title_block(doc) -> dict:
    """The drawing's details from its title block (a block of many prompted texts): DEXPI
    MetaData fields. The latest revision letter filled in is the revision."""
    meta: dict = {}
    for ins in doc.modelspace().query("INSERT"):
        asked = _prompted(doc, ins)
        if len(asked) < 6:
            continue
        hits = [(f, v) for q, v in asked for rx, f in TITLE_FIELDS if q and rx.search(q)]
        if len({f for f, _ in hits}) < 3:
            continue   # an equipment data table, not a title block
        for f, v in hits:
            if not v or _PLACEHOLDER.match(v):
                continue
            if f == "revisionNumber" or f not in meta:
                meta[f] = v
    return meta


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


def _pl_points(pl) -> list:
    """A polyline's points, its arc segments (bulges) flattened."""
    if not any(b for *_, b in pl.get_points("xyb")):
        return [(p[0], p[1]) for p in pl.get_points()]
    from ezdxf import path as dpath
    return [(v.x, v.y) for v in dpath.make_path(pl).flattening(0.2)]


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


def _is_bowtie(e) -> bool:
    return _bowtie_box(e) is not None


def _bowtie_box(e):
    """A valve body: two triangles tip to tip - four corners of a rectangle drawn with its two
    end sides and a diagonal between them (a plain box never steps corner to opposite corner).
    The body can be part of a longer stroke (a spring or stem drawn on in one go)."""
    if e.dxftype() != "LWPOLYLINE":
        return False
    pts = [(p[0], p[1]) for p in e.get_points()]
    if e.closed:
        pts = pts + pts[:3]
    for q in zip(pts, pts[1:], pts[2:], pts[3:]):
        if len({(round(x, 2), round(y, 2)) for x, y in q}) != 4:
            continue
        c = (sum(x for x, _ in q) / 4, sum(y for _, y in q) / 4)
        r = [math.dist(p, c) for p in q]
        if min(r) < 0.5 or max(r) - min(r) > 0.05 * max(r):
            continue   # not the corners of a rectangle
        diag = 2 * r[0]
        sides = sorted(math.dist(a, b) for i, a in enumerate(q) for b in q[i + 1:])[:4]
        if sides[0] < 0.3 * diag:
            continue   # a thin bar (a flange), not a body
        steps = [math.dist(a, b) for a, b in zip(q, q[1:])]
        if any(abs(d - diag) < 0.05 * diag for d in steps):
            xs, ys = [x for x, _ in q], [y for _, y in q]
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def _split_assembly(doc, ins, k, tags) -> list:
    """A block holding several valves (a double root valve: two tags, two valve bodies and the
    piping between them) as one valve item per body, named by the tag written nearest it; its
    lines go in as piping (on the first item), nested symbols as their own items."""
    from ezdxf import bbox
    bodies, lines, nested = [], [], []
    for e in ins.virtual_entities():
        if e.dxftype() == "INSERT":
            nested.append(e)
        elif _is_bowtie(e):
            bodies.append(e)
        elif e.dxftype() in ("LINE", "LWPOLYLINE"):
            if e.dxf.get("layer", "0") == "0":
                e.dxf.layer = ins.dxf.layer
            lines.append(e)
    if len(bodies) < len(tags):
        return []
    spots = {a.dxf.text.replace("%%U", "").strip(): (a.dxf.insert.x, a.dxf.insert.y) for a in ins.attribs}
    out = []
    for j, b in enumerate(bodies):
        ext = bbox.extents([b])
        box = (ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y)
        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        tag = min(tags, key=lambda t: math.dist(spots.get(t, c), c))
        tags = [t for t in tags if t != tag] or [""]
        kind, tag, body, extra, why = _classify([tag]) if tag else ("valve", "", "", [], "")
        out.append(Item(f"b{k}_{j}", "valve", box, tag=tag, block=ins.dxf.name, ents=[b],
                        why=f"one of {len(bodies)} valves in one block ({why})", future=_is_future(doc, [b]),
                        style=f"valveType={body};" if body else ""))
    for j, e in enumerate(nested):
        parts = _parts(e)
        ext = bbox.extents(parts) if parts else None
        if ext is not None and ext.has_data:
            out.append(Item(f"b{k}_n{j}", "inline", (ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y),
                            block=e.dxf.name, why="symbol without text", ents=parts, future=_is_future(doc, parts)))
    out[0].lines = lines
    return out


def _text_only(doc, ins) -> bool:
    blk = doc.blocks.get(ins.dxf.name)
    return blk is not None and all(e.dxftype() in TEXT_TYPES for e in blk)


def _classify(fields: list[str]) -> tuple[str, str, str, list, str]:
    """(kind, tag, valve body type, extra fields, why) from a block's text fields."""
    if len(fields) >= 2 and _LETTERS.match(fields[0]) and _NUMBER.match(fields[1]):
        return "instrument", f"{fields[0]} {fields[1]}", "", fields[2:], "tag in two fields: an instrument bubble"
    tag = fields[0]
    if _REDUCER.match(tag):
        return "reducer", tag, "", fields[1:], "size change (reducer)"
    # "LC-AFS0221": a valve tag behind a locked-position mark (details() reads the mark)
    m = re.match(r"^(?:(?:LO|LC|CSO|CSC)[-\s]+)?([A-Z]+)-?\s*\d", tag)
    prefix = m.group(1) if m else ""
    if prefix in VALVE_WORDS:
        return "valve", tag, VALVE_WORDS[prefix], fields[1:], f"valve abbreviation {prefix}"
    kind, cls, _ = classify_tag(tag)
    if kind:
        return kind, tag, "", fields[1:], f"tag prefix ({cls})"
    return "unknown", tag, "", fields[1:], "tag not recognised"


# what a block's prompt says its text is (when the text itself is not a known tag)
PROMPT_KINDS = [
    (re.compile(r"pomp|pump", re.I), "equipment", "Pump"),
    (re.compile(r"apparaat|equipment|vessel|vat\b", re.I), "equipment", ""),
    (re.compile(r"function identifier|instrument", re.I), "instrument", ""),
    (re.compile(r"verloop|reducer", re.I), "reducer", ""),
    (re.compile(r"omschrijving|description|destination|bestemming", re.I), "stream", ""),
]


def _by_prompt(asked, kind, why):
    for q, v in asked:
        if not v:
            continue
        for rx, k, _ in PROMPT_KINDS:
            if q and rx.search(q):
                return k, f"the CAD block asks for it as '{q.strip()}'"
    return kind, why


def _within(e, box) -> bool:
    from ezdxf import bbox
    ext = bbox.extents([e])
    return ext.has_data and _inside(ext.extmin, box, 0.01) and _inside(ext.extmax, box, 0.01)


def symbol_svg(ents, box, faded: bool = False, thumb: bool = False) -> str:
    """The symbol as drawn in the CAD file, as an SVG image of exactly its box (y up -> down).

    thumb: for the legend table - a white tile and lines thick enough to read at thumbnail size."""
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
    sw = max(0.25, max(vw, vh) / 45) if thumb else 0.25
    tile = f'<rect width="{vw:.2f}" height="{vh:.2f}" fill="#fff"/>' if thumb else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:.2f} {vh:.2f}" width="{vw:.2f}mm" '
            f'height="{vh:.2f}mm">{tile}<g stroke="{"#8a8a8a" if faded else "#000"}" stroke-width="{sw:.2f}" '
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
    items, labels, databoxes = [], [], []
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
        valve_tags = [f for f in fields if _classify([f])[0] == "valve"]
        if len(valve_tags) >= 2:
            split = _split_assembly(doc, ins, k, valve_tags)
            if split:
                items += split
                continue
        if len(fields) >= 3 and not (_LETTERS.match(fields[0]) and _NUMBER.match(fields[1])):
            databoxes.append((box, fields))   # a data box beside a symbol (a safety valve's set pressure)
            continue
        kind, tag, body, extra, why = _classify(fields)
        asked = _prompted(doc, ins)
        if kind == "unknown":
            kind, why = _by_prompt(asked, kind, why)
        items.append(Item(iid, kind, box, tag=tag, extra=extra, block=ins.dxf.name, why=why, rot=round(ins.dxf.rotation) % 360, ents=parts,
                          future=_is_future(doc, parts), prompts=asked,
                          style=f"valveType={body};" if body else ""))
    # a data box touching an untagged symbol names it: "PSV-0001 / 1-12345 / 5 BARG" on a
    # safety valve's symbol makes it PSV-0001, set at 5 bar
    for dbox, fields in databoxes:
        near = min(((_box_gap(dbox, it.box), it) for it in items if not it.tag), key=lambda t: t[0], default=(None, None))
        if near[1] is None or near[0] > 1.5:
            continue
        it = near[1]
        kind, tag, body, _, why = _classify(fields[:1])
        it.kind, it.tag, it.why = (kind if kind != "unknown" else "valve"), tag, f"named by its data box ({why})"
        it.style = f"valveType={body};" if body else it.style
        for f in fields[1:]:
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*BAR", f, re.I)
            if m:
                it.specs.setdefault("dexpi", {})["setPressureHigh"] = {"value": float(m.group(1).replace(",", ".")), "unit": "bar"}
            else:
                it.specs.setdefault("custom", {})["DataBox"] = f
    # equipment drawn as an outline, named by a label inside it; outlines around symbols: frames
    frames = []
    for k, pl in enumerate(msp.query("LWPOLYLINE")):
        if not pl.closed:
            continue
        pts = _pl_points(pl)   # with its arcs (a vessel's heads are bulges), not only its corners
        box = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
        # well inside: a nozzle's flange on the outline is not "inside" it
        inside_syms = [it for it in items if _inside(((it.box[0] + it.box[2]) / 2, (it.box[1] + it.box[3]) / 2), box, -3)]
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
    # an open outline (three sides, a U) with an equipment name inside: equipment too - a drip
    # tray (lekbak) or bund under a tank; pipes end on its drawn sides, not its whole box
    taken = {f[0] for it in items for f in [[it.tag]]}
    for k, pl in enumerate(msp.query("LWPOLYLINE")):
        if pl.closed:
            continue
        pts = _pl_points(pl)
        if len(pts) < 4 or math.dist(pts[0], pts[-1]) < 1:
            continue
        box = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
        if box[2] - box[0] < 10 or box[3] - box[1] < 5 or (box[2] - box[0]) > 0.25 * W:
            continue
        names = [f for (x, y, f) in labels if _inside((x, y), box, 6) and f[0] not in taken
                 and not re.search(r"leiding|line", f[0], re.I)]
        if len(names) != 1:
            continue
        tag = names[0][0]
        cls = "CustomVessel" if re.search(r"lekbak|drip|bund|opvang|tray", tag, re.I) else (classify_tag(tag)[1] or "CustomEquipment")
        items.append(Item(f"o{k}", "equipment", box, tag=tag, extra=names[0][1:], cls=cls, ents=[pl], outline=pts,
                          why="an open outline named " + tag + (": a drip tray" if cls == "CustomVessel" else ""),
                          future=_is_future(doc, [pl]),
                          specs={"custom": {"Description": "drip tray (lekbak)"}} if cls == "CustomVessel" else {}))
        taken.add(tag)
    return items, labels, frames


_ALARM = re.compile(r"^[HL]{1,2}[A-Z]{1,2}$")
_LOCK = {"LO": "locked open", "LC": "locked closed", "CSO": "car sealed open", "CSC": "car sealed closed"}


def details(items: list[Item]) -> dict:
    """What the drawing writes beside a symbol, carried into DEXPI: an instrument's alarms and
    where it is read (a plain bubble: field; with a line or a box around it: control room), a
    valve's locked position (LO/LC), the drawing a stream comes from or goes to."""
    n = {"alarms": 0, "locked": 0, "drawings": 0, "located": 0}
    for it in items:
        custom = it.specs.setdefault("custom", {})
        if it.kind == "instrument":
            alarms = [v for v in it.extra if _ALARM.match(v)] + [v for q, v in it.prompts
                                                                  if re.search(r"alarm", q, re.I) and v and v not in it.extra]
            if alarms:
                custom["Alarms"] = ", ".join(dict.fromkeys(alarms))
                n["alarms"] += 1
            ring = any(e.dxftype() in ("ELLIPSE", "CIRCLE") for e in it.ents)
            marked = any(e.dxftype() == "LINE" or (e.dxftype() == "LWPOLYLINE" and e.closed) for e in it.ents)
            if ring:
                it.specs.setdefault("dexpi", {})["location"] = "central" if marked else "field"
                n["located"] += 1
        elif it.kind == "valve":
            m = re.match(r"^(LO|LC|CSO|CSC)[-\s]+(.+)$", it.tag)
            if m:   # "LC-AFS0221": valve AFS0221, locked closed
                custom["LockedPosition"], it.tag = _LOCK[m.group(1)], m.group(2)
            lock = [_LOCK[v.upper()] for v in it.extra if v.upper() in _LOCK]
            if lock:
                custom["LockedPosition"] = lock[0]
            n["locked"] += "LockedPosition" in custom
        elif it.kind == "stream":
            for q, v in it.prompts:
                if v and re.search(r"p\s*&?\s*i\s*d|tekening|drawing", q, re.I):
                    custom["ConnectedDrawing"] = v
                    n["drawings"] += 1
            extra = [v for q, v in it.prompts if v and re.search(r"extra|info", q, re.I)]
            if extra:
                custom["Note"] = "; ".join(extra)
        if not custom:
            it.specs.pop("custom")
    return n


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
    inner = [e for it in items for e in it.lines]   # piping drawn inside assembly blocks
    for e in list(msp.query("LINE LWPOLYLINE")) + inner:
        if e.dxftype() == "LINE":
            pts = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
        else:
            if e.closed:
                continue  # outlines: equipment and frames (read_symbols)
            pts = _pl_points(e)
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
        if it.outline:   # an open outline: its drawn sides only
            return min(_seg_dist(p, a, b) for a, b in zip(it.outline, it.outline[1:]))
        if it.kind != "unit":
            return _dist_to_box(p, it.box)
        b = it.box   # a unit is a frame: only its border counts (things drawn inside are its own)
        if _inside(p, b):
            return min(p[0] - b[0], b[2] - p[0], p[1] - b[1], b[3] - p[1])
        return _dist_to_box(p, b)
    def aims_at(p, q, it):   # the line's last stretch q -> p, carried on, runs into the box
        d = math.dist(p, q)
        if d == 0 or it.kind == "unit":
            return False
        ux, uy = (p[0] - q[0]) / d, (p[1] - q[1]) / d
        return any(_inside((p[0] + ux * s, p[1] + uy * s), it.box, 0.3) for s in (0.5, 1, 1.5, 2, 2.5, 3))
    for c in chains:
        pts = c["pts"]
        for end, p, q in (("a", pts[0], pts[1]), ("b", pts[-1], pts[-2])):
            best = min(((dist(p, it), it.id, it) for it in items), default=(None, None, None), key=lambda t: t[0])
            if best[0] is not None and (best[0] <= ATTACH or (best[0] <= AIM and aims_at(p, q, best[2]))):
                c[end] = best[1]


def _is_tick(it) -> bool:
    """A "//" mark: a small text-free symbol of short diagonal strokes only."""
    w, h = it.box[2] - it.box[0], it.box[3] - it.box[1]
    if it.tag or max(w, h) > 4 or not it.ents:
        return False
    for e in it.ents:
        if e.dxftype() != "LINE":
            return False
        a, b = e.dxf.start, e.dxf.end
        if not (abs(a.x - b.x) > 0.2 and abs(a.y - b.y) > 0.2):
            return False
    return True


def _zigzag(e, body=None) -> bool:
    """A spring: an open polyline zigzagging (sharp turns to alternate sides, 2 or more) outside
    the valve body (box `body`), whose own outline turns back too; a triangle turns one way."""
    if e.dxftype() != "LWPOLYLINE" or e.closed:
        return False
    pts = [(p[0], p[1]) for p in e.get_points()]
    sides = []
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        if body is not None and _inside(b, body, 0.1):
            continue
        u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu and nv and (u[0] * v[0] + u[1] * v[1]) / (nu * nv) < -0.3:
            sides.append(u[0] * v[1] - u[1] * v[0] > 0)
    return len(sides) >= 2 and all(x != y for x, y in zip(sides, sides[1:]))


def _chevron(ents):
    """+1 for an open "^" (apex up), -1 for "v" (apex down), else None: one 3-point open stroke."""
    if len(ents) != 1 or ents[0].dxftype() != "LWPOLYLINE" or ents[0].closed:
        return None
    pts = [(p[0], p[1]) for p in ents[0].get_points()]
    if len(pts) != 3 or any(b for *_, b in ents[0].get_points("xyb")):
        return None
    a, m, c = pts
    if abs(a[1] - c[1]) > 0.3 or abs(m[0] - (a[0] + c[0]) / 2) > 0.3 or abs(m[1] - a[1]) < 1:
        return None
    return 1 if m[1] > a[1] else -1


def shape_rules(chains: list[dict], items: list[Item]) -> list[str]:
    """What an untagged symbol is, from how it is drawn (general P&ID drawing conventions):
    a valve body (bow tie) with a spring is a safety valve, with an actuator (a stem, or a signal
    line or arrow onto it) a control valve, else a hand valve; a wave of arcs is a hose; at the open
    end of a pipe, "^" or a hook is a vent and "v" a funnel. Kept as items (DEXPI classes)."""
    ends = lambda it: [c for c in chains for e in ("a", "b") if c[e] == it.id]
    arrows = [it for it in items if it.kind == "inline" and _pointing(it.ents) is not None]
    found = []
    for it in items:
        if it.tag or it.kind not in ("inline", "unknown") or not it.ents:
            continue
        on = ends(it)
        cls = why = ""
        bodies = [b for b in map(_bowtie_box, it.ents) if b]
        if bodies:
            if any(_zigzag(e, bodies[0]) for e in it.ents):
                cls, why = "SpringLoadedGlobeSafetyValve", "a valve body with a spring: a safety valve"
            elif (len(it.ents) > 1 or any(c["dashed"] for c in on)
                  or any(not _inside((p[0], p[1]), bodies[0], 0.3) for e in it.ents
                         if e.dxftype() == "LWPOLYLINE" for p in e.get_points())
                  or any(_box_gap(a.box, it.box) <= 0.5 for a in arrows if a is not it)):
                cls, why = "OperatedValve", "a valve body with an actuator: a control valve"
            else:
                cls, why = "GlobeValve", "a valve body (bow tie)"
        elif sum(e.dxftype() in ("ELLIPSE", "ARC") for e in it.ents) >= 3:
            cls, why = "Hose", "a wave of arcs: a hose"
        elif len(on) <= 1 or (_chevron(it.ents) == -1 and (it.box[2] - it.box[0]) >= 5):
            ch = _chevron(it.ents)
            hook = (len(it.ents) == 1 and it.ents[0].dxftype() == "LWPOLYLINE" and not it.ents[0].closed
                    and any(b for *_, b in it.ents[0].get_points("xyb")))
            if ch == 1 or hook:
                cls, why = "VentilationDevice", "an open pipe end turned up (^ or a hook): a vent"
            elif ch == -1 and (it.box[2] - it.box[0]) >= 5:
                cls, why = "Funnel", "an open \"v\" at a pipe end: a funnel (drain)"
        if cls:
            it.kind, it.cls, it.why = "valve", cls, why
            found.append(cls)
    return found


def heat_tracing(chains: list[dict], items: list[Item]) -> list[str]:
    """A dashed line running alongside a pipe (within 3.5 mm, most of its length) is that pipe's
    heat tracing, not a signal line: it goes, and the pipes it runs along get heatTracingType
    (electrical when it starts at an "EL" supply, which goes too). The pipes' tags, for the report."""
    pipes = [c for c in chains if not c["dashed"] and not c.get("future")]

    def along(p, d):   # pipes a point lies alongside (parallel, 0.4-3.5 mm away)
        out = []
        for c in pipes:
            for a, b in zip(c["pts"], c["pts"][1:]):
                L = math.dist(a, b)
                if L == 0:
                    continue
                u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
                if abs(u[0] * d[1] - u[1] * d[0]) > 0.1:
                    continue
                if 0.4 <= _seg_dist(p, a, b) <= 3.5:
                    out.append(c)
                    break
        return out
    supplies = [it for it in items if (it.tag or "").strip().upper() == "EL"]
    traced, gone = [], []
    for c in chains:
        if not c["dashed"]:
            continue
        samples, hits, near = 0, 0, []
        for a, b in zip(c["pts"], c["pts"][1:]):
            L = math.dist(a, b)
            if L == 0:
                continue
            d = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
            for k in range(int(L) + 1):
                p = (a[0] + d[0] * min(k, L), a[1] + d[1] * min(k, L))
                samples += 1
                ps = along(p, d)
                if ps:
                    hits += 1
                    near.extend(ps)
        if samples < 5 or hits < 0.6 * samples:
            continue
        gone.append((c, near))
    # an "EL" supply on the tracing, or on a short dashed stub to it: electrical tracing
    stubs = []
    for c, near in gone:
        sup = [it.id for it in supplies if c["a"] == it.id or c["b"] == it.id]
        for st in chains:
            if st["dashed"] and st is not c and all(st is not g for g, _ in gone):
                feeds = [st[e] for e in ("a", "b") if st[e] in {it.id for it in supplies}]
                other = st["pts"][-1] if st["a"] in feeds else st["pts"][0]
                if feeds and any(_seg_dist(other, a, b) < 0.5 for a, b in zip(c["pts"], c["pts"][1:])):
                    sup += feeds
                    stubs.append(st)
        kind = "electrical heat tracing system" if sup else "heat tracing system"
        for it in [it for it in items if it.id in sup]:
            items.remove(it)
        for pc in {id(x): x for x in near}.values():
            pc.setdefault("specs", {}).setdefault("dexpi", {})["heatTracingType"] = kind
            traced.append(pc)
    for c in [g for g, _ in gone] + stubs:
        if c in chains:
            chains.remove(c)
    return [c.get("label", "") for c in {id(x): x for x in traced}.values()]


def signal_by_ticks(chains: list[dict], items: list[Item]) -> int:
    """A line carrying "//" marks is a signal line, whatever its colour (a grey one: planned)."""
    ticks = [((it.box[0] + it.box[2]) / 2, (it.box[1] + it.box[3]) / 2) for it in items if _is_tick(it)]
    n = 0
    for c in chains:
        if c["dashed"]:
            continue
        segs = list(zip(c["pts"], c["pts"][1:]))
        if any(_seg_dist(t, a, b) < 1.0 for t in ticks for a, b in segs):
            c["dashed"] = True
            n += 1
    # a planned line branching off a planned signal line is one too (not one meeting it at an
    # instrument: that can be the instrument's tap on a planned pipe)
    changed = True
    while changed:
        changed = False
        sig = [c for c in chains if c["dashed"] and c.get("future")]
        for c in chains:
            if c["dashed"] or not c.get("future"):
                continue
            ends = [c["pts"][0], c["pts"][-1]]
            if any(_seg_dist(p, a, b) < 0.5 for p in ends for s in sig for a, b in zip(s["pts"], s["pts"][1:])):
                c["dashed"] = changed = True
                n += 1
    return n


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
            specs = {**(c2.get("specs") or {}), **(c1.get("specs") or {})}
            c1.update(pts=p1 + p2, a=far1, b=far2, votes=votes, specs=specs)
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
            if e.closed and len(pts) > 3:
                return None   # a box or a body: a component, not an arrow
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
    if it.kind == "reducer":
        return "PipeReducer"
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


def apply_legend(items: list[Item], legend: dict) -> None:
    """A person's symbol legend ({block name: DEXPI type or a LEGEND_SPECIAL choice}) over the
    general rules, for every symbol of that type."""
    from pid_dexpi_aas.review import EQUIPMENT_CLASSES, INSTRUMENT_CLASSES, OFF_PAGE
    for it in list(items):
        choice = legend.get(it.block)
        if not choice:
            continue
        it.why = f"symbol legend: {choice}"
        if choice == LEAVE_OUT:
            items.remove(it)
        elif choice in (IN_LINE, FLOW_ARROW):
            it.kind, it.cls = "inline", ""
        elif choice == OFF_PAGE:
            it.kind, it.cls = "stream", ""
        else:
            it.cls = choice
            it.kind = ("instrument" if choice in INSTRUMENT_CLASSES else
                       "equipment" if choice in EQUIPMENT_CLASSES else "valve")


def dxf_to_drawio(doc, title: str = "drawing", legend: dict | None = None) -> tuple[str, dict]:
    """(draw.io XML, report) for an ezdxf document. legend: a person's choice per symbol type
    (apply_legend); the report's "legend" lists every symbol type found, for making one."""
    from collections import Counter
    from ezdxf import bbox
    items, labels, frames = read_symbols(doc)
    every = list(items)   # for the legend: types a legend leaves out stay listed (to be put back)
    if legend:
        apply_legend(items, legend)
    pieces = read_lines(doc, items)
    chains = join_pieces(pieces)
    attach(chains, items)
    traced = heat_tracing(chains, items)
    signal_by_ticks(chains, items)
    shaped = shape_rules(chains, items)
    found = [(it, sum(1 for c in chains for e in ("a", "b") if c[e] == it.id)) for it in every if it.block]
    # unknown text where a pipe ends: a stream in or out of the drawing; else a note
    ends_on = {c[e] for c in chains for e in ("a", "b") if c[e]}
    for it in items:
        if it.kind == "unknown" and it.id in ends_on and not any(c["a"] == it.id and c["b"] == it.id for c in chains):
            on = [c for c in chains for e in ("a", "b") if c[e] == it.id]
            if len(on) == 1 and not on[0]["dashed"]:
                it.kind, it.why = "stream", "text where a pipe ends: a stream"
            elif on and all(c["dashed"] for c in on):
                # text at the end of a signal line: instrumentation, not a stream
                it.kind, it.cls, it.why = "instrument", "ProcessInstrumentationFunction", "text at the end of a signal line"
    # one block, one meaning: a block used mostly as a stream is a stream everywhere (a stream
    # named like equipment - "V0205" on the same arrow box as MEDIUM 01 - is still a stream)
    by_block: dict = {}
    for it in items:
        if it.block:
            by_block.setdefault(it.block, []).append(it)
    for same in by_block.values():
        streams = [it for it in same if it.kind == "stream"]
        if len(streams) * 2 > len(same):
            for it in same:
                if it.kind != "stream" and sum(1 for c in chains for e in ("a", "b") if c[e] == it.id) <= 1:
                    it.kind, it.cls, it.why = "stream", "", f"drawn with the same symbol as the streams ({len(streams)} of {len(same)})"
    ends_on = {c[e] for c in chains for e in ("a", "b") if c[e]}
    units = [it for it in items if it.kind == "unit"]
    for u in units:
        if u.id not in ends_on:
            items.remove(u)   # a frame only (nothing piped to it)
            continue
        for it in items:   # what is drawn inside it is part of it
            if it is not u and _inside(((it.box[0] + it.box[2]) / 2, (it.box[1] + it.box[3]) / 2), u.box):
                it.section = u.tag
    # an instrument with no line of its own drawn against a process pipe (within TAP mm) measures
    # on it: a short branch from the bubble to the pipe, which the draw.io route reads as a tap
    taps = 0
    TAP = 8.0
    ends_on = {c[e] for c in chains for e in ("a", "b") if c[e]}
    for it in items:
        if it.kind != "instrument" or it.id in ends_on or it.future:
            continue
        centre = ((it.box[0] + it.box[2]) / 2, (it.box[1] + it.box[3]) / 2)
        # a bubble standing on an in-line symbol (an orifice plate whose block draws the line up to
        # the bubble) measures where that symbol is on the pipe, however far that is
        on = [x for x in items if x.kind == "inline" and not x.tag and _box_gap(x.box, it.box) <= ATTACH]
        best = None
        for c in chains:
            if c["dashed"] or c.get("future"):
                continue
            for q, r in zip(c["pts"], c["pts"][1:]):
                ax, ay = r[0] - q[0], r[1] - q[1]
                L = ax * ax + ay * ay
                t = 0.0 if L == 0 else max(0.0, min(1.0, ((centre[0] - q[0]) * ax + (centre[1] - q[1]) * ay) / L))
                foot = (q[0] + t * ax, q[1] + t * ay)
                d = _dist_to_box(foot, it.box)
                if (d <= TAP or any(_dist_to_box(foot, x.box) <= 0.5 for x in on)) and (best is None or d < best[0]):
                    best = (d, foot)
        if best is not None and it.tag[:2].upper() == "FO":
            for x in on:   # FO: a flow orifice - the symbol it stands on is the orifice plate
                x.kind, x.cls, x.why = "valve", "RestrictionOrifice", f"the symbol {it.tag} stands on"
        if best is None:
            continue
        foot = best[1]
        edge = (min(max(foot[0], it.box[0]), it.box[2]), min(max(foot[1], it.box[1]), it.box[3]))
        chains.append({"pts": [edge, foot], "dashed": False, "future": False, "a": it.id, "b": None})
        taps += 1
    tables = read_tables(doc)
    line_numbers = 0
    for x, y, fields in labels:   # "LEIDING-001-D80" beside a pipe: its line number and size
        text = fields[0]
        if classify_tag(text)[0] or not re.search(r"\d", text) or "-" not in text or len(text) < 6:
            continue
        best = None
        for c in chains:
            if c["dashed"]:
                continue
            for q, r in zip(c["pts"], c["pts"][1:]):
                d = _seg_dist((x, y), q, r)
                if d <= 8.0 and (best is None or d < best[0]):
                    best = (d, c)
        if best is None:
            continue
        c = best[1]
        dexpi = c.setdefault("specs", {}).setdefault("dexpi", {})
        dexpi["segmentNumber"] = text
        m = re.search(r"\bD(?:N)?\s?(\d{2,4})\b", text)
        if m:
            dexpi["nominalDiameterRepresentation"] = f"DN{m.group(1)}"
        c["label"] = text
        line_numbers += 1
    for it in items:
        if it.kind == "equipment" and it.tag in tables:
            it.specs = tables[it.tag]
    noted = details(items)
    meta = read_title_block(doc)
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
    root = " ".join(f"{k}={quoteattr(v)}" for k, v in meta.items())
    cells = [f'<object id="0" label="" {root}><mxCell/></object>' if meta else '<mxCell id="0"/>',
             '<mxCell id="1" parent="0"/>']
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
            ("strokeColor=#9a9a9a;dashed=1;dashPattern=6 4;" if c.get("future") else "")
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
        data = {}
        if c.get("future"):   # a planned / out-of-service pipe: in the drawing, left out of DEXPI
            data.update(in_dexpi="0", notes="drawn dashed and grey: planned or out of service, left out of DEXPI")
        if c.get("specs"):
            import json
            data["specs"] = json.dumps(c["specs"], sort_keys=True)
        if c.get("label"):
            data["label"] = c["label"]
            style += "fontSize=9;labelBackgroundColor=#ffffff;"
        if data:
            attrs = " ".join(f"{k}={quoteattr(v)}" for k, v in data.items())
            cells.append(f'<object id="{c["id"]}" {attrs}><mxCell style="{style}" edge="1" parent="1"{terms}>'
                         f'<mxGeometry relative="1" as="geometry">'
                         f'<mxPoint x="{X(pts[0][0])}" y="{Y(pts[0][1])}" as="sourcePoint"/>'
                         f'<mxPoint x="{X(pts[-1][0])}" y="{Y(pts[-1][1])}" as="targetPoint"/>'
                         + (f'<Array as="points">{inner}</Array>' if inner else "") + "</mxGeometry></mxCell></object>")
            continue
        cells.append(f'<mxCell id="{c["id"]}" style="{style}" edge="1" parent="1"{terms}><mxGeometry relative="1" as="geometry">'
                     f'<mxPoint x="{X(pts[0][0])}" y="{Y(pts[0][1])}" as="sourcePoint"/>'
                     f'<mxPoint x="{X(pts[-1][0])}" y="{Y(pts[-1][1])}" as="targetPoint"/>'
                     + (f'<Array as="points">{inner}</Array>' if inner else "") + "</mxGeometry></mxCell>")
    # the symbol legend: every symbol type, what it became, a picture of it
    emitted = {it.id: it for it in items}
    rows: dict = {}
    for it, n_ends in found:
        row = rows.setdefault(it.block, {"block": it.block, "count": 0, "became": Counter(), "why": it.why,
                                         "svg": None, "example": it.tag})
        row["count"] += 1
        if it.id in emitted:
            became = emitted[it.id].cls or ("(off-page connector)" if emitted[it.id].kind == "stream" else "")
        elif (legend or {}).get(it.block) in LEGEND_SPECIAL:
            became = legend[it.block]
        elif not it.tag and _pointing(it.ents) is not None:
            became = FLOW_ARROW
        elif n_ends:
            became = IN_LINE
        else:
            became = LEAVE_OUT
        row["became"][became] += 1
        if row["svg"] is None and it.ents:
            row["svg"] = symbol_svg(it.ents, it.box, thumb=True)
        if not row["example"] and it.tag:
            row["example"] = it.tag
    legend_rows = []
    for row in sorted(rows.values(), key=lambda r: -r["count"]):
        choice, _ = row["became"].most_common(1)[0]
        legend_rows.append({"block": row["block"], "count": row["count"], "dexpi_class": choice,
                            "why": row["why"], "example": row["example"] or "", "svg": row["svg"] or ""})
    W, H = X(ext.extmax.x), Y(ext.extmin.y)
    xml = (f'<mxfile><diagram name={quoteattr(title)} id="cad"><mxGraphModel dx="0" dy="0" grid="1" gridSize="10" '
           f'page="1" pageWidth="{W}" pageHeight="{H}"><root>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
    kinds: dict = {}
    for it in items:
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    report = {"symbols": kinds, "pipes": sum(1 for c in chains if not c["dashed"]),
              "heat_traced": len(traced), "details": noted, "title_block": meta, "by_shape": dict(Counter(shaped)), "signal_lines": sum(1 for c in chains if c["dashed"]), "through": through, "bridged": bridged, "tees": tees, "flow_from_arrows": with_flow,
              "units": [it.tag for it in items if it.kind == "unit"], "future": sum(it.future for it in items),
              "tables": sorted(tables), "line_numbers": line_numbers, "taps_added": taps,
              "legend": legend_rows,
              "frames": len(frames), "unrecognised": sorted({it.tag or it.block for it in items if it.cls})}
    return xml, report
