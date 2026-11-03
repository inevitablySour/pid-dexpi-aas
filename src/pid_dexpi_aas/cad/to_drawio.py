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


def symbol_svg(ents, box) -> str:
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
            f'height="{vh:.2f}mm"><g stroke="#000" stroke-width="0.25" stroke-linecap="round" '
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
                              rot=round(ins.dxf.rotation) % 360, ents=parts))
            continue
        if len(fields) >= 3 and not (_LETTERS.match(fields[0]) and _NUMBER.match(fields[1])):
            continue  # a data box beside a symbol (a safety valve's set pressure), not a symbol
        kind, tag, body, extra, why = _classify(fields)
        items.append(Item(iid, kind, box, tag=tag, extra=extra, block=ins.dxf.name, why=why, rot=round(ins.dxf.rotation) % 360, ents=parts,
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
            continue
        for f in names:
            kind, cls, _ = classify_tag(f[0])
            if kind == "equipment":
                inner = [e for e in msp.query("LINE LWPOLYLINE ARC CIRCLE ELLIPSE")
                         if e is not pl and _within(e, box)]
                items.append(Item(f"q{k}", "equipment", box, tag=f[0], extra=f[1:], why=f"outline named {f[0]}",
                                  ents=[pl] + inner))
                break
    return items, labels, frames


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
        lt = e.dxf.linetype if e.dxf.hasattr("linetype") else "BYLAYER"
        if lt.upper() == "BYLAYER":
            lt = layer_lt.get(e.dxf.layer, "")
        out.append({"pts": pts, "dashed": bool(DASHED.search(lt or ""))})
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
                nxt = [(j, e) for j, e in here if j not in used and pieces[j]["dashed"] == dashed]
                if len(here) != 2 or len(nxt) != 1:
                    break
                j, e = nxt[0]
                used.add(j)
                seg = pieces[j]["pts"] if e == 0 else pieces[j]["pts"][::-1]
                if direction == 1:
                    pts.extend(seg[1:])
                else:
                    pts[:0] = seg[::-1][:-1]
        chains.append({"pts": pts, "dashed": dashed, "a": None, "b": None})
    return chains


def attach(chains: list[dict], items: list[Item]) -> None:
    """A chain end within ATTACH mm of a symbol's outline is on that symbol (the nearest)."""
    for c in chains:
        for end, p in (("a", c["pts"][0]), ("b", c["pts"][-1])):
            best = min(((_dist_to_box(p, it.box), it.id) for it in items), default=(None, None))
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
            c1.update(pts=p1 + p2, a=far1, b=far2)
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
    through = pass_through(chains, items)
    bridged = bridge_ends(chains, items)
    for it in items:   # what is left unknown keeps its place, typed as generically as DEXPI allows
        n_on = sum(1 for c in chains for e in ("a", "b") if c[e] == it.id)
        if it.kind == "inline" and n_on:
            it.cls, it.why = "CustomPipeFitting", f"symbol without text, {n_on} pipe ends"
        elif it.kind == "unknown" and n_on:
            it.cls, it.why = "CustomPipingComponent", f"tag {it.tag} not recognised"
    ends_on = {c[e] for c in chains for e in ("a", "b") if c[e]}
    items = [it for it in items if it.kind not in ("unknown", "inline") or it.id in ends_on]
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
        svg = base64.b64encode(symbol_svg(it.ents, it.box).encode()).decode()
        style = (f"shape=image;image=data:image/svg+xml,{svg};imageAspect=0;html=1;fontSize=10;"
                 + ("verticalLabelPosition=middle;verticalAlign=middle;" if it.kind == "instrument"
                    else "verticalLabelPosition=bottom;verticalAlign=top;"))
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
        style = "endArrow=none;html=1;rounded=0;edgeStyle=none;" + ("dashed=1;" if c["dashed"] else "")
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
              "signal_lines": sum(1 for c in chains if c["dashed"]), "through": through, "bridged": bridged, "tees": tees,
              "frames": len(frames), "unrecognised": sorted({it.tag or it.block for it in items if it.cls})}
    return xml, report
