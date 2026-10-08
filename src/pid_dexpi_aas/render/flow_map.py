"""The flow map: the drawing with every pipe traced along its drawn route, coloured by how
its flow direction is known, with arrowheads pointing downstream - so a reviewer can see at
a glance what the converter understood. Fully automatic for a draw.io file:

  - the routes come from draw.io's own SVG export (line_routes.py), the picture from its
    PNG export, both at the same origin - no manual tracing, no image alignment;
  - tees, gap crossings and hand-drawn dashed lines come from the parser;
  - flow directions from flow_direction.py (drawn arrows, then implied ones).

Two steps: build_layers() describes everything to draw (each line with the review-table
rows it belongs to and a hover text), then either flow_map() rasterises it to a PNG
(render_labels.py, tests) or map_markup() writes an interactive SVG for the review app -
hover for what a line or box is, rows ticked in the tables highlighted and zoomed to.
image_layers() gives the same for image-only mode, from the model's boxes and connections.
In the app the map is a st.components.v2 component (map_markup + MAP_JS/MAP_CSS): clicking
a line or box selects its table rows.

Needs the draw.io desktop app (ml/tools/drawio.AppImage) for the draw.io map; build_layers
returns None without it.
"""
from __future__ import annotations

import base64
import html
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pid_dexpi_aas.drawio.parser import ParsedDiagram
from pid_dexpi_aas.render.routing import route

# One colour per meaning, nothing shared: grey is only "flow unknown", red only "left out".
FLOW_COLORS = {"arrow": (215, 80, 0), "implied": (150, 50, 200), "reviewer": (0, 150, 120), "": (95, 95, 95)}
FLOW_WORD = {"arrow": "arrow", "implied": "implied", "reviewer": "reviewer", "": "?"}
FLOW_TEXT = {"arrow": "flow from an arrow drawn on the line", "implied": "flow implied by neighbouring arrows",
             "reviewer": "flow direction set by the reviewer", "": "flow direction unknown"}
SIGNAL = (0, 140, 255)
TAP = (110, 160, 190)
GUESSED = (230, 160, 0)  # amber: "check this", like items to check
COMPARTMENT = (40, 160, 60)
EXCLUDED = (205, 40, 40)  # left out of DEXPI: red, dashed - items and lines alike
MODEL_LINE = (150, 95, 45)  # brown, dashed: only the model / only the picture says so
# item boxes
BOX_IN = (0, 140, 70)
BOX_CHECK = (225, 150, 0)
BOX_OUT = EXCLUDED
BOX_PICTURE = MODEL_LINE
OCR_TAG = (110, 90, 170)
BOX_LEGEND = [("item in DEXPI", BOX_IN, "box-in", "box"), ("item to check", BOX_CHECK, "box-check", "box"),
              ("item left out of DEXPI", BOX_OUT, "box-out", "box")]
HIGHLIGHT = (230, 0, 200)
SECTION = (120, 120, 170)  # plant section outlines
MEMBER = (30, 110, 255)  # items of a selected section: theirs, not one object with it
LEGEND_BAND = 120


@dataclass
class Line:
    points: list  # sheet pixels
    color: tuple
    width: int
    rows: set  # review-table connection row ids this line belongs to
    tip: str  # hover text
    dashed: bool = False
    arrows: list = field(default_factory=list)  # (tip point, unit direction)
    word: tuple | None = None  # (text, position)
    dots: list = field(default_factory=list)  # tee points
    guessed: list = field(default_factory=list)  # (line end, symbol edge) jumps
    ends: tuple = ()  # symbol ids at its ends (highlighted with the line)
    kind: str = "pipe"  # pipe | signal | tap | excluded | compartment - the map toolbar toggles each


@dataclass
class Box:
    id: str
    box: tuple  # sheet pixels
    color: tuple
    label: str
    width: int = 3
    tip: str = ""
    dashed: bool = False  # not from the drawing: a model-only detection or an OCR-read tag
    kind: str = "item"  # item | model | ocr - the map toolbar toggles each


@dataclass
class Layers:
    background: Path
    size: tuple
    lines: list
    boxes: list
    stats: dict
    legend: list  # (text, colour, kind the map toolbar can hide | None)
    origin: tuple = (0.0, 0.0)  # sheet pixel = draw.io coordinate + origin (draw.io maps); (0, 0) for pictures


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _along(route, frac):
    """Point and unit direction at a fraction of the route's length."""
    seg = [(a, b, math.dist(a, b)) for a, b in zip(route, route[1:])]
    total = sum(L for _, _, L in seg) or 1.0
    target, acc = frac * total, 0.0
    for a, b, L in seg:
        if L and acc + L >= target:
            t = (target - acc) / L
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        acc += L
    a, b, L = seg[-1]
    return b, ((b[0] - a[0]) / (L or 1), (b[1] - a[1]) / (L or 1))


def _arrow_polygon(tip, u, size=16):
    px, py = -u[1], u[0]
    base = (tip[0] - u[0] * size, tip[1] - u[1] * size)
    return [tip, (base[0] + px * size * 0.55, base[1] + py * size * 0.55),
            (base[0] - px * size * 0.55, base[1] - py * size * 0.55)]


def _dash_segments(pts, dash=14, gap=10):
    out = []
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        k = 0.0
        while k < L:
            t0, t1 = k / (L or 1), min(1.0, (k + dash) / (L or 1))
            out.append(((a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1])),
                        (a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1]))))
            k += dash + gap
    return out


def _nearest_on_box(p, b):
    return (min(max(p[0], b[0]), b[2]), min(max(p[1], b[1]), b[3]))


def _arrows_on(run, flow):
    length = sum(math.dist(a, b) for a, b in zip(run, run[1:]))
    n = max(1, int(length // 350)) if flow else 0
    return [_along(run, (k + 0.5) / n + 8 / max(length, 1)) for k in range(n)], length


# ------------------------------------------------------------------------- layers ---
def build_layers(diagram: ParsedDiagram, drawio_path: str | Path, boxes: list | None = None,
                 overrides: dict | None = None, names: dict | None = None) -> Layers | None:
    """Everything the draw.io flow map shows, in sheet pixels.

    boxes: [Box | (id, (x1, y1, x2, y2) in draw.io units, colour, label, width)] to outline;
      default: every symbol, coloured by category.
    overrides: {edge id: (include, flow sign relative to the edge's source -> target,
      basis)} from the reviewer's edits (review_overrides); an excluded line is drawn faint,
      a reviewer-set direction in teal.
    names: {shape id: what to call it in hover texts} (default: its text).
    """
    from pid_dexpi_aas.drawio.line_routes import routes_for, sheet_png
    from pid_dexpi_aas.dexpi.mapping import Category, map_shape

    got = sheet_png(drawio_path, diagram.edges)
    routes = routes_for(drawio_path, diagram.edges)
    if not got or not routes:
        return None
    png, (dx, dy) = got
    with Image.open(png) as im:
        size = im.size
    P = lambda p: (p[0] + dx, p[1] + dy)
    shapes = {s.id: s for s in diagram.shapes}
    category = {s.id: map_shape(s.shape_token, s.text).category for s in diagram.shapes if s.shape_token}
    instruments = {sid for sid, c in category.items() if c == Category.INSTRUMENTATION}
    pbox = {sid: (s.x + dx, s.y + dy, s.x + s.width + dx, s.y + s.height + dy) for sid, s in shapes.items()}
    ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    ends = {eid: (r[0], r[-1]) for eid, r in routes.items()}
    overrides = overrides or {}
    names = names or {}
    name = lambda sid: (names.get(sid) or (shapes[sid].text if sid in shapes else "") or sid or "a loose end").strip()
    # connection rows of each pipe network ("net3:<a>:<b>"), so one network line lights up with any of them
    net_rows: dict[str, set] = {}
    for e in diagram.edges:
        if e.style == "pipe network":
            net_rows.setdefault(e.id.split(":", 1)[0], set()).add(e.id)
    net_ends: dict[str, set] = {}
    for e in diagram.edges:
        if e.merged_into:
            net_ends.setdefault(e.merged_into, set()).update(x for x in (e.source, e.target) if x)
    stats = {"arrow": 0, "implied": 0, "reviewer": 0, "": 0, "signal": 0, "tap": 0, "excluded": 0}
    lines: list[Line] = []

    for e in diagram.edges:
        if e.style == "adjacent":
            if e.source in pbox and e.target in pbox:
                lines.append(Line([ctr(pbox[e.source]), ctr(pbox[e.target])], COMPARTMENT, 5, {e.id},
                                  f"{name(e.source)} | {name(e.target)}: compartments sharing a wall",
                                  ends=(e.source, e.target), kind="compartment"))
            continue
        r = routes.get(e.id)
        if e.is_decorative or not r or len(r) < 2:
            continue
        pts = [P(p) for p in r]
        rows = net_rows.get(e.merged_into, set()) if e.merged_into else {e.id}
        if e.merged_into:
            members = sorted(name(s) for s in net_ends.get(e.merged_into, ()))
            what = f"pipe network {e.merged_into} joining {', '.join(members)}"
            line_ends = tuple(net_ends.get(e.merged_into, ()))
        else:
            what = f"{name(e.source)} - {name(e.target)}"
            line_ends = tuple(x for x in (e.source, e.target) if x)
        include, flow, basis = overrides.get(e.id, (True, e.flow, e.flow_basis if e.flow else ""))
        if not include:
            lines.append(Line(pts, EXCLUDED, 3, rows, f"{what}: left out of DEXPI", dashed=True, ends=line_ends,
                              kind="excluded"))
            stats["excluded"] += 1
            continue
        if e.is_dashed:
            lines.append(Line(pts, SIGNAL, 3, rows, f"{what}: signal line", dashed=True, ends=line_ends, kind="signal"))
            stats["signal"] += 1
            continue
        if not flow and (e.source in instruments or e.target in instruments):
            lines.append(Line(pts, TAP, 3, rows, f"{what}: instrument on the pipe (no process flow)", ends=line_ends,
                              kind="tap"))
            stats["tap"] += 1
            continue
        basis = basis if flow else ""
        stats[basis] += 1
        run = pts if flow >= 0 else pts[::-1]
        arrows, length = _arrows_on(run, flow)
        if flow and not e.merged_into:
            up, down = (e.source, e.target) if flow > 0 else (e.target, e.source)
            what = f"{name(up)} -> {name(down)}"
        line = Line(pts, FLOW_COLORS[basis], 5, rows, f"{what}: process pipe, {FLOW_TEXT[basis]}",
                    arrows=arrows, ends=line_ends, kind=f"flow-{basis or 'unknown'}")
        if length >= 80:  # a stub has no room for a word
            mid, u = _along(run, 0.5)
            line.word = (FLOW_WORD[basis], (mid[0] - u[1] * 16 + 4, mid[1] + u[0] * 16 - 8))
        for end, sid, res in ((pts[0], e.source, e.source_resolution), (pts[-1], e.target, e.target_resolution)):
            if sid and res.startswith("nearest") and sid in pbox:
                line.guessed.append((end, _nearest_on_box(end, pbox[sid])))
        # a tee only where the line ends on the MIDDLE of another; two pieces of one pipe
        # meeting end to end get no dot
        for end, junction in ((r[0], e.source_junction), (r[-1], e.target_junction)):
            if junction and junction in ends and min(math.dist(end, q) for q in ends[junction]) > 10:
                line.dots.append(P(end))
        lines.append(line)

    # plant sections (frames drawn around groups, sections.py): outline + name, own layer
    for sec in getattr(diagram, "sections", None) or []:
        pts = [P(p) for p in sec.polygon] + [P(sec.polygon[0])]
        top = min(pts, key=lambda p: (p[1], p[0]))
        lines.append(Line(pts, SECTION, 3, {f"section:{sec.id}"}, f"Plant section: {sec.name} ({len(sec.members)} items)",
                          dashed=True, word=(sec.name, (top[0] + 6, top[1] + 4)), kind="section",
                          ends=tuple(sec.members)))
    if boxes is None:
        cat_color = {Category.EQUIPMENT: (240, 50, 230), Category.PIPING: (0, 128, 128),
                     Category.INSTRUMENTATION: (67, 99, 216)}
        boxes = [(sid, (s.x, s.y, s.x + s.width, s.y + s.height), cat_color[category[sid]], s.text, 4)
                 for sid, s in shapes.items() if category.get(sid) in cat_color]
    out_boxes = []
    for b in boxes:
        if isinstance(b, Box):
            out_boxes.append(Box(b.id, (b.box[0] + dx, b.box[1] + dy, b.box[2] + dx, b.box[3] + dy),
                                 b.color, b.label, b.width, b.tip, b.dashed, b.kind))
        else:
            sid, (x1, y1, x2, y2), color, label, width = b
            out_boxes.append(Box(sid, (x1 + dx, y1 + dy, x2 + dx, y2 + dy), color, label, width, label))
    legend = [(f"flow from an arrow drawn on the line ({stats['arrow']})", FLOW_COLORS["arrow"], "flow-arrow"),
              (f"flow implied by neighbouring arrows ({stats['implied']})", FLOW_COLORS["implied"], "flow-implied"),
              (f"flow direction unknown ({stats['']})", FLOW_COLORS[""], "flow-unknown"),
              (f"signal line ({stats['signal']})", SIGNAL, "signal"),
              (f"instrument on a pipe ({stats['tap']})", TAP, "tap"),
              ("tee", (0, 0, 0), None), ("guessed line end", GUESSED, None),
              ("compartments sharing a wall", COMPARTMENT, "compartment")]
    if getattr(diagram, "sections", None):
        legend.append((f"plant sections ({len(diagram.sections)})", SECTION, "section"))
    if stats["reviewer"]:
        legend.insert(2, (f"direction set by you ({stats['reviewer']})", FLOW_COLORS["reviewer"], "flow-reviewer"))
    if stats["excluded"]:
        legend.append((f"line left out of DEXPI ({stats['excluded']})", EXCLUDED, "excluded"))
    return Layers(Path(png), size, lines, out_boxes, stats, legend, origin=(dx, dy))


def image_layers(image_path: str | Path, connections: list, boxes: list[Box]) -> Layers:
    """Image mode: pipes traced on the picture along their path, coloured by how their flow
    direction is known (as on the draw.io map); the model's untraced predictions as
    straight lines between box centres.
    connections: [(row id, from id, to id, line type, include, hover text, path | None,
    flow basis "arrow" | "implied" | "reviewer" | "")] - flow runs from -> to."""
    with Image.open(image_path) as im:
        size = im.size
    centre = {b.id: ((b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2) for b in boxes}
    box_of = {b.id: b.box for b in boxes}
    all_boxes = [b.box for b in boxes if b.id]
    lines, n = [], {"traced": 0, "model": 0, "signal": 0, "excluded": 0}
    for rid, a, b, kind, include, tip, path, basis in connections:
        if a not in centre or b not in centre:
            continue
        # traced pipes follow the picture; the rest is routed around the items (routing.py)
        pts = [tuple(p) for p in path] if path and len(path) >= 2 else route(box_of[a], box_of[b], all_boxes)
        if not include:
            n["excluded"] += 1
            lines.append(Line(pts, EXCLUDED, 3, {rid}, tip, dashed=True, ends=(a, b), kind="excluded"))
            continue
        if kind == "signal":
            n["signal"] += 1
            lines.append(Line(pts, SIGNAL, 3, {rid}, tip, dashed=True, ends=(a, b), kind="signal"))
            continue
        traced = bool(path)
        n["traced" if traced else "model"] += 1
        n[f"flow-{basis or 'unknown'}" if traced else "x"] = n.get(f"flow-{basis or 'unknown'}" if traced else "x", 0) + 1
        color = FLOW_COLORS[basis] if traced else MODEL_LINE
        arrows, _ = _arrows_on(pts, 1 if basis else 0)
        lines.append(Line(pts, color, 5 if traced else 3, {rid}, tip, dashed=not traced, arrows=arrows, ends=(a, b),
                          kind=f"flow-{basis or 'unknown'}" if traced else "model-line"))
    legend = [(f"traced pipe, flow from an arrowhead ({n.get('flow-arrow', 0)})", FLOW_COLORS["arrow"], "flow-arrow"),
              (f"traced pipe, flow implied ({n.get('flow-implied', 0)})", FLOW_COLORS["implied"], "flow-implied"),
              (f"traced pipe, flow unknown ({n.get('flow-unknown', 0)})", FLOW_COLORS[""], "flow-unknown"),
              (f"predicted by the model, not traced ({n['model']})", MODEL_LINE, "model-line"),
              (f"signal line ({n['signal']})", SIGNAL, "signal")]
    if n["excluded"]:
        legend.append((f"line left out of DEXPI ({n['excluded']})", EXCLUDED, "excluded"))
    return Layers(Path(image_path), size, lines, boxes, n, legend)


# ------------------------------------------------------------------------ raster ---
def rasterize(layers: Layers, max_side: int | None = 3200) -> Image.Image:
    im = _flatten(Image.open(layers.background))
    im = Image.blend(im, Image.new("RGB", im.size, "white"), 0.6)  # fade the drawing under the overlay
    canvas = Image.new("RGB", (im.width, im.height + LEGEND_BAND), "white")
    canvas.paste(im, (0, LEGEND_BAND))
    im = canvas
    d = ImageDraw.Draw(im)
    S = lambda p: (p[0], p[1] + LEGEND_BAND)
    small, f = font(17), font(20)
    for ln in layers.lines:
        pts = [S(p) for p in ln.points]
        if ln.dashed:
            for a, b in _dash_segments(pts):
                d.line([a, b], fill=ln.color, width=ln.width)
        else:
            d.line(pts, fill=ln.color, width=ln.width, joint="curve")
        for a, b in ln.guessed:
            for x, y in _dash_segments([S(a), S(b)], dash=6, gap=6):
                d.line([x, y], fill=GUESSED, width=3)
        for x, y in (S(p) for p in ln.dots):
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(0, 0, 0))
        for tip, u in ln.arrows:
            d.polygon(_arrow_polygon(S(tip), u), fill=ln.color)
        if ln.word:
            d.text(S(ln.word[1]), ln.word[0], fill=ln.color, font=small)
    for b in layers.boxes:
        x1, y1 = S(b.box[:2])
        x2, y2 = S(b.box[2:])
        if b.dashed:
            for a, c in _dash_segments([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], dash=8, gap=6):
                d.line([a, c], fill=b.color, width=b.width)
        else:
            d.rectangle((x1, y1, x2, y2), outline=b.color, width=b.width)
        if b.label:
            d.text((x1, y1 - 24), b.label, fill=b.color, font=f)
    lx, ly = 10, 10
    for name, c, *_ in layers.legend:
        w = 50 + 11 * len(name)
        if lx + w > im.width:
            lx, ly = 10, ly + 28
        d.line([lx, ly + 10, lx + 30, ly + 10], fill=c, width=5)
        d.text((lx + 36, ly - 2), name, fill=c, font=f)
        lx += w
    if max_side:
        im.thumbnail((max_side, max_side))
    return im


def flow_map(diagram: ParsedDiagram, drawio_path: str | Path, boxes: list | None = None,
             overrides: dict | None = None, max_side: int | None = 3200) -> tuple[Image.Image, dict] | None:
    """(PNG image, counts) of the draw.io flow map, or None without the draw.io app."""
    layers = build_layers(diagram, drawio_path, boxes, overrides)
    if layers is None:
        return None
    return rasterize(layers, max_side), layers.stats


# --------------------------------------------------------------------- interactive ---
def _rgb(c, a=None):
    return f"rgb({c[0]},{c[1]},{c[2]})" if a is None else f"rgba({c[0]},{c[1]},{c[2]},{a})"


def _flatten(im: Image.Image) -> Image.Image:
    """Dark ink on white, as the model saw it (sheet_image.to_dark_on_white)."""
    from pid_dexpi_aas.image.sheet_image import to_dark_on_white
    return to_dark_on_white(im)


def _background_b64(path: Path) -> tuple[str, str]:
    with Image.open(path) as im:
        fixed = _flatten(im)
        if im.mode not in ("RGBA", "LA", "P") and fixed.tobytes() == im.convert("RGB").tobytes():
            mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            return mime, base64.b64encode(path.read_bytes()).decode()  # already dark on white
        buf = io.BytesIO()
        fixed.save(buf, "PNG")
    return "image/png", base64.b64encode(buf.getvalue()).decode()


def add_edits(layers: Layers, edits: list, replaced_rows: set) -> Layers:
    """Show the reviewer's line edits on the draw.io map: an added or re-pointed line is
    drawn between its new ends (teal, with flow arrows when known), and a drawn route
    whose line was re-pointed fades out - so it is clear what the DEXPI file will hold.
    edits: [(row id, from item id, to item id, flow sign +1/0, hover text)]."""
    boxes = {b.id: b for b in layers.boxes}
    centre = lambda b: ((b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2)
    for ln in layers.lines:
        if ln.rows and ln.rows <= replaced_rows:
            ln.color, ln.dashed, ln.kind, ln.arrows, ln.word, ln.dots = EXCLUDED, True, "excluded", [], None, []
            ln.tip = "original drawn line - you re-pointed it (see the teal line)"
    n = 0
    for rid, a, b, flow, tip in edits:
        if a not in boxes or b not in boxes:
            continue
        # routed like a draw.io connector: around the other items, edge to edge (routing.py)
        pts = route(boxes[a].box, boxes[b].box, [x.box for x in layers.boxes])
        arrows, _ = _arrows_on(pts, flow)
        layers.lines.append(Line(pts, FLOW_COLORS["reviewer"], 5, {rid}, tip, arrows=arrows, ends=(a, b), kind="edits"))
        n += 1
    if n:
        layers.legend.append((f"line you added or changed ({n})", FLOW_COLORS["reviewer"], "edits"))
    return layers


def hide(layers: Layers, symbols: set, rows: set) -> Layers:
    """The layers without the items and lines the reviewer hid. A line goes once every
    table row it stands for is hidden (a pipe network's piece stays while any pair of
    its ends is still shown). Only the map changes - the DEXPI file is not affected."""
    if not symbols and not rows:
        return layers
    return Layers(layers.background, layers.size,
                  [ln for ln in layers.lines if not (ln.rows and ln.rows <= rows)],
                  [b for b in layers.boxes if b.id not in symbols], layers.stats, layers.legend)


def map_markup(layers: Layers, highlight_rows: set | None = None, highlight_symbols: set | None = None,
               zoom_to_highlight: bool = True, key: str = "map", height: int = 720,
               names: str = "auto", refs: dict | None = None) -> dict:
    """The interactive map's content for MAP_JS (st.components.v2): {"html", "view", "full", "key"}.

    Every line and box carries its review-table rows (data-rows / data-sym) so a click can
    select them, and a kind (k-pipe, k-signal, k-model, ...) the toolbar can hide. Selected
    rows are drawn in magenta, the rest fades, and the view starts zoomed to them
    (zoom_to_highlight). Names are drawn with a white halo; `names` is the toolbar's starting
    choice: "auto" (once zoomed in far enough to read them), "on" or "off" - "off" suits a
    draw.io drawing, which prints its own names. Selected items are always named. Line
    widths stay constant on screen at any zoom. `refs` ({item id: "#12"}) tags every box
    (data-ref) so browsing an item dropdown can outline it (MAP_JS)."""
    hr, hs = set(highlight_rows or ()), set(highlight_symbols or ())
    lit_lines = [ln for ln in layers.lines if ln.rows & hr]
    # a selected connection lights its ends too; a selected SECTION lights its items in their own
    # colour - they belong to it, they are not one object with it
    members = {s for ln in lit_lines if ln.kind == "section" for s in ln.ends}
    hs |= {s for ln in lit_lines if ln.kind != "section" for s in ln.ends}
    members -= hs
    # only what this map draws: a selection made on another tab (a picture-only item while
    # the drawing renders) has nothing here to light up or zoom to
    hs &= {b.id for b in layers.boxes}
    members &= {b.id for b in layers.boxes}
    any_lit = bool(lit_lines or hs)
    W, H = layers.size
    mime, bg = _background_b64(layers.background)
    parts = [f'<image href="data:{mime};base64,{bg}" x="0" y="0" width="{W}" height="{H}"/>',
             f'<rect x="0" y="0" width="{W}" height="{H}" fill="white" opacity="0.55"/>']
    NS = ' vector-effect="non-scaling-stroke"'

    def label_svg(x, y, text, col, size=20, cls="lbl"):
        return (f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" fill="{col}" font-size="{size}" font-weight="600" '
                f'paint-order="stroke" stroke="white" stroke-width="5" stroke-linejoin="round">{html.escape(text)}</text>')

    def line_svg(ln: Line, color=None, width=None, lit=False):
        col = _rgb(color or ln.color)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in ln.points)
        dash = ' stroke-dasharray="7 5"' if ln.dashed else ""
        w = width or max(1.5, ln.width * 0.5)
        out = [f'<polyline class="vis" points="{pts}" fill="none" stroke="{col}" stroke-width="{w}" '
               f'stroke-linejoin="round"{dash}{NS}/>',
               # a wide transparent copy, so the line is easy to hover and click
               f'<polyline points="{pts}" fill="none" stroke="transparent" stroke-width="14"{NS}/>']
        for a, b in ln.guessed:
            out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                       f'stroke="{_rgb(GUESSED)}" stroke-width="2" stroke-dasharray="4 4"{NS}/>')
        for x, y in ln.dots:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="black"/>')
        for tip, u in ln.arrows:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in _arrow_polygon(tip, u, 20 if lit else 15))
            out.append(f'<polygon points="{poly}" fill="{col}"/>')
        if ln.word and not lit:
            x, y = ln.word[1]
            if ln.kind == "section":
                out.append(label_svg(x, y + 18, ln.word[0], col, 22, "sec"))
            else:
                out.append(label_svg(x, y + 14, ln.word[0], col, 15, "word"))
        rows = "|".join(sorted(ln.rows))
        tip = html.escape(ln.tip + ("  (click to select)" if ln.rows else ""))
        return f'<g class="item k-{ln.kind}" data-rows="{html.escape(rows)}"><title>{tip}</title>{"".join(out)}</g>'

    def box_svg(b: Box, color=None, width=None, lit=False):
        x1, y1, x2, y2 = b.box
        col = _rgb(color or b.color)
        dash = ' stroke-dasharray="6 4"' if b.dashed else ""
        lab = label_svg(x1, y1 - 7, b.label, col, 22 if lit else 19, "lbl lit" if lit else "lbl") if b.label else ""
        tip = html.escape(f"{b.tip or b.label or b.id}" + ("  (click to select)" if b.id else ""))
        ref = (refs or {}).get(b.id, "")
        return (f'<g class="item k-{b.kind}" data-sym="{html.escape(b.id)}" data-ref="{html.escape(ref)}"><title>{tip}</title>'
                f'<rect class="vis" x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" height="{y2 - y1:.1f}" '
                f'fill="rgba(255,255,255,0.01)" stroke="{col}" stroke-width="{width or 2}"{dash}{NS}/>{lab}</g>')

    dim = ' class="dim"' if any_lit else ""
    # boxes under lines: a line crossing a box stays clickable
    parts.append(f"<g{dim}>" + "".join(box_svg(b) for b in layers.boxes if b.id not in hs and b.id not in members) +
                 "".join(line_svg(ln) for ln in layers.lines if not (ln.rows & hr)) + "</g>")
    glow = _rgb(HIGHLIGHT, 0.3)
    for ln in lit_lines:  # selected on top: a wide glow, then the line in magenta
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in ln.points)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{glow}" stroke-width="14" '
                     f'stroke-linejoin="round"{NS}/>')
        parts.append(line_svg(ln, HIGHLIGHT, 4, lit=True))
    for b in layers.boxes:
        if b.id in members:  # a selected section's items: their own colour, each its own box
            x1, y1, x2, y2 = b.box
            parts.append(f'<rect x="{x1 - 4:.1f}" y="{y1 - 4:.1f}" width="{x2 - x1 + 8:.1f}" height="{y2 - y1 + 8:.1f}" '
                         f'fill="{_rgb(MEMBER, 0.12)}" stroke="{_rgb(MEMBER, 0.35)}" stroke-width="8"{NS}/>')
            parts.append(box_svg(b, MEMBER, 3))
    for b in layers.boxes:
        if b.id in hs:
            x1, y1, x2, y2 = b.box
            parts.append(f'<rect x="{x1 - 6:.1f}" y="{y1 - 6:.1f}" width="{x2 - x1 + 12:.1f}" height="{y2 - y1 + 12:.1f}" '
                         f'fill="{_rgb(HIGHLIGHT, 0.1)}" stroke="{glow}" stroke-width="10"{NS}/>')
            parts.append(box_svg(b, HIGHLIGHT, 3.5, lit=True))

    full = [0, 0, W, H]
    view = None  # None: keep the user's current zoom
    focus = None  # what +/- zoom towards: the selection
    if any_lit:
        lit_boxes = hs | members
        xs = [x for ln in lit_lines for x, _ in ln.points] + [v for b in layers.boxes if b.id in lit_boxes for v in (b.box[0], b.box[2])]
        ys = [y for ln in lit_lines for _, y in ln.points] + [v for b in layers.boxes if b.id in lit_boxes for v in (b.box[1], b.box[3])]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        focus = [cx, cy]
        if zoom_to_highlight:
            w, h = max(x2 - x1, 700), max(y2 - y1, 450)
            view = [cx - w * 0.65, cy - h * 0.65, w * 1.3, h * 1.3]

    chips = []
    for text, c, kind, *shape in layers.legend:
        if shape and shape[0] == "box":
            dash = "dashed" if c == BOX_OUT else "solid"
            sw = f'<i style="width:12px;height:10px;border:2px {dash} {_rgb(c)};background:transparent"></i>'
        else:
            sw = f'<i style="background:{_rgb(c)}"></i>'
        if kind:  # a layer the toolbar can hide
            chips.append(f'<button class="chip" data-kind="{kind}" title="Show / hide">{sw}{html.escape(text)}</button>')
        else:
            chips.append(f'<span class="chip static">{sw}{html.escape(text)}</span>')
    toolbar = (
        '<div class="bar">'
        '<button data-act="in" title="Zoom in (or scroll on the map)">+</button>'
        '<button data-act="out" title="Zoom out">&minus;</button>'
        '<button data-act="fit" title="Show the whole sheet">Fit</button>'
        '<button data-act="add" class="add" title="Draw a box around a symbol that was missed, to add it as an item">'
        '&#xFF0B; Add item</button>'
        '<span class="sep"></span>'
        '<label title="Names of items: shown when zoomed in far enough to read them">Names '
        '<select data-opt="labels"><option value="auto">when zoomed in</option><option value="on">always</option>'
        '<option value="off">never</option></select></label>'
        '<label title="The words \'arrow\' / \'implied\' / \'?\' next to each pipe">'
        '<input type="checkbox" data-opt="words"> Flow words</label>'
        '<span class="hint">Scroll to zoom · drag to move · click an item or line to select it · ＋ Add item, then drag a box around a missed item</span>'
        '</div>'
        f'<div class="chips">{"".join(chips)}</div>')
    markup = (f'{toolbar}<div class="frame" style="height:{height}px">'
              f'<svg class="map" viewBox="{" ".join(map(str, view or full))}" preserveAspectRatio="xMidYMid meet">'
              f'{"".join(parts)}</svg></div>')
    return {"html": markup, "view": view, "full": full, "key": key, "focus": focus, "names": names,
            "origin": list(layers.origin)}


MAP_CSS = """
/* Colours follow the Streamlit theme (light or dark); the drawing itself stays on white. */
.mapwrap{--fg:var(--st-text-color,#31333f);--bg2:var(--st-secondary-background-color,#f0f2f6);
  --line:var(--st-border-color,rgba(49,51,63,.2));--accent:var(--st-primary-color,#ff4b4b);
  font-family:var(--st-font,sans-serif);color:var(--fg)}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:6px;font-size:13px;padding:2px 0 6px}
.bar button{min-width:34px;height:30px;border:1px solid var(--line);border-radius:6px;background:var(--bg2);
  color:var(--fg);font:600 15px var(--st-font,sans-serif);cursor:pointer}
.bar button:hover{border-color:var(--accent);color:var(--accent)}
.bar label{display:flex;align-items:center;gap:4px;color:var(--fg)}
.bar select{height:28px;border:1px solid var(--line);border-radius:6px;background:var(--bg2);color:var(--fg);
  font:13px var(--st-font,sans-serif)}
.bar input[type=checkbox]{accent-color:var(--accent)}
.bar .sep{width:1px;height:22px;background:var(--line);margin:0 4px}
.bar button.add{padding:0 10px;font-size:13px}
.bar button.add.on{background:var(--accent);color:#fff;border-color:var(--accent)}
svg.map.adding{cursor:crosshair}
.bar .hint{opacity:.6;margin-left:auto;font-size:12px}
.chips{display:flex;flex-wrap:wrap;gap:4px 6px;padding:0 0 6px;font-size:12px}
.chip{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--line);border-radius:12px;padding:2px 8px;
  background:var(--bg2);color:var(--fg);font:12px var(--st-font,sans-serif)}
.chip i{display:inline-block;width:16px;height:4px;border-radius:2px;box-shadow:0 0 0 1px var(--line)}
button.chip{cursor:pointer}
button.chip:hover{border-color:var(--accent)}
button.chip.off{opacity:.4;text-decoration:line-through}
.frame{border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#fff}
svg.map{width:100%;height:100%;display:block;cursor:grab;user-select:none}
svg.map.dragging{cursor:grabbing}
.item{cursor:pointer}
.item:hover .vis{filter:drop-shadow(0 0 4px #e600c8)}
.dim{opacity:.45}
svg.map.nolabels .lbl:not(.lit){display:none}
svg.map.nowords .word{display:none}
"""

# Toolbar (zoom, fit, names, flow words, layer chips), wheel zoom, drag to pan, and
# click-to-select (trigger "clicked") - a click that dragged does not select. View and
# toolbar settings survive re-renders per map (key); Python sends a view only to zoom to a
# selection made in the tables.
MAP_JS = """
const memory = {};
export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  const mem = memory[data.key] || (memory[data.key] = { view: null, hidden: [], labels: data.names || 'auto', words: false });
  let root = parentElement.querySelector('.mapwrap');
  if (!root) { root = document.createElement('div'); root.className = 'mapwrap'; parentElement.appendChild(root); }
  root.innerHTML = data.html;
  const s = root.querySelector('svg.map');
  let vb = (data.view || mem.view || data.full).slice();
  const applyLod = () => {
    const pxPerUnit = s.clientWidth / vb[2];
    const show = mem.labels === 'on' || (mem.labels === 'auto' && pxPerUnit > 0.4);
    s.classList.toggle('nolabels', !show);
    s.classList.toggle('nowords', !mem.words);
  };
  const set = () => { s.setAttribute('viewBox', vb.join(' ')); mem.view = vb.slice(); applyLod(); };
  const zoomAt = (k, px, py) => { vb = [px - (px - vb[0]) * k, py - (py - vb[1]) * k, vb[2] * k, vb[3] * k]; set(); };
  const centre = () => data.focus || [vb[0] + vb[2] / 2, vb[1] + vb[3] / 2];
  // layers
  const applyHidden = () => {
    root.querySelectorAll('button.chip').forEach((b) => b.classList.toggle('off', mem.hidden.includes(b.dataset.kind)));
    s.querySelectorAll('.item').forEach((g) => {
      const k = [...g.classList].find((c) => c.startsWith('k-'));
      g.style.display = k && mem.hidden.includes(k.slice(2)) ? 'none' : '';
    });
  };
  root.querySelectorAll('button.chip').forEach((b) => b.addEventListener('click', () => {
    const k = b.dataset.kind;
    mem.hidden = mem.hidden.includes(k) ? mem.hidden.filter((x) => x !== k) : mem.hidden.concat([k]);
    applyHidden();
  }));
  // +/-: zoom towards the selection (centred on it), else around the middle of the view
  const zoomBtn = (k) => { const [cx, cy] = centre(); vb = [cx - vb[2] * k / 2, cy - vb[3] * k / 2, vb[2] * k, vb[3] * k]; set(); };
  root.querySelector('[data-act=in]').onclick = () => zoomBtn(1 / 1.4);
  root.querySelector('[data-act=out]').onclick = () => zoomBtn(1.4);
  root.querySelector('[data-act=fit]').onclick = () => { vb = data.full.slice(); set(); };
  const sel = root.querySelector('[data-opt=labels]'); sel.value = mem.labels;
  sel.onchange = () => { mem.labels = sel.value; applyLod(); };
  const words = root.querySelector('[data-opt=words]'); words.checked = mem.words;
  words.onchange = () => { mem.words = words.checked; applyLod(); };
  // wheel zoom, drag pan, click select
  s.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = s.getBoundingClientRect();
    // the viewBox is letterboxed (meet): map the cursor through the scale actually used
    const sc = Math.max(vb[2] / r.width, vb[3] / r.height);
    const ox = vb[0] + vb[2] / 2 - (r.width / 2) * sc, oy = vb[1] + vb[3] / 2 - (r.height / 2) * sc;
    zoomAt(e.deltaY > 0 ? 1.15 : 1 / 1.15, ox + (e.clientX - r.left) * sc, oy + (e.clientY - r.top) * sc);
  }, { passive: false });
  // "Add item": drag a box; sent to Python as the trigger "drawn" in map coordinates
  let adding = false, drawStart = null, rubber = null;
  const addBtn = root.querySelector('[data-act=add]');
  const toMap = (e) => { const pt = s.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const m = pt.matrixTransform(s.getScreenCTM().inverse()); return [m.x, m.y]; };
  addBtn.onclick = () => { adding = !adding; addBtn.classList.toggle('on', adding); s.classList.toggle('adding', adding); };
  s.addEventListener('mousedown', (e) => {
    if (!adding) return;
    e.stopImmediatePropagation(); drawStart = toMap(e);
    rubber = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rubber.setAttribute('fill', 'rgba(255,75,75,0.08)'); rubber.setAttribute('stroke', '#ff4b4b');
    rubber.setAttribute('stroke-width', '2'); rubber.setAttribute('stroke-dasharray', '6 4');
    rubber.setAttribute('vector-effect', 'non-scaling-stroke'); s.appendChild(rubber);
  }, true);
  s.addEventListener('mousemove', (e) => {
    if (!adding || !drawStart) return;
    e.stopImmediatePropagation(); const [x, y] = toMap(e);
    rubber.setAttribute('x', Math.min(x, drawStart[0])); rubber.setAttribute('y', Math.min(y, drawStart[1]));
    rubber.setAttribute('width', Math.abs(x - drawStart[0])); rubber.setAttribute('height', Math.abs(y - drawStart[1]));
  }, true);
  s.addEventListener('mouseup', (e) => {
    if (!adding || !drawStart) return;
    e.stopImmediatePropagation(); const [x, y] = toMap(e), [x0, y0] = drawStart; drawStart = null;
    adding = false; addBtn.classList.remove('on'); s.classList.remove('adding');
    if (Math.abs(x - x0) < 4 || Math.abs(y - y0) < 4) { rubber.remove(); return; }
    setTriggerValue('drawn', JSON.stringify({ box: [Math.min(x, x0), Math.min(y, y0), Math.max(x, x0), Math.max(y, y0)],
                                               t: Date.now() }));
  }, true);
  let drag = null, moved = false;
  s.addEventListener('mousedown', (e) => { drag = [e.clientX, e.clientY, vb[0], vb[1]]; moved = false; });
  s.addEventListener('mousemove', (e) => {
    if (!drag) return;
    if (Math.abs(e.clientX - drag[0]) + Math.abs(e.clientY - drag[1]) > 4) { moved = true; s.classList.add('dragging'); }
    const r = s.getBoundingClientRect(), sc = Math.max(vb[2] / r.width, vb[3] / r.height);
    vb[0] = drag[2] - (e.clientX - drag[0]) * sc;
    vb[1] = drag[3] - (e.clientY - drag[1]) * sc; set();
  });
  s.addEventListener('mouseup', (e) => {
    const wasDrag = moved; drag = null; s.classList.remove('dragging');
    if (wasDrag) return;
    const item = e.target.closest('.item');
    if (!item) { setTriggerValue('clicked', JSON.stringify({ clear: true, t: Date.now() })); return; }
    if (!item.dataset.rows && !item.dataset.sym) return;  // an OCR tag: no table row to select
    setTriggerValue('clicked', JSON.stringify({
      rows: (item.dataset.rows || '').split('|').filter(Boolean),
      sym: item.dataset.sym || null, t: Date.now() }));
  });
  s.addEventListener('mouseleave', () => { drag = null; s.classList.remove('dragging'); });
  s.addEventListener('dblclick', () => { vb = data.full.slice(); set(); });
  set(); applyHidden();
  // Browsing an item dropdown (Connect to, Merge with, From / To, Bypass ...): the item under
  // the mouse or the arrow keys is outlined in orange - entries start with the item's "#12".
  // The view does not move (it may be off screen) and the selection stays lit.
  if (mem.peekOff) mem.peekOff();
  const SVGNS = 'http://www.w3.org/2000/svg';
  let peekRef = null;
  const showPeek = (ref) => {
    if (ref === peekRef) return;
    peekRef = ref;
    s.querySelectorAll('.peekmark').forEach((n) => n.remove());
    if (!ref) return;
    s.querySelectorAll(`g.item[data-ref="${ref}"] rect.vis`).forEach((r) => {
      const x = +r.getAttribute('x'), y = +r.getAttribute('y'), w = +r.getAttribute('width'), h = +r.getAttribute('height');
      for (const [pad, sw, fill, stroke] of [[12, 14, 'rgba(255,140,0,0.14)', 'rgba(255,140,0,0.4)'], [5, 4, 'none', '#ff8c00']]) {
        const m = document.createElementNS(SVGNS, 'rect');
        m.setAttribute('class', 'peekmark');
        m.setAttribute('x', x - pad); m.setAttribute('y', y - pad);
        m.setAttribute('width', w + 2 * pad); m.setAttribute('height', h + 2 * pad);
        m.setAttribute('fill', fill); m.setAttribute('stroke', stroke); m.setAttribute('stroke-width', sw);
        m.setAttribute('vector-effect', 'non-scaling-stroke');
        m.style.pointerEvents = 'none';
        s.appendChild(m);
      }
    });
  };
  const refOf = (o) => { const m = o && o.textContent.trim().match(/^(#\d+)(\s|$)/); return m ? m[1] : null; };
  const current = () => refOf(document.querySelector('[role=listbox] [role=option][data-hovered=true]')
    || document.querySelector('[role=listbox] [role=option][data-focused=true]'));
  const watch = new MutationObserver(() => showPeek(current()));
  watch.observe(document.body, { subtree: true, childList: true, attributes: true,
    attributeFilter: ['data-focused', 'data-hovered'] });
  // Hovering a line button in the side panel (key "peekline__<row id>__..."): that pipe is
  // outlined the same way. Streamlit's key class turns other characters into "-".
  let peekLine = null;
  const showLinePeek = (id) => {
    if (id === peekLine) return;
    peekLine = id;
    s.querySelectorAll('.linepeek').forEach((n) => n.remove());
    if (!id) return;
    const san = (r) => r.replace(/[^a-zA-Z0-9_-]/g, '-');
    s.querySelectorAll('g.item[data-rows]').forEach((g) => {
      if (!g.dataset.rows.split('|').some((r) => san(r) === id)) return;
      const pl = g.querySelector('polyline.vis');
      if (!pl) return;
      for (const [sw, stroke] of [[16, 'rgba(255,140,0,0.4)'], [5, '#ff8c00']]) {
        const m = document.createElementNS(SVGNS, 'polyline');
        m.setAttribute('class', 'linepeek');
        m.setAttribute('points', pl.getAttribute('points'));
        m.setAttribute('fill', 'none'); m.setAttribute('stroke', stroke); m.setAttribute('stroke-width', sw);
        m.setAttribute('stroke-linejoin', 'round'); m.setAttribute('stroke-linecap', 'round');
        m.setAttribute('vector-effect', 'non-scaling-stroke');
        m.style.pointerEvents = 'none';
        s.appendChild(m);
      }
    });
  };
  const lineOf = (t) => {
    const b = t.closest && t.closest('[class*="st-key-peekline__"]');
    const m = b && b.className.match(/st-key-peekline__(.+?)__/);
    return m ? m[1] : null;
  };
  const over = (e) => {
    const o = e.target.closest && e.target.closest('[role=option]');
    if (o) showPeek(refOf(o));
    showLinePeek(lineOf(e.target));
  };
  document.addEventListener('mouseover', over);
  mem.peekOff = () => { watch.disconnect(); document.removeEventListener('mouseover', over); showPeek(null); showLinePeek(null); };
}
"""


def review_overrides(review) -> dict:
    """The reviewer's connection edits, in build_layers' `overrides` form."""
    from pid_dexpi_aas.review import FLOW_KNOWN

    out = {}
    edges = {e.id: e for e in review.drawio.edges} if review.drawio else {}
    for c in review.connections:
        e = edges.get(c.id)
        if e is None:
            continue
        sign = (1 if c.from_id == e.source else -1) if c.flow in FLOW_KNOWN else 0
        basis = {"drawn arrow": "arrow", "implied": "implied", "reviewer": "reviewer"}.get(c.flow, "")
        out[c.id] = (c.include, sign, basis)
    return out
