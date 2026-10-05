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
import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pid_dexpi_aas.parser_drawio import ParsedDiagram

FLOW_COLORS = {"arrow": (215, 80, 0), "implied": (150, 50, 200), "reviewer": (0, 150, 120), "": (130, 130, 130)}
FLOW_WORD = {"arrow": "arrow", "implied": "implied", "reviewer": "reviewer", "": "?"}
FLOW_TEXT = {"arrow": "flow from an arrow drawn on the line", "implied": "flow implied by neighbouring arrows",
             "reviewer": "flow direction set by the reviewer", "": "flow direction unknown"}
SIGNAL = (0, 140, 255)
TAP = (110, 160, 190)
GUESSED = (220, 0, 0)
COMPARTMENT = (40, 160, 60)
EXCLUDED = (200, 200, 200)
MODEL_LINE = (255, 140, 0)
HIGHLIGHT = (230, 0, 200)
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


@dataclass
class Box:
    id: str
    box: tuple  # sheet pixels
    color: tuple
    label: str
    width: int = 3
    tip: str = ""
    dashed: bool = False  # not from the drawing: a model-only detection or an OCR-read tag


@dataclass
class Layers:
    background: Path
    size: tuple
    lines: list
    boxes: list
    stats: dict
    legend: list  # (text, colour)


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
    from pid_dexpi_aas.line_routes import routes_for, sheet_png
    from pid_dexpi_aas.mapping import Category, map_shape

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
                                  ends=(e.source, e.target)))
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
            lines.append(Line(pts, EXCLUDED, 3, rows, f"{what}: excluded by the reviewer", dashed=True, ends=line_ends))
            stats["excluded"] += 1
            continue
        if e.is_dashed:
            lines.append(Line(pts, SIGNAL, 3, rows, f"{what}: signal line", dashed=True, ends=line_ends))
            stats["signal"] += 1
            continue
        if not flow and (e.source in instruments or e.target in instruments):
            lines.append(Line(pts, TAP, 3, rows, f"{what}: instrument on the pipe (no process flow)", ends=line_ends))
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
                    arrows=arrows, ends=line_ends)
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

    if boxes is None:
        cat_color = {Category.EQUIPMENT: (240, 50, 230), Category.PIPING: (0, 128, 128),
                     Category.INSTRUMENTATION: (67, 99, 216)}
        boxes = [(sid, (s.x, s.y, s.x + s.width, s.y + s.height), cat_color[category[sid]], s.text, 4)
                 for sid, s in shapes.items() if category.get(sid) in cat_color]
    out_boxes = []
    for b in boxes:
        if isinstance(b, Box):
            out_boxes.append(Box(b.id, (b.box[0] + dx, b.box[1] + dy, b.box[2] + dx, b.box[3] + dy),
                                 b.color, b.label, b.width, b.tip, b.dashed))
        else:
            sid, (x1, y1, x2, y2), color, label, width = b
            out_boxes.append(Box(sid, (x1 + dx, y1 + dy, x2 + dx, y2 + dy), color, label, width, label))
    legend = [(f"flow from an arrow drawn on the line ({stats['arrow']})", FLOW_COLORS["arrow"]),
              (f"flow implied by neighbouring arrows ({stats['implied']})", FLOW_COLORS["implied"]),
              (f"flow direction unknown ({stats['']})", FLOW_COLORS[""]),
              (f"signal line ({stats['signal']})", SIGNAL),
              (f"instrument tap, no process flow ({stats['tap']})", TAP), ("tee", (0, 0, 0)),
              ("guessed line end", GUESSED), ("compartments sharing a wall", COMPARTMENT)]
    if stats["reviewer"]:
        legend.insert(2, (f"direction set by the reviewer ({stats['reviewer']})", FLOW_COLORS["reviewer"]))
    if stats["excluded"]:
        legend.append((f"excluded by the reviewer ({stats['excluded']})", EXCLUDED))
    return Layers(Path(png), size, lines, out_boxes, stats, legend)


def image_layers(image_path: str | Path, connections: list, boxes: list[Box]) -> Layers:
    """Image-only mode: the model's connections as straight lines between box centres.
    connections: [(row id, from id, to id, line type, include, hover text)]."""
    with Image.open(image_path) as im:
        size = im.size
    centre = {b.id: ((b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2) for b in boxes}
    lines, n = [], {"process": 0, "signal": 0, "excluded": 0}
    for rid, a, b, kind, include, tip in connections:
        if a not in centre or b not in centre:
            continue
        key = "excluded" if not include else kind
        n[key] = n.get(key, 0) + 1
        color = EXCLUDED if not include else SIGNAL if kind == "signal" else MODEL_LINE
        lines.append(Line([centre[a], centre[b]], color, 3, {rid}, tip, dashed=kind == "signal" or not include,
                          ends=(a, b)))
    legend = [(f"predicted pipe, centre to centre ({n['process']})", MODEL_LINE),
              (f"predicted signal line ({n['signal']})", SIGNAL)]
    if n["excluded"]:
        legend.append((f"excluded by the reviewer ({n['excluded']})", EXCLUDED))
    return Layers(Path(image_path), size, lines, boxes, n, legend)


# ------------------------------------------------------------------------ raster ---
def rasterize(layers: Layers, max_side: int | None = 3200) -> Image.Image:
    im = Image.open(layers.background).convert("RGB")
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
    for name, c in layers.legend:
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


def map_markup(layers: Layers, highlight_rows: set | None = None, highlight_symbols: set | None = None,
               zoom_to_highlight: bool = True) -> dict:
    """The interactive map's content for MAP_JS (st.components.v2): {"html", "view", "full"}.
    Every line and box carries its review-table rows (data-rows / data-sym), so a click can
    select them; highlighted rows are drawn in magenta, everything else fades, and the view
    starts zoomed to them (zoom_to_highlight)."""
    hr, hs = set(highlight_rows or ()), set(highlight_symbols or ())
    lit_lines = [ln for ln in layers.lines if ln.rows & hr]
    hs |= {s for ln in lit_lines for s in ln.ends}  # a highlighted connection lights its ends too
    any_lit = bool(lit_lines or hs)
    W, H = layers.size
    mime = "image/jpeg" if layers.background.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    bg = base64.b64encode(layers.background.read_bytes()).decode()
    parts = [f'<image href="data:{mime};base64,{bg}" x="0" y="0" width="{W}" height="{H}"/>',
             f'<rect class="bg" x="0" y="0" width="{W}" height="{H}" fill="white" opacity="0.6"/>']

    def line_svg(ln: Line, color=None, width=None):
        col = _rgb(color or ln.color)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in ln.points)
        dash = ' stroke-dasharray="14 10"' if ln.dashed else ""
        out = [f'<polyline class="vis" points="{pts}" fill="none" stroke="{col}" stroke-width="{width or ln.width}" '
               f'stroke-linejoin="round"{dash}/>',
               # a wide transparent copy, so the line is easy to hover and click
               f'<polyline points="{pts}" fill="none" stroke="transparent" stroke-width="18"/>']
        for a, b in ln.guessed:
            out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                       f'stroke="{_rgb(GUESSED)}" stroke-width="3" stroke-dasharray="6 6"/>')
        for x, y in ln.dots:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="black"/>')
        for tip, u in ln.arrows:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in _arrow_polygon(tip, u, 22 if width else 16))
            out.append(f'<polygon points="{poly}" fill="{col}"/>')
        if ln.word and not width:
            x, y = ln.word[1]
            out.append(f'<text x="{x:.1f}" y="{y + 14:.1f}" fill="{col}" font-size="17" font-weight="bold">'
                       f'{html.escape(ln.word[0])}</text>')
        rows = "|".join(sorted(ln.rows))
        tip = html.escape(ln.tip + (f"  [click: select row {', '.join(sorted(ln.rows))}]" if ln.rows else ""))
        return f'<g class="item" data-rows="{html.escape(rows)}"><title>{tip}</title>{"".join(out)}</g>'

    def box_svg(b: Box, color=None, width=None):
        x1, y1, x2, y2 = b.box
        col = _rgb(color or b.color)
        lab = (f'<text x="{x1:.1f}" y="{y1 - 6:.1f}" fill="{col}" font-size="20" font-weight="bold">'
               f'{html.escape(b.label)}</text>') if b.label else ""
        tip = html.escape(f"{b.tip or b.label or b.id}" + (f"  [click: select row {b.id}]" if b.id else ""))
        dash = ' stroke-dasharray="8 6"' if b.dashed else ""
        return (f'<g class="item" data-sym="{html.escape(b.id)}"><title>{tip}</title>'
                f'<rect class="vis" x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" height="{y2 - y1:.1f}" '
                f'fill="rgba(255,255,255,0.01)" stroke="{col}" stroke-width="{width or b.width}"{dash}/>{lab}</g>')

    dim = ' opacity="0.3"' if any_lit else ""
    # boxes under lines: a line crossing a box stays clickable
    parts.append(f"<g{dim}>" + "".join(box_svg(b) for b in layers.boxes if b.id not in hs) +
                 "".join(line_svg(ln) for ln in layers.lines if not (ln.rows & hr)) + "</g>")
    glow = _rgb(HIGHLIGHT, 0.35)
    for b in layers.boxes:
        if b.id in hs:
            x1, y1, x2, y2 = b.box
            parts.append(f'<rect x="{x1 - 8:.1f}" y="{y1 - 8:.1f}" width="{x2 - x1 + 16:.1f}" height="{y2 - y1 + 16:.1f}" '
                         f'fill="{_rgb(HIGHLIGHT, 0.12)}" stroke="{glow}" stroke-width="12"/>')
            parts.append(box_svg(b, HIGHLIGHT, 6))
    for ln in lit_lines:  # highlighted on top: a wide glow, then the line in magenta
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in ln.points)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{glow}" stroke-width="22" stroke-linejoin="round"/>')
        parts.append(line_svg(ln, HIGHLIGHT, 7))

    full = [0, 0, W, H]
    view = None  # None: keep the user's current zoom
    if any_lit and zoom_to_highlight:
        xs = [x for ln in lit_lines for x, _ in ln.points] + [v for b in layers.boxes if b.id in hs for v in (b.box[0], b.box[2])]
        ys = [y for ln in lit_lines for _, y in ln.points] + [v for b in layers.boxes if b.id in hs for v in (b.box[1], b.box[3])]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        w, h = max(x2 - x1, 700), max(y2 - y1, 450)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        view = [cx - w * 0.65, cy - h * 0.65, w * 1.3, h * 1.3]
    legend = "".join(f'<span class="lg" style="color:{_rgb(c)}"><b style="background:{_rgb(c)}"></b> '
                     f'{html.escape(t)}</span>' for t, c in layers.legend)
    note = ("Magenta = selected rows (the 'show' column). " if any_lit else "") + \
        "Click a line or box to select its table row · click empty space to clear · hover for what it is · " \
        "scroll to zoom · drag to pan · double-click: whole sheet"
    markup = (f'<div class="legend">{legend}</div><div class="note">{note}</div>'
              f'<svg class="map" viewBox="{" ".join(map(str, view or full))}" preserveAspectRatio="xMidYMid meet">'
              f'{"".join(parts)}</svg>')
    return {"html": markup, "view": view, "full": full}


MAP_CSS = """
.legend{font:12px sans-serif;line-height:20px;padding:4px 6px;color:#333}
.lg{white-space:nowrap;margin-right:14px}
.lg b{display:inline-block;width:22px;height:4px;vertical-align:middle}
.note{font:12px sans-serif;color:#666;padding:0 6px 4px}
svg.map{width:100%;height:640px;border:1px solid #ddd;cursor:grab;background:#fff;display:block}
.item{cursor:pointer}
.item:hover .vis{filter:drop-shadow(0 0 5px #e600c8)}
"""

# Zoom (wheel), pan (drag), whole sheet (double-click); a click that did not drag selects
# the clicked line's or box's rows (trigger "clicked"), or clears on empty space. The
# user's zoom survives re-renders unless Python sends a view to zoom to.
MAP_JS = """
let savedView = null;
export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  let root = parentElement.querySelector('.mapwrap');
  if (!root) { root = document.createElement('div'); root.className = 'mapwrap'; parentElement.appendChild(root); }
  root.innerHTML = data.html;
  const s = root.querySelector('svg.map');
  let vb = (data.view || savedView || data.full).slice();
  const set = () => { s.setAttribute('viewBox', vb.join(' ')); savedView = vb.slice(); };
  set();
  s.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = s.getBoundingClientRect(), k = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    const px = vb[0] + (e.clientX - r.left) / r.width * vb[2], py = vb[1] + (e.clientY - r.top) / r.height * vb[3];
    vb = [px - (px - vb[0]) * k, py - (py - vb[1]) * k, vb[2] * k, vb[3] * k]; set();
  }, { passive: false });
  let drag = null, moved = false;
  s.addEventListener('mousedown', (e) => { drag = [e.clientX, e.clientY, vb[0], vb[1]]; moved = false; });
  s.addEventListener('mousemove', (e) => {
    if (!drag) return;
    if (Math.abs(e.clientX - drag[0]) + Math.abs(e.clientY - drag[1]) > 4) moved = true;
    const r = s.getBoundingClientRect();
    vb[0] = drag[2] - (e.clientX - drag[0]) / r.width * vb[2];
    vb[1] = drag[3] - (e.clientY - drag[1]) / r.height * vb[3]; set();
  });
  s.addEventListener('mouseup', (e) => {
    const wasDrag = moved; drag = null;
    if (wasDrag) return;
    const item = e.target.closest('.item');
    if (!item) { setTriggerValue('clicked', JSON.stringify({ clear: true, t: Date.now() })); return; }
    if (!item.dataset.rows && !item.dataset.sym) return;  // an OCR tag: no table row to select
    setTriggerValue('clicked', JSON.stringify({
      rows: (item.dataset.rows || '').split('|').filter(Boolean),
      sym: item.dataset.sym || null, t: Date.now() }));
  });
  s.addEventListener('mouseleave', () => { drag = null; });
  s.addEventListener('dblclick', () => { vb = data.full.slice(); set(); });
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
