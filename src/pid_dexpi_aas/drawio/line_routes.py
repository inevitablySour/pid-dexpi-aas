"""Follow draw.io lines along their real drawn route, so lines that END ON ANOTHER LINE
(T-junctions, branches) connect the symbols at both ends.

The .drawio file stores only a line's attached shapes and any user-placed bend points;
draw.io routes most lines automatically, so the real path exists only once draw.io
renders it. Its SVG export tags every drawn line with its cell id (data-cell-id) and
gives the routed path, so this module exports the SVG with the draw.io desktop app and
reads each line's route back (shifted from image to draw.io coordinates).

Why it matters: in the team's drawings many pipes end on another pipe instead of on a
symbol (LightUp: the three feeds into V101, P101 <-> V101, the PC/PI/FI 102 branch...).
Without routes such an end is either left unconnected or guessed onto the nearest
symbol - wrong both ways.

Without the draw.io app (another machine, CI) routes are unavailable and the parser
falls back to its previous behaviour.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET



def _find_drawio() -> Path:
    """The draw.io desktop app: $DRAWIO_APP, the repo's ml/tools/drawio.AppImage
    (scripts/get_drawio.sh), or a normal install on Linux, macOS or Windows."""
    candidates = [os.environ.get("DRAWIO_APP"),
                  Path(__file__).resolve().parents[3] / "ml" / "tools" / "drawio.AppImage",
                  shutil.which("drawio"), shutil.which("draw.io"),
                  "/Applications/draw.io.app/Contents/MacOS/draw.io",
                  Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "draw.io" / "draw.io.exe",
                  Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "draw.io" / "draw.io.exe"]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path(candidates[1])


DRAWIO_APP = _find_drawio()
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pid_dexpi_aas" / "routes"
_SVG = "{http://www.w3.org/2000/svg}"
_NUM = re.compile(r"(-?[\d.]+)[ ,](-?[\d.]+)")

Route = list[tuple[float, float]]


def available() -> bool:
    return DRAWIO_APP.exists()


def _export_svg(drawio_path: Path) -> str | None:
    key = hashlib.sha256(drawio_path.read_bytes()).hexdigest()[:20]
    cached = CACHE / f"{key}.svg"
    if cached.exists():
        return cached.read_text()
    if not available():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.svg"
        r = subprocess.run([str(DRAWIO_APP), "--no-sandbox", "-x", "-f", "svg", "-b", "0", "-s", "1",
                            "-o", str(out), str(drawio_path)], capture_output=True)
        if r.returncode != 0 or not out.exists():
            return None
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(out.read_text())
    return cached.read_text()


def _raw_routes(text: str) -> dict[str, Route]:
    """{cell id: polyline in image pixels} from a draw.io SVG export."""
    root = ET.fromstring(text)
    raw: dict[str, Route] = {}
    for g in root.iter(f"{_SVG}g"):
        cid = g.get("data-cell-id")
        if not cid or cid in raw:
            continue
        p = g.find(f".//{_SVG}path")
        if p is None or not p.get("d"):
            continue
        # M/L give the corners; a Q (rounded corner) contributes its end point.
        d = p.get("d")
        pts = []
        for cmd, body in re.findall(r"([MLQ])([^MLQZ]*)", d):
            nums = [(float(a), float(b)) for a, b in _NUM.findall(body)]
            if nums:
                pts.append(nums[-1])
        if len(pts) >= 2:
            raw[cid] = pts
    return raw


def _shift(raw: dict[str, Route], edges, shapes=()) -> tuple[int, int] | None:
    """Image pixel = draw.io coordinate + shift, calibrated from the loose line ends the
    file stores in both systems (LightUp: 175 of 219 agree exactly). A drawing with every
    line attached at both ends (one made in the Editor tab) has none: then from the shapes -
    a symbol's outline starts at the left / top of its box, so the most common gap between
    its drawn outline and its box (else the smallest) is the shift, per axis."""
    shifts = Counter()
    for e in edges:
        r = raw.get(e.id)
        if not r:
            continue
        # an end with an arrowhead is drawn short (the line stops where the head starts):
        # it would put the shift a head's length off
        for end, pt, arrow in ((r[0], e.source_point, e.has_start_arrow), (r[-1], e.target_point, e.has_end_arrow)):
            if pt and not arrow:
                shifts[(round(end[0] - pt[0]), round(end[1] - pt[1]))] += 1
    if shifts:
        return shifts.most_common(1)[0][0]
    xs, ys = Counter(), Counter()
    for sh in shapes:
        r = raw.get(sh.id)
        if r and sh.width and sh.height:
            xs[round(min(x for x, _ in r) - sh.x)] += 1
            ys[round(min(y for _, y in r) - sh.y)] += 1
    if not xs:
        return None
    pick = lambda c: (lambda v, n: v if n >= 2 else min(c))(*c.most_common(1)[0])
    return pick(xs), pick(ys)


def routes_for(drawio_path: str | Path, edges, shapes=()) -> dict[str, Route] | None:
    """{edge id: polyline in draw.io coordinates} for every line draw.io drew, or None."""
    text = _export_svg(Path(drawio_path))
    if text is None:
        return None
    raw = _raw_routes(text)
    shift = _shift(raw, edges, shapes)
    if shift is None:
        return None
    dx, dy = shift
    ids = {e.id for e in edges}
    return {cid: [(x - dx, y - dy) for x, y in pts] for cid, pts in raw.items() if cid in ids}


def sheet_png(drawio_path: str | Path, edges, shapes=()) -> tuple[Path, tuple[int, int]] | None:
    """The drawing rendered by draw.io (PNG, cached) and the shift from draw.io coordinates
    to its pixels (same export origin as the SVG the routes come from)."""
    drawio_path = Path(drawio_path)
    text = _export_svg(drawio_path)
    shift = _shift(_raw_routes(text), edges, shapes) if text else None
    if shift is None:
        return None
    key = hashlib.sha256(drawio_path.read_bytes()).hexdigest()[:20]
    png = CACHE / f"{key}.png"
    if not png.exists():
        r = subprocess.run([str(DRAWIO_APP), "--no-sandbox", "-x", "-f", "png", "-b", "0", "-s", "1",
                            "-o", str(png), str(drawio_path)], capture_output=True)
        if r.returncode != 0 or not png.exists():
            return None
    return png, shift


def _dist_to_route(p: tuple[float, float], route: Route) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in zip(route, route[1:]):
        vx, vy = x2 - x1, y2 - y1
        L = vx * vx + vy * vy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - x1) * vx + (p[1] - y1) * vy) / L))
        best = min(best, math.hypot(p[0] - (x1 + t * vx), p[1] - (y1 + t * vy)))
    return best


def _out_dir(route: Route, end: int) -> tuple[float, float]:
    """Unit vector pointing OUT of the route at one end."""
    a, b = (route[0], route[1]) if end == 0 else (route[-1], route[-2])
    vx, vy = a[0] - b[0], a[1] - b[1]
    n = math.hypot(vx, vy) or 1.0
    return vx / n, vy / n


def find_junctions(edges, routes: dict[str, Route], tol: float = 8.0, gap: float = 30.0) -> dict[tuple[str, int], str]:
    """{(edge id, 0=source end | 1=target end): id of the line it joins} for unattached
    line ends. Two cases, checked in this order:

      1. Two loose ends meeting each other: one pipe drawn in pieces -> joined to each
         other. If they meet head-on in a straight line, any line passing through that
         point is a CROSSING, not a junction (LightUp draws the GV drops into V101 split
         exactly where they cross the XV101-XV102 line).
      2. A single loose end lying on another line's route: a T-junction / branch.

    tol: 8 draw.io units (the grid is 10) - LightUp ends meet 5.7 and 6.4 units apart.
    Lines that merely cross have no end on each other and are never joined.
    """
    live = [e for e in edges if not e.is_decorative and e.id in routes]
    loose = []
    for e in live:
        r = routes[e.id]
        for end, attached, stored in ((0, e.source, e.source_point), (1, e.target, e.target_point)):
            if attached is None:
                # the end point the file stores is exact; the drawn line stops short of it
                # where it carries an arrowhead
                loose.append((e.id, end, stored or (r[0] if end == 0 else r[-1]), _out_dir(r, end)))
    joins: dict[tuple[str, int], str] = {}
    straight_through = set()
    # an end attached to a line in the file (parser._ends_on_lines): on that line, no guessing
    on_line = {(e.id, end): (e.data or {}).get(f"_on_line_{end}") for e in live for end in (0, 1)}
    for (eid, end), line in on_line.items():
        if line and line in routes:
            joins[(eid, end)] = line
    loose = [lo for lo in loose if (lo[0], lo[1]) not in joins]
    # 0. A pipe drawn with a GAP where it crosses another line (a "line jump"): two loose
    #    ends facing each other on one straight line, up to `gap` apart. LightUp draws the
    #    GV101/GV102 drops into V101 this way across the XV101-XV102 line; without this the
    #    piece ends were read as T-junctions on the crossed line.
    for i, (eid, end, p, d) in enumerate(loose):
        best = None
        for j, (fid, fend, q, fd) in enumerate(loose):
            if j == i or fid == eid:
                continue
            vx, vy = q[0] - p[0], q[1] - p[1]
            dist = math.hypot(vx, vy)
            if not (tol < dist <= gap):
                continue
            along = (vx * d[0] + vy * d[1]) / dist      # partner lies straight ahead
            facing = d[0] * fd[0] + d[1] * fd[1]          # partner points back at us
            lateral = abs(vx * d[1] - vy * d[0])          # offset from our line
            if along > 0.98 and facing < -0.98 and lateral <= 3 and (best is None or dist < best[0]):
                best = (dist, fid)
        if best:
            joins[(eid, end)] = best[1]
            straight_through.add((eid, end))
    for i, (eid, end, p, d) in enumerate(loose):
        if (eid, end) in joins:
            continue
        partners = [(math.dist(p, q), fid, fend, fd) for j, (fid, fend, q, fd) in enumerate(loose)
                    if j != i and fid != eid and math.dist(p, q) <= tol]
        if partners:
            _, fid, fend, fd = min(partners)
            joins[(eid, end)] = fid
            if len(partners) == 1 and d[0] * fd[0] + d[1] * fd[1] < -0.9:
                straight_through.add((eid, end))
    for eid, end, p, _ in loose:
        if (eid, end) in joins and (eid, end) in straight_through:
            continue  # a pipe continued across another line: not a junction with it
        if (eid, end) in joins:
            continue
        hits = [(d, f.id) for f in live if f.id != eid for d in [_dist_to_route(p, routes[f.id])] if d <= tol]
        if hits:
            joins[(eid, end)] = min(hits)[1]
    return joins


def find_dash_chains(edges, routes: dict[str, Route], max_piece: float = 16.0, max_gap: float = 14.0,
                     min_pieces: int = 4) -> list[list[str]]:
    """Lines drawn as a row of separate short dashes (LightUp's TIC 203 -> H201 signal line:
    ~20 pieces 10 units long, 10 apart). Each piece alone looks like decorative linework,
    and the junction rules would join the row into a PROCESS pipe; together they are one
    dashed (signal) line. Returns groups of edge ids: >= min_pieces short straight pieces,
    each end within max_gap of the next piece's end, on one line (a symbol's hatch strokes
    are parallel but side by side, so they never chain)."""
    short = {}
    for e in edges:
        r = routes.get(e.id)
        if r and len(r) == 2 and math.dist(r[0], r[1]) <= max_piece:
            short[e.id] = r
    ids = list(short)
    parent = {i: i for i in ids}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        ra = short[a]
        for b in ids[i + 1:]:
            rb = short[b]
            gap = min(math.dist(p, q) for p in ra for q in rb)
            if gap > max_gap:
                continue
            # same straight line: b's end points lie on a's line (perpendicular offset <= 2)
            vx, vy = ra[1][0] - ra[0][0], ra[1][1] - ra[0][1]
            n = math.hypot(vx, vy) or 1.0
            if all(abs((q[0] - ra[0][0]) * vy - (q[1] - ra[0][1]) * vx) / n <= 2 for q in rb):
                parent[root(a)] = root(b)
            elif gap <= 3:  # a corner of the dashed route (pieces meeting at right angles)
                parent[root(a)] = root(b)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) >= min_pieces]


# --- the shapes as draw.io draws them --------------------------------------------------
# Each symbol's own geometry from the same SVG export (its <g data-cell-id> group): paths,
# ellipses, rectangles, polylines - not its text. So a DEXPI file's drawing and the "after"
# picture use the very symbol on the drawing (the draw.io P&ID stencil as drawn), not a
# stand-in. A stroke is (points, closed, filled) in draw.io coordinates.

def _matrix(t: str):
    """An SVG transform attribute as (a, b, c, d, e, f)."""
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def mul(p, q):
        a, b, c, d, e, f = p
        A, B, C, D, E, F = q
        return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D, a * E + c * F + e, b * E + d * F + f)
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", t or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "translate":
            q = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            q = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            a = math.radians(v[0])
            cx, cy = (v[1], v[2]) if len(v) > 2 else (0, 0)
            q = mul(mul((1, 0, 0, 1, cx, cy), (math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0, 0)),
                    (1, 0, 0, 1, -cx, -cy))
        elif name == "matrix" and len(v) == 6:
            q = tuple(v)
        else:
            continue
        m = mul(m, q)
    return m


def _path_points(d: str) -> list[tuple[list, bool]]:
    """An SVG path's subpaths as ([(x, y)...], closed), curves and arcs flattened."""
    toks = re.findall(r"[A-Za-z]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", d or "")
    out, pts = [], []
    x = y = sx = sy = 0.0
    lc = None   # last control point (for S/T)
    i, cmd = 0, None

    def num():
        nonlocal i
        i += 1
        return float(toks[i - 1])
    while i < len(toks):
        if re.match(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if pts:
                    out.append((pts, True))
                pts, x, y = [], sx, sy
                continue
        if cmd is None:
            break
        rel = cmd.islower()
        C = cmd.upper()
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if C == "M":
            if len(pts) > 1:
                out.append((pts, False))
            x, y = ox + num(), oy + num()
            sx, sy, pts = x, y, [(x, y)]
            cmd = "l" if rel else "L"
        elif C == "L":
            x, y = ox + num(), oy + num()
            pts.append((x, y))
        elif C == "H":
            x = (x if rel else 0) + num()
            pts.append((x, y))
        elif C == "V":
            y = (y if rel else 0) + num()
            pts.append((x, y))
        elif C in "CSQT":
            if C == "C":
                c1 = (ox + num(), oy + num())
                c2 = (ox + num(), oy + num())
            elif C == "S":
                c1 = (2 * x - lc[0], 2 * y - lc[1]) if lc else (x, y)
                c2 = (ox + num(), oy + num())
            elif C == "Q":
                q = (ox + num(), oy + num())
                c1 = (x + 2 / 3 * (q[0] - x), y + 2 / 3 * (q[1] - y))
                c2 = None
            else:
                q = (2 * x - lc[0], 2 * y - lc[1]) if lc else (x, y)
                c1, c2 = None, None
            ex, ey = ox + num(), oy + num()
            if C in "QT":
                q = q if C == "T" else q
                c1 = (x + 2 / 3 * (q[0] - x), y + 2 / 3 * (q[1] - y))
                c2 = (ex + 2 / 3 * (q[0] - ex), ey + 2 / 3 * (q[1] - ey))
                lc = q
            else:
                lc = c2
            for k in range(1, 13):
                t = k / 12
                u = 1 - t
                pts.append((u ** 3 * x + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * ex,
                            u ** 3 * y + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * ey))
            x, y = ex, ey
            continue
        elif C == "A":
            rx, ry, phi, large, sweep = abs(num()), abs(num()), math.radians(num()), num(), num()
            ex, ey = ox + num(), oy + num()
            pts.extend(_arc(x, y, rx, ry, phi, bool(large), bool(sweep), ex, ey))
            x, y = ex, ey
        else:
            i += 1
        lc = None
    if len(pts) > 1:
        out.append((pts, False))
    return out


def _arc(x1, y1, rx, ry, phi, large, sweep, x2, y2):
    """An SVG elliptical arc as points (SVG spec, F.6.5)."""
    if rx == 0 or ry == 0:
        return [(x2, y2)]
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = c * dx + s * dy, -s * dx + c * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num_ = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    k = math.sqrt(max(0.0, num_ / den)) if den else 0.0
    if large == sweep:
        k = -k
    cxp, cyp = k * rx * y1p / ry, -k * ry * x1p / rx
    cx, cy = c * cxp - s * cyp + (x1 + x2) / 2, s * cxp + c * cyp + (y1 + y2) / 2
    ang = lambda ux, uy, vx, vy: math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(4, int(abs(dt) / (math.pi / 16)))
    return [(cx + rx * math.cos(t1 + dt * j / n) * c - ry * math.sin(t1 + dt * j / n) * s,
             cy + rx * math.cos(t1 + dt * j / n) * s + ry * math.sin(t1 + dt * j / n) * c) for j in range(1, n + 1)]


def _filled(el) -> bool:
    fill = (el.get("fill") or "").lower()
    style = (el.get("style") or "").lower()
    m = re.search(r"fill:\s*([^;]+)", style)
    fill = m.group(1).strip() if m and "var(" not in m.group(1) else fill
    return fill not in ("", "none", "transparent", "#ffffff", "#fff", "white") and not fill.startswith("url")


def _shape_strokes(group) -> list:
    """The drawn strokes of one cell's <g>: [(points, closed, filled)] in export pixels."""
    out = []

    def walk(el, m):
        tag = el.tag.split("}")[-1]
        if tag in ("foreignObject", "text", "image", "switch"):
            return
        m = _mul(m, _matrix(el.get("transform")))
        if (el.get("stroke") in ("none", "transparent") and not _filled(el)) or el.get("visibility") == "hidden":
            return
        P = lambda p: (m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5])
        f = lambda k: float(el.get(k, 0) or 0)
        if tag == "path":
            for pts, closed in _path_points(el.get("d")):
                out.append(([P(p) for p in pts], closed, closed and _filled(el)))
        elif tag in ("ellipse", "circle"):
            rx, ry = (f("r"), f("r")) if tag == "circle" else (f("rx"), f("ry"))
            pts = [(f("cx") + rx * math.cos(t * math.pi / 18), f("cy") + ry * math.sin(t * math.pi / 18)) for t in range(36)]
            out.append(([P(p) for p in pts], True, _filled(el)))
        elif tag == "rect":
            x, y, w, h = f("x"), f("y"), f("width"), f("height")
            out.append(([P(p) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))], True, _filled(el)))
        elif tag in ("polyline", "polygon"):
            v = [float(t) for t in re.split(r"[\s,]+", (el.get("points") or "").strip()) if t]
            pts = list(zip(v[::2], v[1::2]))
            if len(pts) > 1:
                out.append(([P(p) for p in pts], tag == "polygon", tag == "polygon" and _filled(el)))
        elif tag == "line":
            out.append(([P((f("x1"), f("y1"))), P((f("x2"), f("y2")))], False, False))
        for ch in el:
            walk(ch, m)
    walk(group, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    return out


def _mul(p, q):
    a, b, c, d, e, f = p
    A, B, C, D, E, F = q
    return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D, a * E + c * F + e, b * E + d * F + f)


def shapes_for(drawio_path: str | Path, edges, shapes=()) -> dict[str, list] | None:
    """{shape id: [(points, closed, filled)]} - each symbol as draw.io draws it, in draw.io
    coordinates (the same export and shift as routes_for); None without the draw.io app."""
    text = _export_svg(Path(drawio_path))
    if text is None:
        return None
    shift = _shift(_raw_routes(text), edges, shapes)
    if shift is None:
        return None
    dx, dy = shift
    ids = {s.id for s in shapes}
    root = ET.fromstring(text)
    out = {}
    for g in root.iter():
        cid = g.get("data-cell-id")
        if g.tag.split("}")[-1] != "g" or cid not in ids or cid in out:
            continue
        strokes = [([(x - dx, y - dy) for x, y in pts], closed, filled) for pts, closed, filled in _shape_strokes(g)]
        if strokes:
            out[cid] = strokes
    return out
