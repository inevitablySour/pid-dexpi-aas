"""Export a parsed drawio diagram as a PID2Graph-schema .graphml file, so the team's own
7 P&ID files can be folded into PID2Graph as additional real, labeled training/eval data.

Schema confirmed by direct inspection of a real sample (PID2Graph/Complete/PID2Graph
OPEN100/3.graphml, downloaded from Zenodo record 14803338) - not guessed from the paper's
prose description, which only said "bounding box and label" and omitted the actual
GraphML key ids/types:

  <key id="d0" for="node" attr.name="label" attr.type="string" />
  <key id="d1" for="node" attr.name="xmin" attr.type="double" />
  <key id="d2" for="node" attr.name="xmax" attr.type="double" />
  <key id="d3" for="node" attr.name="ymin" attr.type="double" />
  <key id="d4" for="node" attr.name="ymax" attr.type="double" />
  <key id="d9" for="edge" attr.name="edge_label" attr.type="string" />

Real files also declare a second, long-typed xmin/xmax/ymin/ymax key set (d5-d8), but
that set is only ever populated on structural "connector"/"crossing" nodes, never on
real symbol nodes - see the module docstring's key ids below for exactly which set this
exporter writes for which node kind.

Node label vocabulary observed in the real OPEN100 subset: arrow, background, connector,
crossing, general, inlet/outlet, instrumentation, pump, tank, valve. Of these, only
{valve, pump, tank, instrumentation, inlet/outlet, general, arrow} are real symbol
classes (the paper's Table 1 taxonomy); background/connector/crossing are structural
node types the graph topology needs (a real edge never runs symbol-to-symbol directly -
every real edge in the dataset runs through at least one connector node - but this
exporter's own topology comes from Edge.source/target resolved directly onto shape ids,
so it emits connector nodes only where genuinely useful: at each edge's midpoint, to
keep the direct symbol-to-symbol connections this pipeline already has instead of
guessing at intermediate waypoints that were never drawn as separate cells).

Edge label vocabulary: "solid" is the only value observed in the real OPEN100 subset,
but the paper's Table 1 also defines "non-solid" for dashed/control lines - mapped from
Edge.is_dashed, since the taxonomy is documented even though this particular real subset
happens not to exercise it.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

from pid_dexpi_aas.dexpi.mapping import Category, classify_tag
from pid_dexpi_aas.drawio.parser import ParsedDiagram, Shape, parse_drawio_file
from pid_dexpi_aas.review import OFF_PAGE, SKIP, category_for_class, from_drawio

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

TANK_CLASSES = {"Tank", "PressureVessel", "CustomVessel", "Vessel", "ProcessColumn"}
PUMP_CLASSES = {"Pump", "CustomPump"}


def pid2graph_class(dexpi_class: str, shape_token: str | None) -> str | None:
    """Review row (DEXPI class + draw.io shape token) -> PID2Graph 7-class label, or None
    for things that are not symbols (text labels, decorative shapes)."""
    token = shape_token or ""
    if token.startswith("plain:"):
        # plain-shape symbol (plain_shapes.py) - its PID2Graph label is in the token,
        # and it is a symbol even when the reviewer/mapping skipped its DEXPI class
        return token.rsplit(":", 1)[1]
    if "arrows2.arrow" in token:
        return "inlet/outlet"  # e.g. "Fume hood [atm]": an off-page destination
    if "opto_electronics.lamp" in token:
        return "general"
    if dexpi_class in (SKIP, OFF_PAGE) or not shape_token:
        return None
    if dexpi_class == "SignalOffPageConnector":
        return "inlet/outlet"
    category = category_for_class(dexpi_class)
    if category == Category.PIPING:
        return "valve"
    if category == Category.INSTRUMENTATION:
        return "instrumentation"
    if dexpi_class in TANK_CLASSES:
        return "tank"
    if dexpi_class in PUMP_CLASSES:
        return "pump"
    return "general"


def rotated_box(shape: Shape) -> tuple[float, float, float, float]:
    """draw.io stores geometry BEFORE rotation; a +-90 deg shape occupies the box with
    width and height swapped around the same centre (off-page arrows, vertical valves,
    heat exchangers). 180 deg leaves the box unchanged."""
    m = re.search(r"(?:^|;)rotation=(-?[\d.]+)", shape.style or "")
    angle = abs(float(m.group(1))) % 180 if m else 0.0
    if 45 < angle < 135:
        cx, cy = shape.x + shape.width / 2, shape.y + shape.height / 2
        return (cx - shape.height / 2, cy - shape.width / 2, cx + shape.height / 2, cy + shape.width / 2)
    return (shape.x, shape.y, shape.x + shape.width, shape.y + shape.height)


def labelled_symbols(diagram: ParsedDiagram, arrows: bool = False, ink=None):
    """The symbols and connections of a drawing in PID2Graph terms - the ONE labelling
    used both for these exports and for the ML training data (make_team_data.py).

    Returns (symbols, edges): symbols = [(node_id, label, (x1, y1, x2, y2), drawio_id)] in
    draw.io coordinates; edges = [(node_id_a, node_id_b, "solid" | "non-solid")], one per
    connected pair. Only real symbols are labelled - the original exporter labelled EVERY
    draw.io cell, text labels included, as "general" (56 of LightUp's 113 nodes), which
    would teach a model to box text as symbols.

    Caveat: connections come from the draw.io lines via the parser; about half of them had
    one loose end resolved to the nearest shape (not verified individually).

    arrows=True also labels every drawn arrowhead as an "arrow" symbol and splits the
    connection it sits on (A - arrow - B), as OPEN100 does (_arrow_labels). Off: the
    labelling of the Team / Team v3 training data. ink(box) -> fraction of dark pixels in a
    draw.io-coordinate box of the rendered drawing; with it the arrow boxes are fitted to
    the drawn heads and arrows hidden under a shape are left out.
    """
    review = from_drawio(diagram)
    rows = {r.id: r for r in review.symbols}
    symbols = []
    for shape in diagram.shapes:
        row = rows.get(shape.id)
        label = pid2graph_class(row.dexpi_class, shape.shape_token) if (row and shape.shape_token) else None
        if label is not None:
            symbols.append((_sanitize_id(shape.id), label, rotated_box(shape), shape.id))
    id_map = {orig: sid for sid, _, _, orig in symbols}
    edges = []  # (a, b, line, group): group = the draw.io line or pipe network it comes from
    for c in review.connections:
        if c.include and c.from_id in id_map and c.to_id in id_map and c.from_id != c.to_id:
            a, b = sorted((id_map[c.from_id], id_map[c.to_id]))
            edges.append((a, b, "non-solid" if c.line_type == "signal" else "solid", c.id.split(":")[0]))
    if arrows:
        _arrow_labels(diagram, symbols, edges, id_map, ink)
    out, seen = [], set()
    for a, b, line, _ in edges:
        key = tuple(sorted((a, b)))
        if a != b and key not in seen:  # parallel lines between one pair -> one connection
            seen.add(key)
            out.append((key[0], key[1], line))
    return symbols, out


ARROW_LENGTH = 10.0  # a draw.io "classic" arrowhead (size 6, stroke 1), drawing units - for finding it
# The box an arrow is LABELLED with, as OPEN100 labels them: its long side 0.35 of the sheet's
# median symbol, its short side 0.73 of that (measured on the 12 OPEN100 sheets). A box the
# size of the drawn head (~0.12 of a symbol) is ~5 px for the model: Team v4 used that and the
# model found 7 of LightUp's 40 arrows (and its connections suffered on OPEN100 too).
ARROW_BOX = (0.35, 0.73)
HEAD_SPAN = (-2, 8)  # where a head lies along its line, past the route end (measured on LightUp)


def _wide_at(ink, x: float, y: float, dx: float, dy: float) -> bool:
    """Is the drawing wider than a plain line at (x, y), across the direction (dx, dy)?"""
    return ink([(x - dy * o, y + dx * o) for o in (-3, -2, 2, 3)]) >= 0.75


def _arrow_labels(diagram: ParsedDiagram, symbols: list, edges: list, id_map: dict, ink=None) -> None:
    """Every arrowhead drawn at a line end becomes an "arrow" symbol: a box over the head.
    The line's connections are split at it - OPEN100 labels arrows as nodes on the pipe (symbol - arrow - symbol), not as decoration. The arrow sits between the
    symbol at its end of the line (or, at a tee, the line's other end) and the rest; on a
    pipe network every connection of that symbol through the network is split.
    symbols / edges are extended in place."""
    import statistics
    sizes = [max(b[2] - b[0], b[3] - b[1]) for _, label, b, _ in symbols if label != "arrow"]
    box_long = ARROW_BOX[0] * (statistics.median(sizes) if sizes else 85.0)
    for e in diagram.edges:
        route = (diagram.routes or {}).get(e.id)
        if e.is_decorative or not route or len(route) < 2:
            continue
        group = e.merged_into or e.id
        for end, drawn in ((0, e.has_start_arrow), (1, e.has_end_arrow)):
            if not drawn:
                continue
            pts = route if end == 1 else route[::-1]
            (tx, ty) = pts[-1]
            prev = next(((x, y) for x, y in reversed(pts[:-1]) if (x, y) != (tx, ty)), None)
            if prev is None:
                continue
            dx, dy = tx - prev[0], ty - prev[1]
            n = (dx * dx + dy * dy) ** 0.5
            dx, dy = dx / n, dy / n
            # draw.io stops the line where an end head starts: the head mostly lies just past
            # the route end; some (start arrows) lie over the line's last stretch instead -
            # with the rendered drawing (ink) the side with the head wins, and a head that
            # is nowhere to be seen (under a shape) is not labelled
            h = ARROW_LENGTH / 2
            lo, hi = HEAD_SPAN
            spots = [(tx + dx * (lo + hi) / 2, ty + dy * (lo + hi) / 2),
                     (tx - dx * h, ty - dy * h)]  # head over the last stretch
            if ink is not None:
                ahead = sum(_wide_at(ink, tx + dx * t, ty + dy * t, dx, dy) for t in range(lo, hi + 1))
                behind = sum(_wide_at(ink, tx - dx * t, ty - dy * t, dx, dy) for t in range(-lo, hi + 3))
                if max(ahead, behind) < 2:
                    continue
                if behind > ahead:
                    spots.reverse()
            cx, cy = spots[0]
            sid = f"{_sanitize_id(e.id)}_arrow{end}"
            along, across = box_long / 2, box_long * ARROW_BOX[1] / 2  # half sizes; long side along the pipe
            hw, hh = (along, across) if abs(dx) >= abs(dy) else (across, along)
            symbols.append((sid, "arrow", (cx - hw, cy - hh, cx + hw, cy + hh), f"{e.id}:arrow{end}"))
            here, other = (e.target, e.source) if end == 1 else (e.source, e.target)
            near = id_map.get(here) or id_map.get(other)
            if near is None:
                continue
            if id_map.get(here) is None and e.merged_into is None:
                # a plain line whose arrow end is on no labelled symbol (a stream label):
                # the arrow is that end
                edges.append((near, sid, "non-solid" if e.is_dashed else "solid", group))
                continue
            for k, (a, b, line, g) in enumerate(list(edges)):
                if g == group and near in (a, b):
                    far = b if a == near else a
                    edges[k] = (sid, far, line, g)
                    edges.append((near, sid, line, g))


POINT_BOX = 8.0  # the waypoint box PID2Graph gives its connector/crossing nodes (drawing units)
JOIN_TOL = 8.0  # line ends this close meet (find_junctions' tolerance)
JUMP_GAP = 30.0  # a pipe drawn with a gap across another line (find_junctions' line jump)


def line_points(diagram: ParsedDiagram, symbols: list) -> tuple[list, list]:
    """The drawn lines as PID2Graph routes them: symbols linked through waypoint nodes -
    "connector" at each corner of a route, "crossing" where lines meet (a tee) - so a model
    can learn the bends and tees of a pipe (the dataset's DATA.LINE_POINTS mode).

    symbols: labelled_symbols' symbols (arrows included). Returns (points, edges): points
    [(node_id, "connector" | "crossing", box, None)], edges [(a, b, line)] for every leg of
    every line. A line end on a labelled symbol joins that symbol; an end on another line
    (a tee) or on another line's end joins a shared point; any other end is left out (a
    stream label, a loose end), as labelled_symbols leaves those connections out.
    Folding the points away again (_collapse_graph) gives labelled_symbols' connections,
    up to its nearest-shape guesses and the arrows at tees (measured in make_team_data)."""
    import math

    routes = diagram.routes or {}
    by_shape = {orig: sid for sid, label, _, orig in symbols if label != "arrow"}
    arrows = {orig: sid for sid, label, _, orig in symbols if label == "arrow"}
    lines = {e.id: e for e in diagram.edges if not e.is_decorative and len(routes.get(e.id) or ()) >= 2}
    where: dict[str, tuple[float, float]] = {}  # point node -> position
    parent: dict[str, str] = {}

    def find(n):
        while parent.get(n, n) != n:
            n = parent[n]
        return n

    def end_of(e, k):
        r = routes[e.id]
        return (e.source_point if k == 0 else e.target_point) or (r[0] if k == 0 else r[-1])

    def junction(e, k):
        return e.source_junction if k == 0 else e.target_junction

    def along(r, p):
        """(distance along route r of p's nearest point, that point, distance off the route)."""
        best, run = None, 0.0
        for (ax, ay), (bx, by) in zip(r, r[1:]):
            seg = math.hypot(bx - ax, by - ay)
            t = 0.0 if seg == 0 else max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / seg ** 2))
            q = (ax + t * (bx - ax), ay + t * (by - ay))
            d = math.dist(p, q)
            if best is None or d < best[2]:
                best = (run + t * seg, q, d)
            run += seg
        return best

    # 1. line ends on lines: a shared point with the other line's end, or a tee on its route
    tees: dict[str, list] = {}  # line id -> [(distance along it, point node)]
    for e in lines.values():
        for k in (0, 1):
            other = lines.get(junction(e, k))
            if other is None:
                continue
            node = f"p_{_sanitize_id(e.id)}_{k}"
            p = end_of(e, k)
            where.setdefault(node, p)
            r = routes[other.id]
            near_end = [j for j in (0, 1) if junction(other, j) is not None
                        and math.dist(p, end_of(other, j)) <= (JUMP_GAP if junction(other, j) == e.id else JOIN_TOL)]
            if near_end:
                j = min(near_end, key=lambda j: math.dist(p, end_of(other, j)))
                mine = f"p_{_sanitize_id(other.id)}_{j}"
                where.setdefault(mine, end_of(other, j))
                parent[find(node)] = find(mine)
            else:
                run, q, _ = along(r, p)
                where[node] = q
                tees.setdefault(other.id, []).append((run, node))

    # 2. each line: its stations in order along the route, linked one to the next
    # corners under a symbol are covered by it; not under a container (an autoclave frame
    # holding other items): pipes drawn inside one still show their corners
    sizes = sorted(max(b[2] - b[0], b[3] - b[1]) for _, label, b, _ in symbols if label != "arrow")
    typical = 3 * sizes[len(sizes) // 2] if sizes else 0
    hidden = [b for _, label, b, _ in symbols if label != "arrow" and max(b[2] - b[0], b[3] - b[1]) <= typical]
    edges: list[tuple[str, str, str]] = []
    legs: list[tuple[list, str]] = []
    corners: set[str] = set()
    for e in lines.values():
        r = routes[e.id]
        total = sum(math.dist(a, b) for a, b in zip(r, r[1:]))
        stations = []
        for k, run in ((0, -1.0), (1, total + 1.0)):
            shape = e.source if k == 0 else e.target
            if shape in by_shape:
                stations.append((run, by_shape[shape]))
            elif junction(e, k) in lines:
                stations.append((run, f"p_{_sanitize_id(e.id)}_{k}"))
            if f"{e.id}:arrow{k}" in arrows:  # an arrowhead sits between the line's end and its symbol
                stations.append((run + (0.5 if k == 0 else -0.5), arrows[f"{e.id}:arrow{k}"]))
        for (bx, by), turn in _corners(r, hidden):
            node = f"p_{_sanitize_id(e.id)}_b{turn}"
            where[node] = (bx, by)
            corners.add(node)
            stations.append((along(r, (bx, by))[0], node))
        stations += tees.get(e.id, [])
        stations.sort(key=lambda st: st[0])
        for (_, a), (_, b) in zip(stations, stations[1:]):  # a tee on a corner: one point
            if a in where and b in where and math.dist(where[a], where[b]) <= JOIN_TOL:
                keep, gone = (a, b) if b in corners else (b, a)
                parent[find(gone)] = find(keep)
        legs.append((stations, "non-solid" if e.is_dashed else "solid"))
    for stations, line in legs:
        nodes = [n if n not in where else find(n) for _, n in stations]
        for a, b in zip(nodes, nodes[1:]):
            if a != b:
                edges.append((a, b, line))
    for e in diagram.edges:  # links without a drawn line (two items drawn touching): direct
        if e.id.startswith("adjacent:") and e.source in by_shape and e.target in by_shape:
            edges.append((by_shape[e.source], by_shape[e.target], "non-solid" if e.is_dashed else "solid"))

    used = {n for a, b, _ in edges for n in (a, b)}
    degree: dict[str, int] = {}
    for a, b, _ in {(min(a, b), max(a, b), l) for a, b, l in edges}:
        degree[a], degree[b] = degree.get(a, 0) + 1, degree.get(b, 0) + 1
    # labelled as PID2Graph does (its author, 2026-10): "crossing" for a corner (an "ankle")
    # and for where lines join; "connector" - never taught - for two line pieces joined
    # straight on (a pipe drawn in pieces: nothing to see there)
    pos = {sid: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for sid, _, b, _ in symbols}
    pos.update(where)
    nbrs: dict[str, set] = {}
    for a_, b_, _ in edges:
        nbrs.setdefault(a_, set()).add(b_)
        nbrs.setdefault(b_, set()).add(a_)

    def straight(n):
        ps = [pos.get(m) for m in nbrs.get(n, ())]
        if len(ps) != 2 or None in ps or n not in pos:
            return False
        (px, py), (ax, ay), (bx, by) = pos[n], ps[0], ps[1]
        u, v = (ax - px, ay - py), (bx - px, by - py)
        nu, nv = math.hypot(*u), math.hypot(*v)
        return nu > 0 and nv > 0 and (u[0] * v[0] + u[1] * v[1]) / (nu * nv) < -0.94

    h = POINT_BOX / 2
    points = []
    for n in sorted(used):
        if n in where and find(n) == n:
            x, y = where[n]
            label = "connector" if n not in corners and degree.get(n, 0) == 2 and straight(n) else "crossing"
            points.append((n, label, (x - h, y - h, x + h, y + h), None))
    return points, edges


JOG = 6.0  # route steps shorter than this are not drawn corners (a 2-5 unit kink in draw.io)


def _corners(route: list, hidden: list) -> list:
    """[(point, index)] of the corners a route visibly turns at: tiny kinks smoothed out
    first, corners inside a symbol's box (covered by it) left out."""
    import math
    pts = [route[0]]
    for i, p in enumerate(route[1:-1], 1):
        if math.dist(p, pts[-1]) >= JOG and math.dist(p, route[-1]) >= JOG:
            pts.append(p)
    pts.append(route[-1])
    out = []
    for i in range(1, len(pts) - 1):
        (ax, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[i + 1]
        u, v = (bx - ax, by - ay), (cx - bx, cy - by)
        if abs(u[0] * v[1] - u[1] * v[0]) / (math.hypot(*u) * math.hypot(*v)) <= 0.17:  # turns < ~10 deg
            continue
        out.append(((bx, by), i))
    # a rounded corner is drawn as two turns ~14 units apart (the corner cut at 10 units
    # each side): one corner, where the straight runs either side of it meet
    merged = []
    for (p, i) in out:
        if merged and merged[-1][1] == i - 1 and math.dist(merged[-1][0], p) <= ROUNDED:
            a, b, c, d = pts[i - 2], pts[i - 1], pts[i], pts[i + 1]
            merged[-1] = (_meet(a, b, c, d) or ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2), i)
        else:
            merged.append((p, i))
    return [(p, i) for p, i in merged
            if not any(b[0] + 2 < p[0] < b[2] - 2 and b[1] + 2 < p[1] < b[3] - 2 for b in hidden)]


ROUNDED = 20.0


def _meet(a, b, c, d):
    """Where the line through a-b meets the line through c-d (None: parallel)."""
    den = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    if abs(den) < 1e-9:
        return None
    t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / den
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _sanitize_id(shape_id: str) -> str:
    """GraphML node/edge ids must be valid XML names; drawio ids are already alnum/-/_
    but are prefixed here to guarantee a non-numeric first character regardless."""
    return f"n{shape_id}"


def diagram_to_pid2graph_xml(diagram: ParsedDiagram) -> ET.Element:
    ET.register_namespace("", GRAPHML_NS)
    root = ET.Element(f"{{{GRAPHML_NS}}}graphml")
    root.set(f"{{{XSI_NS}}}schemaLocation", f"{GRAPHML_NS} {GRAPHML_NS}/1.0/graphml.xsd")

    def add_key(key_id: str, for_: str, name: str, type_: str) -> None:
        key = ET.SubElement(root, f"{{{GRAPHML_NS}}}key")
        key.set("id", key_id)
        key.set("for", for_)
        key.set("attr.name", name)
        key.set("attr.type", type_)

    add_key("d0", "node", "label", "string")
    add_key("d1", "node", "xmin", "double")
    add_key("d2", "node", "xmax", "double")
    add_key("d3", "node", "ymin", "double")
    add_key("d4", "node", "ymax", "double")
    add_key("d9", "edge", "edge_label", "string")

    graph = ET.SubElement(root, f"{{{GRAPHML_NS}}}graph")
    graph.set("edgedefault", "undirected")

    def add_data(parent: ET.Element, key_id: str, value: str) -> None:
        data = ET.SubElement(parent, f"{{{GRAPHML_NS}}}data")
        data.set("key", key_id)
        data.text = value

    symbols, edges = labelled_symbols(diagram)
    for node_id, label, (x1, y1, x2, y2), _ in symbols:
        node = ET.SubElement(graph, f"{{{GRAPHML_NS}}}node")
        node.set("id", node_id)
        add_data(node, "d0", label)
        add_data(node, "d1", str(x1))
        add_data(node, "d2", str(x2))
        add_data(node, "d3", str(y1))
        add_data(node, "d4", str(y2))
    for a, b, line in edges:
        edge_el = ET.SubElement(graph, f"{{{GRAPHML_NS}}}edge")
        edge_el.set("source", a)
        edge_el.set("target", b)
        add_data(edge_el, "d9", line)

    return root


def export_pid2graph_graphml(diagram: ParsedDiagram) -> str:
    root = diagram_to_pid2graph_xml(diagram)
    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def export_drawio_file_to_pid2graph(drawio_path: str | Path, output_path: str | Path) -> None:
    """Accepts a real .drawio file (compressed diagram, decoded first) or already-decoded
    mxGraph XML. Reading a .drawio directly as XML finds no cells and writes an empty
    graph - which is exactly what happened when regenerating these exports."""
    from pid_dexpi_aas.drawio.decode import decode_drawio
    from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved

    path = Path(drawio_path)
    if path.suffix == ".drawio":
        diagram = parse_drawio_xml_resolved(decode_drawio(path)[0], source_file=path.name, drawio_path=path)
    else:
        diagram = parse_drawio_file(path)
    xml_text = export_pid2graph_graphml(diagram)
    Path(output_path).write_text(xml_text, encoding="utf-8")
