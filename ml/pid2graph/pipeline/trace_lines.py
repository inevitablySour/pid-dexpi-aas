"""Trace pipes as straight line pieces, not pixels (the alternative to trace_pipes.py).

trace_pipes.py thins all the ink to 1-px lines and joins whatever touches. Most of its
errors come from that: a hop over a pipe or a crossing reads as a tee, a tee beside a
symbol is lost in the symbol's margin, arrowheads and symbol bits make blobs. P&IDs are
drawn almost entirely in horizontal and vertical lines, so here:

  1. linework as in trace_pipes (colour and light gray included, text and the symbols
     with their margins erased);
  2. horizontal and vertical runs at least `min_run` long are taken out separately (a
     morphological opening with a line) - text bits, symbol fragments and arrowheads
     mostly drop out; each run is one segment: a row (column) and its two ends. A thick run
     (a filled area) is no line;
  3. dashes too short for 2. that sit in a row with regular gaps become a dashed segment;
     what is left that is thin and long enough - diagonal pipes, curves, slanted lines - is
     thinned to its centre line: a free-shaped segment (its path keeps the drawn shape);
  4. how segments meet is decided geometrically:
       - an end facing the end of a segment in line with it, across a short gap (dashes,
         a break), across erased text (a label written on the pipe), across a symbol's margin
         without entering the symbol (a pipe running past it), or across a pipe it crosses (a
         line jump: an arc or a gap) - one line;
       - otherwise an end at another segment, perpendicular: a bend (near its end) or a tee
         (on its body);
       - two segments passing through each other: a crossing - not joined;
       - an end pointing into a symbol within the margin (or `near_miss` of a symbol size
         when nothing else took it): the line ends at that symbol;
  5. every symbol's lines are followed through the joined segments to the other symbols they
     reach (the fewest segments), giving the path as its corner points; flow from
     arrowheads where a line meets a symbol, and from flow arrows drawn on the pipe.
Symbols close together (stacked instruments) use trace_pipes' direct check across the gap.
Same inputs and output as trace_pipes.trace_connections. Image pixels; boxes (x1, y1, x2, y2).
"""
from __future__ import annotations

import functools
import math
from collections import deque

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from trace_pipes import _arrow_tip, _arrowhead_at, _erase_text, _ink, _short_links, _symbol_map


def _runs(mask: np.ndarray, axis: int, wmax: int) -> list:
    """Segments from a mask of long runs along `axis` (1: horizontal, 0: vertical):
    [{"o": "h"|"v", "c": row (col), "a": start, "b": end, "w": thickness}]. A component's
    line is its rows (cols) holding most of it - an arrowhead's or a symbol bit's extra
    rows are left out; too many such rows: a filled area, no line."""
    lab, n = ndi.label(mask)
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == k
        if axis == 0:
            sub = sub.T
        counts = sub.sum(axis=1)
        core = np.nonzero(counts >= 0.6 * counts.max())[0]
        if len(core) > wmax:
            continue
        cols = np.nonzero(sub[core].any(axis=0))[0]
        across, along = (sl[0].start, sl[1].start) if axis == 1 else (sl[1].start, sl[0].start)
        out.append({"o": "h" if axis == 1 else "v", "c": float(across + core.mean()),
                    "a": float(along + cols[0]), "b": float(along + cols[-1]), "w": len(core), "dashed": False})
    return out


def _dashes(pieces: np.ndarray, min_run: int, size: float) -> list:
    """Dashed segments: small thin pieces in a row (column) with short gaps, at least 3 of
    them and `min_run` long in all."""
    lab, n = ndi.label(pieces, structure=np.ones((3, 3)))
    cand = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) >= 1.5 * min_run or min(h, w) > 3 or (lab[sl] == k).sum() < 2:
            continue
        cand.append(((sl[0].start + sl[0].stop - 1) / 2, (sl[1].start + sl[1].stop - 1) / 2,
                     sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1, h, w))
    gap = max(5.0, 0.2 * size)
    out = []
    for o in ("h", "v"):
        # (across, along start, along end) of the pieces that could lie along this direction
        items = sorted(((yc, x0, x1) if o == "h" else (xc, y0, y1))
                       for yc, xc, y0, y1, x0, x1, h, w in cand if (w >= h if o == "h" else h >= w))
        rows, cur = [], []
        for it in items:  # pieces of one row: across within 1.5 px
            if cur and it[0] - cur[-1][0] > 1.5:
                rows.append(cur)
                cur = []
            cur.append(it)
        if cur:
            rows.append(cur)
        for row in rows:
            row.sort(key=lambda it: it[1])
            chain = [row[0]]
            for it in row[1:] + [None]:
                if it is not None and it[1] - chain[-1][2] <= gap:
                    chain.append(it)
                    continue
                if len(chain) >= 3 and chain[-1][2] - chain[0][1] >= min_run:
                    out.append({"o": o, "c": float(np.median([c[0] for c in chain])), "a": float(chain[0][1]),
                                "b": float(chain[-1][2]), "w": 1, "dashed": True})
                if it is not None:
                    chain = [it]
    return out


def _free_lines(mask: np.ndarray, min_len: float, wmax: int) -> list:
    """Lines that are neither horizontal nor vertical (diagonal pipes, curves, slanted signal
    lines): thin pieces of what is left after the straight runs, at least `min_len` long,
    thinned to a centre line - its longest path end to end is the segment
    {"o": "d", "pts": [(x, y)...]}. A thick piece (a fill, an arrowhead) is no line."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < min_len:
            continue
        comp = lab[sl] == k
        sk = skeletonize(comp)
        n_sk = int(sk.sum())
        if n_sk < min_len or comp.sum() > 1.2 * wmax * n_sk:
            continue
        ys, xs = np.nonzero(sk)
        pix = set(zip(ys.tolist(), xs.tolist()))

        def far(start):   # the skeleton pixel farthest from start (along the skeleton), and the way there
            prev, todo, last = {start: None}, deque([start]), start
            while todo:
                last = q = todo.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        r = (q[0] + dy, q[1] + dx)
                        if r in pix and r not in prev:
                            prev[r] = q
                            todo.append(r)
            way = [last]
            while prev[way[-1]] is not None:
                way.append(prev[way[-1]])
            return last, way
        one, _ = far(next(iter(pix)))
        _, way = far(one)
        if len(way) < min_len:
            continue
        pts = [(float(x + sl[1].start), float(y + sl[0].start)) for y, x in way[::3] + [way[-1]]]
        out.append({"o": "d", "pts": pts, "w": 2, "dashed": False, "a": 0.0, "b": float(len(way)), "c": 0.0})
    return out


def _nearest(seg: dict, p):
    """The point of the segment nearest to p."""
    if seg["o"] != "d":
        x, y = p
        if seg["o"] == "h":
            return (min(max(x, seg["a"]), seg["b"]), seg["c"])
        return (seg["c"], min(max(y, seg["a"]), seg["b"]))
    best, best_d = seg["pts"][0], float("inf")
    for u, v in zip(seg["pts"], seg["pts"][1:]):
        dx, dy = v[0] - u[0], v[1] - u[1]
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else min(1.0, max(0.0, ((p[0] - u[0]) * dx + (p[1] - u[1]) * dy) / L))
        q = (u[0] + t * dx, u[1] + t * dy)
        dd = math.dist(q, p)
        if dd < best_d:
            best, best_d = q, dd
    return best


def _end(seg: dict, e: int):
    """((x, y) of end e, its outward direction (dx, dy))."""
    if seg["o"] == "d":
        pts = seg["pts"] if e == 1 else seg["pts"][::-1]
        p, q = pts[-1], pts[max(0, len(pts) - 4)]   # the last ~10 px give its heading
        L = math.dist(p, q) or 1.0
        return p, ((p[0] - q[0]) / L, (p[1] - q[1]) / L)
    along = seg["a"] if e == 0 else seg["b"]
    sign = -1.0 if e == 0 else 1.0
    if seg["o"] == "h":
        return (along, seg["c"]), (sign, 0.0)
    return (seg["c"], along), (0.0, sign)


def _dist_to(seg: dict, p) -> tuple[float, float]:
    """(distance from p to the segment, how far along it the nearest point is from its
    nearer end - 0 at an end)."""
    if seg["o"] == "d":
        q = _nearest(seg, p)
        return math.dist(q, p), min(math.dist(q, seg["pts"][0]), math.dist(q, seg["pts"][-1]))
    x, y = p
    along, across = (x, y) if seg["o"] == "h" else (y, x)
    t = min(max(along, seg["a"]), seg["b"])
    return math.hypot(along - t, across - seg["c"]), min(t - seg["a"], seg["b"] - t)


def trace_lines(img: np.ndarray, boxes: list, text_boxes: list | None = None, *, dark: int = 210,
                faint: int = 235, margin: float | None = None, gap: float | None = None, near_miss: float = 0.6,
                containers: bool = True, keep_lines_in_text: bool = True, arrows: set | None = None,
                missed: list | None = None, **_ignored) -> list:
    """Connections between symbols along the drawn lines - see the module docstring. Returns
    [{"a", "b" (a < b), "line_type", "bridges", "length", "path" [(x, y)] a -> b, "flow"}]
    like trace_pipes.trace_connections (`missed`: its unboxed-symbol check)."""
    full_ink = _ink(img, dark, faint)
    H, W = full_ink.shape
    size = float(np.median([max(b[2] - b[0], b[3] - b[1]) for b in boxes])) if boxes else 40.0
    margin = margin if margin is not None else max(3.0, 0.12 * size)
    gap = gap if gap is not None else max(6.0, 0.35 * size)
    min_run = max(8, int(0.15 * size))
    wmax = max(5, int(0.1 * size))
    tol = max(4.0, wmax)          # a bend or tee: an end this near the other line
    bend = max(6.0, 0.3 * size)   # rounded bends: the two lines' ends this far apart

    ink = _erase_text(full_ink.copy(), text_boxes, size, keep_lines_in_text)
    sym = _symbol_map(boxes, size, margin, (H, W), containers)
    lines = ink & (sym == 0)
    hmask = ndi.binary_opening(lines, structure=np.ones((1, min_run), bool))
    vmask = ndi.binary_opening(lines, structure=np.ones((min_run, 1), bool))
    segs = _runs(hmask, 1, wmax) + _runs(vmask, 0, wmax)
    rest = lines & ~ndi.binary_dilation(hmask | vmask, iterations=1)
    segs += _dashes(rest, min_run, size)
    # what is left: diagonal and curved lines (not dashes - those are short pieces)
    segs += _free_lines(rest & ~ndi.binary_dilation(hmask | vmask, iterations=2), max(2 * min_run, 0.5 * size), wmax)
    ink_parts, _ = ndi.label(lines, structure=np.ones((3, 3), bool))   # what ink joins what

    @functools.lru_cache(maxsize=None)
    def part_at(x, y) -> int:
        """The ink piece at (x, y) or right next to it (0: none)."""
        ya, xa = max(0, int(round(y)) - 2), max(0, int(round(x)) - 2)
        win = ink_parts[ya:int(round(y)) + 3, xa:int(round(x)) + 3]
        v = win[win > 0]
        return int(np.bincount(v).argmax()) if v.size else 0

    big = {k for k, b in enumerate(boxes) if containers and max(b[2] - b[0], b[3] - b[1]) >= 3.0 * size}
    inside = np.zeros((H, W), np.int32)  # the symbols themselves (frames: their border), no margins
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        ya, yb, xa, xb = max(0, int(y1)), int(y2) + 1, max(0, int(x1)), int(x2) + 1
        if k in big:
            for sl in ((slice(ya, ya + 3), slice(xa, xb)), (slice(max(ya, yb - 3), yb), slice(xa, xb)),
                       (slice(ya, yb), slice(xa, xa + 3)), (slice(ya, yb), slice(max(xa, xb - 3), xb))):
                inside[sl] = k + 1
        else:
            inside[ya:yb, xa:xb] = k + 1
    text_mask = np.zeros((H, W), bool)
    for tb in text_boxes or []:
        x1, y1, x2, y2 = [int(round(v)) for v in tb]
        text_mask[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = True

    def along_line(p, q):
        """Every pixel between p and q (a container's border is only a few px wide)."""
        n = max(2, int(math.dist(p, q)) + 1)
        return [(int(round(p[1] + (q[1] - p[1]) * t)), int(round(p[0] + (q[0] - p[0]) * t)))
                for t in np.linspace(0.0, 1.0, n)]

    def hits(mask, p, q) -> bool:
        return any(0 <= y < H and 0 <= x < W and mask[y, x] for y, x in along_line(p, q))

    side = max(2, wmax // 2 + 1)  # a line a few px beside a small symbol's box still runs into it

    def symbol_ahead(p, d, reach):
        """The symbol a line end at p pointing d runs into within `reach` px (within `side`
        px of the line's own row / column), or None."""
        for t in np.arange(0.0, reach + 1.0, 1.0):
            for o in (0, *[v for k in range(1, side + 1) for v in (k, -k)]):
                x, y = int(round(p[0] + d[0] * t - d[1] * o)), int(round(p[1] + d[1] * t + d[0] * o))
                if 0 <= y < H and 0 <= x < W and inside[y, x]:
                    return int(inside[y, x]) - 1
        return None

    n = len(segs)
    joins: dict[int, list] = {i: [] for i in range(n)}   # i -> [(j, point, bridge counted)]
    ends_at: dict[tuple[int, int], tuple] = {}             # (seg, end) -> (symbol, point)
    taken = set()                                          # (seg, end) already continued / joined

    def join(i, j, p, bridge=False):
        joins[i].append((j, p, bridge))
        joins[j].append((i, p, bridge))

    by_o = {o: [i for i, s in enumerate(segs) if s["o"] == o] for o in ("h", "v", "d")}

    def across(i):
        """The segments an end of segment i can meet at a bend or tee: the other direction, and
        the free-shaped lines (for a free-shaped one: all others)."""
        o = segs[i]["o"]
        if o == "d":
            return [j for j in range(len(segs)) if j != i]
        return by_o["v" if o == "h" else "h"] + by_o["d"]

    far_reach = max(0.6 * size, bend, tol) + 1.0
    # a grid of the segments (cells of far_reach): an end looks only at the segments near it
    cell = max(8.0, far_reach)
    grid: dict = {}
    for j, sg in enumerate(segs):
        if sg["o"] == "d":
            xs, ys = [q[0] for q in sg["pts"]], [q[1] for q in sg["pts"]]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        elif sg["o"] == "h":
            x0, x1, y0, y1 = sg["a"], sg["b"], sg["c"], sg["c"]
        else:
            x0, x1, y0, y1 = sg["c"], sg["c"], sg["a"], sg["b"]
        for gx in range(int(x0 // cell), int(x1 // cell) + 1):
            for gy in range(int(y0 // cell), int(y1 // cell) + 1):
                grid.setdefault((gx, gy), []).append(j)

    def near(i, p):
        """across(i), only the segments within a cell of p."""
        gx, gy = int(p[0] // cell), int(p[1] // cell)
        found = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                found.update(grid.get((gx + dx, gy + dy), ()))
        o = segs[i]["o"]
        return [j for j in found if j != i and (o == "d" or segs[j]["o"] in ("d", "v" if o == "h" else "h"))]


    def on_body(i, e) -> bool:
        """End e of segment i lies on another line's body (not near its ends): it ends there,
        a tee - never carried on in line across that line."""
        p, _ = _end(segs[i], e)
        for j in near(i, p):
            dist, from_end = _dist_to(segs[j], p)
            if dist <= max(2.0, segs[j]["w"] / 2 + 1.5) and from_end > tol:
                return True
        return False
    # 4a. in line: an end facing the next segment along the same row (column)
    for o in ("h", "v"):
        idx = sorted(by_o[o], key=lambda i: (segs[i]["c"], segs[i]["a"]))
        for i in idx:
            s = segs[i]
            if on_body(i, 1):
                continue
            p, d = _end(s, 1)
            best = None
            for j in by_o[o]:
                t = segs[j]
                if j == i or abs(t["c"] - s["c"]) > max(2.0, (s["w"] + t["w"]) / 2 + 1) or t["b"] <= s["b"]:
                    continue
                g = t["a"] - s["b"]
                if g < -tol or (best is not None and g >= best[0]):
                    continue
                best = (g, j)
            if best is None:
                continue
            g, j = best
            if on_body(j, 0):
                continue
            q, _ = _end(segs[j], 0)
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            other = [k for k in near(i, mid) if segs[k]["o"] != o] if g <= 2 * cell else by_o["v" if o == "h" else "h"]
            crossed = any(_dist_to(segs[k], mid)[0] <= max(2.0, g / 2 + 1) and _dist_to(segs[k], mid)[1] > tol
                          for k in other)
            over_text = hits(text_mask, p, q)
            past = hits(sym > 0, p, q) and not hits(inside > 0, p, q)
            if hits(inside > 0, p, q):
                continue            # a symbol between them: each end goes to the symbol
            limit = 3.0 * size if over_text else 0.6 * size if crossed else gap
            if g <= limit or past:
                join(i, j, ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), bridge=g > 2 and not (over_text or crossed or past))
                taken.add((i, 1))
                taken.add((j, 0))

    def meet(wide: bool) -> None:
        """An end at another, perpendicular segment: a bend (near that one's end - rounded
        bends leave the two ends apart, `wide`) or a tee (on its body)."""
        for i, s in enumerate(segs):
            for e in (0, 1):
                if (i, e) in taken:
                    continue
                p, d = _end(s, e)
                best = None
                for j in near(i, p):
                    t = segs[j]
                    dist, from_end = _dist_to(t, p)
                    if dist > far_reach:   # beyond any bend or tee reach: not worth a closer look
                        continue
                    if wide and from_end <= bend and "d" not in (s["o"], t["o"]):
                        # a rounded bend: this end heads towards that line, and that line's
                        # near end towards this one
                        ahead = (t["c"] - p[0]) * d[0] if s["o"] == "h" else (t["c"] - p[1]) * d[1]
                        near_end = t["a"] if abs(t["a"] - s["c"]) < abs(t["b"] - s["c"]) else t["b"]
                        back = (s["c"] - near_end) * (1 if near_end == t["b"] else -1)
                        joined = part_at(*p) and part_at(*p) == part_at(*((t["c"], near_end) if t["o"] == "v" else (near_end, t["c"])))
                        reach = (0.6 * size if joined else bend) if ahead >= -tol and back >= -tol else tol
                    elif wide and "d" in (s["o"], t["o"]):
                        # a free-shaped line's end and another line a little apart (the straight
                        # runs' surroundings are left out of the free-shaped ones): one drawn
                        # line - ink joins the two and the end heads that way
                        foot = _nearest(t, p)
                        heads = (foot[0] - p[0]) * d[0] + (foot[1] - p[1]) * d[1] >= -tol
                        reach = bend if heads and part_at(*p) and part_at(*p) == part_at(*foot) else 0.0
                    else:
                        # a tee (or a sharp bend): touching that line - or a little short of it
                        # with ink joining the two (an end the opening shortened)
                        touch = max(3.0, (s["w"] + t["w"]) / 2 + 1.5)
                        if dist <= touch:
                            reach = touch
                        else:
                            foot = _nearest(t, p)
                            reach = tol if part_at(*p) and part_at(*p) == part_at(*foot) else touch
                    if dist <= reach and (best is None or dist < best[0]):
                        best = (dist, j)
                if best:
                    t = segs[best[1]]
                    if "d" in (s["o"], t["o"]):
                        corner = _nearest(t, p)
                    else:
                        corner = (t["c"], s["c"]) if s["o"] == "h" else (s["c"], t["c"])
                    join(i, best[1], corner)
                    taken.add((i, e))

    # 4b. an end touching another line: a bend or a tee - before a symbol just beyond it
    meet(wide=False)
    # 4c. symbols: an end pointing into one within the margin
    for i, s in enumerate(segs):
        for e in (0, 1):
            if (i, e) in taken:
                continue
            p, d = _end(s, e)
            k = symbol_ahead(p, d, margin + tol)
            if k is not None:
                ends_at[(i, e)] = (k, p)
                taken.add((i, e))
    # 4d. rounded bends: the two lines' ends a little apart
    meet(wide=True)

    # 4e. a loose end pointing at a symbol a little further on
    if near_miss:
        for i, s in enumerate(segs):
            for e in (0, 1):
                if (i, e) in taken:
                    continue
                p, d = _end(s, e)
                k = symbol_ahead(p, d, near_miss * size)
                if k is not None:
                    ends_at[(i, e)] = (k, p)

    # the pipe networks: segments joined to each other (a network with tees links several symbols)
    net = list(range(n))

    def root(i):
        while net[i] != i:
            net[i] = net[net[i]]
            i = net[i]
        return i
    for i, js in joins.items():
        for j, _, _ in js:
            net[root(i)] = root(j)

    # 5. from every symbol along the joined segments to the symbols reached
    at_symbol: dict[int, list] = {}
    for (i, e), (k, p) in ends_at.items():
        at_symbol.setdefault(k, []).append((i, e, p))
    seg_ends: dict[int, list] = {}
    for (i, e), (k, p) in ends_at.items():
        seg_ends.setdefault(i, []).append((k, e, p))
    found: dict[tuple[int, int], dict] = {}
    for a, starts in at_symbol.items():
        prev = {}
        queue = deque()
        for i, e, p in starts:
            if i not in prev:
                prev[i] = (None, p, False, e)
                queue.append(i)
        while queue:
            i = queue.popleft()
            for k, e, p in seg_ends.get(i, []):
                if k == a:
                    continue
                key = (min(a, k), max(a, k))
                # the path back to a: corner points, bridges counted, dashed segments
                pts, bridges, dashed, trail, j = [p], 0, False, [], i
                while j is not None:
                    back, q, br, _ = prev[j]
                    dashed |= segs[j]["dashed"]
                    trail.append(j)
                    pts.append(q)
                    bridges += br
                    j = back
                # along a free-shaped segment the path follows its drawn shape, not a chord
                shaped = [pts[0]]
                for t_i, seg_i in enumerate(trail):
                    sg = segs[seg_i]
                    if sg["o"] == "d" and t_i + 1 < len(pts):
                        u, v = pts[t_i], pts[t_i + 1]
                        iu = min(range(len(sg["pts"])), key=lambda k: math.dist(sg["pts"][k], u))
                        iv = min(range(len(sg["pts"])), key=lambda k: math.dist(sg["pts"][k], v))
                        step = 1 if iv >= iu else -1
                        shaped += sg["pts"][iu:iv + step:step] if iv != iu else []
                    if t_i + 1 < len(pts):
                        shaped.append(pts[t_i + 1])
                pts = shaped[::-1]
                length = sum(math.dist(u, v) for u, v in zip(pts, pts[1:]))
                if key not in found or len(trail) < found[key]["_n"]:
                    found[key] = {"a": key[0], "b": key[1], "bridges": bridges + (2 if dashed else 0),
                                  "length": length, "path": pts if a == key[0] else pts[::-1], "_n": len(trail),
                                  "net": root(i),
                                  "_from": a,
                                  "_first": trail[-1], "_last": i, "_e_last": e}
            for j, q, br in joins[i]:
                if j not in prev:
                    prev[j] = (i, q, br, None)
                    queue.append(j)

    raw_ink = full_ink

    def arrow_at(i, e) -> bool:
        s = segs[i]
        if s["o"] == "d":
            pts = np.array([(int(round(y)), int(round(x))) for x, y in s["pts"]])
            return len(pts) >= 2 and bool(_arrowhead_at(raw_ink, pts, e, margin))
        lo, hi = int(s["a"]), int(s["b"])
        c = int(round(s["c"]))
        pts = np.array([(c, v) for v in range(lo, hi + 1)] if s["o"] == "h" else [(v, c) for v in range(lo, hi + 1)])
        return len(pts) >= 2 and bool(_arrowhead_at(raw_ink, pts, e, margin))

    for (a, b), link in _short_links(raw_ink, boxes, size, margin).items():
        if (a, b) not in found:
            found[(a, b)] = link

    out = []
    for c in found.values():
        if "_n" not in c:
            out.append(c)
            continue
        # flow: an arrowhead where the line meets b (+1) or a (-1)
        start = c["_from"]
        first, last, e_last = c["_first"], c["_last"], c["_e_last"]
        e_first = next((e for i, e, p in at_symbol[start] if i == first), None)
        votes = int(arrow_at(last, e_last)) - (int(arrow_at(first, e_first)) if e_first is not None else 0)
        if start != c["a"]:
            votes = -votes
        c["flow"] = 1 if votes > 0 else -1 if votes < 0 else 0
        path = c["path"]
        if arrows and c["flow"] == 0 and len(path) >= 2:
            for end_idx, sign in ((c["b"], 1), (c["a"], -1)):
                if end_idx in arrows:
                    tip = _arrow_tip(raw_ink, boxes[end_idx])
                    near, far = (path[-1], path[-2]) if sign == 1 else (path[0], path[1])
                    v = (near[0] - far[0], near[1] - far[1])
                    if tip is not None and math.hypot(*v):
                        along = (v[0] * tip[0] + v[1] * tip[1]) / math.hypot(*v)
                        if abs(along) > 0.5:
                            c["flow"] = sign if along > 0 else -sign
                            break
        c["line_type"] = "signal" if c["bridges"] >= 2 else "process"
        out.append({k: v for k, v in c.items() if not k.startswith("_")})
    if missed is not None:  # symbols on a pipe with no box: trace_pipes' outline-knot check
        from trace_pipes import trace_connections
        trace_connections(img, boxes, text_boxes, dark=dark, faint=faint, arrows=None, missed=missed)
    return out
