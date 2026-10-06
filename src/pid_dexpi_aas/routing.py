"""Orthogonal routing for lines that have no drawn route: lines the reviewer added or
re-pointed, bypasses, and the model's predicted connections the tracer did not follow.

Like draw.io's own connectors: horizontal and vertical runs only, leaving and entering the
items at the sides facing each other, going around every other item, with as few bends as
possible. A* over a grid (cell ~ a tenth of a typical item) inside the area around the two
items; when no way around is found the plain one-bend elbow is used.
"""
from __future__ import annotations

import heapq
import math
from functools import lru_cache

Box = tuple  # (x1, y1, x2, y2)


def _side_point(box: Box, toward: tuple[float, float], gap: float) -> tuple[tuple[float, float], tuple[int, int]]:
    """Mid-point of the side of `box` facing `toward`, and the outward direction."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if abs(dx) * (y2 - y1) >= abs(dy) * (x2 - x1):  # leave sideways
        return ((x2 + gap, cy), (1, 0)) if dx > 0 else ((x1 - gap, cy), (-1, 0))
    return ((cx, y2 + gap), (0, 1)) if dy > 0 else ((cx, y1 - gap), (0, -1))


def _elbow(a: Box, b: Box) -> list:
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    pa, da = _side_point(a, cb, 0)
    pb, db = _side_point(b, ca, 0)
    corner = (pb[0], pa[1]) if da[0] else (pa[0], pb[1])
    return [pa, corner, pb]


def route(a: Box, b: Box, obstacles: list[Box], cell: float | None = None, margin: float = 160.0) -> list:
    """Points from item box a to item box b (both excluded from the obstacles)."""
    return list(_route(tuple(map(float, a)), tuple(map(float, b)),
                       tuple(sorted(tuple(map(float, o)) for o in obstacles)), cell, margin))


def _corners(pts: list) -> list:
    """Drop repeated points and points in the middle of a straight run."""
    out = []
    for p in pts:
        if out and math.dist(p, out[-1]) < 1e-6:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            if (abs(a[0] - b[0]) < 1e-6 and abs(b[0] - p[0]) < 1e-6) or (abs(a[1] - b[1]) < 1e-6 and abs(b[1] - p[1]) < 1e-6):
                out[-1] = p
                continue
        out.append(p)
    return out


@lru_cache(maxsize=4096)
def _route(a: Box, b: Box, obstacles: tuple, cell: float | None, margin: float) -> tuple:
    sizes = [max(o[2] - o[0], o[3] - o[1]) for o in obstacles] or [max(a[2] - a[0], a[3] - a[1])]
    sizes.sort()
    cell = cell or max(6.0, sizes[len(sizes) // 2] / 8)
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    pa, da = _side_point(a, cb, cell)
    pb, db = _side_point(b, ca, cell)
    # the grid is aligned so the start point sits on a cell centre (no jog at the start)
    gx0 = pa[0] - cell * (math.ceil((pa[0] - min(a[0], b[0]) + margin) / cell) + 0.5)
    gy0 = pa[1] - cell * (math.ceil((pa[1] - min(a[1], b[1]) + margin) / cell) + 0.5)
    gx1 = max(a[2], b[2]) + margin
    gy1 = max(a[3], b[3]) + margin
    W, H = int((gx1 - gx0) / cell) + 1, int((gy1 - gy0) / cell) + 1
    to_cell = lambda p: (min(W - 1, max(0, int((p[0] - gx0) / cell))), min(H - 1, max(0, int((p[1] - gy0) / cell))))
    pad = cell * 0.6
    blocked = set()
    for o in obstacles:
        if o in (a, b) or o[2] < gx0 or o[0] > gx1 or o[3] < gy0 or o[1] > gy1:
            continue
        i0, j0 = to_cell((o[0] - pad, o[1] - pad))
        i1, j1 = to_cell((o[2] + pad, o[3] + pad))
        blocked.update((i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))
    s, t = to_cell(pa), to_cell(pb)
    blocked.discard(s)
    blocked.discard(t)
    # A* over (cell, heading); a bend costs as much as a few steps
    BEND = 4
    start = (s, da)
    best = {start: 0}
    came = {}
    heap = [(abs(s[0] - t[0]) + abs(s[1] - t[1]), 0, start)]
    goal = None
    steps = 0
    while heap and steps < 60_000:
        steps += 1
        _, g, (c, d) = heapq.heappop(heap)
        if c == t:
            goal = (c, d)
            break
        if g > best.get((c, d), float("inf")):
            continue
        for nd in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if nd == (-d[0], -d[1]):
                continue
            n = (c[0] + nd[0], c[1] + nd[1])
            if not (0 <= n[0] < W and 0 <= n[1] < H) or n in blocked:
                continue
            ng = g + 1 + (BEND if nd != d else 0)
            if ng < best.get((n, nd), float("inf")):
                best[(n, nd)] = ng
                came[(n, nd)] = (c, d)
                heapq.heappush(heap, (ng + abs(n[0] - t[0]) + abs(n[1] - t[1]), ng, (n, nd)))
    if goal is None:
        return tuple(_elbow(a, b))
    cells = [goal[0]]
    k = goal
    while k in came:
        k = came[k]
        cells.append(k[0])
    cells.reverse()
    centre = lambda c: (gx0 + (c[0] + 0.5) * cell, gy0 + (c[1] + 0.5) * cell)
    pts = [centre(c) for c in cells]
    # snap the first / last run onto the side mid-points, then keep only the corners
    pts[0] = (pa[0], pts[0][1]) if da[1] else (pts[0][0], pa[1])
    pts[-1] = (pb[0], pts[-1][1]) if db[1] else (pts[-1][0], pb[1])
    ea = (pa[0] - da[0] * cell, pa[1] - da[1] * cell)
    eb = (pb[0] - db[0] * cell, pb[1] - db[1] * cell)
    full = [ea, pa] + pts + [pb, eb]
    out = [full[0]]
    for p, q in zip(full[1:], full[2:]):
        prev = out[-1]
        if not ((abs(prev[0] - p[0]) < 1e-6 and abs(p[0] - q[0]) < 1e-6) or
                (abs(prev[1] - p[1]) < 1e-6 and abs(p[1] - q[1]) < 1e-6)):
            out.append(p)
    out.append(full[-1])
    # make every run axis-aligned (cell centres vs side points can differ slightly)
    fixed = [out[0]]
    for p in out[1:]:
        q = fixed[-1]
        if abs(p[0] - q[0]) > 1e-6 and abs(p[1] - q[1]) > 1e-6:
            fixed.append((p[0], q[1]))
        fixed.append(p)
    fixed = _corners(fixed)
    # a jog shorter than a cell just before the end (cell centre vs side point): move the whole
    # last run onto the end's line - its start sits on a perpendicular run, so it stays square
    if len(fixed) >= 5:
        p, q, r = fixed[-4], fixed[-3], fixed[-2]
        if math.dist(q, r) < cell:
            if abs(p[1] - q[1]) < 1e-6 and abs(q[0] - r[0]) < 1e-6:  # horizontal run, vertical jog
                fixed[-4], fixed[-3] = (p[0], r[1]), r
            elif abs(p[0] - q[0]) < 1e-6 and abs(q[1] - r[1]) < 1e-6:  # vertical run, horizontal jog
                fixed[-4], fixed[-3] = (r[0], p[1]), r
    clean = _corners(fixed)
    return tuple(clean)
