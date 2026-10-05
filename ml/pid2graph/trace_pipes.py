"""Trace pipes across a WHOLE sheet to find which symbols they connect.

The model only sees one tile at a time, so it can only link symbols that share a tile;
a pipe longer than a tile is invisible to it. This module does not tile: it follows the
drawn lines over the full sheet with classical image processing.

  1. ink = dark pixels; text (OCR boxes) and the inside of every symbol box are erased,
     leaving only linework; specks smaller than `min_component` px are dropped.
  2. The linework is thinned to a 1-px skeleton and cut at junction pixels into segments.
  3. Each segment end is attached to whatever it touches: a symbol (its box grown by
     `margin`), a junction, or nothing (a free end).
  4. Free ends that point at each other across a short gap are bridged - dashed signal
     lines are long runs of such gaps, so a path that needed bridges is a signal line.
  5. From every symbol, walk the segments: at a 4-way junction whose arms form two
     straight lines (pipes merely drawn across each other) continue straight on only;
     at any other junction (a real branch / T) continue down every arm. Reaching another
     symbol = a connection.

Runs in the ML venv (needs scipy + scikit-image). Coordinates are image pixels.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def _direction(points: np.ndarray, from_end: int) -> tuple[float, float]:
    """Unit vector pointing OUT of a segment at one end, from up to 12 px of its pixels."""
    pts = points if from_end == 0 else points[::-1]
    a, b = pts[0], pts[min(len(pts) - 1, 12)]
    v = (a[0] - b[0], a[1] - b[1])
    n = math.hypot(*v) or 1.0
    return (v[0] / n, v[1] / n)


def _order(pixels: np.ndarray) -> np.ndarray:
    """Order the pixels of a simple (non-branching) skeleton segment end to end."""
    if len(pixels) <= 2:
        return pixels
    s = {tuple(p) for p in pixels}
    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (p[0] + dy, p[1] + dx) in s]
    start = next((p for p in s if len(nbrs(p)) <= 1), next(iter(s)))
    out, prev, cur = [start], None, start
    while True:
        nxt = [q for q in nbrs(cur) if q != prev and q not in out[-3:]]
        if not nxt or len(out) > len(s):
            break
        prev, cur = cur, nxt[0]
        out.append(cur)
    return np.array(out)


def trace_connections(gray: np.ndarray, boxes: list, text_boxes: list | None = None, *,
                      dark: int = 140, margin: float | None = None, gap: float | None = None,
                      min_component: int = 40, straight_tol_deg: float = 25.0):
    """gray: HxW uint8 sheet. boxes: [(x1, y1, x2, y2)] symbols. Returns a list of
    {"a": i, "b": j, "line_type": "process"|"signal", "bridges": n, "length": px}."""
    H, W = gray.shape
    sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes] or [40.0]
    size = float(np.median(sizes))
    margin = margin if margin is not None else max(3.0, 0.12 * size)
    gap = gap if gap is not None else max(6.0, 0.35 * size)

    ink = gray < dark
    for tb in text_boxes or []:
        x1, y1, x2, y2 = [int(round(v)) for v in tb]
        ink[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2] = False
    sym = np.zeros((H, W), np.int32)
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        sym[max(0, int(y1 - margin)):int(y2 + margin) + 1, max(0, int(x1 - margin)):int(x2 + margin) + 1] = k + 1
    lines = ink & (sym == 0)
    lab, n = ndi.label(lines, structure=np.ones((3, 3)))
    if n:
        counts = np.bincount(lab.ravel())
        keep = counts >= min_component
        keep[0] = False
        lines = keep[lab]
    skel = skeletonize(lines)
    deg = ndi.convolve(skel.astype(np.uint8), _NEIGH, mode="constant") * skel
    junction_px = skel & (deg >= 3)
    jlab, n_j = ndi.label(ndi.binary_dilation(junction_px, iterations=2) & skel, structure=np.ones((3, 3)))
    seg_mask = skel & (jlab == 0)
    slab, n_s = ndi.label(seg_mask, structure=np.ones((3, 3)))
    # Symbol regions grown by 2 px so a segment ending at the margin edge touches them.
    sym_touch = ndi.grey_dilation(sym, size=(5, 5))
    j_touch = ndi.grey_dilation(jlab, size=(5, 5))

    seg_pixels = ndi.find_objects(slab)
    ends: dict[int, list] = {}  # seg -> [(kind, id, direction, (y, x)) for each end]
    for s in range(1, n_s + 1):
        sl = seg_pixels[s - 1]
        ys, xs = np.nonzero(slab[sl] == s)
        pts = _order(np.stack([ys + sl[0].start, xs + sl[1].start], axis=1))
        if len(pts) < 2:
            continue
        seg_ends = []
        for e, (y, x) in ((0, pts[0]), (1, pts[-1])):
            if sym_touch[y, x]:
                seg_ends.append(("sym", int(sym_touch[y, x]) - 1, _direction(pts, e), (y, x)))
            elif j_touch[y, x]:
                seg_ends.append(("junc", int(j_touch[y, x]), _direction(pts, e), (y, x)))
            else:
                seg_ends.append(("free", None, _direction(pts, e), (y, x)))
        ends[s] = (seg_ends, len(pts))

    # 4. bridge free ends across short, aligned gaps (dashes; small breaks in scans)
    free = [(s, e) for s, (se, _) in ends.items() for e in (0, 1) if se[e][0] == "free"]
    bridge: dict[tuple[int, int], tuple[int, int]] = {}
    pts_arr = np.array([ends[s][0][e][3] for s, e in free], dtype=float).reshape(-1, 2)
    for i, (s, e) in enumerate(free):
        if (s, e) in bridge:
            continue
        y, x = ends[s][0][e][3]
        d = ends[s][0][e][2]
        dist = np.hypot(pts_arr[:, 0] - y, pts_arr[:, 1] - x)
        best = None
        for j in np.argsort(dist):
            if dist[j] > gap:
                break
            s2, e2 = free[j]
            if s2 == s or (s2, e2) in bridge:
                continue
            v = ((pts_arr[j, 0] - y) / (dist[j] or 1), (pts_arr[j, 1] - x) / (dist[j] or 1))
            d2 = ends[s2][0][e2][2]
            # the gap must continue the line (pointing out of this end, into the other)
            if v[0] * d[0] + v[1] * d[1] > 0.8 and d[0] * d2[0] + d[1] * d2[1] < -0.8:
                best = (s2, e2)
                break
        if best:
            bridge[(s, e)] = best
            bridge[best] = (s, e)

    # junction -> incident (segment, end) arms, and pass-through pairing for 4-way crossings
    arms: dict[int, list] = defaultdict(list)
    for s, (se, _) in ends.items():
        for e in (0, 1):
            if se[e][0] == "junc":
                arms[se[e][1]].append((s, e))
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    for j, a in arms.items():
        if len(a) != 4:
            continue
        ang = sorted(((math.degrees(math.atan2(ends[s][0][e][2][0], ends[s][0][e][2][1])) + 180) % 360, (s, e))
                     for s, e in a)
        opp = lambda i, k: abs(((ang[k][0] - ang[i][0]) % 360) - 180) <= straight_tol_deg
        if opp(0, 2) and opp(1, 3):
            for i, k in ((0, 2), (1, 3)):
                partner[ang[i][1]] = ang[k][1]
                partner[ang[k][1]] = ang[i][1]

    # 5. walk from every symbol
    found: dict[tuple[int, int], dict] = {}
    for s0, (se0, _) in ends.items():
        for e0 in (0, 1):
            if se0[e0][0] != "sym":
                continue
            start = se0[e0][1]
            stack = [(s0, e0, 0, 0)]  # entered segment s0 at end e0; bridges, length so far
            seen = set()
            while stack:
                s, e_in, nb, length = stack.pop()
                if (s, e_in) in seen:
                    continue
                seen.add((s, e_in))
                seg_ends, seg_len = ends[s]
                e_out = 1 - e_in
                kind, ident, _, _ = seg_ends[e_out]
                length += seg_len
                if kind == "sym":
                    if ident != start:
                        key = (min(start, ident), max(start, ident))
                        prev = found.get(key)
                        if prev is None or length < prev["length"]:
                            found[key] = {"a": key[0], "b": key[1], "bridges": nb, "length": length}
                elif kind == "free" and (s, e_out) in bridge:
                    s2, e2 = bridge[(s, e_out)]
                    stack.append((s2, e2, nb + 1, length))
                elif kind == "junc":
                    if (s, e_out) in partner:
                        stack.append((*partner[(s, e_out)], nb, length))
                    else:
                        for s2, e2 in arms[ident]:
                            if (s2, e2) != (s, e_out):
                                stack.append((s2, e2, nb, length))
    out = []
    for c in found.values():
        c["line_type"] = "signal" if c["bridges"] >= 2 else "process"
        out.append(c)
    return out
