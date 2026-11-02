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
     A gap where an erased label sat on the line ("17 bar" written on a pipe) is bridged
     too, however wide the label.
  5. From every symbol, walk the segments: at a 4-way junction whose arms form two
     straight lines (pipes merely drawn across each other) continue straight on only;
     at any other junction (a real branch / T) continue down every arm. Reaching another
     symbol = a connection.
  6. Symbols close together (stacked instruments, a gas bottle under its valve) have a
     line shorter than the erased margins, so nothing is left to walk: facing sides less
     than ~0.6 of a symbol apart are checked directly for a thin straight line across -
     solid, or dashed (instrument chains: TT over TIC; a signal line).
  7. Dotted / finely dashed lines (instrument signal lines) have dashes smaller than
     `min_component`: runs of them, straight or bent, are joined up into a signal line.
  8. Frames and containers (boxes >= 3x the usual symbol) keep only their border, so lines
     inside reach the symbols there (`containers`).
  9. A flow arrow drawn mid-pipe (a box of class arrow, `arrows`): which way its tip points,
     against the pipe coming in, gives the flow where no arrowhead at a symbol did.
 10. A symbol drawn on a pipe that has no box: the pipe runs through its outline, and the
     skeleton shows a knot - junctions joined by short strokes with exactly two lines
     leaving. Reported in `missed` (box + the connection through it) for review.

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
                      dark: int = 210, margin: float | None = None, gap: float | None = None,
                      min_component: int = 15, straight_tol_deg: float = 25.0, graze: bool = True,
                      near_miss: float = 0.6, dotted: bool = True, loose: dict | None = None,
                      keep_lines_in_text: bool = True, containers: bool = True, arrows: set | None = None,
                      missed: list | None = None, faint: int = 235):
    """gray: HxW uint8 sheet, or HxWx3 colour (lines in colour are then found too - _ink). boxes: [(x1, y1, x2, y2)] symbols.

    Tuned 2026-10-15 on the 6 team sheets (446 connections, true boxes): ink < 210 (was 140:
    thin, anti-aliased lines were lost), specks >= 15 px kept (was 40), and a line ending just
    short of a symbol, pointing at it, ends at it (near_miss, x the median symbol size):
    recall 0.72 -> 0.80 at the same precision 0.93 (F1 0.81 -> 0.86). Joining dotted lines
    (dotted=True) gained nothing there, so it is off.

    keep_lines_in_text: a line running straight through a text (an instrument bubble's tag
    on the line between PI and PC) is not erased with the text.
    containers: a box much bigger than a symbol (a reactor with compartments) is a frame -
    only its border is a symbol, so lines inside reach what is in it.
    arrows: indices of flow-arrow boxes - a connection ending at one gets the arrow's
    direction (flow +1: a -> b), read from its drawing (the tip side is narrower).
    missed (a list, filled in if given): {"box", "a", "b"} where a traced pipe crosses
    linework too dense to be a pipe - a symbol no box was given for (the walk would join
    its two neighbours straight across it).

    loose (a dict, filled in if given): {symbol index: number of lines leaving it that the
    walk lost - ended loose, nowhere} - where a connection the tracer could not follow is
    likely (review.from_image weighs the model's own predictions by it).

    Returns a list of
    {"a": i, "b": j, "line_type": "process"|"signal", "bridges": n, "length": px}."""
    full_ink = _ink(gray, dark, faint)
    H, W = full_ink.shape
    sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes] or [40.0]
    size = float(np.median(sizes))
    margin = margin if margin is not None else max(3.0, 0.12 * size)
    gap = gap if gap is not None else max(6.0, 0.35 * size)

    ink = _erase_text(full_ink.copy(), text_boxes, size, keep_lines_in_text)
    sym = _symbol_map(boxes, size, margin, (H, W), containers)
    big = {k for k, b in enumerate(boxes) if containers and max(b[2] - b[0], b[3] - b[1]) >= 3.0 * size}
    lines = ink & (sym == 0)
    lab, n = ndi.label(lines, structure=np.ones((3, 3)))
    dotted_mask = np.zeros((H, W), bool)
    if n:
        counts = np.bincount(lab.ravel())
        keep = counts >= min_component
        keep[0] = False
        if dotted:
            # dotted / finely dashed lines: their dots are specks, dropped below - unless
            # several sit in a row: then the row, joined up, is a line (a signal line)
            dots = (~keep[lab]) & lines & (counts[lab] >= 2)
            joined = ndi.binary_dilation(dots, iterations=2)
            jl, jn = ndi.label(joined, structure=np.ones((3, 3)))
            for k, sl in enumerate(ndi.find_objects(jl), start=1):
                if sl is None:
                    continue
                h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                n_dots = len(np.unique(lab[sl][(jl[sl] == k) & dots[sl]]))
                area = int((jl[sl] == k).sum())
                # long enough, and thin: a straight or bent run of dots (its area about the
                # joined-up line's width times its length), not a cluster of specks
                if h + w >= 0.6 * size and area <= 7 * (h + w) and n_dots >= 4:
                    dotted_mask[sl] |= jl[sl] == k
        lines = keep[lab] | dotted_mask
    skel = skeletonize(lines)
    deg = ndi.convolve(skel.astype(np.uint8), _NEIGH, mode="constant") * skel
    junction_px = skel & (deg >= 3)
    jlab, n_j = ndi.label(ndi.binary_dilation(junction_px, iterations=2) & skel, structure=np.ones((3, 3)))
    seg_mask = skel & (jlab == 0)
    slab, n_s = ndi.label(seg_mask, structure=np.ones((3, 3)))
    # Symbol regions grown by 2 px so a segment ending at the margin edge touches them.
    sym_touch = ndi.grey_dilation(sym, size=(5, 5))
    # An arrowhead sticking out past the margin thins into a little junction blob right
    # at the symbol: a line ending in such a blob next to a symbol ends at the symbol.
    sym_near = ndi.grey_dilation(sym, size=(15, 15))
    j_touch = ndi.grey_dilation(jlab, size=(5, 5))

    seg_pixels = ndi.find_objects(slab)
    dotted_seg = set()
    ends: dict[int, list] = {}  # seg -> [(kind, id, direction, (y, x)) for each end]
    seg_pts: dict[int, np.ndarray] = {}  # seg -> its pixels in order, end 0 -> end 1
    for s in range(1, n_s + 1):
        sl = seg_pixels[s - 1]
        ys, xs = np.nonzero(slab[sl] == s)
        pts = _order(np.stack([ys + sl[0].start, xs + sl[1].start], axis=1))
        if len(pts) < 2:
            continue
        seg_ends = []
        for e, (y, x) in ((0, pts[0]), (1, pts[-1])):
            if graze and sym_touch[y, x] and _grazes(boxes[int(sym_touch[y, x]) - 1], (y, x), _direction(pts, e), margin):
                # maybe running past the symbol's side (a frame, another pipe): decided in step 4
                seg_ends.append(("graze", int(sym_touch[y, x]) - 1, _direction(pts, e), (y, x)))
            elif sym_touch[y, x]:
                seg_ends.append(("sym", int(sym_touch[y, x]) - 1, _direction(pts, e), (y, x)))
            elif j_touch[y, x] and sym_near[y, x] and not _grazes(boxes[int(sym_near[y, x]) - 1], (y, x), _direction(pts, e), margin):
                # (heading into the symbol: an arrowhead's blob; running past it: a tee beside it)
                seg_ends.append(("sym", int(sym_near[y, x]) - 1, _direction(pts, e), (y, x)))
            elif j_touch[y, x]:
                seg_ends.append(("junc", int(j_touch[y, x]), _direction(pts, e), (y, x)))
            else:
                seg_ends.append(("free", None, _direction(pts, e), (y, x)))
        ends[s] = (seg_ends, len(pts))
        seg_pts[s] = pts
        if dotted and dotted_mask[pts[:, 0], pts[:, 1]].mean() > 0.5:
            dotted_seg.add(s)

    # 4. bridge free ends across short, aligned gaps (dashes; small breaks in scans), and
    # across erased labels written on the line, whatever their width
    text_mask = np.zeros((H, W), bool)
    for tb in text_boxes or []:
        x1, y1, x2, y2 = [int(round(v)) for v in tb]
        text_mask[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = True
    text_gap = max(gap, 3.0 * size)

    def over_text(p, q) -> bool:
        return any(text_mask[int(round(p[0] + (q[0] - p[0]) * t)), int(round(p[1] + (q[1] - p[1]) * t))]
                   for t in np.linspace(0.1, 0.9, 9))

    inside = np.zeros((H, W), bool)  # the symbols themselves, without their margins
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        if k not in big:
            inside[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1] = True

    def past_symbol(p, q) -> bool:
        """The gap runs through a symbol's margin but not into the symbol: a line that
        was cut where it passes alongside it."""
        samples = [(int(round(p[0] + (q[0] - p[0]) * t)), int(round(p[1] + (q[1] - p[1]) * t)))
                   for t in np.linspace(0.05, 0.95, 19)]
        return any(sym[y, x] for y, x in samples) and not any(inside[y, x] for y, x in samples)

    free = [(s, e) for s, (se, _) in ends.items() for e in (0, 1) if se[e][0] in ("free", "graze")]
    bridge: dict[tuple[int, int], tuple[int, int]] = {}
    text_bridge: set = set()  # bridges over a label: not a dash
    pts_arr = np.array([ends[s][0][e][3] for s, e in free], dtype=float).reshape(-1, 2)
    for i, (s, e) in enumerate(free):
        if (s, e) in bridge:
            continue
        y, x = ends[s][0][e][3]
        d = ends[s][0][e][2]
        dist = np.hypot(pts_arr[:, 0] - y, pts_arr[:, 1] - x)
        best = None
        for j in np.argsort(dist):
            if dist[j] > text_gap:
                break
            s2, e2 = free[j]
            if s2 == s or (s2, e2) in bridge:
                continue
            v = ((pts_arr[j, 0] - y) / (dist[j] or 1), (pts_arr[j, 1] - x) / (dist[j] or 1))
            d2 = ends[s2][0][e2][2]
            # the gap must continue the line (pointing out of this end, into the other)
            if v[0] * d[0] + v[1] * d[1] > 0.8 and d[0] * d2[0] + d[1] * d2[1] < -0.8:
                grazing = "graze" in (ends[s][0][e][0], ends[s2][0][e2][0])
                if past_symbol((y, x), pts_arr[j]) or (not grazing and (dist[j] <= gap or over_text((y, x), pts_arr[j]))):
                    best, best_d = (s2, e2), dist[j]
                    break
        if best:
            bridge[(s, e)] = best
            bridge[best] = (s, e)
            if best_d > gap:
                text_bridge.add((s, e))
                text_bridge.add(best)

    # a grazing end with a continuation past the symbol is a line running by: free (bridged);
    # without one it heads into the symbol after all (along its edge, or bending in the margin)
    for s, (se, n_px) in list(ends.items()):
        for e in (0, 1):
            if se[e][0] == "graze":
                if (s, e) in bridge:
                    se[e] = ("free", None, se[e][2], se[e][3])
                else:
                    se[e] = ("sym", *se[e][1:])

    # a line stopping just short of a symbol, pointing at it (a loose end, often with its
    # arrowhead short of the box): it ends at that symbol - draw.io's own reading of loose ends
    if near_miss:
        reach = near_miss * size
        for s, (se, n_px) in ends.items():
            for e in (0, 1):
                kind, _, d, (y, x) = se[e]
                if kind != "free" or (s, e) in bridge:
                    continue
                best = None
                for k, (x1, y1, x2, y2) in enumerate(boxes):
                    cy, cx = min(max(y, y1), y2), min(max(x, x1), x2)   # nearest point of the box
                    dist = math.hypot(cy - y, cx - x)
                    if 0 < dist <= reach and ((cy - y) * d[0] + (cx - x) * d[1]) / dist > 0.7 and (best is None or dist < best[0]):
                        best = (dist, k)
                if best:
                    se[e] = ("sym", best[1], d, (y, x))

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
        # two straight lines drawn across each other - or one straight line with the other
        # hopping over it (a jump-over arc: the arc's arms leave the top of the hump bending
        # away, not straight); a P&ID has no 4-way joins, so the other two arms are one line
        if opp(0, 2) or opp(1, 3):
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
            # entered segment s0 at end e0; bridges, length so far, trail of (segment, entry end)
            stack = [(s0, e0, 0, 0, ())]
            seen = set()
            while stack:
                s, e_in, nb, length, trail = stack.pop()
                if (s, e_in) in seen:
                    continue
                seen.add((s, e_in))
                trail = trail + ((s, e_in),)
                if s in dotted_seg:
                    nb += 2  # drawn dotted: a signal line
                seg_ends, seg_len = ends[s]
                e_out = 1 - e_in
                kind, ident, _, _ = seg_ends[e_out]
                length += seg_len
                if kind == "sym":
                    if ident != start:
                        key = (min(start, ident), max(start, ident))
                        prev = found.get(key)
                        if prev is None or length < prev["length"]:
                            found[key] = {"a": key[0], "b": key[1], "bridges": nb, "length": length,
                                          "_from": start, "_trail": trail}
                elif kind == "free" and (s, e_out) in bridge:
                    s2, e2 = bridge[(s, e_out)]
                    stack.append((s2, e2, nb + ((s, e_out) not in text_bridge), length, trail))
                elif kind == "free" and loose is not None and length > 0.5 * size:
                    loose[start] = loose.get(start, 0) + 1  # a line out of start, lost on the way
                elif kind == "junc":
                    if (s, e_out) in partner:
                        stack.append((*partner[(s, e_out)], nb, length, trail))
                    else:
                        for s2, e2 in arms[ident]:
                            if (s2, e2) != (s, e_out):
                                stack.append((s2, e2, nb, length, trail))
    raw_ink = full_ink  # arrowheads are measured on the full ink, symbols not erased
    arrow_cache: dict[tuple[int, int], bool] = {}

    def has_arrow(seg: int, end: int) -> bool:
        # only where a line meets a symbol: at a tee the crossing pipe looks just as wide
        if ends[seg][0][end][0] != "sym":
            return False
        if (seg, end) not in arrow_cache:
            arrow_cache[(seg, end)] = bool(_arrowhead_at(raw_ink, seg_pts[seg], end, margin))
        return arrow_cache[(seg, end)]

    # 6. symbols close together: a thin straight line across the gap between facing sides
    for (a, b), link in _short_links(raw_ink, boxes, size, margin).items():
        if (a, b) not in found:
            found[(a, b)] = link

    knot_boxes = _knots(ends, arms, seg_pts, size, bridge) if missed is not None else []
    out = []
    for c in found.values():
        if "_trail" not in c:  # a short link: no walked segments
            out.append(c)
            continue
        trail = c.pop("_trail")
        start = c.pop("_from")
        # the path, oriented start -> end of the walk, every 3rd pixel, as (x, y)
        path, votes = [], 0
        for seg, e_in in trail:
            pts = seg_pts[seg] if e_in == 0 else seg_pts[seg][::-1]
            path += [(float(x), float(y)) for y, x in pts[::3]] + [(float(pts[-1][1]), float(pts[-1][0]))]
            votes += int(has_arrow(seg, 1 - e_in)) - int(has_arrow(seg, e_in))  # arrow at the far end: flows onward
        if start != c["a"]:  # report a -> b
            path, votes = path[::-1], -votes
        c["path"] = path
        c["flow"] = 1 if votes > 0 else -1 if votes < 0 else 0  # +1 a -> b, -1 b -> a, 0 unknown
        if arrows and c["flow"] == 0 and path:
            # a flow arrow at one end: which way it points, against the pipe coming in
            for end_idx, sign in ((c["b"], 1), (c["a"], -1)):
                if end_idx in arrows:
                    tip = _arrow_tip(raw_ink, boxes[end_idx])
                    near = path[-1] if sign == 1 else path[0]
                    far = path[max(0, len(path) - 6)] if sign == 1 else path[min(len(path) - 1, 5)]
                    v = (near[0] - far[0], near[1] - far[1])   # the pipe's direction into the arrow
                    if tip is not None and math.hypot(*v):
                        along = (v[0] * tip[0] + v[1] * tip[1]) / math.hypot(*v)
                        if abs(along) > 0.5:
                            c["flow"] = sign if along > 0 else -sign   # tip pointing on: flowing into it
                            break
        if missed is not None:
            for box in knot_boxes:
                if any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in path):
                    missed.append({"box": box, "a": c["a"], "b": c["b"]})
        c["line_type"] = "signal" if c["bridges"] >= 2 else "process"
        out.append(c)
    return out


def _erase_text(ink: np.ndarray, text_boxes, size: float, keep_lines_in_text: bool = True) -> np.ndarray:
    """ink with the text (OCR boxes) erased - except a row (column) inked well beyond both
    sides of the text: a line passing through it (keep_lines_in_text)."""
    H, W = ink.shape
    reach = max(4, int(0.25 * size))
    for tb in text_boxes or []:
        x1, y1, x2, y2 = [int(round(v)) for v in tb]
        ya, yb, xa, xb = max(0, y1 - 2), min(H, y2 + 2), max(0, x1 - 2), min(W, x2 + 2)
        keep = np.zeros((max(0, yb - ya), max(0, xb - xa)), bool)
        if keep_lines_in_text and yb > ya and xb > xa:
            left, right = ink[ya:yb, max(0, xa - reach):xa], ink[ya:yb, xb:min(W, xb + reach)]
            rows = (left.mean(axis=1) > 0.8 if left.size else False) & (right.mean(axis=1) > 0.8 if right.size else False)
            top, bottom = ink[max(0, ya - reach):ya, xa:xb], ink[yb:min(H, yb + reach), xa:xb]
            cols = (top.mean(axis=0) > 0.8 if top.size else False) & (bottom.mean(axis=0) > 0.8 if bottom.size else False)
            keep[np.asarray(rows, bool) if np.ndim(rows) else np.zeros(yb - ya, bool), :] = True
            keep[:, np.asarray(cols, bool) if np.ndim(cols) else np.zeros(xb - xa, bool)] = True
        ink[ya:yb, xa:xb] &= keep
    return ink


def _symbol_map(boxes: list, size: float, margin: float, shape, containers: bool = True) -> np.ndarray:
    """(H, W) int map: k + 1 over symbol k's box grown by `margin`; a frame / container (>= 3x
    the usual symbol) only over its border band, so what is inside stays (painted over it)."""
    H, W = shape
    sym = np.zeros((H, W), np.int32)
    big = [k for k, b in enumerate(boxes) if containers and max(b[2] - b[0], b[3] - b[1]) >= 3.0 * size]
    order = big + [k for k in range(len(boxes)) if k not in big]   # frames first: what is inside paints over
    for k in order:
        x1, y1, x2, y2 = boxes[k]
        ya, yb = max(0, int(y1 - margin)), int(y2 + margin) + 1
        xa, xb = max(0, int(x1 - margin)), int(x2 + margin) + 1
        if k in big:  # a frame: its border band only
            band = int(2 * margin) + 1
            for sl in ((slice(ya, ya + band), slice(xa, xb)), (slice(max(ya, yb - band), yb), slice(xa, xb)),
                       (slice(ya, yb), slice(xa, xa + band)), (slice(ya, yb), slice(max(xa, xb - band), xb))):
                sym[sl] = k + 1
        else:
            sym[ya:yb, xa:xb] = k + 1
    return sym


def _ink(img: np.ndarray, dark: int, faint: int) -> np.ndarray:
    """The drawing's linework, also where a line changes colour along the way. img: gray
    (H, W) or colour (H, W, 3).
      - dark: brightness below `dark` (as always);
      - coloured: a colour picture's darkest channel below `dark` - a yellow, cyan or light
        green line, whose brightness (a weighted mix of the channels) is near white;
      - light gray: brightness below `faint`.
    Coloured and light gray pixels only count as thin line pieces: not a fill (a coloured
    frame's, a shaded area) and not the soft edge of a dark line (that would only thicken it)."""
    img = np.asarray(img)
    if img.ndim == 3:
        gray = (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114)
        level = img.min(axis=2)
    else:
        gray = level = img
    ink = gray < dark
    extra = (level < dark) | (gray < faint) if faint and faint > dark else (level < dark)
    extra &= ~ink
    if extra.any():
        extra &= ~ndi.binary_opening(extra | ink, np.ones((5, 5)))     # thin: not a fill
        extra &= ~ndi.binary_dilation(ink, iterations=2)              # not a dark line's edge
        lab, n = ndi.label(extra, structure=np.ones((3, 3)))
        if n:
            big = np.bincount(lab.ravel()) >= 15   # a line piece, not a speck
            big[0] = False
            ink |= big[lab]
    return ink


def _grazes(box, end, direction, margin) -> bool:
    """A line end in a symbol's margin that does not head into the symbol: followed on
    for the margin's width plus a bit, it never enters the box (it runs along a side)."""
    x1, y1, x2, y2 = box
    y, x = end
    for t in np.arange(0, 3 * margin + 4, 1.0):
        py, px = y + direction[0] * t, x + direction[1] * t
        if x1 + 1 <= px <= x2 - 1 and y1 + 1 <= py <= y2 - 1:
            return False
    return True


def _short_links(ink: np.ndarray, boxes: list, size: float, margin: float) -> dict:
    """{(i, j): connection} for symbols whose facing sides are less than 0.6 symbol apart
    with a thin line spanning the gap: >= 90% ink along a row (or column) between them, in
    a band no thicker than a line (a filled area between two blocks is not a pipe), and no
    other symbol in between."""
    H, W = ink.shape
    out = {}
    limit = 0.6 * size
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            for axis in (0, 1):  # 0: side by side (gap along x), 1: one above the other
                lo, hi = (a, b) if a[axis] <= b[axis] else (b, a)
                g0, g1 = int(np.ceil(lo[axis + 2])), int(np.floor(hi[axis]))
                if not (2 <= g1 - g0 <= limit):
                    continue
                o0, o1 = int(max(a[1 - axis], b[1 - axis])), int(min(a[3 - axis], b[3 - axis]))
                if o1 - o0 < 3:
                    continue  # the sides do not face each other
                corridor = (g0, o0, g1, o1) if axis == 0 else (o0, g0, o1, g1)
                if any(k not in (i, j) and c[0] < corridor[2] and c[2] > corridor[0] and c[1] < corridor[3]
                       and c[3] > corridor[1] and max(c[2] - c[0], c[3] - c[1]) < 3.0 * size
                       for k, c in enumerate(boxes)):  # (a frame / container around them is not in the way)
                    continue
                runs = [ink[y, g0:g1] for y in range(max(0, o0), min(H, o1 + 1))] if axis == 0 else \
                       [ink[g0:g1, x] for x in range(max(0, o0), min(W, o1 + 1))]
                full = [o0 + t for t, r in enumerate(runs) if r.mean() >= 0.9]
                dashed = False
                if not full:  # a dashed signal line (instrument to instrument): >= 3 dashes, half inked
                    full = [o0 + t for t, r in enumerate(runs)
                            if r.mean() >= 0.4 and np.count_nonzero(np.diff(r.astype(np.int8)) == 1) + r[0] >= 3]
                    dashed = bool(full)
                if not full or len(full) > max(6, 0.15 * size):
                    continue
                m = float(np.median(full))
                path = [(float(g0), m), (float(g1), m)] if axis == 0 else [(m, float(g0)), (m, float(g1))]
                if lo is not a:
                    path = path[::-1]  # report a -> b
                out[(i, j)] = {"a": i, "b": j, "bridges": 2 if dashed else 0, "length": float(g1 - g0), "path": path,
                               "flow": 0, "line_type": "signal" if dashed else "process"}
    return out


def _arrowhead_at(ink: np.ndarray, pts: np.ndarray, end: int, margin: float) -> bool:
    """A filled arrowhead where a line meets a symbol: across the line, the ink there is
    several times wider than the line further back. The line was cut `margin` px before
    the symbol (its box was grown by that much and erased), and draw.io's arrowhead sits
    in exactly that gap - so the unerased ink is measured from a few px back along the
    line to just short of the symbol's outline."""
    H, W = ink.shape
    pts = pts if end == 0 else pts[::-1]
    if len(pts) < 12:
        return False
    u = _direction(pts, 0)  # pointing out of this end
    n = (-u[1], u[0])
    y0, x0 = pts[0]

    def width(k: float) -> int:
        cy, cx = y0 - u[0] * k, x0 - u[1] * k  # k > 0: back along the line; k < 0: beyond the end
        run = 0
        for side in (1, -1):
            for t in range(0 if side == 1 else 1, 14):
                y, x = int(round(cy + side * n[0] * t)), int(round(cx + side * n[1] * t))
                if not (0 <= y < H and 0 <= x < W) or not ink[y, x]:
                    break
                run += 1
        return run

    line_w = np.median([width(k) for k in range(min(len(pts) - 1, 24), min(len(pts) - 1, 40))] or [2])
    near = max(width(k) for k in range(-int(margin) + 2, 5))  # k < 0: into the gap, short of the outline
    return near >= max(5, 3 * max(line_w, 1))


def _arrow_tip(ink: np.ndarray, box) -> tuple[float, float] | None:
    """Which way a flow arrow points: (dx, dy) unit, or None. A filled triangle is widest at its
    base and narrows to its tip: the width profile along each axis is fitted with a line, and
    the axis with the steeper (relative) slope is the arrow's, its narrowing side the tip."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = ink[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
    if crop.size == 0 or crop.sum() < 6:
        return None
    best = None
    for axis, unit in ((0, (1.0, 0.0)), (1, (0.0, 1.0))):   # 0: widths per column (along x)
        prof = crop.sum(axis=axis).astype(float)
        idx = np.nonzero(prof)[0]
        if len(idx) < 4:
            continue
        xs, ys = idx.astype(float), prof[idx]
        slope = np.polyfit(xs, ys, 1)[0] / max(1.0, ys.max()) * len(idx)
        if best is None or abs(slope) > abs(best[0]):
            best = (slope, unit)
    if best is None or abs(best[0]) < 0.4:
        return None
    sgn = -1.0 if best[0] > 0 else 1.0   # widths growing along +axis: the tip is at -axis
    return (best[1][0] * sgn, best[1][1] * sgn)


KNOT_STROKES = 2  # outline strokes a knot needs (1 found 2 more of 108 hidden on OPEN100, also without false alarms)


def _knots(ends: dict, arms: dict, seg_pts: dict, size: float, bridge: dict) -> list:
    """Symbols drawn on a pipe that no box was given for: the skeleton runs through their
    outline as a knot - junctions joined by outline strokes (short segments from junction to
    junction, or loose stubs; a check valve's triangle, a valve's bowtie) with exactly two
    lines leaving it (the pipe, in and out). A tee or a crossing is one junction with no
    outline strokes; two tees close together have four lines leaving. -> [box]."""
    limit = 1.2 * size

    def stroke(s, e):  # segment s, seen from its end e at a junction: part of an outline?
        se, n_px = ends[s]
        other = se[1 - e]
        return n_px < limit and (other[0] == "junc" or other[0] == "free" and (s, 1 - e) not in bridge)

    link = defaultdict(set)
    for j, a in arms.items():
        for s, e in a:
            if stroke(s, e) and ends[s][0][1 - e][0] == "junc":
                link[j].add(ends[s][0][1 - e][1])
    out, done = [], set()
    for j0 in arms:
        if j0 in done:
            continue
        group, todo = set(), [j0]
        while todo:
            j = todo.pop()
            if j not in group:
                group.add(j); todo += list(link[j])
        done |= group
        strokes = {s for j in group for s, e in arms[j] if stroke(s, e)}
        exits = [(s, e) for j in group for s, e in arms[j] if s not in strokes]
        if len(strokes) < KNOT_STROKES or len(exits) != 2:
            continue
        pts = np.concatenate([seg_pts[s] for s in strokes])
        y1, x1 = pts.min(0); y2, x2 = pts.max(0)
        if not 0.2 * size <= max(x2 - x1, y2 - y1) <= 1.5 * size:
            continue
        pad = 0.1 * size
        out.append([float(x1 - pad), float(y1 - pad), float(x2 + pad), float(y2 + pad)])
    return out
