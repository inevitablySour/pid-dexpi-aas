"""Clean up the model's symbols with the picture itself and the OCR tags, after the tiles
are merged and before the pipe tracer runs. Four fixes, measured on the team sheets
(eval_refine.py), where tanks were the weakest class:

  1. tag: a tag read inside a box says what it is (P101 a pump, V101 a tank, FI 102 an
     instrument) - the model calls pumps on unfamiliar sheets something else;
  2. snap: a tank's box is fitted to the closed outline drawn around it - large vessels
     seen in one tile, or cut at a tile edge, came out loose (scored at IoU 0.5-0.95);
  3. merge: tank boxes that snap to the same outline are one tank (a tall vessel came out
     as 5 boxes);
  4. containers: a closed outline holding several symbols with a pipe ending on it is a
     piece of equipment the model never boxes (the Lumen reactor around its Mixing /
     Heating / Cooling compartments). Pipes only crossing it, or nothing touching it,
     make it a section frame instead - same rule as sections.py for draw.io files.

Never by colour or line weight. Runs in the ML venv (numpy + scipy); tags use the main
package's ocr_tags / mapping (stdlib only). Coordinates are image pixels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# a tag's DEXPI reading -> PID2Graph class
_TAG_KIND = {"valve": "valve", "instrument": "instrumentation"}
_TAG_EQUIPMENT = {"Pump": "pump", "Vessel": "tank", "Tank": "tank", "PressureVessel": "tank",
                  "ProcessColumn": "tank"}


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def _area(b) -> float:
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def _inside(inner, outer, frac=0.8) -> bool:
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)
    return w > 0 and h > 0 and w * h / area >= frac


def refine(gray: np.ndarray, symbols: list[dict], tokens: list[dict] | None = None, *, dark: int = 140,
           fixes: tuple = ("tag", "snap", "merge", "container")) -> tuple[list[dict], dict]:
    """Returns (symbols, renamed): new symbol list (dicts as infer_sheet.detect makes, plus
    "refined": what changed) and {old id: kept id} for symbols merged into another."""
    syms = [dict(s, box=list(s["box"])) for s in symbols]
    if not syms:
        return syms, {}
    ink = gray < dark
    size = float(np.median([max(s["box"][2] - s["box"][0], s["box"][3] - s["box"][1]) for s in syms]))
    renamed: dict = {}
    if "tag" in fixes and tokens:
        _classes_from_tags(syms, tokens)
    if "snap" in fixes:
        for s in syms:
            if s["class"] == "tank":
                box = _snap(ink, s["box"], size)
                # a shrink must only cut off thin stubs (nozzles), never a part as wide as the
                # tank (a dome not closed off from the body); and never swallow a sizeable
                # non-tank symbol (a heater drawn against a reactor) - specks are fine
                if box is not None and (
                        (_area(box) < 0.9 * _area(s["box"]) and _cuts_wide_part(ink, s["box"], box)) or any(
                        o is not s and o["class"] != "tank" and _inside(o["box"], box, 0.3)
                        and _area(o["box"]) >= 0.08 * _area(box) for o in syms)):
                    box = None
                if box is not None and _iou(box, s["box"]) < 0.98:
                    s["refined"] = s.get("refined", []) + ["box fitted to outline"]
                    s["box"] = box
    if "merge" in fixes:
        syms, renamed = _merge_tanks(syms)
    if "container" in fixes:
        n = len(syms)
        for k, (box, cls) in enumerate(_containers(ink, syms, size)):
            syms.append({"id": f"c{k}", "box": box, "class": cls, "score": 0.5, "class_votes": {cls: 1},
                         "n_tiles": 0, "refined": ["closed outline around symbols, piped into" if cls == "tank"
                                                   else "closed box inside a container (compartment)"]})
        if tokens and len(syms) > n and "tag" in fixes:
            _classes_from_tags(syms, tokens, only=set(range(n, len(syms))))
    return syms, renamed


# ------------------------------------------------------------------------------- tags ---
def _classes_from_tags(syms: list[dict], tokens: list[dict], only: set | None = None) -> None:
    """A tag read INSIDE a symbol's box sets its class when the tag's kind is clear. Only
    inside: a tag next to a box may belong to the neighbour."""
    from pid_dexpi_aas.dexpi.mapping import classify_tag
    from pid_dexpi_aas.image.ocr_tags import assemble_tags, attach_tags

    tags = assemble_tags(tokens)
    boxes = {k: tuple(s["box"]) for k, s in enumerate(syms) if only is None or k in only}
    attached = attach_tags(boxes, tags, classes={k: syms[k]["class"] for k in boxes})
    for k, (tag, how) in attached.items():
        if how != "inside" or syms[k]["class"] == "inlet/outlet":
            continue  # text in an off-page arrow names where the line goes, not the arrow
        kind, dexpi_class, _ = classify_tag(tag.text)
        new = _TAG_KIND.get(kind) or (_TAG_EQUIPMENT.get(dexpi_class) if kind == "equipment" else None)
        if new and new != syms[k]["class"]:
            syms[k]["refined"] = syms[k].get("refined", []) + [f"class {syms[k]['class']} -> {new} from tag {tag.text}"]
            syms[k]["class"] = new


# ------------------------------------------------------------------------------- snap ---
def _snap(ink: np.ndarray, box, size: float):
    """The closed outline around a tank: inside a window around the box, closed shapes are
    filled, thin lines (pipes on it) opened away, and the filled shape that overlaps the box
    most is the tank. None when no closed shape fits (open drawing, touching symbols). A
    wider window is tried when the first one cuts the outline (a fragment of a tall vessel)."""
    for f in (0.3, 1.2):
        box2 = _snap_in(ink, box, size, f * max(box[2] - box[0], box[3] - box[1]) + 0.2 * size)
        if box2 is not None:
            return box2
    return None


def _snap_in(ink: np.ndarray, box, size: float, pad: float):
    H, W = ink.shape
    x1, y1, x2, y2 = box
    X1, Y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    X2, Y2 = min(W, int(x2 + pad) + 1), min(H, int(y2 + pad) + 1)
    win = ink[Y1:Y2, X1:X2]
    if not win.any():
        return None
    k = max(5, int(0.06 * size))
    filled = ndi.binary_opening(ndi.binary_fill_holes(win), structure=np.ones((k, k)))
    lab, n = ndi.label(filled)
    best, best_iou = None, 0.0
    for sl in ndi.find_objects(lab):
        if sl is None:
            continue
        cand = [X1 + sl[1].start, Y1 + sl[0].start, X1 + sl[1].stop, Y1 + sl[0].stop]
        v = _iou(cand, box)
        if v > best_iou:
            best, best_iou = cand, v
    if best is None or best_iou < 0.4:
        return None
    ratio = (best[2] - best[0]) * (best[3] - best[1]) / max((x2 - x1) * (y2 - y1), 1e-6)
    if not 0.4 <= ratio <= 2.5:
        return None
    return [float(v) for v in best]


def _cuts_wide_part(ink: np.ndarray, old, new) -> bool:
    """Does shrinking `old` to `new` cut off ink spanning more than half the box's width
    (or height, for a cut at a side)? A nozzle stub is a few px wide; a dome is not."""
    x1, y1, x2, y2 = (int(round(v)) for v in old)
    nx1, ny1, nx2, ny2 = (int(round(v)) for v in new)
    strips = []
    if ny1 - y1 > 2:
        strips.append((ink[y1:ny1, x1:x2], 1, x2 - x1))
    if y2 - ny2 > 2:
        strips.append((ink[ny2:y2, x1:x2], 1, x2 - x1))
    if nx1 - x1 > 2:
        strips.append((ink[y1:y2, x1:nx1], 0, y2 - y1))
    if x2 - nx2 > 2:
        strips.append((ink[y1:y2, nx2:x2], 0, y2 - y1))
    for strip, axis, span in strips:
        if strip.size:
            cols = np.flatnonzero(strip.any(axis=0 if axis == 1 else 1))
            if len(cols) and cols[-1] - cols[0] > 0.5 * span:
                return True
    return False


# ------------------------------------------------------------------------------ merge ---
def _merge_tanks(syms: list[dict]) -> tuple[list[dict], dict]:
    """Tank boxes on one outline (after snapping: near-equal boxes, or one inside the
    other) become one: the highest-scoring box, with the others' tile votes."""
    tanks = sorted((k for k, s in enumerate(syms) if s["class"] == "tank"),
                   key=lambda k: -((syms[k]["box"][2] - syms[k]["box"][0]) * (syms[k]["box"][3] - syms[k]["box"][1])))
    into: dict[int, int] = {}
    for i, a in enumerate(tanks):
        if a in into:
            continue
        for b in tanks[i + 1:]:
            if b in into:
                continue
            A, B = syms[a]["box"], syms[b]["box"]
            if _iou(A, B) >= 0.6 or (_inside(B, A) and "box fitted to outline" in syms[a].get("refined", [])):
                into[b] = a
    if not into:
        return syms, {}
    for b, a in into.items():
        keep = syms[a]
        for c, v in syms[b].get("class_votes", {}).items():
            keep.setdefault("class_votes", {})[c] = keep["class_votes"].get(c, 0) + v
        keep["n_tiles"] = keep.get("n_tiles", 0) + syms[b].get("n_tiles", 0)
        keep["score"] = max(keep["score"], syms[b]["score"])
        keep["refined"] = keep.get("refined", []) + [f"merged {syms[b]['id']}"]
    renamed = {syms[b]["id"]: syms[a]["id"] for b, a in into.items()}
    return [s for k, s in enumerate(syms) if k not in into], renamed


# ------------------------------------------------------------------------- containers ---
def _containers(ink: np.ndarray, syms: list[dict], size: float) -> list[tuple[list[float], str]]:
    """Rectangular outlines (square or rounded corners) around >= 2 things that no symbol
    box covers - symbols, or closed boxes inside that the model did not box either (the
    reactor's Mixing / Heating / Cooling compartments) - with a pipe ending on the outline
    (a line outside that does not continue inside) and no more lines only crossing it than
    ending on it. Returns (box, class): the container as a tank, its undetected inner boxes
    as general (compartments)."""
    H, W = ink.shape
    rects = _rectangles(ink, size)
    # not the drawing border (a quarter of the sheet or more), and nothing sharing a side
    # with a much larger rectangle - title-block cells share the border's sides, a
    # reactor's compartments are set in from its walls
    def on_bigger(r):
        return any(_area(o[:4]) >= 3 * _area(r[:4]) and any(abs(a - b) <= 6 for a in (r[0], r[2]) for b in (o[0], o[2])
                                                             if o[1] - 6 <= r[1] and r[3] <= o[3] + 6)
                   or _area(o[:4]) >= 3 * _area(r[:4]) and any(abs(a - b) <= 6 for a in (r[1], r[3]) for b in (o[1], o[3])
                                                               if o[0] - 6 <= r[0] and r[2] <= o[2] + 6)
                   for o in rects)
    rects = [r for r in rects if _area(r[:4]) < 0.25 * H * W and not on_bigger(r)]
    boxes = [[r[0] - r[4] / 2, r[1] - r[4] / 2, r[2] + r[4] / 2, r[3] + r[4] / 2] for r in rects]
    covered = lambda b: any(_iou(s["box"], b) >= 0.5 for s in syms)
    found: list[tuple[list[float], str]] = []
    for (x1, y1, x2, y2, lw), box in zip(rects, boxes):
        if covered(box) or any(_iou(box, f) >= 0.8 for f, _ in found):
            continue
        held = [s for s in syms if _inside(s["box"], box, 0.9) and _iou(s["box"], box) < 0.5]
        inner = [b for b in boxes if b is not box and _inside(b, box, 1.0) and _area(b) < 0.8 * _area(box)
                 and not covered(b)]
        inner = [b for k, b in enumerate(inner) if not any(_iou(b, o) >= 0.8 for o in inner[:k])]
        if len(held) + len(inner) < 2:
            continue
        ended, crossed = _side_lines(ink, (x1, y1, x2, y2), lw, [s["box"] for s in syms], size)
        # a section frame holds symbols the model boxes and pipes cross it; a container
        # (reactor) has pipes ending on it, or holds compartments the model does not box
        if ended >= 1 and (ended >= crossed or len(inner) >= 2):
            found.append((box, "tank"))
            for b in inner:
                for part in _split_at_walls(ink, b, size):
                    if not any(_iou(part, f) >= 0.8 for f, _ in found):
                        found.append((part, "general"))
    return found


def _segments(ink: np.ndarray, min_len: int) -> list[tuple[float, int, int, int]]:
    """Long horizontal ink runs, joined across neighbouring rows (a thick line):
    (row centre, start, end, thickness)."""
    segs: list = []
    open_: dict = {}
    for y in range(ink.shape[0]):
        row = np.concatenate([[False], ink[y], [False]])
        d = np.flatnonzero(np.diff(row.astype(np.int8)))
        runs = [(a, b) for a, b in zip(d[::2], d[1::2]) if b - a >= min_len]
        nxt = {}
        for a, b in runs:
            key = next((k for k in open_ if abs(k[0] - a) <= 3 and abs(k[1] - b) <= 3), None)
            if key is not None:
                y0, n = open_.pop(key)
                nxt[(a, b)] = (y0, n + 1)
            else:
                nxt[(a, b)] = (y, 1)
        for (a, b), (y0, n) in open_.items():
            segs.append((y0 + (n - 1) / 2, a, b, n))
        open_ = nxt
    for (a, b), (y0, n) in open_.items():
        segs.append((y0 + (n - 1) / 2, a, b, n))
    return [s for s in segs if s[3] <= 12]  # thicker is a filled area, not a line


def _rectangles(ink: np.ndarray, size: float) -> list[tuple[float, float, float, float, int]]:
    """Outlines made of two horizontal and two vertical lines, corners square or rounded:
    (x1, y1, x2, y2, line width), on the line centres."""
    min_len = int(1.5 * size)
    hs = _segments(ink, min_len)
    vs = [(x, a, b, n) for x, a, b, n in _segments(ink.T, min_len)]  # (column, top, bottom, width)
    out = []
    for i, (ty, ta, tb, tn) in enumerate(hs):
        for by, ba, bb, bn in hs:
            if by - ty < min_len or abs(ta - ba) > 6 or abs(tb - bb) > 6:
                continue
            r = 0.25 * min(by - ty, tb - ta) + 6  # room for rounded corners
            span = by - ty
            left = [v for v in vs if ta - r <= v[0] <= ta + 6 and v[1] <= ty + r and v[2] >= by - r
                    and (min(v[2], by) - max(v[1], ty)) >= 0.6 * span]
            right = [v for v in vs if tb - 6 <= v[0] <= tb + r and v[1] <= ty + r and v[2] >= by - r
                     and (min(v[2], by) - max(v[1], ty)) >= 0.6 * span]
            # the wall that belongs to this outline: a corner is as round across as down
            # (square: both 0), so the top line starts as far in from the wall as the wall
            # starts below the top line - a compartment wall just inside does not fit
            left = [v for v in left if abs((ta - v[0]) - (v[1] - ty)) <= 8]
            right = [v for v in right if abs((v[0] - tb) - (v[1] - ty)) <= 8]
            if left and right:
                lx = min(left, key=lambda v: abs((ta - v[0]) - (v[1] - ty)))[0]
                rx = min(right, key=lambda v: abs((v[0] - tb) - (v[1] - ty)))[0]
                out.append((float(lx), float(ty), float(rx), float(by), int(max(tn, bn))))
    return out


def _split_at_walls(ink: np.ndarray, box, size: float) -> list[list[float]]:
    """A box divided by a full-height (or full-width) inner wall is two compartments
    (Mixing | Heating drawn as one rectangle with a line between)."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    m = max(4, int(0.05 * size))
    cols = [x for x in range(x1 + m, x2 - m) if ink[y1 + m:y2 - m, x].mean() >= 0.95]
    rows = [y for y in range(y1 + m, y2 - m) if ink[y, x1 + m:x2 - m].mean() >= 0.95]
    for walls, vertical in ((cols, True), (rows, False)):
        groups, cur = [], []
        for w in walls:
            if cur and w != cur[-1] + 1:
                groups.append(cur)
                cur = []
            cur.append(w)
        if cur:
            groups.append(cur)
        cuts = [g[len(g) // 2] for g in groups if len(g) <= 12]
        if cuts:
            edges = [x1 if vertical else y1] + cuts + [x2 if vertical else y2]
            parts = [[a, y1, b, y2] if vertical else [x1, a, x2, b] for a, b in zip(edges, edges[1:])]
            return [[float(v) for v in p] for p in parts if (p[2] - p[0]) >= 0.5 * size and (p[3] - p[1]) >= 0.5 * size]
    return [list(map(float, box))]


def _side_lines(ink: np.ndarray, rect, lw: int, symbol_boxes=(), size: float = 0.0) -> tuple[int, int]:
    """Lines meeting a rectangular outline: (ending on it, crossing it). Looked for a few
    px outside the outline, where an arrowhead has narrowed back to the line; a line there
    that also shows a few px inside crosses it - as does one reaching a symbol that sits
    right inside the outline (NV201 against the Autoclave frame)."""
    H, W = ink.shape
    x1, y1, x2, y2 = (int(round(v)) for v in rect)
    h = lw // 2
    out0, out1 = h + 10, h + 18
    in0, in1 = h + 3, h + 10
    corner = max(10, int(0.15 * min(x2 - x1, y2 - y1)))
    clip = lambda a, b, n: (max(0, min(n, a)), max(0, min(n, b)))
    sides = (
        (range(x1 + corner, x2 - corner), lambda p: ink[slice(*clip(y1 - out1, y1 - out0, H)), p],
         lambda p: ink[slice(*clip(y1 + in0, y1 + in1, H)), p]),
        (range(x1 + corner, x2 - corner), lambda p: ink[slice(*clip(y2 + out0, y2 + out1, H)), p],
         lambda p: ink[slice(*clip(y2 - in1, y2 - in0, H)), p]),
        (range(y1 + corner, y2 - corner), lambda p: ink[p, slice(*clip(x1 - out1, x1 - out0, W))],
         lambda p: ink[p, slice(*clip(x1 + in0, x1 + in1, W))]),
        (range(y1 + corner, y2 - corner), lambda p: ink[p, slice(*clip(x2 + out0, x2 + out1, W))],
         lambda p: ink[p, slice(*clip(x2 - in1, x2 - in0, W))]),
    )
    ended = crossed = 0
    for side, (positions, outside, inside) in enumerate(sides):
        hits = [p for p in positions if outside(p).size and outside(p).all()]
        runs, cur = [], []
        for p in hits:
            if cur and p != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(p)
        if cur:
            runs.append(cur)
        for r in runs:
            if len(r) > 8:  # a wide band, not a line meeting the outline
                continue
            mid = r[len(r) // 2]
            strip = inside(mid)
            if (strip.size and strip.all()) or _symbol_just_inside(symbol_boxes, (x1, y1, x2, y2), side, mid, size):
                crossed += 1
            else:
                ended += 1
    return ended, crossed


def _symbol_just_inside(symbol_boxes, rect, side: int, pos: int, size: float) -> bool:
    """A symbol box within 0.6 symbol of the outline, inside it, in line with `pos`."""
    x1, y1, x2, y2 = rect
    reach = 0.6 * size
    for b in symbol_boxes:
        if side in (0, 1) and b[0] <= pos <= b[2]:
            if (side == 0 and 0 <= b[1] - y1 <= reach) or (side == 1 and 0 <= y2 - b[3] <= reach):
                return True
        if side in (2, 3) and b[1] <= pos <= b[3]:
            if (side == 2 and 0 <= b[0] - x1 <= reach) or (side == 3 and 0 <= x2 - b[2] <= reach):
                return True
    return False
