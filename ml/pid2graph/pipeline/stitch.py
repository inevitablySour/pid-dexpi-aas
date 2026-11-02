"""Merging the detections of overlapping tiles into one set of symbols for the sheet.

The sheet is read in overlapping tiles, so each symbol is detected by several of them.
"greedy" (the first method): highest score first, a detection overlapping a kept box by IoU
>= 0.5 joins it, else it is a new symbol. A symbol cut by a tile's edge gives a partial box
there; against the full box from the next tile its IoU is often below 0.5, so the symbol
came out twice ("doubled boxes").

"wbf" (from PID2Graph's author, who found it worked best, with the border weighting "a huge
boost" in merging accuracy):
  - a detection near an inner tile edge (not the sheet's own edge) counts less: its score is
    scaled down towards the edge, half at the edge itself;
  - a detection touching an inner tile edge (cut) also joins a symbol whose box mostly
    contains it - the cut-off part of a symbol seen whole in the next tile;
  - the symbol's box is the score-weighted average of its uncut detections (all cut: their
    union), its class the score-weighted vote.
Line points (bends, junctions; tiny boxes) are matched by centre distance in both.
"""
from __future__ import annotations

import numpy as np

EDGE_MARGIN = 0.05   # of the tile: detections nearer an inner tile edge than this count less
CUT = 4.0            # px: a box this near an inner tile edge is cut by it
CONTAINED = 0.7      # a cut box mostly inside another symbol's box is part of it


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside(a, b) -> float:
    """How much of box a lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def _centre_dist(a, b) -> float:
    return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))


def edge_distance(box, tile_rect, size) -> float:
    """Distance from box to the nearest INNER edge of its tile (an edge that is not the
    sheet's own edge - nothing is cut there); inf for a tile covering the whole sheet."""
    x0, y0, x1, y1 = tile_rect
    W, H = size
    d = []
    if x0 > 0:
        d.append(box[0] - x0)
    if y0 > 0:
        d.append(box[1] - y0)
    if x1 < W:
        d.append(x1 - box[2])
    if y1 < H:
        d.append(y1 - box[3])
    return max(0.0, min(d)) if d else float("inf")


def merge_detections(dets: list, point_cls: set, size, method: str = "wbf") -> tuple[list, dict]:
    """dets: {"box", "cls", "score", "tile", "idx", "rect" (the tile, sheet px)}. -> (merged,
    cluster_of): merged = [{"box", "score", "votes" [cls...], "weights", "point"}],
    cluster_of = {(tile, idx): merged index}."""
    for d in dets:
        d["edge"] = edge_distance(d["box"], d["rect"], size)
        tile_w = d["rect"][2] - d["rect"][0]
        margin = EDGE_MARGIN * tile_w
        d["w"] = d["score"] * (0.5 + 0.5 * min(1.0, d["edge"] / margin)) if method == "wbf" else d["score"]
        d["cut"] = method == "wbf" and d["edge"] < CUT
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["w"])
    cluster_of, merged = {}, []
    for i in order:
        d = dets[i]
        point = d["cls"] in point_cls
        best, best_val = None, 0.0
        for m_idx, m in enumerate(merged):
            if m["point"] != point:
                continue
            if point:  # a line point: tiles see it a few px apart - matched by its centre
                ok = _centre_dist(d["box"], m["box"]) <= max(d["box"][2] - d["box"][0], m["box"][2] - m["box"][0])
                v = 1.0 if ok else 0.0
            else:
                v = box_iou(d["box"], m["box"])
                if method == "wbf":
                    if v < 0.5 and d["cut"] and _inside(d["box"], m["box"]) >= CONTAINED:
                        v = 0.5  # a cut-off part of this symbol
                    elif v < 0.5 and m["all_cut"] and _inside(m["box"], d["box"]) >= CONTAINED:
                        v = 0.5  # the whole symbol, its cut-off part kept first
                if method == "greedy" and v >= 0.5:
                    best = m_idx
                    break
            if v >= 0.5 and v > best_val:
                best, best_val = m_idx, v
        if best is None:
            cluster_of[(d["tile"], d["idx"])] = len(merged)
            merged.append({"box": list(d["box"]), "score": d["score"], "votes": [d["cls"]], "weights": [d["w"]],
                           "point": point, "members": [d], "all_cut": d["cut"]})
            continue
        m = merged[best]
        cluster_of[(d["tile"], d["idx"])] = best
        m["votes"].append(d["cls"])
        m["weights"].append(d["w"])
        m["members"].append(d)
        if method == "wbf" and not point:
            whole = [x for x in m["members"] if not x["cut"]]
            m["all_cut"] = not whole
            if whole:
                w = np.array([x["w"] for x in whole]) + 1e-9
                m["box"] = list(np.average(np.array([x["box"] for x in whole]), axis=0, weights=w))
                m["score"] = max(x["score"] for x in whole)
            else:  # only cut pieces: together they cover the symbol
                bs = np.array([x["box"] for x in m["members"]])
                m["box"] = [bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()]
    for m in merged:
        m["box"] = [float(v) for v in m["box"]]
        m["runs"] = sorted({x.get("run", 0) for x in m.pop("members")})  # the ensemble runs that found it
    return merged, cluster_of
