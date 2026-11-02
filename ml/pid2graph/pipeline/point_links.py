"""Line points (bends, junctions) read by a DATA.LINE_POINTS model, folded back into the
symbol-to-symbol connections the rest of the pipeline uses. No torch: tested in the main venv."""
from __future__ import annotations

import numpy as np


def centre_dist(a, b) -> float:
    return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))


ARMS = {"bend": 2, "junction": 4}   # a bend joins 2 pipe pieces, a tee 3 (a cross 4)
MIN_POINT_SCORE = 0.5


def prune_points(nodes: list, is_point: list, probs: dict) -> dict:
    """A point is what a bend or a tee physically is: a bend keeps only its 2 most likely links,
    a junction its 4; a weak point (score < MIN_POINT_SCORE) or one left with too few links (a
    bend with < 2, a junction with < 3: it joins nothing) is dropped, with its links. Without
    this, every false point joined everything around it: v6 fold 0 predicted 275 points for
    121 real ones and 1,861 connections for 381 (precision 0.17)."""
    probs = dict(probs)
    for _ in range(3):   # dropping a point can leave a neighbour short of links
        links: dict[int, list] = {}
        for (a, b), (p, c) in probs.items():
            links.setdefault(a, []).append((p, (a, b)))
            links.setdefault(b, []).append((p, (a, b)))
        drop = set()
        for k, own in links.items():
            if not is_point[k]:
                continue
            kind = nodes[k].get("class")
            if kind not in ARMS:
                continue
            need = 2 if kind == "bend" else 3
            if nodes[k].get("score", 1.0) < MIN_POINT_SCORE or len(own) < need:
                drop.update(e for _, e in own)
                continue
            drop.update(e for _, e in sorted(own, reverse=True)[ARMS.get(kind, 4):])
        if not drop:
            break
        for e in drop:
            probs.pop(e, None)
    return probs


def contract(nodes: list, is_point: list, probs: dict, min_prob: float) -> tuple[list, list]:
    """Symbol-to-symbol connections from a graph with line points: two symbols are linked
    when a chain of bend/junction nodes joins them - the training labels' own collapse
    (dataset_pid2graph.contract_points), so a line-point model scores like any other. A
    chain is as sure as its weakest link: the pair's probability is the best path's lowest
    link (and that link's line type). Symbols keep their ids ("m<n>")."""
    import heapq

    probs = prune_points(nodes, is_point, probs)
    adj: dict[int, list] = {}
    for (a, b), (p, c) in probs.items():
        adj.setdefault(a, []).append((b, p, c))
        adj.setdefault(b, []).append((a, p, c))
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for s in (k for k in range(len(nodes)) if not is_point[k]):
        # widest path from s, passing only through points
        width = {s: (1.0, 1)}
        heap = [(-1.0, s, 1)]
        while heap:
            w, n, c = heapq.heappop(heap)
            w = -w
            if w < width.get(n, (0.0,))[0]:
                continue
            if n != s and not is_point[n]:
                key = (min(s, n), max(s, n))
                if w > best.get(key, (0.0, 1))[0]:
                    best[key] = (w, c)
                continue  # a symbol ends the chain
            for m, p, mc in adj.get(n, ()):
                nw, ncls = (p, mc) if p < w else (w, c if n != s else mc)
                if nw > width.get(m, (0.0,))[0]:
                    width[m] = (nw, ncls)
                    heapq.heappush(heap, (-nw, m, ncls))
    symbols = [n for k, n in enumerate(nodes) if not is_point[k]]
    connections = [{"from": f"m{a}", "to": f"m{b}", "prob": p, "line_type": c}
                   for (a, b), (p, c) in sorted(best.items()) if p >= min_prob]
    return symbols, connections
