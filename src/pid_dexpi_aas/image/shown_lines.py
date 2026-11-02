"""Score what the review SHOWS against a labelled drawing - the lines a person sees in the
editor and has to fix, every step included (traced pipes as network trees, lines across flow
arrows, possible missed items, the model's own lines).

Pair scores (eval_sheets.py, the tracer comparisons) count every two items a pipe network
joins - a header with ten branches is 45 pairs - and leave out the review's own steps; they
can improve while the drawing a person gets is worse (2026-10-20). This counts per drawing:
  - lines shown (included in DEXPI);
  - wrong ones (to delete): no labelled connection between their two items, or an end on an
    item that is not in the labels (a false item);
  - real connections missing (to draw): labelled pairs between found items that the shown
    lines do not give;
  - of those, missing only because an item has no DEXPI type yet (its lines are unticked
    until someone sets one) - a different job from drawing a line;
  - joined pairs: every two items the review connects - a pipe network joins all its items,
    as DEXPI builds it - and how many of them are wrong. One false line that merges two
    networks is one wrong line but joins every item of one to every item of the other: those
    wrong pairs are what the DEXPI file gets, and are counted here.

Compared the way the review and the DEXPI build mean them:
  - a pipe network (rows with a "net": a traced network shown as a tree, lines across a flow
    arrow joining the networks on both sides) joins all its items - DEXPI builds it once,
    with tees;
  - flow arrows are no items: the review joins the items on either side straight through
    (pump -> arrow -> tank is pump -> tank). Labels that keep arrows as items (OPEN100) are
    compared the same way: every labelled arrow (gt_labels "arrow"), or chain of arrows, is
    replaced by links between the items around it.
"""
from __future__ import annotations

import collections


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def match(review, gt_boxes: list, skip: set = frozenset()) -> dict:
    """{review item id: labelled item index}, one to one, at IoU >= 0.5 (best first);
    labelled items in `skip` (arrows) are not matched."""
    pairs = sorted(((box_iou(s.box, g), s.id, j) for s in review.symbols if s.box
                    for j, g in enumerate(gt_boxes) if j not in skip), key=lambda p: -p[0])
    out, used = {}, set()
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def labelled_links(gt_edges: list, arrows: set) -> set:
    """The labelled connections with flow arrows joined straight through: item - arrow(s) -
    item becomes item - item (every two items around one arrow, or chain of arrows)."""
    links: dict = collections.defaultdict(set)
    for a, b in gt_edges:
        if a != b:
            links[a].add(b)
            links[b].add(a)
    out = {frozenset((a, b)) for a, b in gt_edges if a != b and a not in arrows and b not in arrows}
    seen = set()
    for start in arrows:
        if start in seen or start not in links:
            continue
        chain, todo, ends = set(), [start], set()
        while todo:
            x = todo.pop()
            if x in chain:
                continue
            chain.add(x)
            for y in links[x]:
                if y in arrows:
                    todo.append(y)   # the chain goes on through the next arrow
                else:
                    ends.add(y)
        seen |= chain
        ends = sorted(ends)
        out |= {frozenset((a, b)) for i, a in enumerate(ends) for b in ends[i + 1:]}
    return out


def _networks(review) -> list[set]:
    """The items each pipe network joins (rows with a "net"; "nets": joined through an arrow)."""
    parent: dict = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    rows = [c for c in review.connections if c.include and (c.attributes or {}).get("net")]
    for c in rows:
        nets = c.attributes.get("nets") or [c.attributes["net"]]
        for n in nets[1:]:
            parent[root(n)] = root(nets[0])
    groups: dict = collections.defaultdict(set)
    for c in rows:
        groups[root(c.attributes["net"])] |= {c.from_id, c.to_id}
    return list(groups.values())


def score(review, gt_boxes: list, gt_edges: list, traced: list | None = None, gt_labels: list | None = None) -> dict:
    """review: a ReviewDiagram (from_image); gt_*: the labels (boxes, [a, b] pairs, classes).
    -> {"shown", "wrong", "missing", "missing_untyped", "labelled", "precision"}.
    (traced: unused - the shown rows say which network each line is part of.)"""
    arrows = {j for j, lab in enumerate(gt_labels or []) if lab == "arrow"}
    m = match(review, gt_boxes, skip=arrows)
    truth = labelled_links(gt_edges, arrows)
    included = {s.id for s in review.symbols if s.include}
    shown = [(c.from_id, c.to_id) for c in review.connections if c.include]

    def label(p):
        a, b = p
        return frozenset((m[a], m[b])) if a in m and b in m and m[a] != m[b] else None

    wrong = sum(1 for p in shown if label(p) not in truth)
    implied = {frozenset(p) for p in shown}
    for items in _networks(review):
        items = sorted(items & included)
        implied |= {frozenset((a, b)) for i, a in enumerate(items) for b in items[i + 1:]}
    got = {label(tuple(p)) for p in implied} - {None}
    # every pair joined, by a line or a network: wrong if the labels do not join them, or an end
    # is no labelled item (a false item)
    joined = [p for p in implied if all(x in included for x in p)]
    joined_wrong = sum(1 for p in joined if label(tuple(p)) not in truth)
    found = set(m.values())
    labelled = {e for e in truth if e <= found}
    missing = labelled - got
    inv = {j: i for i, j in m.items()}
    untyped = sum(1 for e in missing if not all(inv[x] in included for x in e))
    return {"shown": len(shown), "wrong": wrong, "missing": len(missing), "missing_untyped": untyped,
            "labelled": len(labelled), "precision": (len(shown) - wrong) / len(shown) if shown else 1.0,
            "joined": len(joined), "joined_wrong": joined_wrong,
            "joined_precision": (len(joined) - joined_wrong) / len(joined) if joined else 1.0}


def summary(results: list[dict]) -> dict:
    """Per drawing averages of score() results, and the precision over all of them."""
    n = max(1, len(results))
    tot = collections.Counter()
    for r in results:
        tot.update({k: v for k, v in r.items() if not k.endswith("precision")})
    return {"drawings": len(results), "shown": tot["shown"] / n, "wrong": tot["wrong"] / n,
            "missing": tot["missing"] / n, "missing_untyped": tot["missing_untyped"] / n,
            "labelled": tot["labelled"] / n,
            "precision": (tot["shown"] - tot["wrong"]) / tot["shown"] if tot["shown"] else 1.0,
            "joined": tot["joined"] / n, "joined_wrong": tot["joined_wrong"] / n,
            "joined_precision": (tot["joined"] - tot["joined_wrong"]) / tot["joined"] if tot["joined"] else 1.0}
