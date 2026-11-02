"""What the review SHOWS, scored against labelled drawings (image/shown_lines.py).

The fixtures (tests/data/shown) are the app's model step (infer_sheet.py: v3 fold models at four
zooms, the line extractor) on sheets the model never saw - LightUp four times (one per fold
model) and OPEN100's twelve sheets, each by its fold's model - with the labels kept as
"_truth". They fix the model's output, so these tests check the review's own steps (network
trees, lines across flow arrows, possible missed items, the model's lines); rerun
ml/pid2graph/eval/eval_shown.py to check the model side. The limits sit just under the
scores of 2026-10-27 (LightUp: 67 lines a drawing, 6.8 wrong, 10.5 of 95 missing, precision
0.90; OPEN100, its flow arrows joined straight through like the review does: 53 lines, 14.5
wrong, 68 of 139 missing - 43 of them only for an item with no DEXPI type yet) - the web of
lines of 2026-10-20 scored 0.81 on LightUp and would fail.
"""
import collections
import json
from pathlib import Path

import pytest

from pid_dexpi_aas.image.shown_lines import score, summary
from pid_dexpi_aas.review import ReviewConnection, ReviewDiagram, ReviewSymbol, from_image

FIXTURES = Path(__file__).parent / "data" / "shown"


def _scored():
    out = collections.defaultdict(list)
    for f in sorted(FIXTURES.glob("*.json")):
        d = json.loads(f.read_text())
        t = d["_truth"]
        review = from_image(d, f.name)
        out[t["kind"]].append((f.name, d, review, score(review, t["boxes"], t["edges"], gt_labels=t.get("labels"))))
    return out


@pytest.fixture(scope="module")
def scored():
    if not FIXTURES.exists():
        pytest.skip("no shown-lines fixtures")
    return _scored()


def test_lightup_shown_lines(scored):
    s = summary([r for *_, r in scored["lightup"]])
    assert s["precision"] >= 0.88, s
    assert s["wrong"] <= 8.0, s      # wrong lines to delete, per drawing
    assert s["missing"] <= 11.5, s   # real connections to draw, per drawing
    assert s["shown"] <= 80, s       # not a web of lines
    assert s["joined_precision"] >= 0.90, s   # every pair joined, networks included (a wrong merge counts in full)


def test_open100_shown_lines(scored):
    s = summary([r for *_, r in scored["open100"]])
    assert s["precision"] >= 0.70, s
    assert s["wrong"] <= 16.0, s
    assert s["missing"] <= 72.0, s
    assert s["missing"] - s["missing_untyped"] <= 28.0, s   # missing although both ends have a DEXPI type
    assert s["joined_precision"] >= 0.72, s


def test_pipe_networks_are_shown_as_trees(scored):
    """A network of n items is n - 1 lines, never every pair of them."""
    for kind in scored.values():
        for name, d, review, _ in kind:
            nets = collections.defaultdict(set)
            for t in d.get("traced", []):
                if t.get("net") is not None:
                    nets[t["net"]] |= {t["from"], t["to"]}
            per_net = collections.Counter()
            for c in review.connections:  # tc<n>: the n-th traced pair (its own network)
                if c.id.startswith("tc") and "." not in c.id:
                    per_net[d["traced"][int(c.id[2:])].get("net")] += 1
            for net, items in nets.items():
                assert per_net[net] <= len(items) - 1, (name, net, per_net[net], len(items))


def test_shown_traced_lines_follow_the_pipes(scored):
    """Traced lines and lines across flow arrows are drawn along the pipe, not straight across."""
    lines = [c for kind in scored.values() for *_, review, _ in kind for c in review.connections
             if c.include and c.source == "picture tracing"]
    along = sum(1 for c in lines if c.path and len(c.path) >= 2)
    assert along / len(lines) >= 0.95, (along, len(lines))


def test_score_counts_trees_false_items_and_misses():
    box = lambda x: [x, 0, x + 40, 40]
    review = ReviewDiagram(source_name="x")
    for i in range(4):
        review.symbols.append(ReviewSymbol(id=f"m{i}", source="image model", box=box(100 * i), detected_as="valve",
                                           dexpi_class="GateValve"))
    review.symbols.append(ReviewSymbol(id="pm0", source="image model", box=box(900), detected_as="possible item",
                                       dexpi_class="GateValve"))  # not in the labels
    net = {"net": "t1"}
    review.connections = [ReviewConnection(id="tc0", from_id="m0", to_id="m1", attributes=dict(net)),
                          ReviewConnection(id="tc1", from_id="m1", to_id="m2", attributes=dict(net)),  # a tree: m0-m2 too
                          ReviewConnection(id="tc2", from_id="m2", to_id="pm0")]                       # to a false item
    gt_boxes = [box(0), box(100), box(200), box(300)]
    gt_edges = [[0, 1], [1, 2], [0, 2], [2, 3]]
    s = score(review, gt_boxes, gt_edges)
    assert s == {"shown": 3, "wrong": 1, "missing": 1, "missing_untyped": 0, "labelled": 4,
                 "precision": pytest.approx(2 / 3), "joined": 4, "joined_wrong": 1,   # m0-m1, m1-m2, m0-m2 (the tree), m2-pm0
                 "joined_precision": pytest.approx(3 / 4)}


def test_score_counts_a_wrong_merge_of_networks_in_full():
    """One false line between two networks of two items each: one wrong line, but four wrong
    joined pairs - what the DEXPI file gets."""
    box = lambda x: [x, 0, x + 40, 40]
    review = ReviewDiagram(source_name="x")
    review.symbols = [ReviewSymbol(id=f"m{i}", source="image model", box=box(100 * i), detected_as="valve",
                                   dexpi_class="GateValve") for i in range(4)]
    net = {"net": "t1"}
    review.connections = [ReviewConnection(id=f"tc{i}", from_id=f"m{i}", to_id=f"m{i + 1}", attributes=dict(net))
                          for i in range(3)]   # m1 - m2 is false: it merges {m0, m1} and {m2, m3}
    s = score(review, [box(100 * i) for i in range(4)], [[0, 1], [2, 3]])
    assert s["wrong"] == 1 and s["joined"] == 6 and s["joined_wrong"] == 4


def test_score_joins_labelled_flow_arrows_straight_through():
    """Labels with the arrow as an item (pump - arrow - tank): the review's pump - tank line
    across the arrow is right, nothing is missing; an item with no DEXPI type is counted apart."""
    box = lambda x: [x, 0, x + 40, 40]
    review = ReviewDiagram(source_name="x")
    review.symbols = [ReviewSymbol(id="p", source="image model", box=box(0), detected_as="pump", dexpi_class="Pump"),
                      ReviewSymbol(id="t", source="image model", box=box(400), detected_as="tank", dexpi_class="Tank"),
                      ReviewSymbol(id="g", source="image model", box=box(800), detected_as="general", include=False)]
    review.connections = [ReviewConnection(id="ac0", from_id="p", to_id="t", confidence="through a flow arrow")]
    gt_boxes = [box(0), box(200), box(400), box(800)]
    gt_labels = ["pump", "arrow", "tank", "general"]
    gt_edges = [[0, 1], [1, 2], [2, 3]]
    s = score(review, gt_boxes, gt_edges, gt_labels=gt_labels)
    assert (s["shown"], s["wrong"], s["missing"], s["missing_untyped"], s["labelled"]) == (1, 0, 1, 1, 2)
