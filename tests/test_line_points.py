"""Line points (bends and tee junctions) as training labels and as model output.

A DATA.LINE_POINTS model learns a pipe's corners and tees as nodes; folding them away again
must give exactly the symbol links the other models learn and are scored on."""
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "pid2graph" / "model"))
sys.path.insert(0, str(ROOT / "ml" / "pid2graph" / "pipeline"))

from dataset_pid2graph import _collapse_graph, _point_graph, contract_points  # noqa: E402
from point_links import contract  # noqa: E402


def _node(g, n, label, x, y, r=4):
    g.add_node(n, label=label, xmin=x - r, xmax=x + r, ymin=y - r, ymax=y + r)


def _sheet():
    """valve A -> corner -> straight waypoint -> tee -> tank B, the tee's branch up to pump C;
    a second pipe D-E passes straight across the A line at a crossing (not a joint)."""
    g = nx.Graph()
    _node(g, "A", "valve", 0, 0, 20)
    _node(g, "c1", "crossing", 100, 0)        # corner (an "ankle": a 2-arm crossing)
    _node(g, "c2", "connector", 100, 50)      # a connector: never a point
    _node(g, "x", "crossing", 100, 100)       # tee
    _node(g, "B", "tank", 100, 300, 40)
    _node(g, "C", "pump", 200, 100, 20)
    _node(g, "D", "valve", 50, 30, 10)
    _node(g, "E", "valve", 150, 30, 10)
    _node(g, "k", "crossing", 100, 30)        # D-E crosses the A line here
    for a, b in [("A", "c1"), ("c1", "k"), ("k", "c2"), ("c2", "x"), ("x", "B"), ("x", "C"), ("D", "k"), ("k", "E")]:
        g.add_edge(a, b, edge_label="solid")
    return g


def test_corners_and_tees_become_points_and_straight_waypoints_do_not():
    labels, boxes, edges = _point_graph(_sheet(), 24)
    assert sorted(l for l in labels if l in ("bend", "junction")) == ["bend", "junction"]
    bend = labels.index("bend")
    assert boxes[bend] == (88.0, -12.0, 112.0, 12.0)  # a point_box square on the corner
    named = {(labels[a], labels[b]) for a, b, _ in edges}
    # the A line runs A - bend - junction (through the crossing and the straight waypoint)
    assert ("valve", "bend") in named and ("bend", "junction") in named


def test_folding_points_away_gives_the_symbol_labels():
    g = _sheet()
    labels, _, edges = _collapse_graph(g)
    plabels, _, pedges = _point_graph(g, 24)
    sym = [i for i, l in enumerate(plabels) if l not in ("bend", "junction")]
    remap = {old: new for new, old in enumerate(sym)}
    assert {(remap[a], remap[b]) for a, b in contract_points(plabels, pedges)} == {(a, b) for a, b, _ in edges}
    names = {tuple(sorted((labels[a], labels[b]))) for a, b, _ in edges}
    assert ("tank", "valve") in names and ("pump", "valve") in names and ("pump", "tank") in names


def test_model_points_fold_into_connections_with_the_weakest_link():
    box = [0, 0, 10, 10]
    nodes = [{"id": f"m{k}", "box": box} for k in range(5)]
    is_point = [False, True, True, False, False]  # m0 - m1 - m2 - m3, m2 - m4 (a tee at m2)
    probs = {(0, 1): (0.9, 1), (1, 2): (0.6, 2), (2, 3): (0.8, 1), (2, 4): (0.3, 1)}
    symbols, conns = contract(nodes, is_point, probs, 0.5)
    assert [s["id"] for s in symbols] == ["m0", "m3", "m4"]
    assert conns == [{"from": "m0", "to": "m3", "prob": 0.6, "line_type": 2}]
    _, ranked = contract(nodes, is_point, probs, 0.0)  # scoring ranks every chain
    assert {(c["from"], c["to"]): c["prob"] for c in ranked} == {("m0", "m3"): 0.6, ("m0", "m4"): 0.3, ("m3", "m4"): 0.3}


def _line(eid, source=None, target=None, sj=None, tj=None):
    from types import SimpleNamespace
    return SimpleNamespace(id=eid, source=source, target=target, source_junction=sj, target_junction=tj,
                           source_point=None, target_point=None, is_decorative=False, is_dashed=False)


def test_drawio_lines_route_through_their_corners_and_tees():
    """A rounded corner (drawn as two cut points) is ONE corner, at the meeting point of the
    straight runs; a branch ending on a line is a tee on it; a link without a drawn line
    between touching items stays direct."""
    from types import SimpleNamespace

    from pid_dexpi_aas.image.export_pid2graph import line_points

    symbols = [("nA", "valve", (-20, -20, 20, 20), "A"), ("nB", "tank", (180, 280, 220, 320), "B"),
               ("nC", "pump", (380, 130, 420, 170), "C"), ("nD", "general", (420, 280, 460, 320), "D"),
               ("nE", "general", (460, 280, 500, 320), "E")]
    routes = {"main": [(20, 0), (190, 0), (200, 10), (200, 280)],  # rounded corner at (200, 0)
              "branch": [(380, 150), (200, 150)]}                  # ends on the main line
    diagram = SimpleNamespace(routes=routes, edges=[_line("main", "A", "B"), _line("branch", "C", None, tj="main"),
                                                    _line("adjacent:D:E", "D", "E")])
    points, edges = line_points(diagram, symbols)
    where = {p[0]: ((p[2][0] + p[2][2]) / 2, (p[2][1] + p[2][3]) / 2) for p in points}
    corner = [p for p in where if "_b" in p]
    assert len(corner) == 1 and where[corner[0]] == (200.0, 0.0)
    tee = [p for p in where if p not in corner]
    assert len(tee) == 1 and where[tee[0]] == (200.0, 150.0)
    links = {frozenset(e[:2]) for e in edges}
    assert links == {frozenset(x) for x in [("nA", corner[0]), (corner[0], tee[0]), (tee[0], "nB"),
                                            ("nC", tee[0]), ("nD", "nE")]}
