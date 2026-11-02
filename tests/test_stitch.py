"""Merging overlapping tiles: a symbol cut by a tile's edge must not come out twice."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "pid2graph" / "pipeline"))
from stitch import edge_distance, merge_detections  # noqa: E402

SIZE = (2100, 1400)
LEFT, RIGHT = (0, 0, 1400, 1400), (700, 0, 2100, 1400)   # two tiles, 700 px overlap


def _det(box, rect, score, tile, cls=1):
    return {"box": box, "cls": cls, "score": score, "tile": tile, "idx": 0, "rect": rect}


def test_a_symbol_cut_by_a_tile_edge_comes_out_once():
    # a valve at x 1360-1460: the left tile sees it cut at its edge (1360-1400), the right whole
    dets = lambda: [_det([1360, 500, 1400, 540], LEFT, 0.9, 0), _det([1362, 501, 1458, 541], RIGHT, 0.8, 1)]
    assert len(merge_detections(dets(), set(), SIZE, "greedy")[0]) == 2      # the doubled box
    merged, cluster_of = merge_detections(dets(), set(), SIZE, "wbf")
    assert len(merged) == 1 and cluster_of[(0, 0)] == cluster_of[(1, 0)] == 0
    assert [round(v) for v in merged[0]["box"]] == [1362, 501, 1458, 541]      # the whole box, not the cut one


def test_the_sheet_edge_is_not_a_cut_and_separate_symbols_stay_apart():
    assert edge_distance([0, 0, 40, 40], LEFT, SIZE) == 1360                  # x=0 is the sheet's edge
    dets = [_det([100, 100, 140, 140], LEFT, 0.9, 0), _det([150, 100, 190, 140], LEFT, 0.9, 0) | {"idx": 1}]
    assert len(merge_detections(dets, set(), SIZE, "wbf")[0]) == 2
