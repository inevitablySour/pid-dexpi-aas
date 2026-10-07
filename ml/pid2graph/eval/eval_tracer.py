"""How well do connections from the picture match the drawing? Scores the pipe tracer
(trace_pipes.py) and the model's own connection predictions against the team sheets'
whole-sheet labels (data/PID2Graph/Complete/Team v3*/), which come from the draw.io files.

Two runs per sheet:
  - tracer on the TRUE symbol boxes: the tracer's own accuracy;
  - tracer / model / both on the MODEL's boxes (+ OCR text erased): what image-only mode gets.
A predicted pair counts when both ends match labelled symbols (box IoU >= 0.5) that the
labels connect. OCR and model runs are cached in --cache.

Usage (ML venv): ml/relationformer/.venv/bin/python ml/pid2graph/eval_tracer.py --checkpoint ml/models/pid2graph_fold1.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from infer_sheet import box_iou, detect, load_sheet, run_ocr  # noqa: E402
from trace_pipes import trace_connections  # noqa: E402

DATA = HERE.parents[1] / "data" / "PID2Graph" / "Complete"


def prf(pred: set, truth: set) -> tuple[int, int, int]:
    return len(pred & truth), len(pred), len(truth)


def fmt(tp, np_, nt):
    p, r = tp / max(np_, 1), tp / max(nt, 1)
    return f"P {p:.2f} R {r:.2f} F1 {2 * p * r / max(p + r, 1e-9):.2f} ({tp}/{np_} pred, {nt} true)"


def match(pred_boxes, true_boxes, thr=0.5) -> dict[int, int]:
    pairs = sorted(((box_iou(p, t), i, j) for i, p in enumerate(pred_boxes) for j, t in enumerate(true_boxes)),
                   reverse=True)
    out, used = {}, set()
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", default="/tmp/eval_tracer_cache")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    tot = {k: [0, 0, 0] for k in ("tracer|true boxes", "tracer|model boxes", "model|model boxes", "both|model boxes")}
    for gml in sorted(list(DATA.glob("Team v3/*.graphml")) + list(DATA.glob("Team v3 test/*.graphml"))):
        sheet = load_sheet(str(gml.with_suffix(".png")))
        g = nx.read_graphml(gml)
        nodes = list(g.nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        tboxes = [tuple(float(g.nodes[n][k]) for k in ("xmin", "ymin", "xmax", "ymax")) for n in nodes]
        truth = {frozenset((idx[u], idx[v])) for u, v in g.edges if u != v}
        cf = cache / f"{gml.stem}.json"
        if cf.exists():
            c = json.loads(cf.read_text())
        else:
            syms, conns = detect(sheet, args.checkpoint, 1400, 700, args.device)
            c = {"symbols": syms, "connections": conns, "ocr": run_ocr(sheet, use_gpu=args.device != "cpu")}
            cf.write_text(json.dumps(c))
        gray = np.asarray(sheet.convert("L"))
        text = [t["box"] for t in c["ocr"]]
        # tracer on the true boxes
        tr = trace_connections(gray, tboxes, text)
        res = {"tracer|true boxes": prf({frozenset((t["a"], t["b"])) for t in tr}, truth)}
        # model boxes
        mboxes = [tuple(s["box"]) for s in c["symbols"]]
        m = match(mboxes, tboxes)
        mid = {s["id"]: i for i, s in enumerate(c["symbols"])}
        to_true = lambda a, b: frozenset((m[a], m[b])) if a in m and b in m and m[a] != m[b] else None
        trm = {to_true(t["a"], t["b"]) for t in trace_connections(gray, mboxes, text)} - {None}
        mod = {to_true(mid[x["from"]], mid[x["to"]]) for x in c["connections"]} - {None}
        # pairs whose ends the model found at all: a fair ceiling for the model-box runs
        reachable = {e for e in truth if all(j in set(m.values()) for j in e)}
        res["tracer|model boxes"] = prf(trm, reachable)
        res["model|model boxes"] = prf(mod, reachable)
        res["both|model boxes"] = prf(trm | mod, reachable)
        print(f"{gml.stem[:38]:38s} symbols found {len(m)}/{len(tboxes)}")
        for k, v in res.items():
            print(f"    {k:20s} {fmt(*v)}")
            tot[k] = [a + b for a, b in zip(tot[k], v)]
    print("ALL SHEETS")
    for k, v in tot.items():
        print(f"    {k:20s} {fmt(*v)}")


if __name__ == "__main__":
    main()
