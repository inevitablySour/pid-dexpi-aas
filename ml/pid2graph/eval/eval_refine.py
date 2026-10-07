"""Measure refine_symbols.py's fixes on the team sheets, one at a time and together:
tank boxes (good / loose / missed), node mAP per class, and the connections model + tracer
get from the refined boxes. Uses eval_tracer.py's cache (model + OCR per sheet), so run
eval_tracer.py once first.

Usage (ML venv): ml/relationformer/.venv/bin/python ml/pid2graph/eval/eval_refine.py --cache <dir>
       ... --set open100 --fold 1 --checkpoint <fold 1 ckpt> --cache <dir>   (runs model + OCR if not cached)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "relationformer"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))  # infer_sheet, trace_pipes, refine_symbols
sys.path.insert(0, str(HERE.parents[2] / "src"))

from eval_sheets import SETS, sheet_paths  # noqa: E402
from eval_tracer import DATA, match, prf  # noqa: E402
from pid2graph_data.dataset_pid2graph import _collapse_graph  # noqa: E402
from evaluator_pid2graph import NodeEdgeMeanAP  # noqa: E402
from infer_sheet import CLASS_NAMES, box_iou, detect, load_sheet, run_ocr, use_fallback_attention  # noqa: E402
from refine_symbols import refine  # noqa: E402
from trace_pipes import trace_connections  # noqa: E402

VARIANTS = {"model only": (), "tag": ("tag",), "snap": ("snap",), "snap+merge": ("snap", "merge"),
            "container": ("container",), "all four": ("tag", "snap", "merge", "container")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--no-tracer", action="store_true")
    ap.add_argument("--set", choices=["team", "open100"], default="team")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--checkpoint", default=None, help="needed when a sheet is not cached yet")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)

    use_fallback_attention(args.device)
    sheets = []
    if args.set == "team":
        paths = [(g.with_suffix(".png"), g) for g in sorted(list(DATA.glob("Team v3/*.graphml"))
                                                              + list(DATA.glob("Team v3 test/*.graphml")))]
        tile = stride = None
    else:
        paths = sheet_paths("open100", args.fold)
        scale = SETS["open100"]["scale"]
        tile, stride = round(1500 / scale), round(750 / scale)
    for png, gml in paths:
        labels, boxes, edges = _collapse_graph(nx.read_graphml(gml))  # symbols only, connectors folded
        tb = [tuple(b) for b in boxes]
        tl = list(labels)
        truth = {frozenset((i, j)) for i, j, _ in edges if i != j}
        cf = Path(args.cache) / f"{gml.stem}.json"
        sheet = load_sheet(str(png))
        if cf.exists():
            c = json.loads(cf.read_text())
        else:  # model + OCR, as eval_tracer.py caches them
            syms, conns = detect(sheet, args.checkpoint, tile or 1400, stride or 700, args.device)
            c = {"symbols": syms, "connections": conns, "ocr": run_ocr(sheet, use_gpu=args.device != "cpu")}
            Path(args.cache).mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps(c))
        gray = np.asarray(sheet.convert("L"))
        sheets.append((gml.stem, gray, tb, tl, truth, c))

    for name, fixes in VARIANTS.items():
        metric = NodeEdgeMeanAP(max_detections=1000)
        tanks = {"good": 0, "loose": 0, "missed": 0}
        conn = [0, 0, 0]
        for stem, gray, tb, tl, truth, c in sheets:
            syms, renamed = refine(gray, c["symbols"], c["ocr"], fixes=fixes) if fixes else (c["symbols"], {})
            pb = np.array([s["box"] for s in syms], dtype=np.float32).reshape(-1, 4)
            pc = np.array([CLASS_NAMES.index(s["class"]) for s in syms], dtype=np.int64)
            ps = np.array([s["score"] for s in syms], dtype=np.float32)
            gb = np.array(tb, dtype=np.float32).reshape(-1, 4)
            gc = np.array([CLASS_NAMES.index(l) if l in CLASS_NAMES else CLASS_NAMES.index("general") for l in tl],
                          dtype=np.int64)
            index = {s["id"]: n for n, s in enumerate(syms)}
            model_pairs = {(index[renamed.get(x["from"], x["from"])], index[renamed.get(x["to"], x["to"])])
                           for x in c["connections"]}
            model_pairs = [p for p in model_pairs if p[0] != p[1]]
            metric.update(([pb], [pc], [ps], [gb], [gc], [{
                "gt_boxes": gb, "gt_edges": np.array([(*sorted(e), 1) for e in truth], dtype=np.int64).reshape(-1, 3),
                "pred_boxes": pb, "node_pairs": model_pairs, "edge_scores": np.ones(len(model_pairs), np.float32)}]))
            for k, lab in enumerate(tl):
                if lab != "tank":
                    continue
                best = max((box_iou(s["box"], tb[k]) for s in syms if s["class"] == "tank"), default=0.0)
                tanks["good" if best >= 0.75 else "loose" if best >= 0.1 else "missed"] += 1
            if not args.no_tracer:
                mboxes = [tuple(s["box"]) for s in syms]
                m = match(mboxes, tb)
                to_true = lambda a, b: frozenset((m[a], m[b])) if a in m and b in m and m[a] != m[b] else None
                trm = {to_true(t["a"], t["b"]) for t in trace_connections(gray, mboxes, [t["box"] for t in c["ocr"]])}
                mod = {to_true(a, b) for a, b in model_pairs}
                reachable = {e for e in truth if all(j in set(m.values()) for j in e)}
                conn = [a + b for a, b in zip(conn, prf((trm | mod) - {None}, truth))]
                conn_reach = reachable  # noqa: F841
        metric.compute()
        r = metric.last_results
        node = next(v for k, v in r.items() if k.startswith("node_mAP_IoU_0.50_0.95"))
        per = {cl: r.get(f"node_{cl}_mAP_IoU_0.50_0.95_0.05_MaxDet_1000") for cl in ("tank", "pump", "valve",
                                                                                      "instrumentation", "general")}
        line = f"{name:12s} tanks {tanks}  node mAP {node:.3f}  " + " ".join(f"{k} {v:.2f}" for k, v in per.items()
                                                                              if v is not None and v >= 0)
        if not args.no_tracer:
            tp, npred, ntrue = conn
            p, rr = tp / max(npred, 1), tp / max(ntrue, 1)
            line += f"  | connections P {p:.2f} R {rr:.2f} F1 {2 * p * rr / max(p + rr, 1e-9):.2f} (of all {ntrue})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
