"""Cross-validated evaluation of the whole picture pipeline, for reporting.

For every fold, that fold's model runs on the drawings it never saw - its 3 held-out OPEN100
sheets and the held-out team drawing (LightUp) - in four configurations:

  A  model only      symbols and connections as the model gives them
  B  + tracer        connections: the model's and those the pipe tracer follows (trace_pipes.py)
  C  + fixes         symbols refined (refine_symbols.py: tags, tank outlines, containers)
  D  full pipeline   C's symbols, B's connections - what the review app does

Metrics, all against the same labels (direct symbol-to-symbol connections):
  node mAP     COCO-style mean AP over IoU 0.50:0.95, per class and overall (+ AP at 0.50)
  edge mAP     connections ranked by score (traced connections count as certain)
  connections  precision / recall / F1 of the final yes/no list (model >= 0.5, or traced)

Reported per fold, and as mean +- standard deviation over the folds. Same tiles and scales as
eval_sheets.py. Model + OCR runs are cached per fold and sheet (--cache), so rerunning is fast.

Caveat for reading the numbers: the refinement rules and tracer fixes were developed on the
team drawings, LightUp included - its C/D gains are optimistic; OPEN100 was never used for that.

Usage (ML venv):
    ml/relationformer/.venv/bin/python ml/pid2graph/eval/eval_pipeline.py \\
        --runs ml/relationformer/trained_weights/runs --tag finetune_v3 --out docs/results/pipeline_v3
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "relationformer"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from eval_sheets import SETS, sheet_paths  # noqa: E402
from evaluator_pid2graph import NodeEdgeMeanAP  # noqa: E402
from infer_sheet import CLASS_NAMES, detect, load_sheet, run_ocr, use_fallback_attention  # noqa: E402
from pid2graph_data.dataset_pid2graph import _collapse_graph  # noqa: E402
from refine_symbols import refine  # noqa: E402
from trace_pipes import trace_connections  # noqa: E402

CONFIGS = {"A model only": (False, False), "B + tracer": (False, True), "C + fixes": (True, False),
           "D full pipeline": (True, True)}
REPORT_CLASSES = ("tank", "pump", "valve", "instrumentation", "general")


def _dist_init():
    import torch.distributed as dist
    if not dist.is_initialized():
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


def _run_sheet(png, ckpt, which, device, cache: Path) -> dict:
    """Model (all connections down to 0.01, for ranking) + OCR, cached."""
    if cache.exists():
        return json.loads(cache.read_text())
    sheet = load_sheet(str(png))
    scale = SETS[which]["scale"]
    syms, conns = detect(sheet, str(ckpt), round(1500 / scale), round(750 / scale), device, min_prob=0.01)
    out = {"symbols": syms, "connections": conns, "ocr": run_ocr(sheet, use_gpu=device != "cpu")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def _match(pred_boxes, true_boxes, thr=0.5) -> dict:
    from infer_sheet import box_iou
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


def evaluate(sheets: list, ckpt: Path, which: str, device: str, cache_dir: Path) -> dict:
    """{config: metrics} over the given sheets for one model."""
    results = {}
    runs = []
    for png, gml in sheets:
        labels, boxes, edges = _collapse_graph(nx.read_graphml(gml))
        c = _run_sheet(png, ckpt, which, device, cache_dir / f"{gml.stem}.json")
        gray = np.asarray(load_sheet(str(png)).convert("L"))
        runs.append((labels, boxes, edges, c, gray))
    for name, (fix, trace) in CONFIGS.items():
        metric = NodeEdgeMeanAP(max_detections=1000)
        tp = npred = ntrue = 0
        for labels, boxes, edges, c, gray in runs:
            syms, renamed = (refine(gray, c["symbols"], c["ocr"]) if fix else (c["symbols"], {}))
            index = {s["id"]: n for n, s in enumerate(syms)}
            score: dict = {}
            for x in c["connections"]:
                a, b = index[renamed.get(x["from"], x["from"])], index[renamed.get(x["to"], x["to"])]
                if a != b:
                    key = (min(a, b), max(a, b))
                    score[key] = max(score.get(key, 0.0), x["prob"])
            if trace:
                for t in trace_connections(gray, [s["box"] for s in syms], [o["box"] for o in c["ocr"]]):
                    score[(min(t["a"], t["b"]), max(t["a"], t["b"]))] = 1.0
            pb = np.array([s["box"] for s in syms], dtype=np.float32).reshape(-1, 4)
            pc = np.array([CLASS_NAMES.index(s["class"]) for s in syms], dtype=np.int64)
            ps = np.array([s["score"] for s in syms], dtype=np.float32)
            gb = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            gc = np.array([CLASS_NAMES.index(l) if l in CLASS_NAMES else CLASS_NAMES.index("general") for l in labels],
                          dtype=np.int64)
            pairs = list(score)
            metric.update(([pb], [pc], [ps], [gb], [gc], [{
                "gt_boxes": gb, "gt_edges": np.array([(i, j, 1) for i, j, _ in edges], dtype=np.int64).reshape(-1, 3),
                "pred_boxes": pb, "node_pairs": pairs, "edge_scores": np.array([score[p] for p in pairs], np.float32)}]))
            # the final yes/no list, as the app keeps it
            m = _match([s["box"] for s in syms], boxes)
            truth = {frozenset((i, j)) for i, j, _ in edges if i != j}
            kept = {frozenset((m[a], m[b])) for (a, b), v in score.items() if v >= 0.5 and a in m and b in m and m[a] != m[b]}
            tp, npred, ntrue = tp + len(kept & truth), npred + len(kept), ntrue + len(truth)
        metric.compute()
        r = metric.last_results
        p, rc = tp / max(npred, 1), tp / max(ntrue, 1)
        results[name] = {
            "node_mAP": r[next(k for k in r if k.startswith("node_mAP_IoU_0.50_0.95"))],
            "node_AP50": next(v for k, v in r.items() if k.startswith("node_AP_IoU_0.50")),
            "edge_mAP": r.get("edge_mAP"),
            "per_class": {cl: r.get(f"node_{cl}_mAP_IoU_0.50_0.95_0.05_MaxDet_1000") for cl in CLASS_NAMES},
            "conn_precision": p, "conn_recall": rc, "conn_F1": 2 * p * rc / max(p + rc, 1e-9),
        }
    return results


def _fmt(values: list) -> str:
    vals = [v for v in values if v is not None and v >= 0]
    if not vals:
        return "-"
    return f"{stats.mean(vals):.3f} ± {stats.stdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(HERE.parents[1] / "relationformer" / "trained_weights" / "runs"))
    ap.add_argument("--tag", default="finetune_v3", help="the run_folds.sh TAG of the fold models")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--cache", default=None, help="model + OCR cache (default: <out>/cache)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="results folder: per-fold JSON, summary.md, summary.csv")
    args = ap.parse_args()
    use_fallback_attention(args.device)
    _dist_init()
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out / "cache"
    per_fold = {}
    for f in args.folds:
        ckpt = max((Path(args.runs) / f"{args.tag}_fold{f}_10" / "models").glob("checkpoint_epoch=*.pt"),
                   key=lambda p: int(p.stem.split("=")[1]))
        per_fold[f] = {}
        for which in ("open100", "team"):
            sheets = sheet_paths(which, f if which == "open100" else None)
            print(f"fold {f} · {which}: {[g.stem for _, g in sheets]} · {ckpt.name}", flush=True)
            per_fold[f][which] = evaluate(sheets, ckpt, which, args.device, cache / f"fold{f}" / which)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"fold{f}.json").write_text(json.dumps(per_fold[f], indent=1))

    rows = []
    md = [f"# Picture pipeline, {len(per_fold)}-fold cross-validation ({args.tag})", "",
          "Each fold's model on the drawings it never saw: its 3 held-out OPEN100 sheets, and LightUp (team).",
          "Mean ± standard deviation over the folds. Connections are direct symbol-to-symbol lines.", ""]
    for which, title in (("open100", "OPEN100 (held-out real sheets)"), ("team", "LightUp (held-out team drawing)")):
        md += [f"## {title}", "",
               "| Configuration | Items mAP | Items AP50 | Lines mAP | Conn. precision | Conn. recall | Conn. F1 | "
               + " | ".join(REPORT_CLASSES) + " |",
               "|---" * (7 + len(REPORT_CLASSES)) + "|"]
        for name in CONFIGS:
            col = lambda key: [per_fold[f][which][name][key] for f in per_fold]
            cls = [_fmt([per_fold[f][which][name]["per_class"].get(c) for f in per_fold]) for c in REPORT_CLASSES]
            cells = [_fmt(col(k)) for k in ("node_mAP", "node_AP50", "edge_mAP", "conn_precision", "conn_recall", "conn_F1")]
            md.append(f"| {name} | " + " | ".join(cells + cls) + " |")
            rows.append([which, name] + cells + cls)
        md.append("")
    md += ["Caveat: the refinement rules (C, D) and the tracer fixes were developed on the team drawings,",
           "LightUp included - its gains there are optimistic. OPEN100 was not used for that."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    import csv
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "configuration", "items_mAP", "items_AP50", "lines_mAP", "conn_precision", "conn_recall",
                    "conn_F1", *REPORT_CLASSES])
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
