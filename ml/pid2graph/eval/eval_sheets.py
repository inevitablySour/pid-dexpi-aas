"""Score a checkpoint on WHOLE sheets (fix #5), comparable to the published baselines.

Patch-level scores cut symbols and pipes at patch borders (measured: symbols at a patch
edge were missed 50% of the time vs 28.5% inside). The published OPEN100 numbers score
complete sheets, so this runs the model over each complete sheet (tiled + merged by
infer_sheet.detect, the same path the app uses), and scores the merged detections
against the sheet's own labels with the same node/edge mAP code as validation.

Scale: OPEN100 patches are the complete sheets enlarged 2.65x (symbol ~39 px on the
sheet, ~103 px in a patch), so tiles are taken at 1500/2.65 px to show the model what it
was trained on. The team's own drawings were patched at 1:1.

Usage (ML venv):
    ml/relationformer/.venv/bin/python ml/pid2graph/eval/eval_sheets.py --checkpoint <ckpt> --set open100 [--fold 0]
    ml/relationformer/.venv/bin/python ml/pid2graph/eval/eval_sheets.py --checkpoint <ckpt> --set team
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RF_DIR = HERE.parents[1] / "relationformer"
sys.path.insert(0, str(RF_DIR))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))  # infer_sheet, trace_pipes, refine_symbols

from evaluator_pid2graph import NodeEdgeMeanAP  # noqa: E402
from infer_sheet import CLASS_NAMES, box_iou, detect, load_sheet  # noqa: E402
from pid2graph_data.dataset_pid2graph import _collapse_graph, real_fold_sheets  # noqa: E402
from train_pid2graph import dict2obj  # noqa: E402

DATA = HERE.parents[2] / "data" / "PID2Graph"
SETS = {
    "open100": {"dir": DATA / "Complete" / "PID2Graph OPEN100", "scale": 2.65},
    # LightUp, held out. "Team v3 test" = the corrected labels (reactor and compartments,
    # tee networks, hand-dashed signal lines); "Team test" (team-v1) had 43 of its 97
    # connections and is kept only to compare with older scores.
    "team": {"dir": DATA / "Complete" / "Team v3 test", "scale": 1.0},
    "team-v1": {"dir": DATA / "Complete" / "Team test", "scale": 1.0},
    # pictures checked in the review app and held out (the app's "held-out test" export, e.g. the
    # MEM4CHEM sheets): already scaled to the team drawings' symbol size
    "reviewed": {"dir": DATA / "Complete" / "Reviewed test", "scale": 1.0},
}


def sheet_paths(which: str, fold: int | None) -> list[tuple[Path, Path]]:
    d = SETS[which]["dir"]
    pairs = [(g.with_suffix(".png"), g) for g in sorted(d.glob("*.graphml")) if g.with_suffix(".png").exists()]
    if which == "open100" and fold is not None:
        config = dict2obj(yaml.safe_load(open(RF_DIR / "configs" / "pid2graph_finetune.yaml")))
        config.DATA.DATA_PATH = str(DATA)
        test = set(real_fold_sheets(config, fold)["test"])
        pairs = [p for p in pairs if p[1].stem in test]
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", choices=sorted(SETS), default="open100")
    ap.add_argument("--fold", type=int, default=None,
                    help="open100 only: score just this fold's 3 test sheets (omit = all 12, valid only for "
                         "a model that never trained on OPEN100)")
    ap.add_argument("--tile", type=int, default=1500, help="tile size at training-patch scale")
    ap.add_argument("--stride", type=int, default=750)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    # The metric code (metrics/boxap.py) gathers across processes even when there is one;
    # give it a single-process group, as training does via ignite.
    import torch.distributed as dist
    if not dist.is_initialized():
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)

    scale = SETS[args.set]["scale"]
    tile, stride = round(args.tile / scale), round(args.stride / scale)
    metric = NodeEdgeMeanAP(max_detections=1000)  # whole sheets hold > 100 symbols
    per_class_found = {c: [0, 0] for c in CLASS_NAMES}  # [found at IoU>=0.5 any class, total]
    sheets = sheet_paths(args.set, args.fold)
    for img_path, g_path in sheets:
        labels, boxes, edges = _collapse_graph(nx.read_graphml(g_path))
        sheet = load_sheet(str(img_path))
        symbols, conns = detect(sheet, args.checkpoint, tile, stride, args.device, min_prob=0.01)
        pb = np.array([s["box"] for s in symbols], dtype=np.float32).reshape(-1, 4)
        pc = np.array([CLASS_NAMES.index(s["class"]) for s in symbols], dtype=np.int64)
        ps = np.array([s["score"] for s in symbols], dtype=np.float32)
        gb = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        gc = np.array([CLASS_NAMES.index(l) if l in CLASS_NAMES else CLASS_NAMES.index("general") for l in labels],
                      dtype=np.int64)
        index = {s["id"]: n for n, s in enumerate(symbols)}
        node_pairs = [(index[c["from"]], index[c["to"]]) for c in conns]
        edge_scores = np.array([c["prob"] for c in conns], dtype=np.float32)
        gt_edges = np.array([(i, j, 1) for i, j, _ in edges], dtype=np.int64).reshape(-1, 3)
        metric.update(([pb], [pc], [ps], [gb], [gc], [{
            "gt_boxes": gb, "gt_edges": gt_edges, "pred_boxes": pb, "node_pairs": node_pairs, "edge_scores": edge_scores,
        }]))
        for k in range(len(gb)):
            per_class_found[CLASS_NAMES[gc[k]]][1] += 1
            if len(pb) and max(box_iou(gb[k], b) for b in pb) >= 0.5:
                per_class_found[CLASS_NAMES[gc[k]]][0] += 1
        print(f"{img_path.name}: {len(gb)} symbols / {len(gt_edges)} connections in labels, "
              f"{len(pb)} detected", flush=True)

    metric.compute()
    results = metric.last_results  # compute() returns only the key scalar
    node_key = next(k for k in results if k.startswith("node_mAP_IoU_0.50_0.95"))
    summary = {
        "set": args.set, "fold": args.fold, "sheets": [p[1].stem for p in sheets], "checkpoint": args.checkpoint,
        "node_mAP": results[node_key], "edge_mAP": results.get("edge_mAP"),
        "node_AP50": next(v for k, v in results.items() if k.startswith("node_AP_IoU_0.50")),
        "edge_AP50": results.get("edge_AP@0.50"),
        "per_class_mAP": {c: results.get(f"node_{c}_mAP_IoU_0.50_0.95_0.05_MaxDet_1000") for c in CLASS_NAMES},
        "found_any_class_IoU50": {c: f"{f}/{t}" for c, (f, t) in per_class_found.items() if t},
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
