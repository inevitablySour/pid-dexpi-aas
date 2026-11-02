"""Step 1 of the shown-lines check (ML venv): the app's model step (infer_sheet.py) on every
sheet a model never saw - OPEN100's by its fold's v3 model, LightUp by each fold model - with
the labels kept in the output ("_truth"). Step 2: eval_shown.py (app venv) scores what the
review shows. Refresh tests/data/shown with --out tests/data/shown after a model / tracer change.

    ml/relationformer/.venv/bin/python ml/pid2graph/eval/shown_gen.py --out <dir> [--tracer lines] [--tile 2500,2000,1500,1000]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ml/pid2graph/eval")); sys.path.insert(0, str(ROOT / "ml/pid2graph/model"))
import networkx as nx  # noqa: E402
from eval_sheets import SETS, sheet_paths  # noqa: E402
from dataset_pid2graph import _collapse_graph  # noqa: E402

RF = ROOT / "ml" / "relationformer"
CK = RF / "trained_weights/runs/finetune_v3_fold{}_10/models/checkpoint_epoch=12.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracer", default="lines")
    ap.add_argument("--tile", default="2500,2000,1500,1000", help="at training-patch scale (eval_sheets.py)")
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    cases = [("open100", f, png, gml) for f in range(4) for png, gml in sheet_paths("open100", f)]
    cases += [("lightup", f, png, gml) for f in range(4) for png, gml in sheet_paths("team", None)]
    for kind, f, png, gml in cases:
        out = out_dir / f"{kind}_{f}_{'lightup' if kind == 'lightup' else gml.stem}.json"
        if out.exists():
            continue
        scale = SETS["open100" if kind == "open100" else "team"]["scale"]
        tiles = ",".join(str(round(int(t) / scale)) for t in a.tile.split(","))
        subprocess.run([str(RF / ".venv/bin/python"), str(ROOT / "ml/pid2graph/pipeline/infer_sheet.py"), "--image", str(png),
                        "--checkpoint", str(CK).format(f), "--out", str(out), "--tile", tiles, "--tracer", a.tracer],
                       check=True, capture_output=True, cwd=RF)
        labels, boxes, edges = _collapse_graph(nx.read_graphml(gml))
        d = json.loads(out.read_text())
        d["image"] = png.name
        d.pop("checkpoint", None)
        d["_truth"] = {"boxes": boxes, "labels": list(labels), "edges": [[x, y] for x, y, _ in edges], "kind": kind}
        out.write_text(json.dumps(d, separators=(",", ":")))
        print("done", out.name, flush=True)


if __name__ == "__main__":
    main()
