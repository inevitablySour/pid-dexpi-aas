"""Step 2 of the shown-lines check (app venv): score what the review SHOWS on shown_gen.py's
outputs (image/shown_lines.py) - per drawing: lines shown, wrong ones to delete, real
connections missing, precision.

    .venv/bin/python ml/pid2graph/eval/eval_shown.py <dir> [--model-only 0.6]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import pid_dexpi_aas.review as rv  # noqa: E402
from pid_dexpi_aas.image.shown_lines import score, summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--model-only", type=float, default=rv.MODEL_ONLY_MIN, help="review.MODEL_ONLY_MIN to try")
    a = ap.parse_args()
    rv.MODEL_ONLY_MIN = a.model_only
    res = collections.defaultdict(list)
    for f in sorted(Path(a.dir).glob("*.json")):
        d = json.loads(f.read_text())
        t = d["_truth"]
        res[t["kind"]].append(score(rv.from_image(d, f.name), t["boxes"], t["edges"], gt_labels=t.get("labels")))
    for kind, r in sorted(res.items()):
        s = summary(r)
        print(f"{kind:8s} {s['drawings']:2d} drawings | per drawing: {s['shown']:6.1f} lines shown, {s['wrong']:5.1f} wrong, "
              f"{s['missing']:5.1f} of {s['labelled']:5.1f} real connections missing ({s['missing_untyped']:.1f} of them: "
              f"an item with no DEXPI type yet) | precision {s['precision']:.2f}\n"
              f"{'':12s} joined pairs (networks included): {s['joined']:6.1f}, {s['joined_wrong']:5.1f} wrong "
              f"| precision {s['joined_precision']:.2f}")


if __name__ == "__main__":
    main()
