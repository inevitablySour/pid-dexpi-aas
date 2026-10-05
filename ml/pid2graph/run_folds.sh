#!/usr/bin/env bash
# Fine-tune all 4 OPEN100 cross-validation folds from a base checkpoint, then score each
# fold on its own 3 held-out OPEN100 sheets and on the held-out team drawing (LightUp),
# and pool the results. Run from anywhere; ~4 h per fold on a 4070 (EPOCH_SAMPLES and
# EPOCHS in configs/pid2graph_finetune.yaml set the length).
#
#   ml/pid2graph/run_folds.sh <base_checkpoint.pt> [folds...]     e.g.  ... ckpt.pt 0 1 2 3
#   DEBUG=1 ml/pid2graph/run_folds.sh <ckpt> 0                    # 30-patch smoke test
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RF="$HERE/../relationformer"
BASE="$(realpath "$1")"; shift
FOLDS=("${@:-0 1 2 3}"); [ $# -eq 0 ] && FOLDS=(0 1 2 3)
PY="$RF/.venv/bin/python"
TAG="${TAG:-finetune}"
EXTRA=(); [ -n "${DEBUG:-}" ] && EXTRA=(--debug --epochs 1) && TAG="${TAG}_debug"
RESULTS="$RF/trained_weights/results_${TAG}"
mkdir -p "$RESULTS"

cp "$HERE/dataset_pid2graph.py" "$RF/pid2graph_data/"
cp "$HERE/train_pid2graph.py" "$HERE/evaluator_pid2graph.py" "$RF/"
cp "$HERE/pid2graph_finetune.yaml" "$RF/configs/"

cd "$RF"
for f in "${FOLDS[@]}"; do
  exp="${TAG}_fold${f}"
  rm -rf "trained_weights/runs/${exp}_10"
  echo "=== fold $f: training ($(date)) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" train_pid2graph.py \
    --config configs/pid2graph_finetune.yaml --exp_name "$exp" -b 4 --fold "$f" \
    --init_from "$BASE" "${EXTRA[@]}" 2>&1 | tee "$RESULTS/train_fold${f}.log" | tr '\r' '\n' | grep -aE "Epoch\[|val_node_mAP|edge_mAP:" || true
  # Last epoch's checkpoint: checkpoint selection on a 1-sheet real val set is too noisy
  # to trust, and the LR schedule is fixed, so the final state is the honest choice.
  ckpt="$(ls -t trained_weights/runs/${exp}_10/models/checkpoint_epoch=*.pt | head -1)"
  echo "=== fold $f: scoring $ckpt ==="
  "$PY" "$HERE/eval_sheets.py" --checkpoint "$ckpt" --set open100 --fold "$f" --out "$RESULTS/open100_fold${f}.json" > /dev/null
  "$PY" "$HERE/eval_sheets.py" --checkpoint "$ckpt" --set team --out "$RESULTS/team_fold${f}.json" > /dev/null
done

"$PY" - "$RESULTS" <<'EOF'
import json, sys, glob, statistics as st
d = sys.argv[1]
for kind in ("open100", "team"):
    rs = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/{kind}_fold*.json"))]
    if not rs:
        continue
    node = [r["node_mAP"] for r in rs]; edge = [r["edge_mAP"] for r in rs]
    print(f"{kind}: node mAP {st.mean(node):.3f} (folds {', '.join(f'{x:.3f}' for x in node)}) | "
          f"edge mAP {st.mean(edge):.3f} (folds {', '.join(f'{x:.3f}' for x in edge)})")
EOF
echo "Results in $RESULTS"
