#!/usr/bin/env bash
# Train the FINAL model on everything: all 12 OPEN100 sheets and every team drawing
# (--fold -1), pumps from real/team patches drawn 3x more often (pid2graph_final.yaml).
# Nothing is held out, so its quality is estimated by the 4-fold run (run_folds.sh) with
# the same settings. The last epoch's checkpoint is copied to ml/models/pid2graph_final.pt.
#
#   ml/pid2graph/run_final.sh <base_checkpoint.pt>          (~4 h on a 4070)
#   DEBUG=1 ml/pid2graph/run_final.sh <base_checkpoint.pt>  # 30-patch smoke test
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RF="$HERE/../relationformer"
BASE="$(realpath "$1")"
PY="$RF/.venv/bin/python"
EXP="final"
EXTRA=(); [ -n "${DEBUG:-}" ] && EXTRA=(--debug --epochs 1) && EXP="final_debug"

cp "$HERE/model/dataset_pid2graph.py" "$RF/pid2graph_data/"
cp "$HERE/model/train_pid2graph.py" "$HERE/model/evaluator_pid2graph.py" "$RF/"
cp "$HERE/pid2graph_final.yaml" "$RF/configs/"

cd "$RF"
rm -rf "trained_weights/runs/${EXP}_10"
mkdir -p trained_weights/results_final
echo "=== final model: training ($(date)) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" train_pid2graph.py \
  --config configs/pid2graph_final.yaml --exp_name "$EXP" -b 4 --fold -1 \
  --init_from "$BASE" "${EXTRA[@]}" 2>&1 | tee "trained_weights/results_final/train_${EXP}.log" | tr '\r' '\n' | grep -aE "Epoch\[|val_node_mAP|Train patches" || true
ckpt="$(ls -t trained_weights/runs/${EXP}_10/models/checkpoint_epoch=*.pt | head -1)"
if [ -z "${DEBUG:-}" ]; then
  mkdir -p "$HERE/../models"
  # the app needs only the weights: drop the optimizer state (475 MB -> ~160 MB); the full
  # checkpoint stays in trained_weights/runs/ for further training
  "$PY" -c "import sys, torch; torch.save({'net': torch.load(sys.argv[1], map_location='cpu')['net']}, sys.argv[2])" \
    "$ckpt" "$HERE/../models/pid2graph_final.pt"
  echo "=== final model: $ckpt -> ml/models/pid2graph_final.pt ($(date)) ==="
else
  echo "=== debug run done: $ckpt ==="
fi
