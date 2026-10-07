#!/usr/bin/env bash
# Reproduce the Relationformer + PID2Graph training environment from scratch.
#
# Every step here encodes something that had to be discovered by trial - see
# ml/README.md for the full list of upstream bugs and why each workaround exists.
# Run from the repo root: ./ml/pid2graph/setup.sh
set -euo pipefail

ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RF_DIR="$ML_DIR/relationformer"
MINE="$ML_DIR/pid2graph"

# 1. Clone the scene_graph branch specifically.
#    NOT road_network_rgb: that branch hardcodes its classification head to
#    nn.Linear(HIDDEN_DIM, 2) - node-exists vs not - so it cannot do multi-class symbol
#    classification (valve vs pump vs tank) no matter what NUM_CLASSES is set to.
#    scene_graph's head is nn.Linear(hidden_dim, NUM_OBJ_CLS(+1)), which is what we need.
if [ ! -d "$RF_DIR" ]; then
  git clone --depth 1 https://github.com/suprosanna/relationformer.git "$RF_DIR"
  cd "$RF_DIR"
  git fetch origin scene_graph:scene_graph --depth 1
  git checkout scene_graph
else
  echo "relationformer/ already present, skipping clone"
fi

cd "$RF_DIR"

# 2. Apply the upstream fixes (3 files). See ml/README.md "Upstream bugs found".
if ! git diff --quiet; then
  echo "working tree already modified - skipping patch (assuming already applied)"
else
  git apply "$MINE/upstream-fixes.patch"
  echo "applied upstream-fixes.patch"
fi

# 3. Copy in our own files. They live outside the clone in git because
#    relationformer's own .gitignore has a '*data' rule that swallows pid2graph_data/.
mkdir -p "$RF_DIR/pid2graph_data"
cp "$MINE/model/dataset_pid2graph.py"    "$RF_DIR/pid2graph_data/"
cp "$MINE/model/train_pid2graph.py"      "$RF_DIR/"
cp "$MINE/model/evaluator_pid2graph.py"  "$RF_DIR/"
cp "$MINE/pid2graph_2d.yaml"       "$RF_DIR/configs/"

# 4. Virtualenv.
#    Check for the interpreter, not the directory: a half-created or copied-in .venv
#    can exist without a working bin/python.
if [ ! -x "$RF_DIR/.venv/bin/python" ]; then
  rm -rf "$RF_DIR/.venv"
  python3 -m venv "$RF_DIR/.venv"
fi
PY="$RF_DIR/.venv/bin/python"
"$PY" -m pip install -q --upgrade pip

# 5. torch MUST match the local CUDA toolkit (check with `nvcc --version`), because the
#    MultiScaleDeformableAttention CUDA extension is compiled against it in step 6 and
#    torch's cpp_extension refuses a major-version mismatch. cu121 wheels build fine
#    against a CUDA 12.0 toolkit. Adjust this line if your nvcc is a different major
#    version (e.g. cu124/cu128 wheels for a CUDA 12.4/12.8 toolkit).
# Without the CUDA toolkit (nvcc) the op in step 6 can't be compiled - that only matters for
# training. The review app's image/hybrid modes run without it (infer_sheet.py falls back to
# a PyTorch op): on an NVIDIA GPU with just the driver (CUDA wheels bundle the runtime), or
# on the CPU (CPU_ONLY=1, or no NVIDIA driver at all).
CPU_ONLY="${CPU_ONLY:-}"
NO_COMPILE=""
if [ -z "$CPU_ONLY" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "no NVIDIA driver - CPU-only install (inference only)"; CPU_ONLY=1
fi
if [ -n "$CPU_ONLY" ] || ! command -v nvcc >/dev/null 2>&1; then
  NO_COMPILE=1
  [ -z "$CPU_ONLY" ] && echo "NVIDIA GPU but no CUDA toolkit (nvcc) - GPU inference without the compiled op (no training)"
fi
if [ -n "$CPU_ONLY" ]; then
  "$PY" -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
else
  "$PY" -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
fi

# monai>=1.6 requires torch>=2.8, which would silently upgrade torch to a cu130 build
# and break the extension compile - pin an older monai and install it with --no-deps.
"$PY" -m pip install -q "monai==1.3.2" --no-deps
"$PY" -m pip install -q ignite pytorch_ignite numpy scipy matplotlib PyYAML \
    scikit-image tensorboardX timm einops networkx pillow ninja \
    tensorboard loguru pycocotools
# EasyOCR reads tags for the review app (ml/pid2graph/pipeline/infer_sheet.py). --no-deps: its
# metadata pulls an unpinned torch, which would replace the cu121 build above.
"$PY" -m pip install -q --no-deps easyocr==1.7.2
"$PY" -m pip install -q opencv-python-headless python-bidi Shapely pyclipper
# tensorboard: torch.utils.tensorboard.SummaryWriter needs it at import time.
# loguru + pycocotools: pulled in by metrics/coco.py, which the mAP evaluator uses.
# None of these three are in the repo's own requirements.txt.

# mmcv / pyvista / skimage.marching_cubes_lewiner from requirements.txt are deliberately
# NOT installed: they're imported at module scope in utils.py but only used by
# road-network/3D-vessel helpers this training path never calls. The patch in step 2
# makes those imports optional instead.

# 6. Compile the deformable-attention CUDA op.
#    -fuse-ld=gold is required: GNU ld 2.42 fails to link against torch's large
#    libtorch_cpu.so with "file too short". The gold linker handles it.
if [ -z "$NO_COMPILE" ]; then
  cd "$RF_DIR/models/ops"
  CFLAGS="-fuse-ld=gold" LDFLAGS="-fuse-ld=gold" "$PY" setup.py install
  cd "$RF_DIR"
fi

echo
echo "Setup complete. Verify with:"
echo "  cd $RF_DIR && .venv/bin/python -c 'from models import build_model; print(\"ok\")'"
echo
echo "Then benchmark before committing to a long run (see ml/README.md):"
echo "  cd $RF_DIR && .venv/bin/python train_pid2graph.py --config configs/pid2graph_2d.yaml --exp_name bench -b 1 --debug"
