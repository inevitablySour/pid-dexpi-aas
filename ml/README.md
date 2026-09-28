# ML track: Relationformer on PID2Graph

Image-to-graph symbol + topology extraction, as an alternative to the heuristic
drawio-parsing pipeline in `src/pid_dexpi_aas/`. The two tracks are independent.

## Status

Working end to end on a 6GB RTX 3060: data loads, forward, all five losses, backward,
optimizer step, evaluation. 150 consecutive steps with no NaN, loss 35.6 -> 15.9. A short
`--debug` run (30 samples) had node mAP climbing 0.027 -> 0.071 over 5 epochs, so the
loop and the metric both respond to learning.

**No real model has been trained yet** - only that the pipeline runs. No weights exist.

---

# Moving this to another machine

The repo is self-contained except for two things that are deliberately gitignored and
must be handled separately: the **14GB dataset** and the **vendored Relationformer
clone + its venv**.

## 1. Copy the repo

Copy the whole `pid-dexpi-aas/` folder. If you want to skip the parts that get rebuilt
anyway (saves ~22GB):

```bash
rsync -av --exclude data/ --exclude .venv/ --exclude ml/relationformer/ \
      pid-dexpi-aas/ /path/on/new/machine/pid-dexpi-aas/
```

Or just copy everything, including `data/` - then skip step 2.

## 2. Get the dataset

Only if you excluded `data/`. From the repo root on the new machine:

```bash
mkdir -p data && cd data
curl -L -o PID2Graph.zip "https://zenodo.org/api/records/14803338/files/PID2Graph.zip/content"
unzip -q PID2Graph.zip && rm PID2Graph.zip
```

9.3GB download, ~14GB extracted. Verify: `ls "data/PID2Graph/Complete/PID2Graph OPEN100"`
should list `0.graphml` through `11.png`.

## 3. Rebuild the Relationformer environment

```bash
./ml/pid2graph/setup.sh
```

This clones the repo, checks out `scene_graph`, applies
`ml/pid2graph/upstream-fixes.patch`, copies in our loader/config/train/eval files, makes
a venv, and compiles the CUDA op.

**Check `nvcc --version` first.** The script installs `torch==2.5.1+cu121`, which needs a
CUDA 12.x toolkit. If your toolkit is a different major version, change the
`--index-url` line in `setup.sh` to the matching wheel (`cu124`, `cu128`, ...), or the
CUDA op won't compile. If `nvcc` is missing entirely, install the CUDA toolkit - the
driver alone is not enough.

Verify:

```bash
cd ml/relationformer
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -c "from models import build_model; print('model ok')"
```

## 4. Optional: the DEXPI pipeline's own venv

Independent of the ML track. From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest tests/ -q     # expect 74 passed
```

---

# Running the training loop

All commands run from `ml/relationformer/`.

## Smoke test first (~2 min)

```bash
.venv/bin/python train_pid2graph.py \
  --config configs/pid2graph_2d.yaml --exp_name smoke -b 1 --debug
```

`--debug` cuts to 30 train / 30 val samples and disables dataloader workers. You should
see parameter count (~39.7M), a loss that decreases, and a
`=== PID2Graph validation, epoch N ===` block after each epoch. Kill it with Ctrl-C once
you have seen a couple of epochs - it will otherwise run all 150.

Then delete the run folder before reusing the name:
`rm -rf trained_weights/runs/smoke_10`

## Find your batch size

Start at 4 on a 12GB card and raise until it OOMs, then drop one step:

```bash
.venv/bin/python train_pid2graph.py \
  --config configs/pid2graph_2d.yaml --exp_name bs4 -b 4 --debug
```

Watch `nvidia-smi` in another terminal. Reference: batch size 1 used 2.3GB on the 3060.
With headroom to spare you can also raise `HIDDEN_DIM` back to 512 and `ENC/DEC_LAYERS`
to 6 in `configs/pid2graph_2d.yaml` (closer to the paper's model), but change one thing
at a time and re-check memory.

## The real run

```bash
nohup .venv/bin/python train_pid2graph.py \
  --config configs/pid2graph_2d.yaml --exp_name pid2graph_run1 -b 4 \
  > train_run1.log 2>&1 &
```

`nohup ... &` matters - this runs for days and you do not want it dying with the
terminal. Follow it with `tail -f train_run1.log`.

Expect **~3.5-6.5 days** for the configured 150 epochs on a 4070 (see the timing table
below). To finish sooner, lower `TRAIN.EPOCHS` - the LR drops at `LR_DROP: 120`, so if
you shorten the schedule, scale that down proportionally (e.g. 50 epochs -> `LR_DROP: 40`).

## Monitoring

```bash
tensorboard --logdir trained_weights/runs
```

Or grep the log:

```bash
grep -E "node_mAP_IoU_0.50_0.95_0.05_MaxDet_100|edge_mAP:" train_run1.log
```

Checkpoints land in `trained_weights/runs/<exp_name>_10/models/`, keeping the 3 best by
node mAP.

## Resuming

```bash
.venv/bin/python train_pid2graph.py \
  --config configs/pid2graph_2d.yaml --exp_name pid2graph_run1 -b 4 \
  --resume trained_weights/runs/pid2graph_run1_10/models/<checkpoint>.pt
```

## Evaluating a checkpoint

```bash
.venv/bin/python train_pid2graph.py \
  --config configs/pid2graph_2d.yaml --exp_name pid2graph_run1 -b 4 \
  --resume trained_weights/runs/pid2graph_run1_10/models/<checkpoint>.pt --eval
```

## What the metrics mean

Each validation epoch prints COCO-style node mAP (overall + per class) and edge mAP:

```
=== PID2Graph validation, epoch 5 ===
  edge_mAP: 0.0000
  node_mAP_IoU_0.50_0.95_0.05_MaxDet_100: 0.0710
  node_valve_mAP_IoU_0.50_0.95_0.05_MaxDet_100: 0.0216
  ...
```

`node_mAP_IoU_0.50_0.95_0.05_MaxDet_100` and `edge_mAP` are the two headline numbers,
directly comparable to the published baselines (multiply by 100):

| | node mAP | edge mAP |
|---|---|---|
| Stürmer et al. 2025 (OPEN100) | 52.14 | 45.89 |
| SynthPID E2, synth-only | 64.2 | 63.8 |
| SynthPID E3, mixed | 71.3 | 72.6 |

Caveat on comparability: those are reported on the 12 **Complete** OPEN100 sheets, while
this config trains and validates on **Patched** data. Same metric, different split - so
treat it as indicative until you evaluate on the held-out Complete sheets directly.

Edge mAP stays near zero for a long time. An edge needs *both* endpoint boxes matched at
IoU >= 0.5 before it can score at all, so it lags node mAP by a wide margin early on.
That is expected, not a bug - the metric returns exactly 1.0 on synthetic perfect input
and 0.0 on wrong edges (unit-tested).

---

# Reference

## Dataset

PID2Graph, Zenodo record 14803338 (9.3GB zip, ~14GB extracted) at `data/PID2Graph/`
(gitignored). Cite: Stürmer, J. M., Graumann, M., & Koch, T. (2025), *From Engineering
Diagrams to Graphs: Digitizing P&IDs with Transformers*, IEEE DSAA 2025.

| subset | graphml | images |
|---|---|---|
| Complete/Dataset PID | 500 | 500 png |
| Complete/PID2Graph OPEN100 | 12 | 12 png |
| Complete/PID2Graph Synthetic | 250 | 250 jpg |
| Patched/Dataset PID | 19,462 | png |
| Patched/PID2Graph OPEN100 | 1,629 | png |
| Patched/PID2Graph Synthetic | 14,708 | png |

Training uses the three **Patched** subsets (35,799 pairs -> 32,219 train / 3,580 val).
Full sheets are 2604x1744; patches are 1500x1500. The 12 unpatched Complete OPEN100
sheets are deliberately held out.

### Real schema (verified, not taken from the paper)

The paper and Zenodo page only say "bounding box and label". The actual GraphML:

- node keys: `d0`=label (string), `d1`-`d4`=xmin/xmax/ymin/ymax (**double**) on real
  symbol nodes; `d5`-`d8`=same names (**long**) on structural nodes. Read by
  `attr.name`, so the two key sets aren't ambiguous.
- edge key: `d9`=edge_label, values `solid` / `non-solid`.
- Boxes are **xyxy**, matching what the model's transforms expect - no conversion.
- Label vocabulary, confirmed by scanning 200 random files across all six subsets:
  7 real classes (`arrow`, `general`, `inlet/outlet`, `instrumentation`, `pump`,
  `tank`, `valve`) + 4 structural (`connector`, `crossing`, `border_node`,
  `background`). No surprises outside this set.

### Connector collapse

Raw edges never run symbol-to-symbol. They route through waypoints:
`valve8 -> connector157 -> crossing158 -> connector203 -> tank12`. Relationformer wants
direct object-to-object triples, so `_collapse_graph` walks each waypoint chain and
emits one edge between the real endpoints. On `Complete/PID2Graph OPEN100/3.graphml`:
280 raw nodes / 279 raw edges -> 88 symbols / 106 real connections.

`border_node` marks a line leaving a patch. It is **not** a symbol - treating it as one
would train the model to detect a phantom object at every tile edge.

### Symbol-free patches (~6.75%)

Some patches contain zero symbols. These are *not* blank: they're tiles covering a pipe
run between equipment, holding only `border_node`/`crossing` waypoints and their edges.
Example, `Patched/PID2Graph OPEN100/0/41.graphml` - a 50KB image with two pipe segments
meeting at a crossing, both running off the tile edge.

They are **kept**, as useful negatives (pipe present, no symbol to detect). This requires
the empty-set guard in `loss_edges`, or the edge loss is `nan` on them - and one `nan`
step propagates through AdamW and destroys every weight.

## Why the `scene_graph` branch

`road_network_rgb` looks closer (simple image+graph loading, 2D, undirected) but its head
is `nn.Linear(HIDDEN_DIM, 2)` - hardcoded to node-exists vs not. It reads
`config.MODEL.NUM_CLASSES` and never wires it in. Road networks only have one node type,
so this is correct *for roads* and useless for P&ID.

`scene_graph` uses `nn.Linear(hidden_dim, NUM_OBJ_CLS(+1))`. Confirmed live: with
`NUM_OBJ_CLS: 7`, `pred_logits` comes out `(1, 99, 8)` = 7 classes + background.

Its *data* pipeline is Visual-Genome-specific (HDF5, proposal files, frequency-bias
`VG.pt`), so we use its model/loss/matcher and supply our own loader. That combination -
their model, our data path - is most likely what Stürmer et al. and SynthPID did too,
since neither published branch reads PID2Graph's format.

## Upstream bugs found

Captured in `pid2graph/upstream-fixes.patch`.

1. **`losses.py` - edge loss never computes without the VG frequency file.** The
   `F.cross_entropy` call and its `return` were both nested inside
   `if len(freq_dist) > 0:`, which is only populated when `FREQ_BIAS` is on. With it off,
   `loss_edges` returned `None` and training crashed at `None * float`. The frequency
   prior is an optional additive term, not a precondition. **This means the published
   `scene_graph` branch cannot train without Visual Genome's `VG.pt`.**
2. **`util/misc.py` - broken version checks.** `float(torchvision.__version__[:3]) < 0.5`
   reads `"0.20.1"[:3]` = `"0.2"` = `0.2`, which is still `< 0.5`. So on every modern
   torchvision it took the pre-0.5 legacy branch and imported the long-removed
   `_NewEmptyTensorOp` / `_new_empty_tensor`. Replaced with a tuple comparison.
3. **`utils.py` - unused heavy imports at module scope.** `mmcv` (only for `get_logger`),
   `pyvista`, and `skimage.marching_cubes_lewiner` (removed from modern scikit-image) are
   imported at import time but only used by road-network/3D-vessel helpers. Made optional.

Plus two environment gotchas, both handled in `setup.sh`:

- GNU ld 2.42 fails to link the CUDA extension against torch's `libtorch_cpu.so` with
  `file too short`. Use `-fuse-ld=gold`.
- `monai>=1.6` requires `torch>=2.8`, silently upgrading torch to a cu130 build and
  breaking the extension compile against a CUDA 12.0 toolkit. Pin `monai==1.3.2
  --no-deps`.

## Config notes

`configs/pid2graph_2d.yaml` is scaled down from the VG config for a single consumer GPU:
`HIDDEN_DIM` 512->256, `ENC/DEC_LAYERS` 6->4, `NUM_QUERIES` 201->100, resnet50 backbone,
batch size 1. ~39.7M parameters. The paper used 4x A6000 (48GB each); this fits 2.3GB.

Raise these once you have measured headroom on the real training card.

## Measured throughput and time estimate

On the 3060 laptop, batch size 1, 576-800px after augmentation: **median 212 ms/step,
2.30GB peak VRAM**.

| | ms/sample | per epoch | 150 epochs |
|---|---|---|---|
| 3060 laptop (measured) | 212 | 114 min | ~12 days |
| 4070 desktop @ bs1 (est.) | ~118 | ~63 min | ~6.6 days |
| 4070 desktop @ bs4 (est.) | ~65 | ~35 min | ~3.6 days |

The 3060 row is measured. **The 4070 rows are extrapolations** assuming ~1.8x throughput
and sublinear batch scaling - treat them as rough. Re-measure on the actual card before
planning around them:

```bash
.venv/bin/python train_pid2graph.py --config configs/pid2graph_2d.yaml --exp_name bench -b 4 --debug
```

12GB should allow a larger batch and/or `HIDDEN_DIM: 512`, which changes these numbers.

## Evaluator

`pid2graph/evaluator_pid2graph.py` replaces the upstream `evaluator_sg.py`, which is
unusable here for three reasons: it reports Visual-Genome scene-graph recall@K rather than
mAP, it requires `config.DATA.LABEL_DATA_DIR` (a VG `idx_to_predicate` json that does not
exist for PID2Graph), and although it wires `MeanBoxAP` as its key metric it feeds that
metric *normalized cxcywh* boxes while the underlying `box_iou_np` expects absolute xyxy -
so its AP figure is not comparable to anything.

Ours reports:

- **Node mAP** - COCO mAP@[.5:.95] over the 7 symbol classes, plus per-class AP/AR, which
  is the same breakdown SynthPID's Table 4 gives.
- **Edge mAP** - a predicted edge is a true positive only if both endpoint boxes match
  distinct GT boxes at the IoU threshold *and* the GT has an edge between those two
  matched nodes, scored by predicted relation probability. Endpoint matching is greedy by
  descending IoU, one prediction per GT node, mirroring COCO's assignment. Class labels
  are deliberately not required to match, since node mAP already scores classification;
  edge mAP measures topology.

The papers describe edge mAP only in prose, so the exact matching rule above is our
interpretation. It is unit-tested against hand-built cases: 1.0 on perfect input, 0.0 on a
wrong edge, 0.0 when endpoint boxes miss, and NaN (excluded from the mean) when a sample
has no GT edges.

## Not done yet

- No real training run. No weights. The only mAP numbers so far are from an untrained or
  30-sample-overfit model and mean nothing.
- Evaluation is on the **Patched** split. The published baselines are on the 12 unpatched
  **Complete** OPEN100 sheets, so a like-for-like comparison still needs a config pointing
  `PID2GRAPH_SUBSETS` at `Complete/PID2Graph OPEN100` for validation.
- The team's own 7 P&IDs are exported as PID2Graph-schema GraphML in
  `exports/pid2graph_export/` but are **graph-only** - no rendered PNGs, so they can't be
  used as image-based eval yet. `subprojects/MEM4CHEM` (pptx) has no parser at all.
- `solid` vs `non-solid` is carried on the target dict but folded to a single
  `connected` predicate class (`NUM_REL_CLS: 1`). Splitting it into two predicate classes
  is a one-line config change plus a loader tweak, if dashed/signal lines matter for the
  DEXPI output.
