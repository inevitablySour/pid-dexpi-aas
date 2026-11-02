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
`=== PID2Graph validation, epoch N ===` block after each epoch. `--debug` stops after 2
epochs; pass `--epochs N` to change that (it also overrides `TRAIN.EPOCHS` without
`--debug`).

Then delete the run folder before reusing the name:
`rm -rf trained_weights/runs/smoke_10`

## Find your batch size

Measured on the 4070 (see the throughput table below): bs4 peaks at 7.3GB, bs8 OOMs.
Use `-b 4` there. On another card, start at 4 and raise until it OOMs, then drop one step:

```bash
.venv/bin/python train_pid2graph.py \
  --config configs/pid2graph_2d.yaml --exp_name bs4 -b 4 --debug
```

Watch `nvidia-smi` in another terminal. Reference: batch size 1 uses ~2.1GB.
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

## Fine-tune with real drawings (cross-validation)

The base run (`sheetsplit_bs4`, 150 epochs, synthetic data only) scores **0.360 node /
0.050 edge mAP on whole OPEN100 sheets** (published baseline: 0.521 / 0.459). The
fine-tune fixes what that run measured as wrong, starting from its weights:

| fix | what | where |
|---|---|---|
| 1 | Crossing pipes no longer counted as connected (56% of training edges were fake) | `_collapse_graph` / `_pass_through_partner` |
| 2, 3 | Real OPEN100 sheets in training, 4-fold CV: 8 train / 1 val / 3 test sheets per fold (inlet/outlet only exists there) | `real_fold_sheets`, `--fold` |
| 4 | Team draw.io drawings rendered + labelled as training data (LightUp held out); grey-fill / line-weight augmentation | `make_team_data.py`, `StyleAugment` |
| 5 | Whole-sheet scoring at the training scale, comparable to the paper | `eval_sheets.py` |
| 6 | Process vs signal lines as 2 relation classes | `NUM_REL_CLS: 2` |
| 7 | Balanced sampling: real 31% / team 6% of draws; arrow 4%->13%, inlet/outlet 0.3%->4% | `sample_weights`, `BALANCED_SAMPLING` |

```bash
# once: team drawings -> data/PID2Graph/{Complete,Patched}/Team[ test] (needs ml/tools/drawio.AppImage)
.venv/bin/python ml/pid2graph/data/make_team_data.py
# all 4 folds (~4 h each on a 4070), then pooled scores:
ml/pid2graph/run_folds.sh ml/relationformer/trained_weights/runs/sheetsplit_bs4_10/models/checkpoint_epoch=150.pt
```

Each fold is scored only on its own 3 held-out sheets, so the pooled number covers all
12 sheets, each by a model that never saw it. Every fold also scores LightUp (team test,
never trained on). `DEBUG=1` runs a 30-patch smoke test of the whole chain.

The model for the app is then trained once on everything - all 12 OPEN100 sheets and
every team drawing, LightUp included (`--fold -1`) - with real/team patches showing a pump
drawn 3x more often (`pid2graph_final.yaml`, `DATA.CLASS_BOOST`; OPEN100 pumps scored ~0).
Nothing is held out, so the 4-fold scores are its estimate:

```bash
ml/pid2graph/run_final.sh ml/relationformer/trained_weights/runs/sheetsplit_bs4_10/models/checkpoint_epoch=150.pt
# -> ml/models/pid2graph_final.pt (~4 h)
```

Team v3 4-fold results (2026-10-07): LightUp node 0.931 / edge 0.489 (team-v1 labels'
run: 0.911 / 0.347), OPEN100 0.689 / 0.281 (0.687 / 0.271).

Notes: OPEN100 patches are the complete sheets enlarged 2.65x, so whole-sheet scoring
tiles at 1500/2.65 px. `docs/examples/pid2graph_export/*.graphml` (draw.io coordinates) and the
training data (`make_team_data.py`, image pixels) share one labelling,
`export_pid2graph.labelled_symbols`: real symbols only. The first exports labelled every
draw.io cell, text labels included, as "general" (LightUp: 113 nodes -> 59 symbols). For the app, point `PID_MODEL_CHECKPOINT` at a fine-tuned checkpoint.

## Evaluating a checkpoint

```bash
.venv/bin/python train_pid2graph.py \
  --config configs/pid2graph_2d.yaml --exp_name pid2graph_run1 -b 4 \
  --resume trained_weights/runs/pid2graph_run1_10/models/<checkpoint>.pt --eval --eval_split test
```

`--eval_split test` scores the held-out OPEN100 sheets (`DATA.PID2GRAPH_TEST_SUBSETS`);
the default `val` scores the sheet-level validation split used for checkpoint selection.

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

Caveat on comparability: those are reported on the 12 **Complete** OPEN100 sheets. Our
test split is the same 12 sheets, never trained on, but scored per **patch** rather than
per whole sheet - same metric and same drawings, different granularity, so treat it as
indicative.

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

Training uses **Patched** Synthetic + Dataset PID, split **by sheet** 90/10: 30,744 train
patches (675 sheets) / 3,426 val (75 sheets). **Patched OPEN100** (1,629 patches, 12 sheets)
is the held-out test set - not trained on, not used for checkpoint selection.

Why by sheet: patches overlap heavily (PID2Graph Synthetic sheet 241: 7000x4500 sheet,
58 patches of 2000x2000 at a 500px stride - neighbours share 75%, each pixel lands in ~7
patches). The first full run split per patch; 100% of val patches shared a sheet with
train and val node mAP read 0.91 after 2 epochs - memorisation, not generalisation. That
run was discarded.

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

### Line points (v6, `DATA.LINE_POINTS`)

With `LINE_POINTS: <box px>` the waypoints a line visibly turns or joins at stay as nodes
(`_point_graph`): `bend` (a corner) and `junction` (a tee, or a 4-way joint that is not
two pipes crossing) - classes 8 and 9, `NUM_OBJ_CLS: 9`. As in PID2Graph (its author, 2026-10): only "crossing" nodes are
points (2 arms: an ankle/bend; more: a junction); "connector" nodes - where a line enters a
symbol - are never used; a 4-arm crossing of two straight pipes is walked straight through. Folding the points away again gives exactly `_collapse_graph`'s
links (checked on 1,212 OPEN100/synthetic/Dataset PID graphs), and the sheet reader does
the same to a model's output (`pipeline/point_links.contract`), so v6 scores compare
with v3/v5. Team drawings get points from their drawn routes
(`export_pid2graph.line_points`, `make_team_data.py --no-arrows --points` -> `Team v6`, i.e. Team v3 + points); rounded
corners, 2-5 unit kinks and corners under a symbol are not points. Training since 2026-10-14 13:40 (4 folds, then final):
`ml/logs/run_v6.sh` (tmux `train-v6`).

### Automatic retraining (auto_retrain.py, nightly)

Sheets added in the app ("Add this drawing to the training data") go to Reviewed (training) or
Reviewed test (held out). Cron runs `auto_retrain.py --if-due` at 01:00. Due when the reviewed
sheets added since the last retraining hold >= 10 % of the hand-labelled units the live model
has (at least 150), a unit being one labelled item or connection - so a big P&ID counts more
than a small one. Then a candidate is trained like the live model (pid2graph_final.yaml +
Patched/Reviewed, ~4 h), both are scored on the held-out sheets (eval_sheets.py --set
reviewed), and the candidate replaces ml/models/pid2graph_final.pt only if no worse on items
and links (within 0.02 - the spread between two identical trainings); the old model goes to ml/models/archive/. With no held-out sheets
the candidate waits for `--approve` (or the app's button). State: ml/models/retrain_state.json.

### Text reading (OCR)

Since 2026-10-14 the sheet reader reads text with **docTR** (db_resnet50 + PARSeq, on this
environment's PyTorch; `--ocr easyocr` for the old reader). Scored on the team drawings' 272
tags (OCR text at the item = its true tag), same tiling / rotations / merge for every engine:

| engine | clean renders | screenshot quality (60 %, blur, JPEG) |
|---|---|---|
| EasyOCR (before) | 81 % | 49 % |
| PaddleOCR PP-OCRv5 mobile | 96 % | 89 % |
| **docTR** | **96 %** | **93 %** |

PaddleOCR is as good on clean drawings but needs its own framework next to PyTorch; Surya
0.22 is now a vision-language reader needing a GPU server. Tag reads are also put right by
grammar (look-alikes on the wrong side of a tag: P5V201 -> PSV201) and snapped to the
drawing's known tags (spec sheet, draw.io file) - look-alike swaps only (`ocr_tags.py`).

### v6 (bends and tees as nodes): stopped 2026-10-14, not adopted

Trained like v3 plus PID2Graph's bend/junction points (Team v6 = Team v3 + points). Fold 0
against v3's fold 0, both with the new tile merge: items better (OPEN100 mAP 0.704 -> 0.728),
links much worse (OPEN100 mAP 0.415 -> 0.298, LightUp 0.515 -> 0.400). Stopped after fold 0.
Diagnosis on fold 0's 3 sheets (381 true links, confidence >= 0.5): v6 found MORE true links
(308 vs v3's 283) but predicted 1,861 links (v3 554; precision 0.17 vs 0.51) - it predicted
275 points for 121 real ones, and its direct symbol links got worse (at >= 0.7: 202 of 321
false). point_links.prune_points (a bend keeps 2 links, a tee 3-4, dangling points dropped)
lifts its F1 0.27 -> 0.49-0.55; v3 is 0.61-0.66. Checkpoint kept:
trained_weights/runs/finetune_v6_fold0_10.

### Tile stitching (pipeline/stitch.py)

Sheets are read in 1400 px tiles at 700 px stride; the tiles' detections are merged. Since
2026-10-14 the merge is PID2Graph's author's advice (by mail): a detection near an inner tile
edge counts less (half at the edge), a cut-off box joins the symbol box that contains it, and
boxes are fused by score-weighted average (WBF). Same v3 fold models, OPEN100, old -> new:
items mAP 0.689 -> 0.719 (AP50 0.730 -> 0.748), links mAP 0.281 -> 0.343 (AP50 0.524 -> 0.604),
better in every fold; LightUp items 0.931 -> 0.957, links 0.489 -> 0.552. `--merge greedy` in
eval_sheets.py scores the old merge. Compare later models with the new numbers.

Note: the paper reports everything at IoU 0.5 (AP50) and scores the graph WITH bend and
crossing nodes; our mAP averages IoU 0.5-0.95 and scores symbol-to-symbol links (harder).

### Several zooms and several models (detect, 2026-10-20)

`detect` takes several tile sizes and several checkpoints: every model reads the sheet at
every tile size, all detections are merged as one (stitch.py), a symbol's score is scaled by
the share of those runs that found it, a connection's probability is its mean over the runs.
Smaller tiles show small symbols larger. v3 fold models, OPEN100 four folds (tile sizes at
training-patch scale; the app uses 2350, 1850, 1400, 950 on its 1:1 pictures):

| tile sizes | items mAP | links mAP | LightUp links |
|---|---|---|---|
| 1500 (before) | 0.720 | 0.343 | 0.552 |
| 2000, 1500, 1000 | 0.765 | 0.423 | 0.608 |
| **2500, 2000, 1500, 1000** | **0.770** | **0.434** | **0.620** |
| 2000, 1500, 1200, 1000 | 0.760 | 0.427 | 0.622 |

LightUp's items mAP seems to drop (0.99 -> 0.85), but only because its labels have no flow
arrows: the extra zooms find the drawn ones, which count as false, and the arrow class's AP
falls from 1 to 0. Without that class 0.992 -> 0.996 (valves 23/24 -> 24/24 found).
Combining models, fold 0 (the same split): v3 + the v3 repeat at four zooms, links 0.533 ->
0.552 (LightUp 0.622 -> 0.633); adding v4 / v5 (arrows labelled differently) is worse
(OPEN100 items 0.687, LightUp links 0.585). The live site has one v3 final model, so it runs
v3 at four zooms; a second v3 final (another ~4 h training) would add the ~0.02.
For the app, `drop_shadows` removes a low-score symbol (seen at one zoom only) lying mostly
inside better ones (a box over two symbols, a piece of one); the rest of the low-score ones
are mostly small symbols the usual zoom misses, flagged in the guided check as unsure.

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

Training steps only (no eval/checkpoint), 4-8 loader workers, 32,219 training patches
per epoch. Measured on the RTX 4070 (12GB):

| config | ms/sample | peak VRAM | per epoch | 150 epochs |
|---|---|---|---|---|
| upstream as shipped (cudnn.benchmark on, gc + empty_cache every step), bs1 | 1350-1760 | 10.0GB | ~12-16 h | months |
| cudnn.benchmark off, gc + empty_cache kept, bs1 | 290 | 2.1GB | ~2.6 h | ~16 days |
| **both fixed, bs1** | 102 | 2.1GB | ~55 min | ~5.7 days |
| **both fixed, bs2** | 91 | 3.9GB | ~49 min | ~5.1 days |
| **both fixed, bs4** | 83 | 7.3GB | ~45 min | ~4.6 days |
| both fixed, bs8 | OOM | - | - | - |

Two upstream settings were responsible for a ~17x slowdown:

- **`cudnn.benchmark = True`** autotunes conv algorithms per input shape. The random
  resize + crop augmentation makes nearly every shape new, so it never stops tuning (and
  the tuning workspaces push VRAM to 10GB at bs1). Now `False` in `train_pid2graph.py`.
- **`gc.collect()` + `torch.cuda.empty_cache()` after every step** in `trainer.py` forces
  the CUDA allocator to re-allocate everything on the next step. Removed via
  `upstream-fixes.patch`.

Batch size gives diminishing returns past 2; bs4 is the fastest that fits, with ~4GB
headroom for the desktop and unusually dense patches. An earlier 3060-laptop figure
(212 ms/sample) was measured differently and is not comparable to this table.

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
- Test scoring is per **patch** on held-out OPEN100. The published baselines score whole
  **Complete** sheets; a like-for-like number needs sheet-level inference (run the patches,
  stitch predictions back into one sheet graph, then score).
- The team's draw.io P&IDs are rendered and labelled by `make_team_data.py` (6 for
  training, LightUp held out). Their connection labels come from the draw.io lines and
  about half have one end resolved to the nearest shape - not individually verified. `subprojects/MEM4CHEM` (pptx) has no parser at all.
- `solid` vs `non-solid` is carried on the target dict but folded to a single
  `connected` predicate class (`NUM_REL_CLS: 1`). Splitting it into two predicate classes
  is a one-line config change plus a loader tweak, if dashed/signal lines matter for the
  DEXPI output.

## Image-only connections: model + pipe tracer

The model only links symbols that share a tile, so long pipes are invisible to it.
`trace_pipes.py` follows the drawn lines over the whole sheet (no tiles): linework only
(text and symbol insides erased), thinned to 1 px, cut at junctions, short gaps bridged
(many bridges = dashed = signal line), then walked from every symbol - straight on at
crossings, down every branch at tees. It returns each pipe's path and its flow direction
from arrowheads where a line meets a symbol. `infer_sheet.py` runs it after the model and
OCR; the review app uses traced pipes first and the model's predictions for pairs the
tracer did not reach.

Scored on the 6 team sheets against their draw.io labels (`eval_tracer.py`):

| connections from | precision | recall | F1 |
|---|---|---|---|
| model only | 0.92 | 0.56 | 0.70 |
| tracer only | 0.82 | 0.60 | 0.69 |
| both | 0.84 | 0.86 | 0.85 |

Flow directions from arrowheads: 95 of 141 pipes with a known direction called, 0 wrong.

2026-10-15, instrument chains, containers, mid-pipe arrows, unboxed symbols (true boxes,
team = 22 Team v3 sheets, OPEN100 = 100 sheets, no OCR):

| change | team P / R / F1 | OPEN100 P / R / F1 |
|---|---|---|
| tuned tracer (before) | 0.93 / 0.80 / 0.86 | 0.88 / 0.66 / 0.76 |
| + frames keep only their border | 0.91 / 0.81 / 0.86 | 0.87 / 0.67 / 0.76 |
| + dashed short links, frames not blocking them | 0.91 / 0.84 / 0.87 | 0.87 / 0.67 / 0.76 |
| + bent dotted runs (`dotted`, on) | 0.89 / 0.85 / 0.87 | 0.87 / 0.67 / 0.76 |

With the model's boxes on the 5 Lumen sheets, instrument links missed (both ends boxed):
33 -> 24. Unboxed symbols on a pipe ("knots", `missed`): hiding every 3rd true inline box,
6 / 33 (team) and 18 / 108 (OPEN100) found, **0 false alarms** with all boxes present (a
density measure tried first: ~5 false alarms a sheet - dropped). On the Zaiput sheet
it flags the 2 unboxed check valves and MFC201, all real; `infer_sheet.py` adds the model's
half-seen symbols (background won, a symbol class >= 0.2) on a traced pipe (M202 there).
The review adds both as flagged "possible missed item" rows and splits the pipe through
them.

Lines that change colour (a black pipe with a light gray or yellow stretch): the tracer gets
the colour picture and counts a pixel as ink by its darkest channel (a yellow line's
brightness is near white) and light gray down to 235 - both only as thin line pieces, not
fills or the soft edges of dark lines (`_ink`). Team / OPEN100 scores unchanged (no such
lines there).

Jump-over crossings (a pipe hopping over another in a small arc) were read as a tee: the
arc's arms bend away from the top of the hump, so the 4-way junction there did not look like
two straight lines. Now a 4-way junction with one straight pair pairs the other two as well
(P&IDs have no 4-way joins). And a line ending in a junction blob beside a symbol was put on
the symbol (meant for arrowheads, which thin into a blob there) even when it runs past the
symbol into a tee right beside it - now only when it heads into the symbol. OPEN100
P / R / F1 0.87 / 0.67 / 0.76 -> 0.88 / 0.69 / 0.78, team unchanged; on Zaiput the PCV's
pipe over the hop no longer joins the BPR300 line.

### Straight-line extractor (pipeline/trace_lines.py, the default from 2026-10-20)

Pipes as straight line pieces, not a pixel skeleton: horizontal and vertical runs taken out
separately (an opening with a line), each a segment with two ends; dashes in a row with
regular gaps a dashed segment; how segments meet decided geometrically - in line across a
short gap / erased text / a symbol's margin / a crossed line (a hop) = one line; an end
touching another line = a tee or bend (rounded bends: ends up to 0.6 symbol apart when ink
joins them); two lines through each other = a crossing. trace_pipes.py stays
(`infer_sheet.py --tracer pixels`; its knot check still gives `missed`). True boxes:

| | team P / R / F1 | OPEN100 P / R / F1 |
|---|---|---|
| trace_pipes (skeleton) | 0.89 / 0.85 / 0.87 | 0.88 / 0.69 / 0.78 |
| trace_lines | 0.90 / 0.87 / 0.89 | 0.88 / 0.69 / 0.77 |

End to end on sheets the models never saw (v3 fold models, OCR, model boxes; F1 of the
review's connections, LightUp / OPEN100):

| pipeline | LightUp | OPEN100 |
|---|---|---|
| before: one zoom, trace_pipes, model lines >= 0.9 | 0.92 | 0.66 |
| line extractor only | 0.87 | 0.68 |
| four zooms only (every item kept) | 0.78 | 0.53 |
| four zooms, items >= 0.75, trace_lines, >= 0.6 | 0.90 | 0.79 |
| **four zooms, items >= 0.5, trace_lines, model lines >= 0.6** | **0.93** | **0.76** |

Four zooms with every item kept is worse end to end - an item seen at one zoom only (a
duplicate, a piece) cuts traced pipes; the scores hardly notice it (ranked low). So items
found at fewer than half the zooms (`SURE`) go to the half-seen list: a traced pipe running
through one makes it a flagged "possible missed item".

**Correction (same day): what the review SHOWS.** The numbers above count pairs: the labels
(and the tracers) join every two items on a pipe network with tees - a header with ten
branches is 45 pairs, all counted. Shown in the editor that was a web of lines (and lines
across flow arrows, drawn straight, which none of the scores above included). Now each pipe
network is drawn as its shortest tree along the pipes (review._network_trees; the other
pairs still count as joined), lines across flow arrows follow the pipe through the arrows,
also as a tree, and items seen at one zoom only are not offered as possible missed items
(on LightUp most were none). Scored on the review's shown lines, every step included
(`shown` scripts; unseen sheets, v3 fold models), per drawing:

| | lines shown | precision | wrong, to delete | real connections missing |
|---|---|---|---|---|
| LightUp before today | 97 | 0.90 | 9.2 | 9.2 of 97 |
| LightUp now | 67 | 0.88 | 8.2 | 10.5 of 95 |
| OPEN100 before today | 66 | 0.46 | - | 114 of 146 |
| OPEN100 now | 52 | 0.51 | 25.5 | 104 of 137 |

Rerun: `ml/relationformer/.venv/bin/python ml/pid2graph/eval/shown_gen.py --out <dir>` (the app's
model step on every unseen sheet), then `.venv/bin/python ml/pid2graph/eval/eval_shown.py <dir>`.
tests/test_shown_lines.py holds these outputs (tests/data/shown) and fails below the numbers
above - judge changes by it, not by pair scores alone.

2026-10-24:
- **Diagonal and curved pipes** (trace_lines.py `_free_lines`): the openings take out only
  horizontal and vertical runs, so a diagonal or curved pipe was dropped. What is left of the
  linework after the straight runs and dashes, if thin and long enough, is thinned to its
  centre line and becomes a free-shaped segment: it meets other lines like the straight ones
  (touching = a tee or bend; a little apart, when ink joins them and the end heads there),
  points into symbols, and paths follow its drawn shape. True boxes: team F1 0.89, OPEN100 0.77
  (unchanged - their pipes are straight; tests cover a diagonal, a curve, a diagonal tee). A
  grid of the segments keeps it fast (a dense OPEN100 sheet 133 s -> 7 s).
- **Pipe networks in DEXPI with tees**: the review shows a traced network as a tree of lines
  (net), and to_dexpi builds each network of 3+ items once with PipeTee fittings, members in
  order along the tree, flows from the lines (review._traced_networks, as draw.io networks);
  lines across a flow arrow join the networks on both sides. Zaiput: 4 networks, 6 tees.
- **Small symbols**: items seen at one zoom only (score < SURE) were measured as candidates on
  the 16 unseen drawings: 709, of which 53 real missed symbols (LightUp 1 of 31). No evidence
  from the picture separates them (on a traced pipe: precision 0.10; the knot check, valve
  shape, size: none) - and small ones are 13 % real; a fifth, closer zoom (700) adds +0.002
  items / +0.011 links on OPEN100. The labelled sets hardly have small symbols the model
  misses (LightUp none; OPEN100 small 11 % vs normal 9 %); MEM4CHEM's tiny icons are a
  drawing-style gap - training on MEM4CHEM is the way. (A plain look-alike search - one symbol
as a template, OpenCV - gave thousands of matches per symbol: pipes and text look like parts of
a valve. Not built.)

**Shown-lines scoring, made fair (2026-10-27)**: OPEN100's labels keep flow arrows as items
(pump - arrow - tank); the review leaves arrows out and joins pump - tank straight through.
The scoring now does the same to the labels (shown_lines.labelled_links), counts a pipe
network's items as joined as DEXPI builds them, and reports separately what is missing only
because an item has no DEXPI type yet. And "possible missed items" are now suggestions (left
out, the pipe not split): splitting made both sets worse. Per drawing, unseen sheets:

| | lines shown | wrong | missing (of them: item without a DEXPI type) | precision |
|---|---|---|---|---|
| LightUp before (2026-10-20) | 97 | 9.2 | 9.2 of 97 (0) | 0.90 |
| LightUp now | 67 | 6.8 | 10.5 of 95 (0) | 0.90 |
| OPEN100 before | 66 | 24.1 | 102.7 of 145 (46.7) | 0.64 |
| OPEN100 now | 53 | 14.5 | 68.0 of 139 (42.8) | 0.73 |

(The "before" rows above were scored with that day's review rules, which split pipes through
possible missed items. With today's review rules for both, and joined pairs - every pair the
review connects, a pipe network joining all its items, so a wrong merge counts in full:

| per drawing | wrong lines | joined pairs wrong (precision) | missing |
|---|---|---|---|
| LightUp, one zoom + trace_pipes | 7.2 | 7.2 (0.92) | 8.2 of 97 |
| LightUp, four zooms + trace_lines | 6.8 | 7.2 (0.92) | 10.5 of 95 |
| OPEN100, one zoom + trace_pipes | 22.7 | 22.7 (0.65) | 102 of 145 |
| OPEN100, four zooms + trace_lines | 14.5 | 23.4 (0.75) | 68 of 139 |

LightUp level on precision, two more connections missing a drawing; OPEN100 clearly better.) Mid-pipe flow arrows give 10 of 13 arrow-bridged lines on Zaiput a direction.

## Before a training run

`check_data.py` checks everything a fine-tune run trains and validates on, through the
training code's own loading: no sheet shared between train / val / test in any fold, the
team test drawing never in training, every label file sane (classes, boxes inside the
image, connections), no patch with more symbols than queries.
