# ML track: Relationformer on PID2Graph

Image-to-graph symbol + topology extraction, as an alternative to the heuristic
drawio-parsing pipeline in `src/pid_dexpi_aas/`. The two tracks are independent.

## Status

Working end to end on this laptop (RTX 3060, 6GB): data loads, model forward, all five
losses, backward, optimizer step, 150 consecutive steps with no NaN and loss descending
35.6 -> 15.9. **No model has actually been trained yet** - only that the pipeline runs.

## Quick start

```bash
./ml/pid2graph/setup.sh          # clone, patch, venv, compile CUDA op
```

Then from `ml/relationformer/`:

```bash
.venv/bin/python train_pid2graph.py --config configs/pid2graph_2d.yaml --exp_name pid2graph_run1 -b 1
```

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

## Not done yet

- No training run. No weights. No mAP numbers.
- Evaluation against the paper's baseline (Stürmer et al. report 52.14 node / 45.89 edge
  mAP on OPEN100) is not wired up - `evaluator_sg.py` computes VG-style recall@K, not
  detection mAP, so scoring against that baseline needs work.
- The team's own 7 P&IDs are exported as PID2Graph-schema GraphML in
  `exports/pid2graph_export/` but are **graph-only** - no rendered PNGs, so they can't be
  used as image-based eval yet. `subprojects/MEM4CHEM` (pptx) has no parser at all.
- `solid` vs `non-solid` is carried on the target dict but folded to a single
  `connected` predicate class (`NUM_REL_CLS: 1`). Splitting it into two predicate classes
  is a one-line config change plus a loader tweak, if dashed/signal lines matter for the
  DEXPI output.
