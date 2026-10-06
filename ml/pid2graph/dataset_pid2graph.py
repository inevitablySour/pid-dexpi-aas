"""PyTorch Dataset for PID2Graph (Zenodo record 14803338), matching the exact
(img, target) contract the scene_graph branch's model/loss/matcher already expect -
see dataset_scene.py's visual_genome_loader.__getitem__ and open_image.py's
OIDataset.__getitem__, both of which return the same target dict shape this module
builds. No Visual-Genome-specific preprocessing (HDF5/proposal files/frequency bias)
is needed since PID2Graph ships plain GraphML + image pairs already.

Real schema confirmed by downloading the dataset and inspecting a genuine sample
(PID2Graph/Complete/PID2Graph OPEN100/3.graphml), not guessed from the paper's prose -
see pid-dexpi-aas/src/pid_dexpi_aas/export_pid2graph.py's module docstring for the
same finding, reused here:
  - node data: d0=label (string), d1-d4=xmin/xmax/ymin/ymax (double) for real symbol
    nodes; connector/crossing/background nodes use d5-d8 (long) instead.
  - edge data: d9=edge_label (string), "solid" or "non-solid".
  - real symbol classes observed: arrow, general, inlet/outlet, instrumentation,
    pump, tank, valve (the paper's Table 1 taxonomy).

Topology collapse: real edges never run directly symbol-to-symbol in the raw file -
they route through connector/crossing waypoint nodes (e.g. valve8 -> connector157 ->
crossing158 -> connector203 -> tank12). Relationformer's target format wants direct
object-to-object relationship triples, not a line-routing graph, so this loader walks
each connected chain of connector/crossing/background nodes and collapses it down to
a single edge between the two real symbol endpoints it connects - the same operation
drawing2's own roadmap.md called "connector-collapse GraphML files (remove routing
waypoints, keep physical components)" as its own required Phase 2 preprocessing step.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Real label vocabulary confirmed against the downloaded dataset (see module docstring).
# Index 0 is reserved for "__background__" per this repo's own convention (see
# visual_genome_loader.__init__: self.num_classes = config.MODEL.NUM_OBJ_CLS + 1) -
# so real classes start at 1, matching what a NUM_OBJ_CLS=7 config expects.
PID2GRAPH_CLASSES = [
    "__background__",
    "arrow",
    "general",
    "inlet/outlet",
    "instrumentation",
    "pump",
    "tank",
    "valve",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(PID2GRAPH_CLASSES)}

# Structural node types that exist only to route a line through waypoints - never a
# real detectable symbol, collapsed away rather than treated as a "general" object.
# "border_node" confirmed present in Patched subset samples (e.g. Patched/PID2Graph
# OPEN100/0/1.graphml) - matches the Zenodo record's own description ("border-nodes
# for the patched data, that marks the location of a line 'leaving' a patch"), i.e. a
# patch-boundary artifact of the tiling process, never a real P&ID symbol. Treating it
# as a real node (the initial version of this loader did, before this was caught by
# direct inspection of a Patched sample) would train the model to detect a fake
# "general" object at every single patch edge.
_STRUCTURAL_LABELS = {"connector", "crossing", "background", "border_node"}

# Edge predicate classes: index 0 reserved the same way as node classes (see above).
# "connected" is the only real predicate PID2Graph's edge_label maps onto once
# solid/non-solid is captured as an edge attribute rather than a class distinction -
# kept as metadata on the target dict (see below) rather than folded into NUM_REL_CLS,
# since the paper's own Table 1 treats solid/non-solid as the full edge taxonomy and
# this loader preserves that instead of guessing at a finer one.
PID2GRAPH_PREDICATES = ["__background__", "connected"]
PREDICATE_TO_IDX = {name: idx for idx, name in enumerate(PID2GRAPH_PREDICATES)}


def _node_bbox(data: dict) -> tuple[float, float, float, float] | None:
    """Real symbol nodes carry d1-d4 (xmin/xmax/ymin/ymax as GraphML attr names, but
    networkx exposes them by their attr.name, not key id) as floats; structural nodes
    use a separate d5-d8 long-typed key set with the SAME attr names, so this is not
    actually ambiguous - whichever of xmin/xmax/ymin/ymax is present wins."""
    try:
        return (float(data["xmin"]), float(data["ymin"]), float(data["xmax"]), float(data["ymax"]))
    except KeyError:
        return None


def _collapse_graph(g: nx.Graph) -> tuple[list[str], list[tuple[float, float, float, float]], list[tuple[int, int, str]]]:
    """Reduce PID2Graph's raw (symbol + connector/crossing waypoint) graph down to
    direct symbol-to-symbol edges. Returns (labels, boxes, edges) where edges are
    (real_node_index_i, real_node_index_j, edge_label) - edge_label is the "solid"/
    "non-solid" value carried by whichever raw edge is picked to represent the
    collapsed connection (arbitrary among the chain's edges when they disagree, which
    real data essentially never does since a single drawn line is one edge_label end
    to end)."""
    if g.is_multigraph():
        # Two drawn lines between the same pair (e.g. the team's draw.io files) make
        # read_graphml return a MultiGraph; for connectivity one edge per pair is enough.
        g = nx.Graph(g)
    real_nodes = [n for n, d in g.nodes(data=True) if d.get("label") not in _STRUCTURAL_LABELS]
    real_index = {n: i for i, n in enumerate(real_nodes)}
    labels = [g.nodes[n]["label"] for n in real_nodes]
    boxes = []
    for n in real_nodes:
        bbox = _node_bbox(g.nodes[n])
        boxes.append(bbox if bbox is not None else (0.0, 0.0, 0.0, 0.0))

    collapsed_edges: set[tuple[int, int, str]] = set()
    partners: dict = {}

    for real_n in real_nodes:
        for neighbor in g.neighbors(real_n):
            if g.nodes[neighbor].get("label") not in _STRUCTURAL_LABELS:
                # Direct real-to-real edge (rare but valid - e.g. very short lines).
                a, b = real_index[real_n], real_index[neighbor]
                if a != b:
                    edge_label = g.edges[real_n, neighbor].get("edge_label", "solid")
                    collapsed_edges.add((min(a, b), max(a, b), edge_label))
                continue
            # Walk the chain of structural nodes until hitting real node(s). The walk
            # remembers which way it came in: at a pass-through crossing (two pipes
            # drawn across each other, see _pass_through_partner) it may only continue
            # straight on along the same pipe, never turn onto the other one.
            chain_edge_label = g.edges[real_n, neighbor].get("edge_label", "solid")
            stack = [(neighbor, real_n)]
            seen = {(neighbor, real_n)}
            while stack:
                current, came_from = stack.pop()
                if current not in partners:
                    partners[current] = _pass_through_partner(g, current)
                partner = partners[current]
                if partner is not None:
                    onward = [partner[came_from]] if came_from in partner else []
                else:
                    onward = [n for n in g.neighbors(current) if n != came_from]
                for nxt in onward:
                    if g.nodes[nxt].get("label") not in _STRUCTURAL_LABELS:
                        a, b = real_index[real_n], real_index[nxt]
                        if a != b:
                            collapsed_edges.add((min(a, b), max(a, b), chain_edge_label))
                    elif (nxt, current) not in seen:
                        seen.add((nxt, current))
                        stack.append((nxt, current))

    return labels, boxes, sorted(collapsed_edges)


def _center(data: dict) -> tuple[float, float] | None:
    bbox = _node_bbox(data)
    return None if bbox is None else ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _pass_through_partner(g: nx.Graph, node, tolerance_deg: float = 20.0) -> dict | None:
    """For a 4-arm crossing whose arms form two straight lines through it, map each
    neighbour to the neighbour straight across - the same pipe continuing. None for
    anything else (bends, T-junctions, 5+ arms, missing geometry), which are walked as
    real junctions.

    Why: in PID2Graph a "crossing" node marks every place lines meet, including two
    pipes merely drawn across each other. Walking through such a node in every
    direction linked every symbol on one pipe to every symbol on the other. Measured on
    a 400-patch training sample: 98% of 4-arm crossings are two straight lines, and
    74% of collapsed training edges existed only because of them. Real OPEN100 sheets
    almost never contain them (7 of 1,926 crossings), so the old labels taught the
    model connections real drawings don't have. In P&ID convention crossing lines
    without a junction dot are not connected.
    """
    if g.nodes[node].get("label") != "crossing" or g.degree(node) != 4:
        return None
    c = _center(g.nodes[node])
    arms = []
    for n in g.neighbors(node):
        p = _center(g.nodes[n])
        if c is None or p is None or p == c:
            return None
        arms.append((math.degrees(math.atan2(p[1] - c[1], p[0] - c[0])) % 360, n))
    arms.sort()
    opposite = lambda i, j: abs(((arms[j][0] - arms[i][0]) % 360) - 180) <= tolerance_deg
    if not (opposite(0, 2) and opposite(1, 3)):
        return None
    (_, a), (_, b), (_, c2), (_, d) = arms
    return {a: c2, c2: a, b: d, d: b}


class PID2GraphDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path]], transforms=None, n_queries: int = 201, n_rel_classes: int = 1):
        """samples: list of (image_path, graphml_path) pairs. n_rel_classes = MODEL.NUM_REL_CLS:
        1 -> every connection is "connected"; 2 -> "solid" (process pipe) vs "non-solid"
        (signal/instrument line) are learned as separate relation classes."""
        self.samples = samples
        self.transforms = transforms
        self.n_queries = n_queries
        self.predicate_to_idx = (
            {"solid": 1, "non-solid": 2} if n_rel_classes >= 2 else {"solid": 1, "non-solid": 1}
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, graphml_path = self.samples[index]
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        g = nx.read_graphml(graphml_path)
        labels, boxes, edges = _collapse_graph(g)

        if len(labels) == 0:
            # A symbol-free patch: ~6.75% of patches, measured. These are NOT blank -
            # they're tiles covering a pipe run between equipment, so the graph holds
            # only border_node/crossing waypoints (line enters the tile, maybe crosses
            # another line, leaves the tile) with no valve/pump/tank in frame. The image
            # has real content, so they're kept as useful negatives (pipe present, no
            # symbol to detect) rather than filtered out. Note this needs the empty-set
            # guard in losses.py loss_edges, or the edge loss is nan for these.
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_classes = np.zeros((0,), dtype=np.int64)
            gt_rels = np.zeros((0, 3), dtype=np.int64)
        else:
            gt_boxes = np.asarray(boxes, dtype=np.float32)
            gt_classes = np.asarray(
                [CLASS_TO_IDX.get(label, CLASS_TO_IDX["general"]) for label in labels], dtype=np.int64
            )
            if edges:
                gt_rels = np.asarray(
                    [(i, j, self.predicate_to_idx.get(label, 1)) for i, j, label in edges],
                    dtype=np.int64,
                )
            else:
                gt_rels = np.zeros((0, 3), dtype=np.int64)

        if len(gt_boxes) > self.n_queries - 2:
            # Same object-count cap dataset_scene.py's visual_genome_loader and
            # open_image.py's OIDataset both apply - the model has a fixed query budget
            # per config.MODEL.DECODER.NUM_QUERIES, so an oversized sample must be
            # subsampled rather than silently overflowing the decoder.
            keep_idx = np.random.choice(len(gt_boxes), self.n_queries - 2, replace=False)
            keep_set = set(keep_idx.tolist())
            remap = {old: new for new, old in enumerate(sorted(keep_set))}
            gt_boxes = gt_boxes[sorted(keep_set)]
            gt_classes = gt_classes[sorted(keep_set)]
            gt_rels = np.asarray(
                [[remap[i], remap[j], p] for i, j, p in gt_rels if i in keep_set and j in keep_set],
                dtype=np.int64,
            ) if len(gt_rels) else np.zeros((0, 3), dtype=np.int64)

        target = {
            "iscrowd": torch.zeros(gt_boxes.shape[0]),
            "boxes": torch.as_tensor(gt_boxes, dtype=torch.float),
            "labels": torch.as_tensor(gt_classes, dtype=torch.int64),
            "edges": torch.as_tensor(gt_rels, dtype=torch.int64),
            "image_id": str(image_path),
            "orig_size": torch.as_tensor([int(h), int(w)]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def _discover_pairs(root: Path, subset_dirs: list[str]) -> list[tuple[Path, Path]]:
    """Walk one or more PID2Graph subset directories (e.g. "Patched/PID2Graph OPEN100")
    and pair every .graphml with its matching image (.png or .jpg - PID2Graph Synthetic
    Complete ships .jpg, every other subset ships .png, confirmed by direct inspection
    of the downloaded dataset)."""
    pairs: list[tuple[Path, Path]] = []
    for subset in subset_dirs:
        subset_path = root / subset
        if not subset_path.exists():
            continue
        for graphml_path in sorted(subset_path.rglob("*.graphml")):
            png = graphml_path.with_suffix(".png")
            jpg = graphml_path.with_suffix(".jpg")
            if png.exists():
                pairs.append((png, graphml_path))
            elif jpg.exists():
                pairs.append((jpg, graphml_path))
    return pairs


class StyleAugment:
    """Make synthetic line drawings look more like real/team P&IDs (train only).

    Measured gaps this targets: the model misses most grey-shaded equipment on the team's
    draw.io drawings (LightUp: V203, X102, V201/V204 undetected) and real sheets differ in
    line weight and scan quality. Applied to the PIL image before resizing, with boxes
    still in absolute pixels: shade the inside of a few equipment boxes grey (lines stay
    dark), thicken lines, blur, shift contrast.
    """

    SHADE_CLASSES = {CLASS_TO_IDX["tank"], CLASS_TO_IDX["general"], CLASS_TO_IDX["pump"]}

    def __init__(self, p_shade=0.4, p_thick=0.3, p_blur=0.2, p_contrast=0.3):
        self.p_shade, self.p_thick, self.p_blur, self.p_contrast = p_shade, p_thick, p_blur, p_contrast

    def __call__(self, img, target):
        import random

        from PIL import ImageEnhance, ImageFilter

        if random.random() < self.p_shade and len(target["boxes"]):
            arr = np.asarray(img).copy()
            cands = [i for i, c in enumerate(target["labels"].tolist()) if c in self.SHADE_CLASSES]
            for i in random.sample(cands, min(len(cands), random.randint(1, 3))):
                x1, y1, x2, y2 = [int(v) for v in target["boxes"][i].tolist()]
                region = arr[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                light = region.mean(axis=2) > 200  # paper, not linework
                top, bottom = random.randint(150, 215), random.randint(150, 215)
                shade = np.linspace(top, bottom, region.shape[0])[:, None].repeat(region.shape[1], 1)
                region[light] = np.stack([shade[light]] * 3, axis=-1).astype(np.uint8)
            img = Image.fromarray(arr)
        if random.random() < self.p_thick:
            # Thicken only: thinning (MaxFilter) erased 1-px symbols entirely in testing,
            # which would train the model to find symbols that are not visible.
            img = img.filter(ImageFilter.MinFilter(3))
        if random.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.2)))
        if random.random() < self.p_contrast:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        return img, target


def _cfg(config, section: str, key: str, default):
    return getattr(getattr(config, section), key, default)


def _sheets(pairs: list[tuple[Path, Path]], subset_paths: set[Path]) -> dict[str, list]:
    """Group pairs by source sheet: Patched/<subset>/<sheet>/<n> -> the sheet dir."""
    groups: dict[str, list] = {}
    for p in pairs:
        g = p[1]
        key = str(g.parent) if g.parent not in subset_paths else str(g)
        groups.setdefault(key, []).append(p)
    return groups


def real_fold_sheets(config, fold: int, seed: int = 10) -> dict[str, list[str]]:
    """The real (OPEN100) sheets of a cross-validation fold: {"test", "val", "train"} ->
    sheet directory names. 12 sheets, 4 folds: 3 test sheets per fold, the next sheet
    (cyclically) as a real validation sheet for checkpoint selection, 8 for training.
    Every sheet is tested by exactly one fold, always by a model that never saw it."""
    import random

    root = Path(config.DATA.DATA_PATH)
    real_dir = root / _cfg(config, "DATA", "REAL_SUBSET", "Patched/PID2Graph OPEN100")
    names = sorted((d.name for d in real_dir.iterdir() if d.is_dir()), key=lambda n: int(n) if n.isdigit() else n)
    random.Random(seed).shuffle(names)
    n_folds = int(_cfg(config, "DATA", "NUM_FOLDS", 4))
    per = len(names) // n_folds
    test = names[fold * per:(fold + 1) * per]
    val = [names[((fold + 1) * per) % len(names)]]
    train = [n for n in names if n not in test and n not in val]
    return {"test": test, "val": val, "train": train}


def build_pid2graph_data(config, mode: str = "split", split: float = 0.9, seed: int = 10, fold: int | None = None):
    """Mirrors dataset_scene.py's build_scene_graph_data interface.

    mode "split" -> (train_ds, val_ds); "test" -> the held-out real test set (with `fold`:
    that fold's 3 OPEN100 sheets; without: DATA.PID2GRAPH_TEST_SUBSETS); "team_test" ->
    the held-out team drawing(s) (DATA.TEAM_TEST_SUBSETS).

    With `fold`, real OPEN100 sheets join training (8 sheets) and validation (1 sheet),
    and DATA.EXTRA_TRAIN_SUBSETS (the team's own drawings) join training. Synthetic data
    is always split 90/10 by sheet. Every split is by whole sheet: patches overlap heavily,
    so a per-patch split leaks near-duplicates (measured: 100% leak, val mAP 0.91).
    """
    import random

    import transforms as T

    root = Path(config.DATA.DATA_PATH)
    n_queries = config.MODEL.DECODER.NUM_QUERIES
    n_rel = int(config.MODEL.NUM_REL_CLS)
    real_subset = _cfg(config, "DATA", "REAL_SUBSET", "Patched/PID2Graph OPEN100")

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    style = [StyleAugment()] if _cfg(config, "DATA", "STYLE_AUGMENT", False) else []
    train_transform = T.Compose(
        style + [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=800),
                T.Compose([T.RandomResize(scales), T.RandomSizeCrop(384, 600), T.RandomResize(scales, max_size=800)]),
            ),
            normalize,
        ]
    )
    val_transform = T.Compose([T.RandomResize([800], max_size=800), normalize])
    ds = lambda pairs, tf: PID2GraphDataset(pairs, tf, n_queries, n_rel)

    if mode == "team_test":
        return ds(_discover_pairs(root, list(_cfg(config, "DATA", "TEAM_TEST_SUBSETS", ["Patched/Team test"]))),
                  val_transform)
    if mode == "test":
        if fold is None:
            return ds(_discover_pairs(root, list(config.DATA.PID2GRAPH_TEST_SUBSETS)), val_transform)
        names = set(real_fold_sheets(config, fold, seed)["test"])
        return ds([p for p in _discover_pairs(root, [real_subset]) if p[1].parent.name in names], val_transform)

    subset_dirs = list(config.DATA.PID2GRAPH_SUBSETS)
    pairs = _discover_pairs(root, subset_dirs)
    if not pairs:
        raise FileNotFoundError(
            f"No .graphml/image pairs found under {root} for subsets {subset_dirs} - "
            "check config.DATA.DATA_PATH points at the extracted PID2Graph/ directory."
        )
    if mode == "train":
        return ds(pairs, train_transform)
    if mode == "val":
        return ds(pairs, val_transform)

    groups = _sheets(pairs, {root / s for s in subset_dirs})
    sheets = sorted(groups)
    random.seed(seed)
    random.shuffle(sheets)
    val_sheets = set(sheets[int(split * len(sheets)):])
    train_pairs = [p for s in sheets if s not in val_sheets for p in groups[s]]
    val_pairs = [p for s in sheets if s in val_sheets for p in groups[s]]

    if fold is not None:
        real = real_fold_sheets(config, fold, seed)
        real_pairs = _discover_pairs(root, [real_subset])
        train_pairs += [p for p in real_pairs if p[1].parent.name in real["train"]]
        val_pairs += [p for p in real_pairs if p[1].parent.name in real["val"]]
        train_pairs += _discover_pairs(root, list(_cfg(config, "DATA", "EXTRA_TRAIN_SUBSETS", [])))

    random.shuffle(train_pairs)
    return ds(train_pairs, train_transform), ds(val_pairs, val_transform)


def sample_weights(samples: list[tuple[Path, Path]], config, cache_path: Path) -> list[float]:
    """Per-patch sampling weight (fix #7 + oversampling of scarce real/team data).

    weight = source weight (DATA.REAL_WEIGHT for OPEN100 patches, DATA.TEAM_WEIGHT for the
    team's drawings, 1 for synthetic) x rarity (the rarest class in the patch, as
    sqrt(median class count / that class's count), capped at DATA.RARITY_CAP). Class
    counts per patch are cached in cache_path - parsing 35k GraphML files takes ~30 s.
    """
    import json
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor

    # keyed by path AND modification time: labels regenerated in place (Team v3) must not
    # be weighted with the class counts of the old file
    key = lambda g: f"{g}|{Path(g).stat().st_mtime_ns}"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [str(g) for _, g in samples if key(g) not in cache]
    if missing:
        with ProcessPoolExecutor() as pool:
            for path, labels in zip(missing, pool.map(_labels_of, missing, chunksize=64)):
                cache[key(path)] = labels
        cache_path.write_text(json.dumps(cache))

    freq = Counter(l for _, g in samples for l in cache[key(g)])
    median = float(np.median(list(freq.values()))) if freq else 1.0
    cap = float(_cfg(config, "DATA", "RARITY_CAP", 4.0))
    real_w = float(_cfg(config, "DATA", "REAL_WEIGHT", 1.0))
    team_w = float(_cfg(config, "DATA", "TEAM_WEIGHT", 1.0))
    real_key = _cfg(config, "DATA", "REAL_SUBSET", "Patched/PID2Graph OPEN100")
    team_keys = list(_cfg(config, "DATA", "EXTRA_TRAIN_SUBSETS", []))
    weights = []
    for _, g in samples:
        labels = cache[key(g)]
        rarity = max((min(cap, (median / freq[l]) ** 0.5) for l in labels), default=1.0)
        source = real_w if real_key in str(g) else team_w if any(k in str(g) for k in team_keys) else 1.0
        weights.append(max(rarity, 1.0) * source)
    return weights


def _labels_of(graphml_path: str) -> list[str]:
    return _collapse_graph(nx.read_graphml(graphml_path))[0]
