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
    real_nodes = [n for n, d in g.nodes(data=True) if d.get("label") not in _STRUCTURAL_LABELS]
    real_index = {n: i for i, n in enumerate(real_nodes)}
    labels = [g.nodes[n]["label"] for n in real_nodes]
    boxes = []
    for n in real_nodes:
        bbox = _node_bbox(g.nodes[n])
        boxes.append(bbox if bbox is not None else (0.0, 0.0, 0.0, 0.0))

    collapsed_edges: set[tuple[int, int, str]] = set()
    visited_structural: set = set()

    for real_n in real_nodes:
        for neighbor in g.neighbors(real_n):
            if g.nodes[neighbor].get("label") not in _STRUCTURAL_LABELS:
                # Direct real-to-real edge (rare but valid - e.g. very short lines).
                a, b = real_index[real_n], real_index[neighbor]
                if a != b:
                    edge_label = g.edges[real_n, neighbor].get("edge_label", "solid")
                    collapsed_edges.add((min(a, b), max(a, b), edge_label))
                continue
            # Walk the chain of structural nodes via BFS until hitting real node(s).
            frontier = [neighbor]
            seen = {neighbor}
            chain_edge_label = g.edges[real_n, neighbor].get("edge_label", "solid")
            while frontier:
                current = frontier.pop()
                visited_structural.add(current)
                for nxt in g.neighbors(current):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    if g.nodes[nxt].get("label") not in _STRUCTURAL_LABELS:
                        a, b = real_index[real_n], real_index[nxt]
                        if a != b:
                            collapsed_edges.add((min(a, b), max(a, b), chain_edge_label))
                    else:
                        frontier.append(nxt)

    return labels, boxes, sorted(collapsed_edges)


class PID2GraphDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path]], transforms=None, n_queries: int = 201):
        """samples: list of (image_path, graphml_path) pairs."""
        self.samples = samples
        self.transforms = transforms
        self.n_queries = n_queries

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
                    [(i, j, PREDICATE_TO_IDX.get(label, PREDICATE_TO_IDX["connected"])) for i, j, label in edges],
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


def build_pid2graph_data(config, mode: str = "split", split: float = 0.9, seed: int = 10):
    """Mirrors dataset_scene.py's build_scene_graph_data / dataset_road_network.py's
    build_road_network_data interface (config in, (train_ds, val_ds) or single ds out)
    so train.py needs only a branch selecting this function, not a rewritten call site.
    """
    import random

    import transforms as T

    root = Path(config.DATA.DATA_PATH)
    subset_dirs = list(config.DATA.PID2GRAPH_SUBSETS)
    n_queries = config.MODEL.DECODER.NUM_QUERIES

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=800),
                T.Compose(
                    [
                        T.RandomResize(scales),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=800),
                    ]
                ),
            ),
            normalize,
        ]
    )
    val_transform = T.Compose([T.RandomResize([800], max_size=800), normalize])

    pairs = _discover_pairs(root, subset_dirs)
    if not pairs:
        raise FileNotFoundError(
            f"No .graphml/image pairs found under {root} for subsets {subset_dirs} - "
            "check config.DATA.DATA_PATH points at the extracted PID2Graph/ directory."
        )

    random.seed(seed)
    random.shuffle(pairs)

    if mode == "train":
        return PID2GraphDataset(pairs, train_transform, n_queries)
    if mode == "val":
        return PID2GraphDataset(pairs, val_transform, n_queries)

    split_idx = int(split * len(pairs))
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]
    train_ds = PID2GraphDataset(train_pairs, train_transform, n_queries)
    val_ds = PID2GraphDataset(val_pairs, val_transform, n_queries)
    return train_ds, val_ds
