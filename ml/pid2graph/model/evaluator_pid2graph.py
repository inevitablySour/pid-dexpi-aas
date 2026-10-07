"""COCO-style node and edge mAP evaluator for PID2Graph.

The upstream scene_graph branch's evaluator_sg.py computes Visual-Genome scene-graph
recall@K (R@20/50/100) and needs config.DATA.LABEL_DATA_DIR - a VG idx_to_predicate json
that doesn't exist for PID2Graph. It also wires MeanBoxAP as its key metric but feeds it
*normalized cxcywh* boxes, while the underlying box_iou_np expects absolute xyxy, and
hardcodes classes=range(150) (VG's class count). So its AP number is not comparable to
anything.

This evaluator reports what the P&ID literature reports, so results can be put next to
the published baselines:

  Stürmer, Graumann & Koch (2025), IEEE DSAA - PID2Graph OPEN100: 52.14 node mAP,
  45.89 edge mAP. (Also the "Reference Modular Digitization" row of SynthPID's Table 3.)
  SynthPID (arXiv 2604.16513) Table 3, same 7-class taxonomy: E1 oracle 67.8 node /
  69.4 edge, E2 synth-only 64.2 / 63.8, E3 mixed 71.3 / 72.6.

Node mAP is COCO mAP@[.5:.95] over the 7 symbol classes. Edge mAP is computed the way
the graph-extraction literature does it: a predicted edge counts as correct only if both
of its endpoint boxes match ground-truth boxes at the IoU threshold AND the ground truth
has an edge between those two matched nodes, scored by the predicted relation
probability. See _edge_average_precision for the exact matching rule, since the papers
describe it only in prose.
"""
from __future__ import annotations

import gc
import os
import time
from typing import Callable, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch
from ignite.engine import Events
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from monai.config import IgniteInfo
from monai.engines import SupervisedEvaluator
from monai.engines.utils import default_prepare_batch
from monai.handlers import CheckpointSaver, StatsHandler, TensorBoardStatsHandler
from monai.inferers import SimpleInferer
from monai.utils import min_version, optional_import
from torch.utils.data import DataLoader

from inference import graph_infer
from metrics.box_ops_np import box_iou_np
from metrics.coco import COCOMetric
from metrics.boxap import BoxAP

Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")

# Must match pid2graph_data/dataset_pid2graph.py's PID2GRAPH_CLASSES, minus background.
PID2GRAPH_CLASS_NAMES = [
    "arrow",
    "general",
    "inlet/outlet",
    "instrumentation",
    "pump",
    "tank",
    "valve",
]


def cxcywh_to_xyxy_abs(boxes: np.ndarray, height: float, width: float) -> np.ndarray:
    """Normalized (cx, cy, w, h) -> absolute (x1, y1, x2, y2).

    Both the ground truth (T.Normalize in transforms.py converts targets to normalized
    cxcywh) and the model's pred_boxes are in normalized cxcywh, while box_iou_np wants
    absolute xyxy. Feeding cxcywh straight into the IoU function - which the upstream
    evaluator does - silently produces meaningless overlaps rather than an error.
    """
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - 0.5 * w) * width
    y1 = (cy - 0.5 * h) * height
    x2 = (cx + 0.5 * w) * width
    y2 = (cy + 0.5 * h) * height
    return np.stack([x1, y1, x2, y2], axis=1)


def _average_precision(scores: np.ndarray, matched: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP, as COCO defines it."""
    if n_gt == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores)
    matched = matched[order].astype(np.float64)
    tp = np.cumsum(matched)
    fp = np.cumsum(1.0 - matched)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # make precision monotonically decreasing
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_thresholds = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, recall_thresholds, side="left")
    interpolated = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    return float(interpolated.mean())


def _edge_average_precision(
    samples: Sequence[dict],
    iou_thresholds: Sequence[float],
) -> Dict[str, float]:
    """Edge mAP: a predicted edge is a true positive iff both endpoints match distinct
    ground-truth nodes at the IoU threshold and a ground-truth edge exists between those
    two nodes.

    Endpoint matching is greedy by descending IoU per image (one prediction per GT node),
    which mirrors how COCO assigns detections and avoids one predicted box claiming
    several GT nodes. Class labels are deliberately NOT required to match for the edge to
    count - the metric measures topology, and node classification is already scored
    separately by node mAP. Averaged over the IoU range, matching COCO's mAP@[.5:.95].
    """
    per_threshold = []
    for iou_threshold in iou_thresholds:
        all_scores = []
        all_matched = []
        total_gt_edges = 0
        for sample in samples:
            gt_boxes = sample["gt_boxes"]
            gt_edges = sample["gt_edges"]
            pred_boxes = sample["pred_boxes"]
            node_pairs = sample["node_pairs"]
            edge_scores = sample["edge_scores"]

            gt_edge_set = set()
            for edge in gt_edges:
                a, b = int(edge[0]), int(edge[1])
                gt_edge_set.add((min(a, b), max(a, b)))
            total_gt_edges += len(gt_edge_set)

            if len(gt_edge_set) == 0 or pred_boxes.shape[0] == 0 or len(node_pairs) == 0:
                continue

            # greedy one-to-one endpoint assignment: pred node index -> gt node index
            iou = box_iou_np(pred_boxes, gt_boxes)
            pred_to_gt = {}
            used_gt = set()
            flat = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
            for p_idx, g_idx in flat:
                if iou[p_idx, g_idx] < iou_threshold:
                    break
                if p_idx in pred_to_gt or g_idx in used_gt:
                    continue
                pred_to_gt[int(p_idx)] = int(g_idx)
                used_gt.add(int(g_idx))

            claimed = set()
            for (p_a, p_b), score in zip(node_pairs, edge_scores):
                p_a, p_b = int(p_a), int(p_b)
                if p_a not in pred_to_gt or p_b not in pred_to_gt:
                    all_scores.append(score)
                    all_matched.append(0)
                    continue
                g_a, g_b = pred_to_gt[p_a], pred_to_gt[p_b]
                key = (min(g_a, g_b), max(g_a, g_b))
                # count a GT edge only once; a second prediction for it is a duplicate
                is_tp = key in gt_edge_set and key not in claimed
                if is_tp:
                    claimed.add(key)
                all_scores.append(score)
                all_matched.append(1 if is_tp else 0)

        per_threshold.append(
            _average_precision(np.asarray(all_scores), np.asarray(all_matched), total_gt_edges)
        )

    valid = [v for v in per_threshold if not np.isnan(v)]
    result = {"edge_mAP": float(np.mean(valid)) if valid else float("nan")}
    for threshold, value in zip(iou_thresholds, per_threshold):
        result[f"edge_AP@{threshold:.2f}"] = value
    return result


class NodeEdgeMeanAP(Metric):
    """Ignite metric computing both node mAP (COCO, per-class) and edge mAP.

    Buffers per-image predictions/targets across the epoch, then reduces in compute().
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        max_detections: int = 100,
        iou_range: Sequence[float] = (0.5, 0.95, 0.05),
        writer=None,
    ):
        self.iou_thresholds = list(np.arange(iou_range[0], iou_range[1] + 1e-9, iou_range[2]))
        self.coco_metric = COCOMetric(
            classes=PID2GRAPH_CLASS_NAMES,
            iou_list=(0.5, 0.75),
            iou_range=iou_range,
            # Same limit as BoxAP below. COCOMetric's own default (50, 100) would silently
            # cap whole-sheet scoring at 100 detections (OPEN100 sheets hold more symbols).
            max_detection=(min(50, max_detections), max_detections),
            per_class=True,
            verbose=False,
        )
        self.node_ap = BoxAP(
            metrics=(self.coco_metric,),
            iou_fn=box_iou_np,
            reduction="none",
            max_detections=max_detections,
        )
        self.writer = writer
        self._edge_samples = []
        super().__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self) -> None:
        self.node_ap.reset()
        self._edge_samples = []

    @reinit__is_reduced
    def update(self, output) -> None:
        pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, edge_samples = output
        self.node_ap(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes)
        self._edge_samples.extend(edge_samples)

    # BoxAP.aggregate() returns the metrics dict directly (its docstring's second tuple
    # element is commented out upstream), keyed as COCOMetric names them.
    NODE_PRIMARY_KEY = "mAP_IoU_0.50_0.95_0.05_MaxDet_100"

    def compute(self):
        node_results = self.node_ap.aggregate()
        if isinstance(node_results, (list, tuple)):
            node_results = node_results[0] if node_results else {}
        if not isinstance(node_results, dict):
            node_results = {}

        edge_results = _edge_average_precision(self._edge_samples, self.iou_thresholds)
        merged = {**{f"node_{k}": v for k, v in node_results.items()}, **edge_results}
        self.last_results = merged

        if self.writer is not None:
            for key, value in merged.items():
                try:
                    if value is not None and not np.isnan(float(value)):
                        self.writer.add_scalar(f"val/{key}", float(value), 0)
                except (TypeError, ValueError):
                    continue

        # Ignite's key_val_metric needs one scalar to rank checkpoints by; node mAP is
        # the primary number. The full dict is printed by the epoch-completed hook.
        primary = merged.get(f"node_{self.NODE_PRIMARY_KEY}")
        if primary is None:
            primary = next(
                (v for k, v in merged.items() if k.startswith("node_mAP") and v is not None),
                float("nan"),
            )
        return float(primary) if primary is not None else float("nan")


class PID2GraphEvaluator(SupervisedEvaluator):
    def __init__(
        self,
        device: torch.device,
        val_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        config=None,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        inferer=None,
        key_val_metric=None,
        additional_metrics=None,
        val_handlers=None,
        amp: bool = False,
        distributed: bool = False,
        writer=None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            inferer=inferer or SimpleInferer(),
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
            amp=amp,
        )
        self.config = config
        self.distributed = distributed
        self.writer = writer
        self.add_emd_rel = config.MODEL.DECODER.ADD_EMB_REL
        self.freq_baseline = kwargs.get("freq_baseline", None)
        self._register_reporting()

    def _iteration(self, engine, batchdata):
        images, gt_datas = list(batchdata[0]), list(batchdata[1])
        images = [image.to(engine.state.device, non_blocking=False) for image in images]

        self.network.eval()
        with torch.no_grad():
            h, out = self.network(images)
            relation_embed = (
                self.network.module.relation_embed if self.distributed else self.network.relation_embed
            )
            inferred = graph_infer(
                h, out, relation_embed, freq=self.freq_baseline, emb=self.add_emd_rel
            )

        pred_boxes_abs, pred_classes, pred_scores = [], [], []
        gt_boxes_abs, gt_classes = [], []
        edge_samples = []

        for i, gt_data in enumerate(gt_datas):
            height, width = [float(v) for v in gt_data["orig_size"].tolist()]

            gt_b = cxcywh_to_xyxy_abs(gt_data["boxes"].cpu().numpy(), height, width)
            # dataset labels are 1-based with 0 = background; COCOMetric indexes classes
            # from 0, matching graph_infer's own pred_boxes_class - 1.0 convention.
            gt_c = gt_data["labels"].cpu().numpy() - 1
            gt_boxes_abs.append(gt_b)
            gt_classes.append(gt_c)

            p_b = cxcywh_to_xyxy_abs(np.asarray(inferred["pred_boxes"][i]), height, width)
            pred_boxes_abs.append(p_b)
            pred_classes.append(np.asarray(inferred["pred_boxes_class"][i]))
            pred_scores.append(np.asarray(inferred["pred_boxes_score"][i]))

            # graph_infer appends a literal None (not an empty array) for any sample with
            # fewer than 2 valid nodes - common early in training, when nearly every
            # query is predicted background - so None has to be handled, not just length.
            node_pairs = inferred["all_node_pairs"][i] if i < len(inferred["all_node_pairs"]) else None
            relation = inferred["all_relation"][i] if i < len(inferred["all_relation"]) else None
            if node_pairs is None or relation is None or len(node_pairs) == 0 or len(relation) == 0:
                node_pairs, edge_scores = [], []
            else:
                relation = np.asarray(relation)
                # column 0 is the "no relation" class; score an edge by the best real
                # predicate probability, matching how graph_infer picks rel ids.
                edge_scores = relation[:, 1:].max(axis=1) if relation.shape[1] > 1 else relation[:, 0]

            edge_samples.append(
                {
                    "gt_boxes": gt_b,
                    "gt_edges": gt_data["edges"].cpu().numpy(),
                    "pred_boxes": p_b,
                    "node_pairs": node_pairs,
                    "edge_scores": edge_scores,
                }
            )

        gc.collect()
        torch.cuda.empty_cache()

        return {
            "images": images,
            "pred_boxes": pred_boxes_abs,
            "pred_boxes_class": pred_classes,
            "pred_boxes_score": pred_scores,
            "boxes": gt_boxes_abs,
            "boxes_class": gt_classes,
            "edge_samples": edge_samples,
        }

    def _register_reporting(self):
        @self.on(Events.EPOCH_COMPLETED)
        def print_full_metrics(engine: Engine) -> None:
            metric = self.key_val_metric_obj if hasattr(self, "key_val_metric_obj") else None
            results = getattr(metric, "last_results", None) if metric is not None else None
            if results is None:
                for value in (engine.state.metrics or {}).values():
                    if isinstance(value, dict):
                        results = value
                        break
            if not results:
                return
            print(f"\n=== PID2Graph validation, epoch {engine.state.epoch} ===")
            for key in sorted(results):
                value = results[key]
                if value is None:
                    continue
                try:
                    print(f"  {key}: {float(value):.4f}")
                except (TypeError, ValueError):
                    continue
            print(
                "  (published baselines on OPEN100 - Stürmer et al. 2025: 52.14 node / "
                "45.89 edge mAP; SynthPID mixed: 71.3 / 72.6)\n"
            )


def build_evaluator(
    val_loader,
    net,
    optimizer,
    scheduler,
    writer,
    config,
    device,
    distributed=False,
    local_rank=0,
    **kwargs,
):
    val_handlers = [
        TensorBoardStatsHandler(
            writer,
            tag_name="val_mAP",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: scheduler.last_epoch,
        ),
    ]
    if local_rank == 0:
        val_handlers.extend(
            [
                StatsHandler(output_transform=lambda x: None),
                CheckpointSaver(
                    save_dir=os.path.join(
                        config.TRAIN.SAVE_PATH,
                        "runs",
                        "%s_%d" % (config.log.exp_name, config.DATA.SEED),
                        "models",
                    ),
                    save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
                    save_key_metric=True,
                    key_metric_n_saved=3,
                    save_interval=1,
                ),
            ]
        )

    metric = NodeEdgeMeanAP(
        output_transform=lambda x: (
            x["pred_boxes"],
            x["pred_boxes_class"],
            x["pred_boxes_score"],
            x["boxes"],
            x["boxes_class"],
            x["edge_samples"],
        ),
        max_detections=100,
        writer=writer,
    )

    evaluator = PID2GraphEvaluator(
        config=config,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        key_val_metric={"val_node_mAP": metric},
        val_handlers=val_handlers,
        amp=False,
        distributed=distributed,
        writer=writer,
        **kwargs,
    )
    evaluator.key_val_metric_obj = metric
    return evaluator
