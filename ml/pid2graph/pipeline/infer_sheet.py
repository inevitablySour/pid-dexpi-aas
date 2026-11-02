"""Run the trained Relationformer + EasyOCR on one whole P&ID image and write JSON.

Runs inside the ML venv (ml/relationformer/.venv) because it needs the compiled
deformable-attention op; the Streamlit app (main venv) calls it as a subprocess and
reads the JSON. Usage:

    ml/relationformer/.venv/bin/python ml/pid2graph/pipeline/infer_sheet.py \
        --image drawing.png --checkpoint <ckpt.pt> --out result.json

The model was trained on 2000px PID2Graph patches, so the sheet is tiled the same way
(2000px windows, 500px stride), each tile resized to 800px like validation. Detections
from overlapping tiles are merged (greedy, class-agnostic, IoU >= 0.5, majority class
vote), and per-tile connection probabilities are mapped onto the merged symbols.

OCR output is raw tokens only - assembling them into tags and attaching tags to symbols
happens in pid_dexpi_aas.image.ocr_tags (main venv), where it is unit-testable.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

RF_DIR = Path(__file__).resolve().parents[2] / "relationformer"
sys.path.insert(0, str(RF_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # pid_dexpi_aas.image.sheet_image
sys.path.insert(0, str(Path(__file__).resolve().parent))  # trace_pipes

from inference import graph_infer  # noqa: E402
from point_links import centre_dist, contract  # noqa: E402
from stitch import merge_detections  # noqa: E402
from models import build_model  # noqa: E402
from train_pid2graph import dict2obj  # noqa: E402
import transforms as T  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
# Must match pid2graph_data/dataset_pid2graph.py's PID2GRAPH_CLASSES minus background.
CLASS_NAMES = ["arrow", "general", "inlet/outlet", "instrumentation", "pump", "tank", "valve"]
# Line points, read by models trained with DATA.LINE_POINTS (classes 8 and 9): a line's
# corners and the joints of its tees. Folded back into symbol-to-symbol links (contract).
POINT_NAMES = ["bend", "junction"]
SURE = 0.5  # several zooms (detect): an item found at fewer than half of them is only half-seen
WEAK = 0.2  # a token whose best symbol class gets this much (but loses to "background"): half-seen


def load_sheet(path: str) -> Image.Image:
    # Dark ink on white, whatever came in: transparent exports, and draw.io dark-theme
    # exports / dark screenshots (light lines - laid on white they vanished).
    from pid_dexpi_aas.image.sheet_image import to_dark_on_white
    return to_dark_on_white(Image.open(path))


def tile_starts(total: int, tile: int, stride: int) -> list[int]:
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    return starts if starts[-1] == total - tile else starts + [total - tile]


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


_MODELS: dict = {}


def _load_model(checkpoint: str, device: str):
    """(model, config, class names) - kept loaded: a sheet, an ensemble and a scoring run
    use the same models many times."""
    key = (str(checkpoint), str(device))
    if key not in _MODELS:
        config = dict2obj(yaml.safe_load(open(RF_DIR / "configs" / "pid2graph_2d.yaml")))
        state = torch.load(checkpoint, map_location="cpu")["net"]
        # The relation head's output size says how many line types the checkpoint learned
        # (1 = "connected"; 2 = process vs signal, fine-tuned models) - build to match.
        config.MODEL.NUM_REL_CLS = int(state["relation_embed.layers.3.weight"].shape[0]) - 1
        # ...and the class head how many classes: 7 symbols, + bend and junction (line points)
        config.MODEL.NUM_OBJ_CLS = int(state["class_embed.0.weight"].shape[0]) - 1
        names = (CLASS_NAMES + POINT_NAMES)[:config.MODEL.NUM_OBJ_CLS]
        model = build_model(config).to(torch.device(device))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint does not match model: missing={missing} unexpected={unexpected}")
        model.eval()
        _MODELS[key] = (model, config, names)
    return _MODELS[key]


def detect(sheet: Image.Image, checkpoint, tile, stride, device: str,
           min_prob: float = 0.5, with_points: bool = False, merge: str = "wbf", ensemble: str = "mean",
           on_tile=None) -> tuple:
    """Symbols + connections for a whole sheet. Connections below min_prob are dropped
    (0.5 = the model's own decision; scoring passes a low value to rank all pairs).
    A line-point model's bends and junctions are folded into the connections (contract);
    with_points also returns them: (symbols, connections, {"points", "links"}).

    checkpoint: one model, or a list (an ensemble); tile / stride: one tile size, or a list
    (the sheet read at several zooms - smaller tiles show small symbols larger). Every
    model reads the sheet at every tile size; all their detections are merged as one
    (stitch.py), then ensemble "mean": a symbol's score is scaled by the share of those
    runs that found it, a connection's probability is its mean over the runs (0 where a
    run did not see it); "max": the best run's."""
    checkpoints = list(checkpoint) if isinstance(checkpoint, (list, tuple)) else [checkpoint]
    tiles = list(tile) if isinstance(tile, (list, tuple)) else [tile]
    strides = list(stride) if isinstance(stride, (list, tuple)) else [stride] * len(tiles)
    runs = [(c, t, st) for c in checkpoints for t, st in zip(tiles, strides)]
    loaded = [_load_model(c, device) for c in checkpoints]
    names = max((m[2] for m in loaded), key=len)
    line_type = {1: "process", 2: "signal"} if min(m[1].MODEL.NUM_REL_CLS for m in loaded) >= 2 else {1: "process"}
    dev = torch.device(device)
    tf = T.Compose([T.RandomResize([800], max_size=800),
                    T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])])

    W, H = sheet.size
    dets, weak, tile_edges, tid = [], [], [], 0
    tile_run = {}
    n_tiles = sum(len(tile_starts(H, t, st)) * len(tile_starts(W, t, st)) for _, t, st in runs)
    for r, (ckpt, tile, stride) in enumerate(runs):
        model = _load_model(ckpt, device)[0]
        for y0 in tile_starts(H, tile, stride):
            for x0 in tile_starts(W, tile, stride):
                crop = sheet.crop((x0, y0, min(x0 + tile, W), min(y0 + tile, H)))
                cw, ch = crop.size
                dummy = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long),
                         "edges": torch.zeros((0, 3), dtype=torch.long), "iscrowd": torch.zeros(0),
                         "orig_size": torch.tensor([ch, cw]), "image_id": "sheet"}
                x, _ = tf(crop, dummy)
                with torch.no_grad():
                    h, out = model([x.to(dev)])
                    inf = graph_infer(h, out, model.relation_embed)
                tile_run[tid] = r
                for k, (cx, cy, bw, bh) in enumerate(inf["pred_boxes"][0]):
                    dets.append({"box": [float(x0 + (cx - bw / 2) * cw), float(y0 + (cy - bh / 2) * ch),
                                         float(x0 + (cx + bw / 2) * cw), float(y0 + (cy + bh / 2) * ch)],
                                 "cls": int(inf["pred_boxes_class"][0][k]),
                                 "score": float(inf["pred_boxes_score"][0][k]), "tile": tid, "idx": k, "run": r,
                                 "rect": (x0, y0, x0 + cw, y0 + ch)})
                # weak: tokens the model half-saw - "background" won, but a symbol class got >= WEAK
                prob = out["pred_logits"][0].softmax(-1).detach().cpu().numpy()
                for k in np.nonzero((prob.argmax(-1) == 0) & (prob[:, 1:].max(-1) >= WEAK))[0]:
                    cls = int(prob[k, 1:].argmax())
                    if cls >= len(names) or names[cls] in POINT_NAMES:
                        continue
                    cx, cy, bw, bh = out["pred_boxes"][0, k, :4].detach().cpu().tolist()
                    weak.append({"box": [float(x0 + (cx - bw / 2) * cw), float(y0 + (cy - bh / 2) * ch),
                                         float(x0 + (cx + bw / 2) * cw), float(y0 + (cy + bh / 2) * ch)],
                                 "cls": cls, "score": float(prob[k, 1 + cls]), "tile": tid, "idx": k, "run": r,
                                 "rect": (x0, y0, x0 + cw, y0 + ch)})
                pairs, rel = inf["all_node_pairs"][0], inf["all_relation"][0]
                if pairs is not None:
                    for (i, j), p in zip(pairs, rel):
                        if i < j:
                            tile_edges.append((tid, int(i), int(j), float(np.max(p[1:])), int(np.argmax(p[1:])) + 1))
                tid += 1
                if on_tile:
                    on_tile(tid, n_tiles)

    point_cls = {k for k, n in enumerate(names) if n in POINT_NAMES}
    # tiles -> one set of symbols (stitch.py: "wbf" border-aware fusion, or the first "greedy")
    merged, cluster_of = merge_detections(dets, point_cls, (W, H), merge)
    nodes = []
    for n, m in enumerate(merged):
        counts = np.bincount(m["votes"], weights=m["weights"] if merge == "wbf" else None, minlength=len(names))
        share = len(m["runs"]) / len(runs) if ensemble == "mean" else 1.0  # found by this share of the runs
        nodes.append({"id": f"m{n}", "box": m["box"], "class": names[int(counts.argmax())],
                      "score": m["score"] * share, "class_votes": {names[c]: round(float(v), 2) for c, v in enumerate(counts) if v},
                      "n_tiles": len(m["votes"])})

    # half-seen symbols, merged the same way, away from the symbols found (a second, weaker
    # look at that symbol) - possible missed items, kept if a traced pipe runs through them
    weak_merged, _ = merge_detections(weak, set(), (W, H), merge)
    weak_nodes = [{"box": m["box"], "class": names[int(np.bincount(m["votes"]).argmax())], "score": m["score"]}
                  for m in weak_merged
                  if not any(box_iou(m["box"], n["box"]) > 0.1 or n["box"][0] <= (m["box"][0] + m["box"][2]) / 2 <= n["box"][2]
                             and n["box"][1] <= (m["box"][1] + m["box"][3]) / 2 <= n["box"][3] for n in nodes)]

    per_run: dict[tuple[int, int, int], tuple[float, int]] = {}   # (run, a, b) -> best (prob, line type)
    for t, i, j, p, rel_cls in tile_edges:
        a, b = cluster_of.get((t, i)), cluster_of.get((t, j))
        if a is None or b is None or a == b:
            continue
        key = (tile_run[t], min(a, b), max(a, b))
        if p > per_run.get(key, (0.0, 1))[0]:
            per_run[key] = (p, rel_cls)
    probs: dict[tuple[int, int], tuple[float, int]] = {}
    for (r, a, b), (p, rel_cls) in per_run.items():
        total, best = probs.get((a, b), (0.0, (0.0, 1)))
        probs[(a, b)] = (total + p if ensemble == "mean" else max(total, p), max(best, (p, rel_cls)))
    probs = {k: ((total / len(runs) if ensemble == "mean" else total), best[1]) for k, (total, best) in probs.items()}
    if not point_cls:
        connections = [{"from": f"m{a}", "to": f"m{b}", "prob": p, "line_type": line_type.get(c, "process")}
                       for (a, b), (p, c) in sorted(probs.items()) if p >= min_prob]
        return (nodes, connections, {"points": [], "links": [], "weak": weak_nodes}) if with_points else (nodes, connections)
    is_point = [m["point"] and nodes[k]["class"] in POINT_NAMES for k, m in enumerate(merged)]
    symbols, connections = contract(nodes, is_point, probs, min_prob)
    for c in connections:
        c["line_type"] = line_type.get(c["line_type"], "process")
    if not with_points:
        return symbols, connections
    ids = {k: f"m{k}" for k in range(len(nodes))}
    lines = {"points": [n for k, n in enumerate(nodes) if is_point[k]],
             "links": [{"from": ids[a], "to": ids[b], "prob": p, "line_type": line_type.get(c, "process")}
                       for (a, b), (p, c) in sorted(probs.items()) if p >= 0.5 and (is_point[a] or is_point[b])],
             "weak": weak_nodes}
    return symbols, connections, lines


# A token "looks right" if it is a plant tag (GV204, PSV-101, TI 201/2), bare ISA letters
# (PI, TIC), a number (102), or a value with a unit (15 bar). Anything else is re-read.
_PLAUSIBLE = re.compile(
    r"^([A-Z]{1,4}[- ]?\d{1,4}[A-Z]?(/\d+)?|[A-Z]{1,4}|\d{1,4}(/\d+)?|\d+(\.\d+)? ?[a-zA-Z/]+.*|\[.*\]|[A-Za-z][a-z ]+)$"
)


@functools.lru_cache(maxsize=None)
def ocr_engine(name: str, use_gpu: bool):
    """The text reader: image (numpy, RGB) -> [(box [x1, y1, x2, y2], text, conf)], one per line
    of text. "doctr" (the default since 2026-10-14) or "easyocr". Scored on the team drawings'
    272 tags (rendered from draw.io, and as screenshot-quality copies: 60 % size, blur, JPEG):
    EasyOCR 81 % read right, docTR 96 % / 93 %; PaddleOCR PP-OCRv5 96 % / 89 % (needs its own
    framework). docTR runs on this environment's PyTorch, on the GPU when there is one."""
    if name == "easyocr":
        import easyocr
        reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

        def read(img, mag=1.5):
            out = []
            for quad, text, conf in reader.readtext(img, mag_ratio=mag):
                xs, ys = [float(p[0]) for p in quad], [float(p[1]) for p in quad]
                out.append(([min(xs), min(ys), max(xs), max(ys)], text.strip(), float(conf)))
            return out
        return read
    from doctr.models import ocr_predictor
    model = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True, assume_straight_pages=True)
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    def read(img, mag=None):
        h, w = img.shape[:2]
        out = []
        with torch.no_grad():
            page = model([img]).pages[0]
        for block in page.blocks:
            for line in block.lines:
                (x1, y1), (x2, y2) = line.geometry
                text = " ".join(wd.value for wd in line.words).strip()
                conf = min((float(wd.confidence) for wd in line.words), default=0.0)
                out.append(([x1 * w, y1 * h, x2 * w, y2 * h], text, conf))
        return out
    return read


def _reread_rotations(read, sheet: Image.Image, box: list[float]) -> tuple[str, float] | None:
    """Re-read one token's crop at 0/90/180/270 degrees; keep the best plausible read.

    A reader's own orientation guess regularly picks an upside-down misread for vertical
    tags (EasyOCR: XV202 -> 'ZOZAX', PSV201 -> 'LOZASd'). Requiring the result to look like
    a tag fixes that.
    """
    pad = 6
    crop = sheet.crop((max(0, box[0] - pad), max(0, box[1] - pad), box[2] + pad, box[3] + pad))
    best = None
    for angle in (0, 90, 180, 270):
        for _, text, conf in read(np.array(crop.rotate(angle, expand=True, fillcolor="white")), 2.0):
            if _PLAUSIBLE.match(text.strip()) and (best is None or conf > best[1]):
                best = (text.strip(), float(conf))
    return best


def _ocr_tiles(read, image: Image.Image, tile: int, stride: int, on_tile=None) -> list[tuple[list[float], str, float]]:
    """OCR over overlapping tiles; returns (box_in_image, text, conf). on_tile(): after each."""
    W, H = image.size
    out = []
    for y0 in tile_starts(H, tile, stride):
        for x0 in tile_starts(W, tile, stride):
            crop = image.crop((x0, y0, min(x0 + tile, W), min(y0 + tile, H)))
            for (x1, y1, x2, y2), text, conf in read(np.array(crop)):
                out.append(([x1 + x0, y1 + y0, x2 + x0, y2 + y0], text, conf))
            if on_tile:
                on_tile()
    return out


def ocr_tile_count(sheet: Image.Image, tile: int = 1000, stride: int = 750) -> int:
    """How many tiles run_ocr reads (upright + the two rotations)."""
    W, H = sheet.size
    upright = len(tile_starts(H, tile, stride)) * len(tile_starts(W, tile, stride))
    return upright + 2 * len(tile_starts(W, tile, stride)) * len(tile_starts(H, tile, stride))


def run_ocr(sheet: Image.Image, use_gpu: bool, tile: int = 1000, stride: int = 750, engine: str = "doctr",
            on_tile=None, on_reread=None) -> list[dict]:
    """Tiled OCR in three orientations, merged.

    A single whole-sheet pass misses most small tags that sit against pipes and symbol
    linework (measured on the LightUp sheet with EasyOCR: GV201/202, CV201, PSV202, V201,
    V203, XV202 all missed, yet each reads at ~0.9-1.0 confidence from a crop). So the sheet
    is read in overlapping tiles - upright, and rotated 90/270 for vertical tags, with boxes
    mapped back to sheet coordinates. Overlapping reads of the same text are merged,
    preferring a plausible tag-like reading, then higher confidence.
    """
    read = ocr_engine(engine, use_gpu)
    sheet = sheet.convert("RGB")
    W, H = sheet.size
    reads = [(b, t, c, 0) for b, t, c in _ocr_tiles(read, sheet, tile, stride, on_tile)]
    for angle in (90, 270):
        rotated = sheet.rotate(angle, expand=True, fillcolor="white")
        for (x1, y1, x2, y2), t, c in _ocr_tiles(read, rotated, tile, stride, on_tile):
            # PIL rotates counter-clockwise. Map rotated-image box back to the sheet.
            if angle == 90:  # rotated (x, y) <- sheet (y, W - x)
                box = [W - y2, x1, W - y1, x2]
            else:  # 270: rotated (x, y) <- sheet (H - y, x)
                box = [y1, H - x2, y2, H - x1]
            reads.append((box, t, c, angle))

    # Single characters are almost always symbol linework read as text ("8", "3").
    reads = [r for r in reads if len(r[1].replace(" ", "")) >= 2]
    reads.sort(key=lambda r: (not _PLAUSIBLE.match(r[1]), r[3] != 0, -r[2]))
    kept: list[tuple[list[float], str, float, int]] = []
    for r in reads:
        if all(box_iou(r[0], k[0]) < 0.3 for k in kept):
            kept.append(r)

    tokens = []
    n_reread = sum(1 for k in kept if not _PLAUSIBLE.match(k[1]))
    done_reread = 0
    if on_reread:
        on_reread(0, n_reread)
    for box, text, conf, angle in kept:
        reread = False
        if not _PLAUSIBLE.match(text):
            better = _reread_rotations(read, sheet, box)
            done_reread += 1
            if on_reread:
                on_reread(done_reread, n_reread)
            if better is not None:
                text, conf, reread = better[0], better[1], True
        tokens.append({"text": text, "conf": conf, "box": [float(v) for v in box], "angle": angle, "reread": reread,
                       "engine": engine})
    return tokens


def drop_shadows(symbols: list, connections: list, low: float = 0.5, floor: float = 0.15) -> tuple[list, list]:
    """Read at several zooms, a symbol seen at only one of them comes out with a low score
    (detect, ensemble "mean") - mostly real symbols the usual zoom misses (small valves),
    but also a box over two symbols or a piece of one. For the app: a low-score symbol
    (< low) mostly (> half) inside better ones is dropped, and anything under `floor`;
    connections to dropped ones go too."""
    def inside(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        area = (a[2] - a[0]) * (a[3] - a[1])
        return ix * iy / area if area > 0 else 0.0
    keep = []
    for s in symbols:
        if s["score"] < floor:
            continue
        if s["score"] < low:
            better = [o["box"] for o in symbols if o is not s and o["score"] > s["score"]]
            if sum(inside(s["box"], b) for b in better) > 0.5:
                continue
        keep.append(s)
    ids = {s["id"] for s in keep}
    return keep, [c for c in connections if c["from"] in ids and c["to"] in ids]


def possibly_missed(weak: list, knots: list, traced: list, ids: list) -> list:
    """Items the model gave no box but a traced pipe runs through: the tracer's knots (a
    symbol's outline on the pipe) and the model's half-seen symbols (weak) on a traced path.
    -> [{"box", "from", "to" (the traced connection it sits on), "class" (weak: the model's
    guess, knots: ""), "score", "how"}], one per place (a weak detection wins: it has a class)."""
    def on_path(box, path):
        return any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in path)
    out = []
    for w in weak:
        t = next((t for t in traced if on_path(w["box"], t["path"])), None)
        if t:
            out.append({"box": [round(v, 1) for v in w["box"]], "from": t["from"], "to": t["to"],
                        "class": w["class"], "score": round(w["score"], 3), "how": "half-seen by the model"})
    for k in knots:
        if any(box_iou(k["box"], o["box"]) > 0 for o in out):
            continue
        out.append({"box": [round(v, 1) for v in k["box"]], "from": ids[k["a"]], "to": ids[k["b"]],
                    "class": "", "score": None, "how": "pipe traced through a drawn symbol"})
    return out


class Progress:
    """The app's progress bar: writes {"step", "done" (0-1), "left" (s, estimated)} to a file.
    Each stage's time is kept per unit of its work (`timings`, from earlier runs on this
    machine and device): seconds per model tile, per text tile, per megapixel - so a sheet's
    stages are weighted by its own size, and the time left comes from how fast this machine
    has been."""
    DEFAULTS = {  # s per unit, measured on the GPU (2026-10-20); the CPU's are learned
        "cuda": {"load": 1.3, "detect": 0.06, "ocr": 0.07, "reread": 0.14, "refine": 0.03, "trace": 0.5, "knots": 0.19},
        "cpu": {"load": 6.0, "detect": 2.5, "ocr": 1.5, "reread": 1.0, "refine": 0.3, "trace": 0.4, "knots": 0.4},
    }

    def __init__(self, path: str | None, timings: str | None, device: str, units: dict,
                 started: float | None = None) -> None:
        """started: when the app launched this script (Python and the libraries starting up
        come before anything here can be measured - counted as an already finished stage)."""
        self.path, self.timings, self.device, self.units = path, timings, device, units
        rates = dict(self.DEFAULTS.get(device, self.DEFAULTS["cpu"]))
        if timings and Path(timings).exists():
            try:
                rates.update(json.loads(Path(timings).read_text()).get(device, {}))
            except (ValueError, OSError):
                pass
        self.rates = rates
        self.cost = {k: rates[k] * units.get(k, 1) for k in rates if k in units}
        self.done_before, self.stage, self.t0, self.took = 0.0, None, time.time(), {}
        if started:
            self.took["startup"] = self.cost["startup"] = self.done_before = max(0.0, time.time() - started)
        self.total = sum(self.cost.values())

    def units_now(self, stage: str, n: float) -> None:
        """A stage whose size is known only once it comes (how much text needs a second
        look): its cost - and the whole run's - from now on."""
        self.units[stage] = n
        self.cost[stage] = self.rates.get(stage, 0.1) * n
        self.total = self.done_before + sum(c for k, c in self.cost.items() if k not in self.took and k != self.stage) \
            + (self.cost[self.stage] if self.stage else 0.0)

    def start(self, stage: str, step: str) -> None:
        if self.stage:
            self.took[self.stage] = time.time() - self.t0
            self.done_before += self.cost[self.stage]
        self.stage, self.t0, self.step = stage, time.time(), step
        self.part(0.0)

    def part(self, frac: float, step: str | None = None) -> None:
        if not self.path:
            return
        done = (self.done_before + self.cost[self.stage] * min(1.0, frac)) / self.total
        # the stage's span of the bar and how long it should take: the app moves the bar on
        # smoothly between these reports (never past the stage's end)
        tmp = Path(self.path).with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "step": step or self.step, "done": round(done, 4), "left": round(max(0.0, self.total * (1 - done)), 1),
            "stage_start": round(self.done_before / self.total, 4),
            "stage_end": round((self.done_before + self.cost[self.stage]) / self.total, 4),
            "stage_est": round(self.cost[self.stage], 2), "stage_t0": self.t0, "total": round(self.total, 1)}))
        tmp.replace(self.path)   # whole, never half-written

    def finish(self) -> None:
        """Remember how fast each stage went (per unit), for the next sheet's estimate."""
        if self.stage:
            self.took[self.stage] = time.time() - self.t0
        if not self.timings:
            return
        try:
            saved = json.loads(Path(self.timings).read_text()) if Path(self.timings).exists() else {}
        except (ValueError, OSError):
            saved = {}
        mine = saved.setdefault(self.device, {})
        mine["_total"] = round(sum(self.took.values()), 1)   # the app's start-up estimate uses it
        for k, t in self.took.items():
            rate = t / max(1e-6, self.units.get(k, 1))
            mine[k] = round(0.5 * mine.get(k, rate) + 0.5 * rate, 4)   # smoothed over runs
        Path(self.timings).write_text(json.dumps(saved, indent=1))


def use_fallback_attention(device: str) -> None:
    """The compiled deformable-attention op is CUDA-only and only built where the CUDA
    toolkit is installed (setup.sh); otherwise use the repo's pure-PyTorch reference
    implementation - on CPU, or on an NVIDIA GPU with just the driver (a laptop)."""
    import models.ops.functions.ms_deform_attn_func as f
    if device != "cpu" and f.MSDA is not None:
        return
    import models.ops.modules.ms_deform_attn as m

    class _CPUFallback:
        @staticmethod
        def apply(value, shapes, level_start, sampling_locations, attention_weights, im2col_step):
            return f.ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)

    m.MSDeformAttnFunction = _CPUFallback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True, help="a checkpoint, or several comma-separated (an ensemble)")
    ap.add_argument("--out", required=True)
    # Training patches show a typical symbol at ~45-55 px after the 800 px resize; a team
    # draw.io symbol is ~85 px, so a 1400 px tile (scale 0.57) matches. 2000 px tiles
    # (the earlier default) shrank symbols to ~34 px.
    # the sheet read at four zooms and merged (detect): scored on OPEN100's four folds, items mAP
    # 0.720 -> 0.770, links 0.343 -> 0.434 against one tile size (ml/README.md)
    ap.add_argument("--tile", default="2350,1850,1400,950", help="tile sizes, comma-separated (stride: half)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--ocr", default="doctr", choices=["doctr", "easyocr"], help="text reader (ocr_engine)")
    ap.add_argument("--no-refine", action="store_true", help="raw model symbols (refine_symbols.py off)")
    ap.add_argument("--tracer", default="lines", choices=["lines", "pixels"],
                    help="pipe tracer: straight line pieces (trace_lines.py) or the skeleton (trace_pipes.py)")
    ap.add_argument("--progress", default=None, help="write progress here (the app's progress bar - Progress)")
    ap.add_argument("--timings", default=None, help="stage speeds from earlier runs, updated after this one")
    ap.add_argument("--started", type=float, default=None, help="when the app launched this (epoch s)")
    args = ap.parse_args()

    use_fallback_attention(args.device)
    sheet = load_sheet(args.image)
    tiles = [int(t) for t in str(args.tile).split(",")]
    checkpoints = args.checkpoint.split(",")
    W, H = sheet.size
    n_tiles = len(checkpoints) * sum(len(tile_starts(H, t, t // 2)) * len(tile_starts(W, t, t // 2)) for t in tiles)
    n_ocr = 0 if args.no_ocr else ocr_tile_count(sheet)
    mpx = W * H / 1e6
    progress = Progress(args.progress, args.timings, "cuda" if args.device != "cpu" else "cpu",
                        {"load": 1, "detect": n_tiles, "ocr": n_ocr, "reread": 0, "refine": mpx, "trace": mpx,
                         "knots": mpx}, started=args.started)
    progress.start("load", "Loading the model")
    for c in checkpoints:
        _load_model(c, args.device)
    if not args.no_ocr:
        ocr_engine(args.ocr, args.device != "cpu")   # (kept loaded: run_ocr uses the same one)
    progress.start("detect", "Finding symbols")
    symbols, connections, lines = detect(
        sheet, checkpoints, tiles, [t // 2 for t in tiles], args.device, with_points=True,
        on_tile=lambda i, n: progress.part(i / n, f"Finding symbols - part {i} of {n} of the sheet"))
    symbols, connections = drop_shadows(symbols, connections)
    # items found at fewer than half the zooms are left out: they would cut traced pipes into
    # pieces (scored end to end on unseen sheets, ml/README.md). Not offered as "possible missed
    # items" either: on LightUp (every symbol labelled) most were none - 28 wrong lines into them
    # over 4 folds, against 10 with the half-seen tokens and pipe knots alone
    unsure = [s for s in symbols if s["score"] < SURE]
    symbols = [s for s in symbols if s["score"] >= SURE]
    ids = {s["id"] for s in symbols}
    connections = [c for c in connections if c["from"] in ids and c["to"] in ids]
    progress.start("ocr", "Reading the text")
    ocr_done = [0]

    def ocr_tick():
        ocr_done[0] += 1
        progress.part(ocr_done[0] / max(1, n_ocr), f"Reading the text - part {min(ocr_done[0], n_ocr)} of {n_ocr}")
    def reread_tick(i, n):
        if i == 0:   # how much text needs a second look is known now
            progress.start("reread", "Reading the text again where it did not look like a tag")
            progress.units_now("reread", n)
        progress.part(i / max(1, n), f"Reading the text again - {i} of {n}")
    tokens = [] if args.no_ocr else run_ocr(sheet, use_gpu=args.device != "cpu", engine=args.ocr, on_tile=ocr_tick,
                                            on_reread=reread_tick)
    progress.start("refine", "Tidying the symbols")
    # Clean up with the picture and the tags (refine_symbols.py): tag-set classes, tank boxes
    # fitted to their outlines and merged, containers and their compartments the model misses.
    if not args.no_refine:
        from refine_symbols import refine
        symbols, renamed = refine(np.asarray(sheet.convert("L")), symbols, tokens)
        connections = [dict(c, **{"from": renamed.get(c["from"], c["from"]), "to": renamed.get(c["to"], c["to"])})
                       for c in connections]
        connections = [c for c in connections if c["from"] != c["to"]]
    # Pipes followed across the whole sheet (trace_pipes.py): the model only links symbols
    # that share a tile; the tracer has no tiles. Paths and arrowhead directions included.
    if args.tracer == "lines":   # straight line pieces (trace_lines.py); "pixels": the skeleton tracer
        from trace_lines import trace_lines as trace_connections
    else:
        from trace_pipes import trace_connections
    progress.start("trace", "Following the pipes")
    ids = [s["id"] for s in symbols]
    knots = []  # symbols on a pipe with no box: the pipe was traced through their outline
    rgb = np.asarray(sheet.convert("RGB"))
    found = trace_connections(rgb, [s["box"] for s in symbols], [t["box"] for t in tokens],
                              arrows={k for k, s in enumerate(symbols) if s["class"] == "arrow"},
                              missed=knots if args.tracer != "lines" else None)
    if args.tracer == "lines":   # symbols on a pipe with no box: trace_pipes' outline-knot check
        progress.start("knots", "Looking for symbols on the pipes with no box")
        from trace_pipes import trace_connections as trace_pixels
        trace_pixels(rgb, [s["box"] for s in symbols], [t["box"] for t in tokens], missed=knots)
    traced = [{"from": ids[t["a"]], "to": ids[t["b"]], "line_type": t["line_type"], "flow": t["flow"],
               "path": [[round(x, 1), round(y, 1)] for x, y in t["path"]], "bridges": t["bridges"],
               "length": round(float(t.get("length", 0.0)), 1), "net": t.get("net")}
              for t in found]
    out = {"image": os.path.abspath(args.image), "image_size": list(sheet.size), "checkpoint": args.checkpoint,
           "symbols": symbols, "connections": connections, "traced": traced, "ocr_tokens": tokens,
           "line_points": lines, "missed": possibly_missed(lines.get("weak", []), knots, traced, ids),
           "unsure": [{"box": s["box"], "class": s["class"], "score": s["score"]} for s in unsure]}
    Path(args.out).write_text(json.dumps(out, indent=1))
    progress.finish()
    print(json.dumps({"symbols": len(symbols), "connections": len(connections), "traced": len(traced), "ocr_tokens": len(tokens)}))


if __name__ == "__main__":
    main()
