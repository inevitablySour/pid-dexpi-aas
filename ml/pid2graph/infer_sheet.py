"""Run the trained Relationformer + EasyOCR on one whole P&ID image and write JSON.

Runs inside the ML venv (ml/relationformer/.venv) because it needs the compiled
deformable-attention op; the Streamlit app (main venv) calls it as a subprocess and
reads the JSON. Usage:

    ml/relationformer/.venv/bin/python ml/pid2graph/infer_sheet.py \
        --image drawing.png --checkpoint <ckpt.pt> --out result.json

The model was trained on 2000px PID2Graph patches, so the sheet is tiled the same way
(2000px windows, 500px stride), each tile resized to 800px like validation. Detections
from overlapping tiles are merged (greedy, class-agnostic, IoU >= 0.5, majority class
vote), and per-tile connection probabilities are mapped onto the merged symbols.

OCR output is raw tokens only - assembling them into tags and attaching tags to symbols
happens in pid_dexpi_aas.ocr_tags (main venv), where it is unit-testable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

RF_DIR = Path(__file__).resolve().parent.parent / "relationformer"
sys.path.insert(0, str(RF_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # pid_dexpi_aas.sheet_image
sys.path.insert(0, str(Path(__file__).resolve().parent))  # trace_pipes

from inference import graph_infer  # noqa: E402
from models import build_model  # noqa: E402
from train_pid2graph import dict2obj  # noqa: E402
import transforms as T  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
# Must match pid2graph_data/dataset_pid2graph.py's PID2GRAPH_CLASSES minus background.
CLASS_NAMES = ["arrow", "general", "inlet/outlet", "instrumentation", "pump", "tank", "valve"]


def load_sheet(path: str) -> Image.Image:
    # Dark ink on white, whatever came in: transparent exports, and draw.io dark-theme
    # exports / dark screenshots (light lines - laid on white they vanished).
    from pid_dexpi_aas.sheet_image import to_dark_on_white
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


def detect(sheet: Image.Image, checkpoint: str, tile: int, stride: int, device: str,
           min_prob: float = 0.5) -> tuple[list, list]:
    """Symbols + connections for a whole sheet. Connections below min_prob are dropped
    (0.5 = the model's own decision; scoring passes a low value to rank all pairs)."""
    config = dict2obj(yaml.safe_load(open(RF_DIR / "configs" / "pid2graph_2d.yaml")))
    state = torch.load(checkpoint, map_location="cpu")["net"]
    # The relation head's output size says how many line types the checkpoint learned
    # (1 = "connected"; 2 = process vs signal, fine-tuned models) - build to match.
    config.MODEL.NUM_REL_CLS = int(state["relation_embed.layers.3.weight"].shape[0]) - 1
    dev = torch.device(device)
    model = build_model(config).to(dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match model: missing={missing} unexpected={unexpected}")
    model.eval()
    tf = T.Compose([T.RandomResize([800], max_size=800),
                    T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])])

    W, H = sheet.size
    dets, tile_edges, tid = [], [], 0
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
            for k, (cx, cy, bw, bh) in enumerate(inf["pred_boxes"][0]):
                dets.append({"box": [float(x0 + (cx - bw / 2) * cw), float(y0 + (cy - bh / 2) * ch),
                                     float(x0 + (cx + bw / 2) * cw), float(y0 + (cy + bh / 2) * ch)],
                             "cls": int(inf["pred_boxes_class"][0][k]),
                             "score": float(inf["pred_boxes_score"][0][k]), "tile": tid, "idx": k})
            pairs, rel = inf["all_node_pairs"][0], inf["all_relation"][0]
            if pairs is not None:
                for (i, j), p in zip(pairs, rel):
                    if i < j:
                        tile_edges.append((tid, int(i), int(j), float(np.max(p[1:])), int(np.argmax(p[1:])) + 1))
            tid += 1

    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    cluster_of, merged = {}, []
    for i in order:
        d = dets[i]
        for m_idx, m in enumerate(merged):
            if box_iou(d["box"], m["box"]) >= 0.5:
                cluster_of[(d["tile"], d["idx"])] = m_idx
                m["votes"].append(d["cls"])
                break
        else:
            cluster_of[(d["tile"], d["idx"])] = len(merged)
            merged.append({"box": d["box"], "score": d["score"], "votes": [d["cls"]]})
    symbols = []
    for n, m in enumerate(merged):
        counts = np.bincount(m["votes"], minlength=len(CLASS_NAMES))
        symbols.append({"id": f"m{n}", "box": m["box"], "class": CLASS_NAMES[int(counts.argmax())],
                        "score": m["score"], "class_votes": {CLASS_NAMES[c]: int(v) for c, v in enumerate(counts) if v},
                        "n_tiles": len(m["votes"])})

    probs: dict[tuple[int, int], tuple[float, int]] = {}
    for t, i, j, p, rel_cls in tile_edges:
        a, b = cluster_of.get((t, i)), cluster_of.get((t, j))
        if a is None or b is None or a == b:
            continue
        key = (min(a, b), max(a, b))
        if p > probs.get(key, (0.0, 1))[0]:
            probs[key] = (p, rel_cls)
    line_type = {1: "process", 2: "signal"} if config.MODEL.NUM_REL_CLS >= 2 else {1: "process"}
    connections = [{"from": f"m{a}", "to": f"m{b}", "prob": p, "line_type": line_type.get(c, "process")}
                   for (a, b), (p, c) in sorted(probs.items()) if p >= min_prob]
    return symbols, connections


# A token "looks right" if it is a plant tag (GV204, PSV-101, TI 201/2), bare ISA letters
# (PI, TIC), a number (102), or a value with a unit (15 bar). Anything else is re-read.
_PLAUSIBLE = re.compile(
    r"^([A-Z]{1,4}[- ]?\d{1,4}[A-Z]?(/\d+)?|[A-Z]{1,4}|\d{1,4}(/\d+)?|\d+(\.\d+)? ?[a-zA-Z/]+.*|\[.*\]|[A-Za-z][a-z ]+)$"
)


def _reread_rotations(reader, sheet: Image.Image, box: list[float]) -> tuple[str, float] | None:
    """Re-read one token's crop at 0/90/180/270 degrees; keep the best plausible read.

    EasyOCR's own rotation_info picks the most *confident* orientation, which for
    vertical tags is regularly an upside-down misread (XV202 -> 'ZOZAX', PSV201 ->
    'LOZASd'). Requiring the result to look like a tag fixes that.
    """
    pad = 6
    crop = sheet.crop((max(0, box[0] - pad), max(0, box[1] - pad), box[2] + pad, box[3] + pad))
    best = None
    for angle in (0, 90, 180, 270):
        for _, text, conf in reader.readtext(np.array(crop.rotate(angle, expand=True, fillcolor="white")),
                                              mag_ratio=2.0):
            if _PLAUSIBLE.match(text.strip()) and (best is None or conf > best[1]):
                best = (text.strip(), float(conf))
    return best


def _ocr_tiles(reader, image: Image.Image, tile: int, stride: int) -> list[tuple[list[float], str, float]]:
    """OCR over overlapping tiles; returns (box_in_image, text, conf)."""
    W, H = image.size
    out = []
    for y0 in tile_starts(H, tile, stride):
        for x0 in tile_starts(W, tile, stride):
            crop = image.crop((x0, y0, min(x0 + tile, W), min(y0 + tile, H)))
            for quad, text, conf in reader.readtext(np.array(crop), mag_ratio=1.5):
                xs = [float(p[0]) + x0 for p in quad]
                ys = [float(p[1]) + y0 for p in quad]
                out.append(([min(xs), min(ys), max(xs), max(ys)], text.strip(), float(conf)))
    return out


def run_ocr(sheet: Image.Image, use_gpu: bool, tile: int = 1000, stride: int = 750) -> list[dict]:
    """Tiled OCR in three orientations, merged.

    A single whole-sheet pass misses most small tags that sit against pipes and symbol
    linework (measured on the LightUp sheet: GV201/202, CV201, PSV202, V201, V203, XV202
    all missed, yet each reads at ~0.9-1.0 confidence from a crop). So the sheet is read
    in overlapping tiles - upright, and rotated 90/270 for vertical tags, with boxes
    mapped back to sheet coordinates. Overlapping reads of the same text are merged,
    preferring a plausible tag-like reading, then higher confidence.
    """
    import easyocr

    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    W, H = sheet.size
    reads = [(b, t, c, 0) for b, t, c in _ocr_tiles(reader, sheet, tile, stride)]
    for angle in (90, 270):
        rotated = sheet.rotate(angle, expand=True, fillcolor="white")
        for (x1, y1, x2, y2), t, c in _ocr_tiles(reader, rotated, tile, stride):
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
    for box, text, conf, angle in kept:
        reread = False
        if not _PLAUSIBLE.match(text):
            better = _reread_rotations(reader, sheet, box)
            if better is not None:
                text, conf, reread = better[0], better[1], True
        tokens.append({"text": text, "conf": conf, "box": box, "angle": angle, "reread": reread})
    return tokens


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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    # Training patches show a typical symbol at ~45-55 px after the 800 px resize; a team
    # draw.io symbol is ~85 px, so a 1400 px tile (scale 0.57) matches. 2000 px tiles
    # (the earlier default) shrank symbols to ~34 px.
    ap.add_argument("--tile", type=int, default=1400)
    ap.add_argument("--stride", type=int, default=700)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-refine", action="store_true", help="raw model symbols (refine_symbols.py off)")
    args = ap.parse_args()

    use_fallback_attention(args.device)
    sheet = load_sheet(args.image)
    symbols, connections = detect(sheet, args.checkpoint, args.tile, args.stride, args.device)
    tokens = [] if args.no_ocr else run_ocr(sheet, use_gpu=args.device != "cpu")
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
    from trace_pipes import trace_connections
    ids = [s["id"] for s in symbols]
    traced = [{"from": ids[t["a"]], "to": ids[t["b"]], "line_type": t["line_type"], "flow": t["flow"],
               "path": [[round(x, 1), round(y, 1)] for x, y in t["path"]], "bridges": t["bridges"]}
              for t in trace_connections(np.asarray(sheet.convert("L")), [s["box"] for s in symbols],
                                         [t["box"] for t in tokens])]
    out = {"image": os.path.abspath(args.image), "image_size": list(sheet.size), "checkpoint": args.checkpoint,
           "symbols": symbols, "connections": connections, "traced": traced, "ocr_tokens": tokens}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"symbols": len(symbols), "connections": len(connections), "traced": len(traced), "ocr_tokens": len(tokens)}))


if __name__ == "__main__":
    main()
