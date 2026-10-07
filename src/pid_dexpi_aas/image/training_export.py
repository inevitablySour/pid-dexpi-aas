"""Reviewed drawings -> PID2Graph training / test data, in the layout the model trains on:

    Complete/<subset>/<name>.png + .graphml          the whole sheet (sheet-level scoring)
    Patched/<subset>/<name>/<n>.png + .graphml       1500 px tiles, 500 px stride (training)

One writer for both sources, so their data cannot drift apart: the team's draw.io drawings
(ml/pid2graph/data/make_team_data.py) and pictures checked in the review app (a .zip the app
offers for download; unzip it into data/PID2Graph).

Labels: symbols with their PID2Graph class (valve, instrumentation, tank, pump, general,
inlet/outlet, arrow) and box; connections between symbols, "solid" or "non-solid" (signal).
Where a connection's drawn path is known (pipes the tracer followed, draw.io routes) it is
kept as an edge attribute "path" - the data for learning pipe bends and tees later.

A picture is first scaled so its symbols are about the size the team tiles show (~85 px),
whatever the scan resolution.
"""
from __future__ import annotations

import io
import json
import statistics
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from pid_dexpi_aas.image.sheet_image import to_dark_on_white

NS = "http://graphml.graphdrawing.org/xmlns"
TARGET_SYMBOL_PX = 85.0  # median symbol size in the team training tiles (draw.io rendered at scale 1)


def graphml_bytes(symbols: list, edges: list) -> bytes:
    """symbols: [(id, label, (x1, y1, x2, y2))]; edges: [(a, b, "solid" | "non-solid")] or
    [(a, b, line, path)] with path = [(x, y), ...] or None."""
    ET.register_namespace("", NS)
    root = ET.Element(f"{{{NS}}}graphml")
    for kid, name, typ in [("d0", "label", "string"), ("d1", "xmin", "double"), ("d2", "xmax", "double"),
                           ("d3", "ymin", "double"), ("d4", "ymax", "double")]:
        ET.SubElement(root, f"{{{NS}}}key", id=kid, attrib={"for": "node", "attr.name": name, "attr.type": typ})
    ET.SubElement(root, f"{{{NS}}}key", id="d9", attrib={"for": "edge", "attr.name": "edge_label", "attr.type": "string"})
    has_paths = any(len(e) > 3 and e[3] for e in edges)
    if has_paths:
        ET.SubElement(root, f"{{{NS}}}key", id="d10", attrib={"for": "edge", "attr.name": "path", "attr.type": "string"})
    graph = ET.SubElement(root, f"{{{NS}}}graph", edgedefault="undirected")
    for sid, label, (x1, y1, x2, y2) in symbols:
        node = ET.SubElement(graph, f"{{{NS}}}node", id=sid)
        for kid, val in (("d0", label), ("d1", x1), ("d2", x2), ("d3", y1), ("d4", y2)):
            ET.SubElement(node, f"{{{NS}}}data", key=kid).text = str(round(val, 2)) if kid != "d0" else val
    for e in edges:
        el = ET.SubElement(graph, f"{{{NS}}}edge", source=e[0], target=e[1])
        ET.SubElement(el, f"{{{NS}}}data", key="d9").text = e[2]
        if len(e) > 3 and e[3]:
            ET.SubElement(el, f"{{{NS}}}data", key="d10").text = json.dumps([[round(x, 1), round(y, 1)] for x, y in e[3]])
    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, xml_declaration=True, encoding="utf-8")
    return buf.getvalue()


def tile_starts(total: int, tile: int, stride: int) -> list[int]:
    if total <= tile:
        return [0]
    s = list(range(0, total - tile + 1, stride))
    return s if s[-1] == total - tile else s + [total - tile]


def write_sheet(write, sheet: Image.Image, name: str, subset: str, symbols: list, edges: list,
                tile: int = 1500, stride: int = 500) -> int:
    """Write one sheet and its tiles through write(relative path, bytes). Tiles keep the
    symbols at least half inside (clipped to the tile) and the connections whose two ends
    are both kept. Returns the number of tiles."""
    def png(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    write(f"Complete/{subset}/{name}.png", png(sheet))
    write(f"Complete/{subset}/{name}.graphml", graphml_bytes(symbols, edges))
    W, H = sheet.size
    n = 0
    for y0 in tile_starts(H, tile, stride):
        for x0 in tile_starts(W, tile, stride):
            x1, y1 = x0 + min(tile, W), y0 + min(tile, H)
            kept = []
            for sid, lab, (a, b, c, d) in symbols:
                ia = max(0.0, min(c, x1) - max(a, x0)) * max(0.0, min(d, y1) - max(b, y0))
                if ia >= 0.5 * (c - a) * (d - b):
                    kept.append((sid, lab, (max(a, x0) - x0, max(b, y0) - y0, min(c, x1) - x0, min(d, y1) - y0)))
            ids = {k[0] for k in kept}
            write(f"Patched/{subset}/{name}/{n}.png", png(sheet.crop((x0, y0, x1, y1))))
            write(f"Patched/{subset}/{name}/{n}.graphml",
                  graphml_bytes(kept, [e[:3] for e in edges if e[0] in ids and e[1] in ids]))
            n += 1
    return n


def normalise_scale(sheet: Image.Image, symbols: list, edges: list):
    """Scale a picture and its labels so the median symbol is ~TARGET_SYMBOL_PX (what the
    model is trained to see), whatever the scan resolution. -> (sheet, symbols, edges, scale)"""
    sizes = [max(b[2] - b[0], b[3] - b[1]) for _, _, b in symbols]
    scale = max(0.4, min(3.0, TARGET_SYMBOL_PX / statistics.median(sizes))) if sizes else 1.0
    if abs(scale - 1.0) <= 0.02:
        return sheet, symbols, edges, 1.0
    sheet = sheet.resize((round(sheet.width * scale), round(sheet.height * scale)), Image.LANCZOS)
    symbols = [(sid, lab, tuple(v * scale for v in b)) for sid, lab, b in symbols]
    edges = [(e[0], e[1], e[2], [(x * scale, y * scale) for x, y in e[3]] if len(e) > 3 and e[3] else None)
             for e in edges]
    return sheet, symbols, edges, scale


def dir_writer(root: Path):
    def write(rel: str, data: bytes) -> None:
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return write


def _corners(points: list, tol: float = 2.0) -> list:
    """A traced path (a point every few px) reduced to its corners (Ramer-Douglas-Peucker):
    the bends a model of pipe geometry would learn, and a much smaller file."""
    if len(points) < 3:
        return points
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5 or 1e-9
    far, k = 0.0, 0
    for i, (x, y) in enumerate(points[1:-1], start=1):
        d = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norm
        if d > far:
            far, k = d, i
    if far <= tol:
        return [points[0], points[-1]]
    return _corners(points[:k + 1], tol)[:-1] + _corners(points[k:], tol)


# ----------------------------------------------------------------- reviewed pictures ---
def review_labels(review) -> tuple[list, list]:
    """A reviewed picture's symbols [(id, label, box)] and connections [(a, b, line, path)],
    in picture pixels. Only what is drawn on the picture: items left out of DEXPI and the
    streams added beside an item (nothing drawn there) are not symbols."""
    from pid_dexpi_aas.image.export_pid2graph import pid2graph_class
    from pid_dexpi_aas.review import OFF_PAGE, SKIP

    symbols, ids = [], {}
    for s in review.symbols:
        if not s.box or s.detected_as == "stream in/out":
            continue
        if s.source == "image model" and s.detected_as == "arrow" and s.dexpi_class == SKIP:
            # flow arrows are never DEXPI items, so nobody reviews them; dropping them would teach
            # the model that arrows are background - kept as the model found them
            label = "arrow"
        elif not s.include or s.dexpi_class == SKIP:
            continue  # left out by the reviewer: a false detection, or not a process item
        elif s.dexpi_class == OFF_PAGE:
            label = "inlet/outlet"
        else:
            label = pid2graph_class(s.dexpi_class, "review") or "general"
        sid = f"n{len(symbols)}"
        ids[s.id] = sid
        symbols.append((sid, label, tuple(float(v) for v in s.box)))
    edges, seen = [], set()
    for c in review.connections:
        a, b = ids.get(c.from_id), ids.get(c.to_id)
        if not c.include or a is None or b is None or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        path = _corners([tuple(p) for p in c.path]) if getattr(c, "path", None) else None
        edges.append((key[0], key[1], "non-solid" if c.line_type == "signal" else "solid", path))
    return symbols, edges


def review_training_zip(review, subset: str = "Reviewed", name: str | None = None,
                        tile: int = 1500, stride: int = 500) -> tuple[bytes, dict]:
    """A reviewed picture as a .zip of PID2Graph data (unzip into data/PID2Graph). The
    picture is scaled so its median symbol is ~TARGET_SYMBOL_PX. -> (zip bytes, summary)."""
    if not review.image_path or not Path(review.image_path).exists():
        raise ValueError("only a review with a picture can become training data")
    symbols, edges = review_labels(review)
    if not symbols:
        raise ValueError("no symbols to export")
    with Image.open(review.image_path) as im:
        sheet = to_dark_on_white(im)  # what the model sees; a plain RGB convert blackens transparency
    sheet, symbols, edges, scale = normalise_scale(sheet, symbols, edges)
    name = (name or Path(review.source_name).stem).strip()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        tiles = write_sheet(z.writestr, sheet, name, subset, symbols, edges, tile, stride)
        z.writestr("README.txt",
                   f"PID2Graph data for '{name}' (subset '{subset}'), made from a review in the P&ID -> DEXPI app.\n"
                   "Add it:   unzip -o <this file> -d data/PID2Graph\n"
                   f"Train on it: add 'Patched/{subset}' to DATA.EXTRA_TRAIN_SUBSETS in the training config.\n"
                   "Held out as a test drawing (never train on it): use a subset ending in 'test' and score with\n"
                   "          ml/pid2graph/eval/eval_sheets.py --set reviewed\n")
    summary = {"name": name, "subset": subset, "symbols": len(symbols), "connections": len(edges),
               "with path": sum(1 for e in edges if e[3]), "tiles": tiles, "scale": round(scale, 3),
               "size": list(sheet.size), "unreviewed arrows": sum(1 for _, l, _ in symbols if l == "arrow"),
               "classes": {lab: sum(1 for _, l, _ in symbols if l == lab)
                                                      for lab in sorted({l for _, l, _ in symbols})}}
    return buf.getvalue(), summary


# -------------------------------------------------- drawings added from the review app ---
REVIEWED_SUBSETS = ("Reviewed", "Reviewed test")


def list_reviewed(root: Path) -> list[dict]:
    """Drawings added as training / held-out data: [{name, subset, tiles}]."""
    out = []
    for subset in REVIEWED_SUBSETS:
        for png in sorted((Path(root) / "Complete" / subset).glob("*.png")):
            tiles = Path(root) / "Patched" / subset / png.stem
            out.append({"name": png.stem, "subset": subset,
                        "tiles": len(list(tiles.glob("*.png"))) if tiles.is_dir() else 0})
    return out


def remove_reviewed(root: Path, subset: str, name: str) -> None:
    """Remove an added drawing (its sheet and its tiles)."""
    import shutil
    if subset not in REVIEWED_SUBSETS or "/" in name or name.startswith("."):
        raise ValueError("not an added drawing")
    for f in (Path(root) / "Complete" / subset).glob(f"{name}.*"):
        f.unlink()
    shutil.rmtree(Path(root) / "Patched" / subset / name, ignore_errors=True)
