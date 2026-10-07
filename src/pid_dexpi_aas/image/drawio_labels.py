"""Label pictures in draw.io: a reviewed picture -> an editable .drawio, and back -> training data.

The review app gets the DEXPI model right (items, types, tags, connections) but cannot edit
the geometry training labels need: exact boxes, pipe paths, solid vs dashed. draw.io can:

  export  the picture on a locked background layer; every item a box coloured by its class
          (the class rides along as a property, so moving / resizing keeps it); every line a
          connector between its two boxes along its traced path - bend points to drag -
          dashed for signal lines. A legend beside the picture holds one box per class: copy
          it (Ctrl+C / Ctrl+V) to add a missed item of that class.
  import  boxes and lines back out (the picture is read from the file itself), each line's
          path from box edge through its bend points to box edge, written with the same
          writer as the team drawings (training_export.write_sheet).

Coordinates: the picture sits at (0, 0) at its pixel size, so draw.io units = picture pixels.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from pid_dexpi_aas.image.sheet_image import to_dark_on_white

MAX_PIXELS = 60_000_000  # a picture inside an uploaded file: larger is refused (memory)

CLASSES = {  # PID2Graph class -> box colour
    "valve": "#00a651", "instrumentation": "#1f6fe5", "tank": "#f28c00", "pump": "#9b30d9",
    "general": "#6b6b6b", "inlet/outlet": "#8b5a2b", "arrow": "#e0004d",
}
LEGEND_GAP = 60


def _box_style(cls: str, font: int = 11) -> str:
    return (f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={CLASSES.get(cls, '#6b6b6b')};"
            f"strokeWidth=2;fontSize={font};fontColor=#000000;verticalAlign=bottom;labelPosition=center;"
            "verticalLabelPosition=top;align=center;labelBackgroundColor=#ffffffcc;")


def _line_style(dashed: bool) -> str:
    return ("edgeStyle=none;rounded=0;html=1;endArrow=none;strokeColor=#d40000;strokeWidth=2;opacity=80;"
            + ("dashed=1;" if dashed else ""))


def review_to_drawio(review, name: str | None = None) -> str:
    """The reviewed picture as a draw.io file (XML) for correcting the labels by hand."""
    from pid_dexpi_aas.image.training_export import review_labels

    if not review.image_path or not Path(review.image_path).exists():
        raise ValueError("only a review with a picture can be labelled in draw.io")
    symbols, edges = review_labels(review)
    with Image.open(review.image_path) as im:
        W, H = im.size
        buf = io.BytesIO()
        to_dark_on_white(im).save(buf, format="PNG")  # transparent pixels would turn black
    data = base64.b64encode(buf.getvalue()).decode()

    root = ET.Element("mxGraphModel", dx="0", dy="0", grid="0", gridSize="10", guides="1", tooltips="1",
                      connect="1", arrows="1", fold="1", page="0", pageScale="1", pageWidth=str(W), pageHeight=str(H),
                      math="0", shadow="0", background="#ffffff")  # white canvas in dark theme too
    cells = ET.SubElement(root, "root")
    ET.SubElement(cells, "mxCell", id="0")
    ET.SubElement(cells, "mxCell", id="picture", value="Picture (locked)", parent="0", style="locked=1;")
    ET.SubElement(cells, "mxCell", id="labels", value="Labels", parent="0")
    pic = ET.SubElement(cells, "mxCell", id="background", value="", parent="picture", vertex="1",
                        style=f"shape=image;imageAspect=0;aspect=fixed;image=data:image/png,{data};")
    ET.SubElement(pic, "mxGeometry", x="0", y="0", width=str(W), height=str(H), **{"as": "geometry"})

    # text and legend sized to the picture's symbols, not fixed pixels: a typical symbol is
    # the unit (small scans and large sheets both get readable, proportionate labels)
    sizes = sorted(max(b[2] - b[0], b[3] - b[1]) for _, _, b in symbols)
    unit = sizes[len(sizes) // 2] if sizes else max(W, H) / 30
    label_font = max(6, min(14, round(unit * 0.13)))
    note_font = max(7, min(16, round(unit * 0.15)))

    def vertex(cid, label, cls, box, parent="labels"):
        obj = ET.SubElement(cells, "object", id=cid, label=html.escape(label or ""), pid2graph_class=cls)
        cell = ET.SubElement(obj, "mxCell", style=_box_style(cls, label_font), vertex="1", parent=parent)
        x1, y1, x2, y2 = box
        ET.SubElement(cell, "mxGeometry", x=f"{x1:.1f}", y=f"{y1:.1f}", width=f"{x2 - x1:.1f}", height=f"{y2 - y1:.1f}",
                      **{"as": "geometry"})

    tags = {s.id: s.tag for s in review.symbols}
    by_box = {tuple(float(v) for v in s.box): s.id for s in review.symbols if s.box}
    for sid, cls, box in symbols:
        vertex(sid, tags.get(by_box.get(tuple(box)), ""), cls, box)
    for n, (a, b, line, path) in enumerate(edges):
        cell = ET.SubElement(cells, "mxCell", id=f"line{n}", value="", style=_line_style(line == "non-solid"),
                             edge="1", parent="labels", source=a, target=b)
        geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        if path and len(path) > 2:
            pts = ET.SubElement(geo, "Array", **{"as": "points"})
            for x, y in path[1:-1]:
                ET.SubElement(pts, "mxPoint", x=f"{x:.1f}", y=f"{y:.1f}")
    # legend: one box per class to copy, and how-to notes, left of the picture (outside it)
    legend = ET.SubElement(cells, "mxCell", id="legend", value="Legend (not labels)", parent="0")
    note_w, gap = 3.2 * unit, 0.6 * unit
    lx = -(note_w + gap)
    note = ET.SubElement(cells, "mxCell", id="howto", parent="legend", vertex="1", style=(
        f"text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize={note_font};"), value=(
        "<b>Correct the labels</b><br>- resize / move boxes to fit each symbol<br>- drag a line's bend points onto "
        "the drawn pipe (click a line, drag the handles; drag the middle of a segment for a new bend)<br>- signal "
        "line: Edit Style, dashed=1 (or the Line dashed option)<br>- missed item: copy the box of its class below, "
        "paste, place<br>- <b>flow arrows</b>: every arrowhead drawn on a pipe - mid-pipe or at its end, also "
        "where it meets equipment - gets a tight pink 'arrow' box<br>- missing line: hover the box it starts at "
        "until small blue arrows appear on its sides, drag one onto the other box (lines must be attached to two "
        "boxes; a free-drawn line is skipped)<br>- wrong item or line: select, Delete<br>The picture layer is "
        "locked."))
    note_h = note_font * 23  # ~19 lines of wrapped text
    ET.SubElement(note, "mxGeometry", x=f"{lx:.1f}", y="0", width=f"{note_w:.1f}", height=f"{note_h:.1f}",
                  **{"as": "geometry"})
    for k, cls in enumerate(CLASSES):
        y = note_h + gap + k * 1.4 * unit
        vertex(f"legend_{k}", cls, cls, (lx, y, lx + unit, y + 0.8 * unit), parent="legend")
    out = ET.Element("mxfile", host="pid-dexpi-aas")
    diagram = ET.SubElement(out, "diagram", id="labels", name=(name or Path(review.source_name).stem))
    diagram.append(root)
    return ET.tostring(out, encoding="unicode")


# --------------------------------------------------------------------------- import ---
def _edge_point(box, toward) -> tuple[float, float]:
    """Where the line from the box's centre towards `toward` leaves the box."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    tx = (x2 - cx) / abs(dx) if dx else float("inf")
    ty = (y2 - cy) / abs(dy) if dy else float("inf")
    t = min(tx, ty)
    return cx + dx * t, cy + dy * t


def drawio_to_labels(xml_text: str):
    """(picture, symbols [(id, class, box)], edges [(a, b, line, path)], notes) from a labelling
    .drawio (as written by review_to_drawio, then edited)."""
    from pid_dexpi_aas.drawio.decode import decode_drawio

    root = ET.fromstring(xml_text)
    model = root.find(".//mxGraphModel")
    if model is None:  # saved compressed by draw.io
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".drawio", delete=False) as fh:
            fh.write(xml_text)
        model = ET.fromstring(decode_drawio(fh.name)[0])
        Path(fh.name).unlink()
    layer_of, parent_of = {}, {}
    picture, symbols, notes = None, {}, []
    edges_raw = []
    for el in model.find("root"):  # top-level cells only: an <object> holds its <mxCell> inside
        cell, attrs = (el.find("mxCell"), el.attrib) if el.tag in ("object", "UserObject") else (el, el.attrib)
        if el.tag not in ("object", "UserObject", "mxCell") or cell is None:
            continue
        cid = attrs.get("id")
        parent = cell.get("parent")
        parent_of[cid] = parent
        style = cell.get("style") or ""
        geo = cell.find("mxGeometry")
        if "image=data:image" in style and geo is not None:
            data = style.split("image=data:image/", 1)[1].split(",", 1)[1].split(";", 1)[0]
            img = Image.open(io.BytesIO(base64.b64decode(data)))
            if img.width * img.height > MAX_PIXELS:
                raise ValueError(f"the picture is too large ({img.width} x {img.height} px)")
            picture = img.convert("RGB")
            continue
        if cell.get("vertex") == "1" and geo is not None and attrs.get("pid2graph_class"):
            if parent == "legend":
                continue
            x, y = float(geo.get("x", 0)), float(geo.get("y", 0))
            w, h = float(geo.get("width", 0)), float(geo.get("height", 0))
            symbols[cid] = (attrs["pid2graph_class"], (x, y, x + w, y + h))
        elif cell.get("vertex") == "1" and geo is not None and parent == "labels" and "text;" not in style:
            notes.append(f"box {cid} has no class (draw your boxes by copying one from the legend) - skipped")
        elif cell.get("edge") == "1":
            pts = [(float(p.get("x")), float(p.get("y"))) for p in (geo.findall("./Array/mxPoint") if geo is not None else [])]
            edges_raw.append((cid, cell.get("source"), cell.get("target"), "dashed=1" in style, pts))
    if picture is None:
        raise ValueError("no picture found in the file - export it from the app first")
    ids = {cid: f"n{k}" for k, cid in enumerate(symbols)}
    out_syms = [(ids[cid], cls, box) for cid, (cls, box) in symbols.items()]
    out_edges, seen = [], set()
    for eid, a, b, dashed, pts in edges_raw:
        if a not in symbols or b not in symbols or a == b:
            notes.append(f"line {eid} is not attached to two item boxes - skipped")
            continue
        key = tuple(sorted((ids[a], ids[b])))
        if key in seen:
            continue
        seen.add(key)
        ba, bb = symbols[a][1], symbols[b][1]
        ca = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2)
        cb = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
        path = [_edge_point(ba, pts[0] if pts else cb)] + pts + [_edge_point(bb, pts[-1] if pts else ca)]
        if ids[a] != key[0]:
            path = path[::-1]
        out_edges.append((key[0], key[1], "non-solid" if dashed else "solid", path))
    return picture, out_syms, out_edges, notes


def safe_name(name: str) -> str:
    """A drawing name usable as a folder name: no path parts, nothing but letters, digits and
    a few separators (an uploaded file names where its data is written)."""
    import re
    stem = Path(str(name)).name
    for ext in (".drawio", ".xml"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    stem = re.sub(r"[^\w .()&+-]", "_", stem).strip(" .") or "drawing"
    return stem[:120]


def preview_png(picture: Image.Image, symbols: list, edges: list, max_side: int = 1800) -> bytes:
    """The labels drawn over the picture, to check them before they become training data."""
    from PIL import ImageDraw

    k = min(1.0, max_side / max(picture.size))
    img = picture.resize((round(picture.width * k), round(picture.height * k))) if k < 1 else picture.copy()
    d = ImageDraw.Draw(img, "RGBA")
    for a, b, line, path in edges:
        pts = [(x * k, y * k) for x, y in path]
        if line == "non-solid":
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):  # dashed
                n = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 8))
                for i in range(0, n, 2):
                    d.line([(x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n),
                            (x1 + (x2 - x1) * (i + 1) / n, y1 + (y2 - y1) * (i + 1) / n)], fill=(212, 0, 0, 220), width=3)
        else:
            d.line(pts, fill=(212, 0, 0, 200), width=3)
    for _, cls, (x1, y1, x2, y2) in symbols:
        col = CLASSES.get(cls, "#6b6b6b")
        rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
        d.rectangle([x1 * k, y1 * k, x2 * k, y2 * k], outline=rgb + (255,), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def drawio_to_training(path: str | Path, out_root: str | Path, subset: str, name: str | None = None) -> dict:
    """Write a corrected labelling .drawio as PID2Graph data (sheet + tiles) under out_root."""
    from pid_dexpi_aas.image.training_export import dir_writer, normalise_scale, write_sheet

    path = Path(path)
    picture, symbols, edges, notes = drawio_to_labels(path.read_text(encoding="utf-8"))
    picture, symbols, edges, scale = normalise_scale(picture, symbols, edges)
    name = safe_name(name or path.stem)
    tiles = write_sheet(dir_writer(Path(out_root)), picture, name, subset, symbols, edges)
    return {"name": name, "subset": subset, "scale": round(scale, 3), "symbols": len(symbols), "connections": len(edges),
            "classes": {c: sum(1 for _, k, _ in symbols if k == c) for c in sorted({k for _, k, _ in symbols})},
            "tiles": tiles, "notes": notes}


def main() -> None:
    """python -m pid_dexpi_aas.image.drawio_labels <file.drawio> [--test] [--out data/PID2Graph]"""
    import argparse

    ap = argparse.ArgumentParser(description="A corrected labelling .drawio (exported from the review app) -> "
                                             "PID2Graph training or held-out test data")
    ap.add_argument("drawio", nargs="+", help="the corrected .drawio file(s)")
    ap.add_argument("--test", action="store_true",
                    help="held out: subset 'Reviewed test', never trained on (eval_sheets.py --set reviewed)")
    ap.add_argument("--subset", default=None, help="subset name (default 'Reviewed', or 'Reviewed test' with --test)")
    ap.add_argument("--out", default="data/PID2Graph")
    args = ap.parse_args()
    subset = args.subset or ("Reviewed test" if args.test else "Reviewed")
    for f in args.drawio:
        summary = drawio_to_training(f, args.out, subset)
        for note in summary.pop("notes"):
            print("  note:", note)
        print(summary)
    print(f"\nTo train on '{subset}': add 'Patched/{subset}' to DATA.EXTRA_TRAIN_SUBSETS in the training config."
          if not subset.endswith("test") else
          "\nHeld out - score any model on it: ml/pid2graph/eval/eval_sheets.py --checkpoint <model> --set reviewed")


if __name__ == "__main__":
    main()
