"""The Editor tab for a picture review (image mode): the model's result in draw.io.

review_to_editor_xml draws the review over its picture - the picture on a locked layer,
each item a box around its symbol (coloured by kind, its tag on top), each line along the
pipe as traced on the picture. The DEXPI panel and the side panel share each cell's details
the same way as for a draw.io drawing (editor_data.editor_patch).

apply_editor_drawing reads a Save back into the review's rows, in place: boxes moved or
resized, items drawn (a box, or a symbol from the P&ID library) or deleted, lines added,
re-pointed, re-routed or deleted, dashed = signal line, the arrowhead = flow, and the
details set in the panel. Nothing is re-run on the picture.
"""
from __future__ import annotations

import html
import json
from xml.etree import ElementTree as ET

from pid_dexpi_aas.drawio.parser import _cells, _extract_shape_token, _strip_html

PICTURE_LAYER, ITEM_LAYER = "picture", "items"
_COLOURS = {"valve": "#0b8a4b", "instrument": "#2456c4", "equipment": "#d9480f", "off-page": "#8a5a2b",
            "left out": "#c92a2a"}


def _kind(s) -> str:
    from pid_dexpi_aas.dexpi.mapping import Category
    from pid_dexpi_aas.review import OFF_PAGE, SKIP, category_for_class
    if not s.include or s.dexpi_class == SKIP:
        return "left out"
    if s.dexpi_class == OFF_PAGE:
        return "off-page"
    cat = category_for_class(s.dexpi_class)
    return {Category.PIPING: "valve", Category.INSTRUMENTATION: "instrument"}.get(cat, "equipment")


def review_to_editor_xml(review, picture_url: str) -> str:
    """The picture review as a draw.io drawing; ids are the review's row ids."""
    W, H = review.image_size or (2000, 1500)
    sizes = sorted(max(b[2] - b[0], b[3] - b[1]) for s in review.symbols if (b := s.box))
    unit = sizes[len(sizes) // 2] if sizes else max(W, H) / 30
    font = max(8, min(28, round(unit * 0.2)))
    stroke = max(2, min(6, round(unit / 30)))

    root = ET.Element("mxGraphModel", grid="0", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1",
                      fold="1", page="0", pageScale="1", pageWidth=str(W), pageHeight=str(H), math="0", shadow="0",
                      background="#ffffff")
    cells = ET.SubElement(root, "root")
    ET.SubElement(cells, "mxCell", id="0")
    ET.SubElement(cells, "mxCell", id=PICTURE_LAYER, value="Picture (locked)", parent="0", style="locked=1;")
    ET.SubElement(cells, "mxCell", id=ITEM_LAYER, value="Items and lines", parent="0")
    pic = ET.SubElement(cells, "mxCell", id="picture-image", value="", parent=PICTURE_LAYER, vertex="1",
                        style=f"shape=image;imageAspect=0;aspect=fixed;image={picture_url};editable=0;movable=0;"
                              "resizable=0;deletable=0;selectable=0;")
    ET.SubElement(pic, "mxGeometry", x="0", y="0", width=str(W), height=str(H), **{"as": "geometry"})

    boxes = {}
    for s in review.symbols:
        if not s.box:
            continue
        x1, y1, x2, y2 = (float(v) for v in s.box)
        boxes[s.id] = (x1, y1, x2, y2)
        colour = _COLOURS[_kind(s)]
        cell = ET.SubElement(cells, "mxCell", id=s.id, value=html.escape(s.tag or ""), parent=ITEM_LAYER, vertex="1",
                             style=f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={colour};"
                                   f"strokeWidth={stroke};fontColor={colour};fontSize={font};fontStyle=1;"
                                   "verticalLabelPosition=top;verticalAlign=bottom;labelBackgroundColor=#ffffffcc;"
                                   + ("dashed=1;" if _kind(s) == "left out" else ""))
        ET.SubElement(cell, "mxGeometry", x=f"{x1:.1f}", y=f"{y1:.1f}", width=f"{x2 - x1:.1f}",
                      height=f"{y2 - y1:.1f}", **{"as": "geometry"})
    from pid_dexpi_aas.review import FLOW_KNOWN
    for c in review.connections:
        if c.from_id not in boxes or c.to_id not in boxes or c.from_id == c.to_id:
            continue
        flow = c.flow in FLOW_KNOWN
        colour = "#c92a2a" if not c.include else "#2456c4" if c.line_type == "signal" else "#d9480f"
        cell = ET.SubElement(cells, "mxCell", id=c.id, value="", parent=ITEM_LAYER, edge="1",
                             source=c.from_id, target=c.to_id,
                             style=f"edgeStyle=none;rounded=0;html=1;strokeColor={colour};strokeWidth={stroke};"
                                   f"endArrow={'classic' if flow else 'none'};startArrow=none;"
                                   + ("dashed=1;" if c.line_type == "signal" else "")
                                   + ("opacity=50;" if not c.include else ""))
        geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        path = [tuple(p) for p in c.path] if c.path else []
        if len(path) > 2:
            from pid_dexpi_aas.image.training_export import _corners
            inner = _corners(path, tol=max(2.0, unit * 0.03))[1:-1]
            if inner:
                pts = ET.SubElement(geo, "Array", **{"as": "points"})
                for x, y in inner:
                    ET.SubElement(pts, "mxPoint", x=f"{x:.1f}", y=f"{y:.1f}")
    return ET.tostring(root, encoding="unicode")


def _on_border(box, toward):
    """The point of `box`'s outline in the direction of `toward` from its centre."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    t = min((x2 - cx) / abs(dx) if dx else float("inf"), (y2 - cy) / abs(dy) if dy else float("inf"))
    return cx + dx * t, cy + dy * t


def apply_editor_drawing(review, xml: str, known_ids: set) -> dict:
    """Read a Save of review_to_editor_xml's drawing back into `review`. `known_ids`: the
    rows the editor was showing - only those can have been deleted there (rows added
    elsewhere meanwhile stay). -> counts of what changed."""
    from pid_dexpi_aas.dexpi.mapping import classify_tag, map_shape
    from pid_dexpi_aas.review import (DEXPI_CLASS_CHOICES, FLOW_KNOWN, SKIP, ReviewConnection, ReviewSymbol,
                                      _instrument_attributes, _mapping_to_class)

    root = ET.fromstring(xml)
    if root.tag != "mxGraphModel":
        root = root.find(".//mxGraphModel")
    vertices, edges = {}, {}
    for cell, cid, label, data in _cells(root):
        if cid in (None, "0", PICTURE_LAYER, ITEM_LAYER, "picture-image") or cell.get("parent") == PICTURE_LAYER:
            continue
        geo = cell.find("mxGeometry")
        if cell.get("vertex") == "1" and geo is not None:
            x, y = float(geo.get("x", 0)), float(geo.get("y", 0))
            w, h = float(geo.get("width", 0)), float(geo.get("height", 0))
            vertices[cid] = {"box": [x, y, x + w, y + h], "label": _strip_html(label), "data": data,
                             "token": _extract_shape_token(cell.get("style", ""))}
        elif cell.get("edge") == "1":
            pts = [] if geo is None else [(float(p.get("x", 0)), float(p.get("y", 0)))
                                          for p in geo.findall("Array[@as='points']/mxPoint")]
            edges[cid] = {"source": cell.get("source"), "target": cell.get("target"), "style": cell.get("style", ""),
                          "points": pts, "data": data}

    style_val = lambda st, k: next((p.split("=", 1)[1] for p in st.split(";") if p.startswith(k + "=")), None)
    counts = {"moved": 0, "added": 0, "removed": 0, "lines added": 0, "lines removed": 0, "lines changed": 0}
    rows = {s.id: s for s in review.symbols}
    for sid, v in vertices.items():
        d, s = v["data"], rows.get(sid)
        if s is None:  # drawn in the editor: a box, or a symbol from the P&ID library
            tag = d.get("tag") if d.get("tag_auto") == "0" else v["label"]
            if v["token"] and not v["token"].startswith("plain:"):
                cls = _mapping_to_class(map_shape(v["token"], tag or "", ""))
            else:
                cls = classify_tag(tag or "")[1] or SKIP
            s = ReviewSymbol(id=sid, source="manual", box=v["box"], detected_as="drawn in the editor", tag=tag or "",
                             tag_source="manual" if tag else "", dexpi_class=cls, identified_by="drawn in the editor",
                             attributes=_instrument_attributes(cls, tag or ""), include=cls != SKIP)
            review.symbols.append(s)
            rows[sid] = s
            counts["added"] += 1
        elif not s.box or max(abs(a - b) for a, b in zip(s.box, v["box"])) > 0.5:  # the drawing keeps 0.1 px
            s.box = v["box"]
            counts["moved"] += 1
        auto = review.auto.get(sid, {})
        if d.get("dexpi_auto") == "0" and d.get("dexpi_class") in DEXPI_CLASS_CHOICES:
            s.dexpi_class = d["dexpi_class"]
        elif d.get("dexpi_auto") == "1" and auto.get("dexpi_class"):
            s.dexpi_class = auto["dexpi_class"]
        if d.get("tag_auto") == "0":
            s.tag = d.get("tag", "")
        elif v["label"] != s.tag and sid in known_ids:
            s.tag = v["label"]  # the tag text edited on the drawing
        if d.get("in_dexpi") in ("0", "1"):
            s.include = d["in_dexpi"] == "1" and s.dexpi_class != SKIP
        if d.get("notes"):
            s.notes = d["notes"]
        specs = json.loads(d.get("specs") or "{}") if d.get("specs") else {}
        for part in ("dexpi", "custom"):
            if part in specs:
                s.attributes = {**(s.attributes or {}), part: specs[part]}
    gone = {sid for sid in known_ids if sid in rows and sid not in vertices}
    if gone:
        review.symbols = [s for s in review.symbols if s.id not in gone]
        counts["removed"] = len(gone)

    conns = {c.id: c for c in review.connections}
    keep = []
    for c in review.connections:
        if c.from_id in gone or c.to_id in gone or (c.id in known_ids and c.id not in edges):
            counts["lines removed"] += 1
            continue
        keep.append(c)
    review.connections = keep
    for eid, e in edges.items():
        a, b = e["source"], e["target"]
        if a not in rows or b not in rows or a == b:
            continue  # a loose end: not a connection
        start, end = style_val(e["style"], "startArrow"), style_val(e["style"], "endArrow")
        to_b, to_a = end not in ("none",), start not in (None, "none")
        frm, to = (b, a) if to_a and not to_b else (a, b)
        ba, bb = rows[frm].box, rows[to].box
        pts = e["points"] if frm == a else e["points"][::-1]
        path = [_on_border(ba, pts[0]), *pts, _on_border(bb, pts[-1])] if pts and ba and bb else None
        c = conns.get(eid)
        if c is None:
            c = ReviewConnection(id=eid, from_id=frm, to_id=to, source="manual",
                                 confidence="drawn in the editor", path=path)
            review.connections.append(c)
            counts["lines added"] += 1
            before = None
        else:
            before = (c.from_id, c.to_id)
            if {c.from_id, c.to_id} != {frm, to} or pts:
                c.path = path  # re-pointed or re-routed (no bends drawn: the map routes it)
            if {c.from_id, c.to_id} != {frm, to}:
                counts["lines changed"] += 1
            c.from_id, c.to_id = frm, to
        c.line_type = "signal" if style_val(e["style"], "dashed") == "1" else "process"
        if not (to_a or to_b):
            c.flow = ""  # no arrowhead: direction unknown
        elif c.flow not in FLOW_KNOWN or before != (frm, to):
            c.flow = "reviewer"  # an arrowhead drawn, or turned round, in the editor
        d = e["data"]
        if d.get("in_dexpi") in ("0", "1"):
            c.include = d["in_dexpi"] == "1"
        if d.get("notes"):
            c.notes = d["notes"]
        specs = json.loads(d.get("specs") or "{}") if d.get("specs") else {}
        for part in ("dexpi", "custom"):
            if part in specs:
                c.attributes = {**(c.attributes or {}), part: specs[part]}
    return counts



def drawing_ids(xml: str) -> set:
    """The item and line ids in an editor drawing (not its layers or picture)."""
    root = ET.fromstring(xml)
    return {cid for cell, cid, _, _ in _cells(root)
            if cid not in (None, "0", PICTURE_LAYER, ITEM_LAYER, "picture-image")
            and cell.get("parent") != PICTURE_LAYER and (cell.get("vertex") == "1" or cell.get("edge") == "1")}
