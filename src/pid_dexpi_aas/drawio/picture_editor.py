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

from pid_dexpi_aas.drawio.parser import _cells, _extract_shape_token, _nesting, _strip_html

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


def sizes(review) -> tuple[float, int, int]:
    """(typical item size, label font size, line width) for drawing a picture review."""
    W, H = review.image_size or (2000, 1500)
    s = sorted(max(b[2] - b[0], b[3] - b[1]) for x in review.symbols if (b := x.box))
    unit = s[len(s) // 2] if s else max(W, H) / 30
    return unit, max(8, min(28, round(unit * 0.2))), max(2, min(6, round(unit / 30)))


def review_to_editor_xml(review, picture_url: str) -> str:
    """The picture review as a draw.io drawing; ids are the review's row ids."""
    W, H = review.image_size or (2000, 1500)
    unit, font, stroke = sizes(review)
    box_of = {s.id: s.box for s in review.symbols if s.box}

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
        group_parts = (s.attributes or {}).get("group_parts")
        if group_parts:  # a group drawn in the editor: the group and its shapes, as drawn
            g = ET.SubElement(cells, "mxCell", id=s.id, value="", parent=ITEM_LAYER, vertex="1", connectable="0",
                              style="group")
            ET.SubElement(g, "mxGeometry", x=f"{x1:.1f}", y=f"{y1:.1f}", width=f"{x2 - x1:.1f}",
                          height=f"{y2 - y1:.1f}", **{"as": "geometry"})
            for part in group_parts:
                try:
                    cells.append(ET.fromstring(part))
                except ET.ParseError:
                    pass
            continue
        cell = ET.SubElement(cells, "mxCell", id=s.id, value=html.escape(s.tag or ""), parent=ITEM_LAYER, vertex="1",
                             style=f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={colour};"
                                   f"strokeWidth={stroke};fontColor={colour};fontSize={font};fontStyle=1;"
                                   "verticalLabelPosition=top;verticalAlign=bottom;labelBackgroundColor=#ffffffcc;"
                                   + ("dashed=1;" if _kind(s) == "left out" else ""))
        ET.SubElement(cell, "mxGeometry", x=f"{x1:.1f}", y=f"{y1:.1f}", width=f"{x2 - x1:.1f}",
                      height=f"{y2 - y1:.1f}", **{"as": "geometry"})
    from pid_dexpi_aas.review import FLOW_KNOWN
    drawn_ids = {c.id for c in review.connections if c.from_id in boxes and c.to_id in boxes and c.from_id != c.to_id}
    for c in review.connections:
        if c.from_id not in boxes or c.to_id not in boxes or c.from_id == c.to_id:
            continue
        tee = (c.attributes or {}).get("tee")
        if tee:  # a branch drawn onto a line: drawn again as drawn, once (its first row)
            if c.id != tee.get("branch") or tee.get("line") not in drawn_ids:
                continue
            item = tee.get("item") or c.from_id
            src, tgt = (item, tee["line"]) if tee.get("at") == "target" else (tee["line"], item)
            colour = "#c92a2a" if not c.include else "#2456c4" if c.line_type == "signal" else "#d9480f"
            cell = ET.SubElement(cells, "mxCell", id=c.id, value="", parent=ITEM_LAYER, edge="1", source=src, target=tgt,
                                 style=f"edgeStyle=none;rounded=0;html=1;strokeColor={colour};strokeWidth={stroke};"
                                       "endArrow=none;startArrow=none;" + ("dashed=1;" if c.line_type == "signal" else ""))
            geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
            if tee.get("points"):
                pts = ET.SubElement(geo, "Array", **{"as": "points"})
                for x, y in tee["points"]:
                    ET.SubElement(pts, "mxPoint", x=f"{x:.1f}", y=f"{y:.1f}")
            continue
        flow = c.flow in FLOW_KNOWN
        colour = "#c92a2a" if not c.include else "#2456c4" if c.line_type == "signal" else "#d9480f"
        cell = ET.SubElement(cells, "mxCell", id=c.id, value="", parent=ITEM_LAYER, edge="1",
                             source=c.from_id, target=c.to_id,
                             style=f"edgeStyle=none;rounded=0;html=1;strokeColor={colour};strokeWidth={stroke};"
                                   f"endArrow={'classic' if flow or c.flow == 'both ways' else 'none'};"
                                   f"startArrow={'classic' if c.flow == 'both ways' else 'none'};"
                                   + ("dashed=1;" if c.line_type == "signal" else "")
                                   + ("opacity=50;" if not c.include else ""))
        geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        path = [tuple(p) for p in c.path] if c.path else []
        if len(path) > 2:
            from pid_dexpi_aas.image.training_export import _corners
            corners = _corners(path, tol=max(2.0, unit * 0.03))
            inner = corners[1:-1]
            # an end short of its item (the pipe meets the item's line at a tee next to it): kept,
            # so the line runs there and then into the item, not across from the last corner
            def short_of(sid, p):
                b = box_of.get(sid)
                return b is not None and not (b[0] - 2 <= p[0] <= b[2] + 2 and b[1] - 2 <= p[1] <= b[3] + 2)
            if short_of(c.from_id, corners[0]):
                inner.insert(0, corners[0])
            if short_of(c.to_id, corners[-1]):
                inner.append(corners[-1])
            if inner:
                pts = ET.SubElement(geo, "Array", **{"as": "points"})
                for x, y in inner:
                    ET.SubElement(pts, "mxPoint", x=f"{x:.1f}", y=f"{y:.1f}")
    return ET.tostring(root, encoding="unicode")


def _nearest_on(route, p):
    """(index of the segment, point) of `route` nearest to `p`."""
    best = None
    for i, (a, b) in enumerate(zip(route, route[1:])):
        vx, vy = b[0] - a[0], b[1] - a[1]
        L = vx * vx + vy * vy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L))
        q = (a[0] + t * vx, a[1] + t * vy)
        d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
        if best is None or d < best[0]:
            best = (d, i, q)
    return best[1], best[2]


def _seg_dist(p, a, b) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    L = vx * vx + vy * vy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L))
    return ((p[0] - a[0] - t * vx) ** 2 + (p[1] - a[1] - t * vy) ** 2) ** 0.5


def _on_border(box, toward):
    """The point of `box`'s outline in the direction of `toward` from its centre."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    t = min((x2 - cx) / abs(dx) if dx else float("inf"), (y2 - cy) / abs(dy) if dy else float("inf"))
    return cx + dx * t, cy + dy * t


def apply_editor_drawing(review, xml: str, known_ids: set, routes: dict | None = None) -> dict:
    """Read a Save of review_to_editor_xml's drawing back into `review`. `known_ids`: the
    rows the editor was showing - only those can have been deleted there (rows added
    elsewhere meanwhile stay). `routes`: {line id: [(x, y), ...]} each line as the editor
    drew it, end to end (its right angles are not in the file) - the lines' paths, and where
    a branch meets a line. -> counts of what changed."""
    routes = {k: [tuple(p) for p in v] for k, v in (routes or {}).items() if v and len(v) >= 2}
    from pid_dexpi_aas.dexpi.mapping import classify_tag, map_shape
    from pid_dexpi_aas.review import (DEXPI_CLASS_CHOICES, FLOW_KNOWN, OFF_PAGE, SKIP, ReviewConnection, ReviewSymbol,
                                      _instrument_attributes, _mapping_to_class)

    root = ET.fromstring(xml)
    if root.tag != "mxGraphModel":
        root = root.find(".//mxGraphModel")
    vertices, edges = {}, {}
    cells = list(_cells(root))
    # a draw.io group (Ctrl+G) is ONE item: its box is the group's, its type its biggest
    # symbol's; its parts (drawn relative to it) are not items, lines to a part end on it
    _, owner = _nesting(cells)
    # each group's shapes as drawn (cells of the drawing, kept on the item: the editor draws
    # the group again when it is opened next - review_to_editor_xml)
    drawn_parts: dict[str, list] = {}
    for holder in root.iter("root"):
        for el in holder:
            if el.get("id") in owner:
                drawn_parts.setdefault(owner[el.get("id")], []).append(ET.tostring(el, encoding="unicode"))
    parts: dict[str, list] = {}
    for cell, cid, label, data in cells:
        if cid in owner and cell.get("vertex") == "1":
            g = cell.find("mxGeometry")
            area = float(g.get("width", 0)) * float(g.get("height", 0)) if g is not None else 0.0
            parts.setdefault(owner[cid], []).append((area, _extract_shape_token(cell.get("style", "")), _strip_html(label)))
    for cell, cid, label, data in cells:
        if cid in (None, "0", PICTURE_LAYER, ITEM_LAYER, "picture-image") or cell.get("parent") == PICTURE_LAYER:
            continue
        if cid in owner:
            continue  # a part of a group
        geo = cell.find("mxGeometry")
        if cell.get("vertex") == "1" and geo is not None and cid in parts:
            x, y = float(geo.get("x", 0)), float(geo.get("y", 0))
            w, h = float(geo.get("width", 0)), float(geo.get("height", 0))
            shaped = [p for p in parts[cid] if p[1]] or parts[cid]
            main = max(shaped, key=lambda p: (map_shape(p[1], p[2], "").dexpi_class_name is not None, p[0]))
            vertices[cid] = {"box": [x, y, x + w, y + h], "data": data, "token": main[1], "text": False,
                             "label": _strip_html(label) or main[2] or " ".join(p[2] for p in parts[cid] if p[2])}
            continue
        if cell.get("vertex") == "1" and geo is not None:
            x, y = float(geo.get("x", 0)), float(geo.get("y", 0))
            w, h = float(geo.get("width", 0)), float(geo.get("height", 0))
            vertices[cid] = {"box": [x, y, x + w, y + h], "label": _strip_html(label), "data": data,
                             "token": _extract_shape_token(cell.get("style", "")),
                             "text": (cell.get("style") or "").split(";")[0] == "text"}
        elif cell.get("edge") == "1":
            pts = [] if geo is None else [(float(p.get("x", 0)), float(p.get("y", 0)))
                                          for p in geo.findall("Array[@as='points']/mxPoint")]
            ends = {} if geo is None else {p.get("as"): (float(p.get("x", 0)), float(p.get("y", 0)))
                                           for p in geo.findall("mxPoint") if p.get("as") in ("sourcePoint", "targetPoint")}
            edges[cid] = {"source": owner.get(cell.get("source"), cell.get("source")),
                          "target": owner.get(cell.get("target"), cell.get("target")), "style": cell.get("style", ""),
                          "points": pts, "data": data, "ends": ends, "route": routes.get(cid)}
    style_val = lambda st, k: next((p.split("=", 1)[1] for p in (st or "").split(";") if p.startswith(k + "=")), None)
    # a loose end lying on another line (dropped there, not attached): a branch of that line
    centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    unit = sizes(review)[0]
    for eid, e in edges.items():
        for key, pt in (("source", e["ends"].get("sourcePoint")), ("target", e["ends"].get("targetPoint"))):
            if e[key] is not None or pt is None:
                continue
            best = None
            for fid, f in edges.items():
                fa, fb = vertices.get(f["source"]), vertices.get(f["target"])
                if fid == eid or fa is None or fb is None:
                    continue
                line = f["route"] or [centre(fa["box"]), *f["points"], centre(fb["box"])]
                d = min(_seg_dist(pt, p, q) for p, q in zip(line, line[1:]))
                if d <= max(8.0, 0.15 * unit) and (best is None or d < best[0]):
                    best = (d, fid)
            if best:
                e[key] = best[1]

    counts = {"moved": 0, "added": 0, "removed": 0, "lines added": 0, "lines removed": 0, "lines changed": 0}
    rows = {s.id: s for s in review.symbols}
    # an item's tag label ("Add a tag label"): its tag, not an item (nor a stream's end)
    tag_label = {v["data"]["tag_cell"]: sid for sid, v in vertices.items()
                 if v["data"].get("tag_cell") in vertices and v["data"]["tag_cell"] != sid}
    for sid, v in vertices.items():
        if sid in tag_label:
            continue
        d, s = v["data"], rows.get(sid)
        if s is None:  # drawn in the editor: a box, or a symbol from the P&ID library
            tag = d.get("tag") if d.get("tag_auto") == "0" else v["label"]
            if v["text"]:  # a text at a line's end: a stream entering or leaving (off-page connector)
                cls = OFF_PAGE
            elif v["token"] and not v["token"].startswith("plain:"):
                cls = _mapping_to_class(map_shape(v["token"], tag or "", ""))
            else:
                cls = classify_tag(tag or "")[1] or SKIP
            s = ReviewSymbol(id=sid, source="manual", box=v["box"], detected_as="drawn in the editor", tag=tag or "",
                             tag_source="manual" if tag else "", dexpi_class=cls, identified_by="drawn in the editor",
                             attributes=_instrument_attributes(cls, tag or ""), include=cls != SKIP, checked=True)
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
        if d.get("tag_cell") in tag_label:
            s.tag = vertices[d["tag_cell"]]["label"]  # its tag label, as it reads now
        elif d.get("tag_auto") == "0":
            s.tag = d.get("tag", "")
        elif v["label"] != s.tag and sid in known_ids:
            s.tag = v["label"]  # the tag text edited on the drawing
        if d.get("in_dexpi") in ("0", "1"):
            s.include = d["in_dexpi"] == "1" and s.dexpi_class != SKIP
        if d.get("notes"):
            s.notes = d["notes"]
        if d.get("checked") in ("0", "1"):
            s.checked = d["checked"] == "1"
        specs = json.loads(d.get("specs") or "{}") if d.get("specs") else {}
        for part in ("dexpi", "custom"):
            if part in specs:
                s.attributes = {**(s.attributes or {}), part: specs[part]}
        if sid in drawn_parts:
            s.attributes = {**(s.attributes or {}), "group_parts": drawn_parts[sid]}
        elif "group_parts" in (s.attributes or {}):  # ungrouped since
            s.attributes = {k: v for k, v in s.attributes.items() if k != "group_parts"}
    gone = {sid for sid in known_ids if sid in rows and sid not in vertices}
    if gone:
        review.symbols = [s for s in review.symbols if s.id not in gone]
        counts["removed"] = len(gone)

    # a branch attached to another line (a tee drawn in the editor): the branch's item joins
    # the items at both ends of that line - pairwise, as the pipe tracer reports tees - along
    # the drawn pipes (branch to the junction, then the line each way). The rows remember the
    # branch ("tee": its line, its bends), so the editor keeps drawing it as drawn.
    tee_of = {}  # branch id -> line id, for the branches seen in this drawing
    for eid, e in list(edges.items()):
        a, b = e["source"], e["target"]
        on = next(((end, x) for end, x in (("source", a), ("target", b)) if x in edges), None)
        if on is None:
            continue
        end, line = on
        item = b if end == "source" else a
        del edges[eid]
        L = edges.get(line)
        if item not in rows or L is None or L["source"] not in vertices or L["target"] not in vertices:
            continue
        tee_of[eid] = line
        line_route = L["route"] or [centre(vertices[L["source"]]["box"]), *L["points"], centre(vertices[L["target"]]["box"])]
        if e["route"]:  # as drawn: from the item to where it meets the line
            drawn = e["route"] if end == "target" else e["route"][::-1]
            bend, towards = drawn[:-1], drawn[-1]
        else:
            bend = e["points"] if end == "target" else e["points"][::-1]
            towards = bend[-1] if bend else centre(vertices[item]["box"])
        k, junction = _nearest_on(line_route, towards)
        legs = [(L["source"], line_route[:k + 1][::-1]), (L["target"], line_route[k + 1:])]
        # flow from the arrowheads: the line's direction says which side feeds the junction;
        # the branch's says whether its item is fed from it or feeds it
        l_end, l_start = style_val(L["style"], "endArrow"), style_val(L["style"], "startArrow")
        l_dir = (1 if l_end != "none" and l_start in (None, "none") else
                 -1 if l_start not in (None, "none") and l_end == "none" else 0)
        upstream = L["source"] if l_dir > 0 else L["target"] if l_dir < 0 else None
        b_end, b_start = style_val(e["style"], "endArrow"), style_val(e["style"], "startArrow")
        at_item = (b_start not in (None, "none")) if end == "target" else (b_end != "none")
        at_line = (b_end != "none") if end == "target" else (b_start not in (None, "none"))
        into_item, out_of_item = at_item and not at_line, at_line and not at_item
        n = 0
        for other, along in legs:
            if other not in rows or other == item:
                continue
            path = [*bend, junction, *along] if e["route"] or L["route"] else [*bend, junction, *along[:-1]]
            flow = (item, other, "")
            if upstream is not None and into_item:
                flow = (other, item, "reviewer") if other == upstream else (item, other, "no flow between them")
            elif upstream is not None and out_of_item:
                flow = (item, other, "reviewer") if other != upstream else (item, other, "no flow between them")
            edges[eid if n == 0 else f"{eid}-tee{n}"] = {
                "source": item, "target": other, "data": e["data"], "ends": {}, "style": e["style"] or "",
                "points": [] if (e["route"] or L["route"]) else path, "route": path if (e["route"] or L["route"]) else None,
                "flow_set": flow, "tee": {"line": line, "branch": eid, "item": item, "at": end,
                        "points": e["points"] or ([list(q) for q in e["route"][1:-1]] if e["route"] else [])}}
            n += 1
    conns = {c.id: c for c in review.connections}
    keep = []
    for c in review.connections:
        tee = (c.attributes or {}).get("tee")
        if tee and tee.get("branch") in known_ids and tee.get("branch") not in tee_of:
            counts["lines removed"] += 1  # its branch was deleted, or no longer ends on a line
            continue
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
        if e.get("flow_set"):  # a tee leg: direction from the line and branch arrowheads
            frm, to = e["flow_set"][:2]
        ba, bb = rows[frm].box, rows[to].box
        pts = e["points"] if frm == a else e["points"][::-1]
        if e.get("route"):  # the line as the editor drew it, end to end
            path = list(e["route"]) if frm == a else list(e["route"])[::-1]
        else:
            path = [_on_border(ba, pts[0]), *pts, _on_border(bb, pts[-1])] if pts and ba and bb else None
        c = conns.get(eid)
        if c is None:
            c = ReviewConnection(id=eid, from_id=frm, to_id=to, source="manual",
                                 confidence="drawn in the editor", path=path, checked=True)
            review.connections.append(c)
            counts["lines added"] += 1
            before = None
        else:
            before = (c.from_id, c.to_id)
            if {c.from_id, c.to_id} != {frm, to} or pts or e.get("route"):
                c.path = path  # as drawn now (without a drawn route or bends: the map routes it)
            if {c.from_id, c.to_id} != {frm, to}:
                counts["lines changed"] += 1
            c.from_id, c.to_id = frm, to
        c.line_type = "signal" if style_val(e["style"], "dashed") == "1" else "process"
        if e.get("flow_set"):
            c.flow = e["flow_set"][2]
        elif not (to_a or to_b):
            c.flow = ""  # no arrowhead: direction unknown
        elif to_a and to_b:
            c.flow = "both ways"  # arrowheads at both ends: a two-way pipe
        elif c.flow not in FLOW_KNOWN or before != (frm, to):
            c.flow = "reviewer"  # an arrowhead drawn, or turned round, in the editor
        if e.get("tee"):
            c.attributes = {**(c.attributes or {}), "tee": e["tee"]}
            c.confidence = c.confidence or "branch drawn in the editor"
        elif "tee" in (c.attributes or {}):
            c.attributes = {k: v for k, v in c.attributes.items() if k != "tee"}
        d = e["data"]
        if d.get("in_dexpi") in ("0", "1"):
            c.include = d["in_dexpi"] == "1"
        if d.get("checked") in ("0", "1"):
            c.checked = d["checked"] == "1"
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
