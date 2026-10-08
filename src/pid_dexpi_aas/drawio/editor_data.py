"""The Editor tab's link between the review rows and the drawing in draw.io.

Every item and line of a draw.io drawing carries its DEXPI details as data fields on its
cell (draw.io's own "Edit Data" mechanism, an <object> wrapper): the editor's DEXPI panel
shows and changes them, and they travel in the saved drawing, where review.from_drawio
reads them back (_apply_editor_data). So the side panel and the editor stay one record.

editor_patch(review) says what each cell should hold now; the editor applies it to the open
drawing (only what differs). Fields:

  items  dexpi_class, dexpi_auto ("1": worked out from the shape - worked out again on
         every read; "0": chosen by a person), auto_class (for display), tag, tag_auto,
         tag_cell (the text cell showing the tag), in_dexpi, notes, specs (JSON), section
  lines  in_dexpi, notes, specs; the line type and a reviewer's flow direction go into the
         drawing itself (dashed, arrowheads), changed ends into the line's terminals
"""
from __future__ import annotations

import json

from pid_dexpi_aas.dexpi.fields import spec_fields
from pid_dexpi_aas.drawio.tag_association import associate_tags


def editor_patch(review) -> dict:
    """{cell id: {"attrs": {...}, "style": {...}, "ends": {"source", "target"}}}."""
    diagram = review.drawio
    if diagram is None:
        return _picture_patch(review) if review.image_path else {}
    shapes = {sh.id: sh for sh in diagram.shapes}
    edges = {e.id: e for e in diagram.edges}
    tag_cell = {}
    for a in associate_tags(diagram):
        if a.target_shape_id:
            tag_cell.setdefault(a.target_shape_id, []).append(a.label_shape_id)
    out: dict = {}
    for s in review.symbols:
        if s.id not in shapes:
            continue
        auto = review.auto.get(s.id, {})
        attrs = _item_attrs(s, auto)
        if len(tag_cell.get(s.id, [])) == 1:
            attrs["tag_cell"] = tag_cell[s.id][0]
        out[s.id] = {"attrs": attrs}
    for c in review.connections:
        e = edges.get(c.id)
        if e is None:
            continue  # pipe networks ("net..."): their member lines are the cells
        auto = review.auto.get(c.id, {})
        entry = {"attrs": {"in_dexpi": "1" if c.include else "0",
                           "notes": c.notes if c.notes != auto.get("notes") else "",
                           "specs": _specs(c.attributes)}}
        if (c.line_type == "signal") != e.is_dashed:
            entry["style"] = {"dashed": "1" if c.line_type == "signal" else "0"}
        if {c.from_id, c.to_id} != {e.source, e.target} and c.from_id in shapes and c.to_id in shapes:
            entry["ends"] = ({"source": c.from_id, "target": c.to_id} if e.source in (c.from_id, None)
                             else {"source": c.to_id, "target": c.from_id})
        if c.flow == "reviewer":  # a person set the direction: draw the arrowhead at the To end
            to_target = (entry.get("ends", {}).get("target") or e.target) == c.to_id
            entry.setdefault("style", {}).update({"endArrow": "classic" if to_target else "none",
                                                  "startArrow": "none" if to_target else "classic"})
        out[c.id] = entry
    return out


def _item_attrs(s, auto: dict) -> dict:
    return {
        "dexpi_class": s.dexpi_class,
        "dexpi_auto": "1" if s.dexpi_class == auto.get("dexpi_class") else "0",
        "auto_class": auto.get("dexpi_class", ""),
        "tag": s.tag,
        "tag_auto": "1" if s.tag == auto.get("tag") else "0",
        "in_dexpi": "1" if s.include else "0",
        "notes": s.notes if s.notes != auto.get("notes") else "",
        "specs": _specs(s.attributes),
        "section": s.section,
    }


def _picture_patch(review) -> dict:
    """A picture review in the editor (picture_editor.review_to_editor_xml): cell ids are row
    ids. Boxes take their kind's colour; lines their type, flow arrowhead and ends."""
    from pid_dexpi_aas.drawio.picture_editor import _COLOURS, _kind
    from pid_dexpi_aas.review import FLOW_KNOWN
    out, boxes = {}, set()
    for s in review.symbols:
        if not s.box:
            continue
        boxes.add(s.id)
        colour = _COLOURS[_kind(s)]
        out[s.id] = {"attrs": _item_attrs(s, review.auto.get(s.id, {})),
                     "style": {"strokeColor": colour, "fontColor": colour,
                               "dashed": "1" if _kind(s) == "left out" else "0"}}
    for c in review.connections:
        if c.from_id not in boxes or c.to_id not in boxes or c.from_id == c.to_id:
            continue
        auto = review.auto.get(c.id, {})
        out[c.id] = {"attrs": {"in_dexpi": "1" if c.include else "0",
                               "notes": c.notes if c.notes != auto.get("notes") else "", "specs": _specs(c.attributes)},
                     "style": {"dashed": "1" if c.line_type == "signal" else "0",
                               "endArrow": "classic" if c.flow in FLOW_KNOWN else "none", "startArrow": "none",
                               "strokeColor": "#c92a2a" if not c.include else "#2456c4" if c.line_type == "signal"
                               else "#d9480f", "opacity": "100" if c.include else "50"},
                     "ends": {"source": c.from_id, "target": c.to_id}}
    return out


def _specs(attributes: dict | None) -> str:
    parts = {k: v for k, v in (attributes or {}).items() if k in ("dexpi", "custom") and v}
    return json.dumps(parts, sort_keys=True) if parts else ""


def spec_schema(class_names) -> dict:
    """{class: [{name, label, kind, units, options, custom}]} for the editor's specs fields."""
    return {cls: [{"name": f.name, "label": f.label, "kind": f.kind, "units": f.units, "options": f.options,
                   "custom": f.custom} for f in spec_fields(cls)] for cls in class_names}
