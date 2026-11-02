"""Reviewable intermediate between "what was detected" and the DEXPI model.

Every input path - a draw.io file, an image run through the ML model + OCR, or both -
becomes a ReviewDiagram: one row per symbol and per connection, each saying what it is,
which DEXPI class it maps to, and *how that was decided* (draw.io shape token, tag
prefix, model class, or a reviewer's edit). The Streamlit app shows these rows as
editable tables; to_dexpi() turns the (edited) rows into a DEXPI model through the
existing build_model pipeline, with the reviewer's choices passed as overrides.

When both a drawing and an image are given, draw.io is the source of truth and the
image only checks it (cross_check): draw.io coordinates are aligned to image pixels via
tags that appear in both, then every draw.io symbol is compared against the model's
detections at the same spot.
"""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field

import numpy as np
from pydexpi.dexpi_classes import equipment as eq
from pydexpi.dexpi_classes import piping

from pid_dexpi_aas.dexpi.build_model import BuildResult, _tag_name, build_model
from pid_dexpi_aas.dexpi.mapping import Category, MappingResult, classify_detection, classify_instrument_tag, map_shape
from pid_dexpi_aas.image.ocr_tags import OcrTag, assemble_tags, attach_tags
from pid_dexpi_aas.drawio.parser import Edge, ParsedDiagram, Shape
from pid_dexpi_aas.drawio.tag_association import build_tag_lookup

SKIP = "(skip - not in DEXPI)"
OFF_PAGE = "(off-page connector)"
EQUIPMENT_CLASSES = sorted(
    n for n in dir(eq)
    if not n.startswith("_") and isinstance(getattr(eq, n), type)
    and issubclass(getattr(eq, n), eq.Equipment) and n != "Equipment"
)
VALVE_CLASSES = sorted(n for n in dir(piping) if not n.startswith("_") and ("Valve" in n or n.endswith("Fitting")))
# every in-line piping component an item can be (valves, strainers, reducers, blinds, flame
# arrestors, sight glasses, ...): not the abstract base, the off-page connectors (streams have
# their own choice) or a tee (worked out from the lines)
PIPING_CLASSES = sorted(
    n for n in dir(piping)
    if not n.startswith("_") and isinstance(getattr(piping, n), type)
    and issubclass(getattr(piping, n), piping.PipingComponent)
    and n not in ("PipingComponent", "PipeTee") and "OffPageConnector" not in n
)
INSTRUMENT_CLASSES = ["ProcessInstrumentationFunction", "SignalOffPageConnector"]
# One dropdown for the reviewer; the DEXPI category is implied by the class picked.
DEXPI_CLASS_CHOICES = [SKIP, OFF_PAGE] + INSTRUMENT_CLASSES + PIPING_CLASSES + EQUIPMENT_CLASSES


def category_for_class(dexpi_class: str | None) -> Category:
    if dexpi_class in EQUIPMENT_CLASSES:
        return Category.EQUIPMENT
    if dexpi_class in PIPING_CLASSES:
        return Category.PIPING
    if dexpi_class in INSTRUMENT_CLASSES:
        return Category.INSTRUMENTATION
    if dexpi_class == OFF_PAGE:
        return Category.NOT_PROCESS_EQUIPMENT
    return Category.UNMAPPED


@dataclass
class ReviewSymbol:
    id: str
    source: str  # "draw.io" | "image model" | "manual"
    box: list[float] | None  # [x1, y1, x2, y2] in image pixels if an image is present, else draw.io units
    detected_as: str  # draw.io shape token or model class
    tag: str = ""
    tag_source: str = ""  # "draw.io label" | "OCR inside" | "OCR nearby" | "manual" | ""
    dexpi_class: str = SKIP
    identified_by: str = ""
    confidence: float | None = None
    attributes: dict = field(default_factory=dict)
    notes: str = ""
    image_check: str = ""  # both-inputs mode only
    section: str = ""  # the plant section (frame) it was drawn in - sections.py; editable
    include: bool = True
    checked: bool = False  # confirmed by a person (the editor's guided check), or drawn by one


@dataclass
class ReviewConnection:
    id: str
    from_id: str
    to_id: str
    line_type: str = "process"  # "process" | "signal"
    source: str = ""  # "draw.io" | "image model" | "manual"
    confidence: str = ""  # draw.io: explicit/inferred resolution; model: probability
    notes: str = ""
    include: bool = True
    # specifications of the pipe / signal line (dexpi/fields.py): {"dexpi": {...}, "custom": {...}},
    # written onto the PipingNetworkSegment(s) or signal line it becomes
    attributes: dict = field(default_factory=dict)
    # Process flow runs From -> To, and how that is known: "drawn arrow" | "implied" (from
    # arrows on neighbouring lines, flow_direction.py) | "reviewer"; "" = unknown;
    # "no flow between them" = two legs on the same side of a tee (both fed, or both feeding)
    flow: str = ""
    # image mode: the pipe as traced on the picture (image pixels, From -> To), for the map
    # and the re-drawn DEXPI picture
    path: list | None = None
    checked: bool = False  # confirmed by a person (the editor's guided check), or drawn by one


def is_flow_arrow(s: ReviewSymbol) -> bool:
    """A flow arrow the picture model found: not an item - only evidence for which way its pipe
    flows (and the pipe is joined through it, _bridge_arrows). Not shown or counted as an
    item; given a real type by a person (it was read wrongly), it is an item like any other."""
    return s.detected_as == "arrow" and s.dexpi_class == SKIP


def look_reasons(s: ReviewSymbol) -> list[str]:
    """Why an item needs a person's look ([] = it does not): the map's "to check" items, and
    the editor's flagged ones."""
    if s.source == "manual" or is_flow_arrow(s):
        return []
    why = []
    if s.dexpi_class == SKIP and s.source != "draw.io":
        why.append("no DEXPI type")
    if s.confidence is not None and s.confidence < 0.7:
        why.append(f"the model is only {s.confidence:.0%} sure")
    if "overrides" in s.notes:
        why.append("its tag suggests another type")
    if s.image_check and not s.image_check.startswith("model agrees"):
        why.append(f"picture: {s.image_check}")
    if s.detected_as == MISSED:
        why.append("possible missed item - the model drew no box here")
    elif s.source == "image model" and not s.tag and s.dexpi_class != SKIP:
        why.append("no tag read")
    return why


def item_why(s: ReviewSymbol) -> str:
    """How an item got its type and tag, in a line."""
    parts = [s.identified_by or s.detected_as]
    if s.confidence is not None and s.source == "image model":
        parts.append(f"model {s.confidence:.0%} sure")
    if s.tag:
        parts.append(f"tag from {s.tag_source or 'the drawing'}")
    return "; ".join(p for p in parts if p)


def line_why(c: ReviewConnection) -> str:
    """How a line was found, in a line."""
    if c.source == "image model":
        return f"predicted by the model ({c.confidence}), not traced on the picture"
    if c.source == "picture tracing":
        return "traced along a drawn line on the picture"
    if c.source == "manual":
        return "added by a person"
    return {"explicit": "draw.io: line attached to both items", "junction": "draw.io: joined at a tee",
            "adjacent": "draw.io: items drawn touching",
            "junction+guess": "draw.io: tee, one end guessed"}.get(c.confidence, "draw.io: one end guessed (loose line end)")


SURE_DRAWIO = ("explicit", "adjacent", "junction")
MODEL_ONLY_MIN = 0.6  # a model prediction the tracer did not confirm is kept from this sure (from_image)


def low_confidence(c: ReviewConnection) -> bool:
    """A connection to check one by one: the picture model's prediction (not traced along a
    drawn line - where its extra connections come from), or a draw.io line with an end guessed
    (a loose end put on the nearest item). Traced lines, stated ends and drawn lines are not."""
    if c.source == "image model":
        return True
    return c.source == "draw.io" and c.confidence not in SURE_DRAWIO


FLOW_KNOWN = ("drawn arrow", "implied", "reviewer")
FLOW_BOTH = "both ways"  # a two-way pipe (arrowheads at both ends; DEXPI "dual flow")


@dataclass
class ReviewDiagram:
    source_name: str
    symbols: list[ReviewSymbol] = field(default_factory=list)
    connections: list[ReviewConnection] = field(default_factory=list)
    image_path: str | None = None
    image_size: tuple[int, int] | None = None
    drawio: ParsedDiagram | None = None  # kept so draw.io-only details (labels) survive building
    unattached_tags: list[dict] = field(default_factory=list)  # OCR tags no symbol claimed
    alignment: dict | None = None  # both-inputs mode: draw.io -> image pixel transform + fit quality
    moved: list = field(default_factory=list)  # both-inputs mode: tags drawn at different places in the two
    # both-inputs mode: what the picture alone shows - {"symbols": [ReviewSymbol] (image px),
    # "connections": [ReviewConnection], "match": {draw.io item id: picture item id}} - for
    # the app's Picture and Compare views (the rows themselves stay draw.io's)
    picture: dict | None = None
    # draw.io mode: what each row was before anyone changed it - {row id: {"dexpi_class",
    # "tag", "notes"}} - so the Editor tab can tell automatic values from ones set by a person
    auto: dict = field(default_factory=dict)
    # items folded into another by a merge: {merged-away id: kept id} - the Editor tab moves
    # their lines (attached or merely ending next to them) and their tag text onto the kept one
    merged: dict = field(default_factory=dict)

    def symbol_rows(self) -> list[dict]:
        return [asdict(s) for s in self.symbols]

    def connection_rows(self) -> list[dict]:
        return [asdict(c) for c in self.connections]


def _short_token(token: str | None) -> str:
    if (token or "").startswith("plain:"):
        return "plain shape"
    return (token or "").split(".")[-1].replace("_", " ") or "text label"


def _mapping_to_class(mapping: MappingResult) -> str:
    if mapping.dexpi_class_name:
        return mapping.dexpi_class_name
    return SKIP


def _instrument_attributes(dexpi_class: str, tag: str) -> dict:
    if dexpi_class != "ProcessInstrumentationFunction" or not tag:
        return {}
    _, meta = classify_instrument_tag(tag)
    return {k: v for k, v in meta.items() if v}


# --------------------------------------------------------------------------- draw.io ---
def from_drawio(diagram: ParsedDiagram) -> ReviewDiagram:
    """Rows for every shaped draw.io cell, plus text labels that a pipe ends on (they
    become off-page connectors, e.g. "Fume hood [atm]"). Plain labels that are just a
    shape's tag are folded into that shape's `tag` instead of getting their own row."""
    from pid_dexpi_aas.drawio.sections import section_of

    tag_lookup = build_tag_lookup(diagram)
    endpoint_ids = {e.source for e in diagram.edges} | {e.target for e in diagram.edges}
    review = ReviewDiagram(source_name=diagram.source_file, drawio=diagram)
    sections = section_of(diagram)

    for shape in diagram.shapes:
        if shape.is_labeled_only and not (shape.id in endpoint_ids and shape.text):
            continue
        tag = _tag_name(shape, tag_lookup) or shape.text
        mapping = map_shape(shape.shape_token, shape.text or tag or "", shape.style)  # same rule as build_model
        if shape.is_labeled_only:
            dexpi_class, identified_by = OFF_PAGE, ("a pipe's loose end: a stream in or out of the drawing"
                                                    if shape.data.get("loose_end") else "text label a pipe ends on")
        else:
            dexpi_class = _mapping_to_class(mapping)
            identified_by = f"draw.io shape '{_short_token(shape.shape_token)}'"
            if mapping.category in (Category.PIPING, Category.INSTRUMENTATION) and tag:
                identified_by += " + tag prefix"
        review.symbols.append(ReviewSymbol(
            id=shape.id, source="draw.io",
            box=[shape.x, shape.y, shape.x + shape.width, shape.y + shape.height],
            detected_as=shape.shape_token or "text label", tag=tag or "",
            tag_source="draw.io label" if tag else "", dexpi_class=dexpi_class,
            identified_by=identified_by, confidence=1.0,
            attributes={**_instrument_attributes(dexpi_class, tag or ""),
                        **({"specs": "; ".join(sp.text for sp in diagram.specs[shape.id])}
                           if diagram.specs.get(shape.id) else {})},
            notes=mapping.reason, include=dexpi_class != SKIP, section=sections.get(shape.id, ""),
        ))

    known = {s.id for s in review.symbols}
    for edge in diagram.edges:
        if edge.is_decorative or edge.merged_into or not edge.source or not edge.target:
            continue
        if edge.source not in known or edge.target not in known:
            continue
        resolution = edge.source_resolution if edge.source_resolution == edge.target_resolution else "mixed"
        a, b = (edge.target, edge.source) if edge.flow == -1 else (edge.source, edge.target)  # From = upstream
        review.connections.append(ReviewConnection(
            id=edge.id, from_id=a, to_id=b,
            flow={"arrow": "drawn arrow", "implied": "implied", "parallel": "no flow between them",
                  "both": FLOW_BOTH}.get(edge.flow_basis, "") if edge.flow or edge.flow_basis in ("parallel", "both") else "",
            line_type="signal" if edge.is_dashed else "process", source="draw.io",
            confidence=resolution,
            notes="" if resolution in ("explicit", "adjacent", "junction") else
            ("pipe network joined at a junction; one end guessed - check" if resolution == "junction+guess"
             else "endpoint inferred from a loose line end - check"),
        ))
    review.auto = {s.id: {"dexpi_class": s.dexpi_class, "tag": s.tag, "notes": s.notes} for s in review.symbols}
    review.auto.update({c.id: {"notes": c.notes} for c in review.connections})
    _apply_editor_data(review, diagram)
    _exclude_dangling(review)
    return review


def _apply_editor_data(review: ReviewDiagram, diagram: ParsedDiagram) -> None:
    """Details set by a person and kept in the drawing (the Editor tab, drawio/editor_data.py):
    a chosen DEXPI type or tag (an automatic one is worked out again - the shape or its label
    may have changed), left out of DEXPI, notes, specifications, plant section."""
    import json
    data = {sh.id: sh.data for sh in diagram.shapes if sh.data}
    data.update({e.id: e.data for e in diagram.edges if e.data})
    texts = {sh.id: sh.text for sh in diagram.shapes}
    for row in [*review.symbols, *review.connections]:
        d = data.get(row.id)
        if not d:
            continue
        if isinstance(row, ReviewSymbol):
            if d.get("dexpi_auto") == "0" and d.get("dexpi_class") in DEXPI_CLASS_CHOICES:
                row.dexpi_class, row.identified_by = d["dexpi_class"], "set in the editor"
                row.include = row.dexpi_class != SKIP
            if d.get("tag_auto") == "0" and "tag" in d:
                row.tag, row.tag_source = d["tag"], "set in the editor"
            if texts.get(d.get("tag_cell")):  # its tag label, as it reads now (edited on the drawing)
                row.tag, row.tag_source = texts[d["tag_cell"]], "tag label"
            if d.get("section"):
                row.section = d["section"]
        if d.get("in_dexpi") == "0":
            row.include = False
        elif d.get("in_dexpi") == "1" and isinstance(row, ReviewSymbol) and row.dexpi_class != SKIP:
            row.include = True
        if d.get("notes"):
            row.notes = d["notes"]
        if d.get("checked") in ("0", "1"):
            row.checked = d["checked"] == "1"
        try:
            specs = json.loads(d.get("specs") or "{}")
        except ValueError:
            specs = {}
        for part in ("dexpi", "custom"):
            if specs.get(part):
                row.attributes = {**(row.attributes or {}), part: specs[part]}


def _exclude_dangling(review: ReviewDiagram) -> None:
    """Untick connections whose endpoint is not converted to DEXPI (build_model would skip
    them anyway); the reviewer can re-include after fixing the endpoint's class."""
    included = {s.id for s in review.symbols if s.include and s.dexpi_class != SKIP}
    for c in review.connections:
        if c.include and (c.from_id not in included or c.to_id not in included):
            c.include = False
            c.notes = "; ".join(x for x in (c.notes, "endpoint not converted to DEXPI") if x)


# ----------------------------------------------------------------------------- image ---
def from_image(infer: dict, source_name: str, known_tags: list[str] | None = None) -> ReviewDiagram:
    """Rows from ml/pid2graph/pipeline/infer_sheet.py output: model symbols + OCR tags + name
    classifier, and the model's predicted connections. known_tags: tags known to be on the
    drawing (a spec sheet's, a draw.io file's) - OCR near-misses snap to them."""
    tags = assemble_tags(infer.get("ocr_tokens", []), known=known_tags)
    boxes = {s["id"]: tuple(s["box"]) for s in infer["symbols"]}
    attached = attach_tags(boxes, tags, classes={s["id"]: s["class"] for s in infer["symbols"]})
    review = ReviewDiagram(source_name=source_name, image_path=infer.get("image"),
                           image_size=tuple(infer["image_size"]))
    claimed = set()
    for s in infer["symbols"]:
        tag_obj, how = attached.get(s["id"], (None, ""))
        tag = tag_obj.text if tag_obj else ""
        if tag_obj:
            claimed.add(id(tag_obj))
        mapping, identified_by = classify_detection(s["class"], tag or None)
        dexpi_class = OFF_PAGE if s["class"] == "inlet/outlet" and mapping.dexpi_class_name is None \
            else _mapping_to_class(mapping)
        votes = ", ".join(f"{k}:{v}" for k, v in s.get("class_votes", {}).items())
        review.symbols.append(ReviewSymbol(
            id=s["id"], source="image model", box=list(s["box"]), detected_as=s["class"],
            tag=tag, tag_source=f"OCR {how}" if tag else "", dexpi_class=dexpi_class,
            identified_by=identified_by, confidence=round(float(s["score"]), 3),
            attributes=_instrument_attributes(dexpi_class, tag),
            notes="; ".join(x for x in (mapping.reason, f"tile votes {votes}" if votes else "",
                                        *s.get("refined", [])) if x),
            include=dexpi_class != SKIP,
        ))
    review.unattached_tags = [
        {"tag": t.text, "box": list(t.box), "conf": round(t.conf, 3)} for t in tags if id(t) not in claimed
    ]
    # Pipes traced on the picture (ml/pid2graph/pipeline/trace_pipes.py) first - they follow the
    # drawn line across the whole sheet and carry their path and arrowhead direction. Then
    # the model's own predictions for pairs the tracer did not reach (scored on the team
    # sheets: tracer alone F1 0.68, model alone 0.70, both 0.84).
    seen = set()
    traced = infer.get("traced", [])
    shown = _network_trees(traced)
    for n, t in enumerate(traced):
        a, b, path = t["from"], t["to"], t.get("path")
        if t.get("flow", 0) < 0:
            a, b, path = b, a, (path or [])[::-1]
        seen.add(frozenset((a, b)))  # joined, shown or not: the model's own link for it is not added
        if n not in shown:
            continue
        review.connections.append(ReviewConnection(
            id=f"tc{n}", from_id=a, to_id=b, source="picture tracing", line_type=t.get("line_type", "process"),
            confidence="dashed line followed" if t.get("line_type") == "signal" else "drawn line followed",
            flow="drawn arrow" if t.get("flow") else "", path=path,
            # the traced pipe network it is part of: DEXPI builds the network once, with tees (to_dexpi)
            attributes={"net": f"t{t['net']}"} if t.get("net") is not None else {},
        ))
    _add_missed(review, infer.get("missed", []), seen)
    # the model's own predictions for pairs the tracer did not join - only the ones it is sure
    # of and that are no "skip" (the tracer already joins the two through another item: the
    # model jumping over an in-line valve). Scored on sheets the model never saw (v3 fold
    # models, OPEN100 + LightUp; 2026-10-20 pipeline: four zooms, items >= 0.5, trace_lines):
    # from 0.6 sure F1 0.76 / 0.93 (from 0.9: 0.70 / 0.91; tracer alone 0.66 / 0.90). The model's
    # probabilities are means over the zooms now, lower than one run's - hence 0.6, not 0.9.
    import networkx as nx
    traced_graph = nx.Graph([tuple(p) for p in seen])
    def skip(a, b) -> bool:
        if a not in traced_graph or b not in traced_graph:
            return False
        try:
            return nx.shortest_path_length(traced_graph, a, b) <= 2
        except nx.NetworkXNoPath:
            return False
    for n, c in enumerate(infer.get("connections", [])):
        if frozenset((c["from"], c["to"])) in seen:
            continue
        if c.get("prob", 1.0) < MODEL_ONLY_MIN or skip(c["from"], c["to"]):
            continue
        review.connections.append(ReviewConnection(
            id=f"mc{n}", from_id=c["from"], to_id=c["to"], source="image model",
            line_type=c.get("line_type", "process"),
            confidence=f"{c['prob']:.2f}", notes="predicted by the model, not traced on the picture - check",
        ))
    _bridge_arrows(review)
    _imply_flow(review)
    _exclude_dangling(review)
    review.auto = {s.id: {"dexpi_class": s.dexpi_class, "tag": s.tag, "notes": s.notes} for s in review.symbols}
    review.auto.update({c.id: {"notes": c.notes} for c in review.connections})
    return review


def _network_trees(traced: list) -> set:
    """The traced pairs to show, as indices. A pipe network with tees joins every item on it
    to every other (the tracer reports, and the labels count, each pair: a header with ten
    branches is 45 pairs). Drawn, that is a web of lines, and it is not how the pipes run: of
    each network (`net`, trace_lines.py) only the shortest tree along the pipes is kept - each
    item joined to its neighbour along the pipe, n - 1 lines. Pairs without a network (short
    links, trace_pipes.py) are all kept."""
    import networkx as nx
    nets: dict = {}
    keep = set()
    for n, t in enumerate(traced):
        if t.get("net") is None:
            keep.add(n)
        else:
            nets.setdefault(t["net"], []).append(n)
    for members in nets.values():
        g = nx.Graph()
        for n in members:
            t = traced[n]
            w = t.get("length") or 0.0
            if not g.has_edge(t["from"], t["to"]) or w < g[t["from"]][t["to"]]["weight"]:
                g.add_edge(t["from"], t["to"], weight=w, n=n)
        keep |= {d["n"] for _, _, d in nx.minimum_spanning_edges(g, data=True)}
    return keep


MISSED = "possible item"  # detected_as of an item the model gave no box (_add_missed)


def _add_missed(review: ReviewDiagram, missed: list, seen: set) -> None:
    """Items the model drew no box for, but a traced pipe runs through (infer_sheet.py
    possibly_missed: a symbol's outline on the pipe, or a symbol the model half saw): offered
    as suggestions - flagged for the guided check, left out of DEXPI, the pipe left as traced.
    Splitting the pipe through each (a -> item -> b) made both labelled sets worse: most are
    no item (LightUp: 1.25 a drawing, nearly all false), and each split a real pipe into two
    wrong lines and broke its network (shown lines, 2026-10-27: LightUp 6.8 -> 8.8 wrong, 10.5
    -> 12.8 missing; OPEN100 68 -> 77 missing). A person who confirms one includes it and
    joins its lines."""
    for n, m in enumerate(missed):
        bx = m["box"]
        line = next((c for c in review.connections if c.source == "picture tracing" and c.path
                     and any(bx[0] <= px <= bx[2] and bx[1] <= py <= bx[3] for px, py in c.path)), None)
        if line is None:
            continue
        guess = m.get("class") or "valve"  # most symbols drawn on a pipe are valves
        mapping, identified_by = classify_detection(guess, None)
        review.symbols.append(ReviewSymbol(
            id=f"pm{n}", source="image model", box=list(m["box"]), detected_as=MISSED,
            dexpi_class=_mapping_to_class(mapping), identified_by=f"{identified_by} (a guess)",
            confidence=m.get("score"), include=False,
            notes=f"possible missed item: {m['how']} - the model drew no box here. If it is an item, "
                  f"include it, check its type and join the pipe to it; otherwise leave it out",
        ))


def _bridge_arrows(review: ReviewDiagram) -> None:
    """Flow arrows are labelled as nodes on the pipe (PID2Graph / OPEN100, and the team data
    from Team v4 on), so the model and the tracer connect pump - arrow - tank. Arrows are
    not DEXPI items: without this the pipe would be dropped with the arrow. The items around an
    arrow (through a chain of arrows) are joined directly, along the pipe through the arrows -
    as a tree, one line per neighbour along the pipe."""
    arrows = {s.id for s in review.symbols if s.detected_as == "arrow"}
    if not arrows:
        return
    links: dict = {}
    for c in review.connections:
        if c.from_id != c.to_id:
            links.setdefault(c.from_id, []).append((c.to_id, c))
            links.setdefault(c.to_id, []).append((c.from_id, c))
    have = {frozenset((c.from_id, c.to_id)) for c in review.connections}
    for arrow in sorted(arrows):
        ends, lines, todo, seen = [], [], [arrow], {arrow}
        while todo:  # the items around this arrow, through any arrows next to it
            for other, c in links.get(todo.pop(), []):
                lines.append(c)
                if other in seen:
                    continue
                seen.add(other)
                (todo if other in arrows else ends).append(other)
        # the arrow's direction, from a line into it with a drawn direction (the tracer reads
        # which way the arrow points): an item the flow leaves towards the arrow is upstream
        up, down = set(), set()
        for c in lines:
            if c.flow != "drawn arrow":
                continue
            if c.to_id in arrows and c.from_id not in arrows:
                up.add(c.from_id)
            elif c.from_id in arrows and c.to_id not in arrows:
                down.add(c.to_id)
        # the chain's own lines: arrow - arrow, item - arrow (the shortest of each pair)
        import math

        import networkx as nx
        box = {x.id: x.box for x in review.symbols}
        centre = lambda k: ((box[k][0] + box[k][2]) / 2, (box[k][1] + box[k][3]) / 2) if box.get(k) else (0.0, 0.0)
        h = nx.Graph()
        for c in lines:
            if c.from_id == c.to_id:
                continue
            pts = c.path or [centre(c.from_id), centre(c.to_id)]
            w = sum(math.dist(u, v) for u, v in zip(pts, pts[1:])) or 1.0
            if not h.has_edge(c.from_id, c.to_id) or w < h[c.from_id][c.to_id]["weight"]:
                h.add_edge(c.from_id, c.to_id, weight=w, line=c)
        chain = {x for x in seen if x in arrows}
        # each pair of items across the chain: the shortest way through its arrows only (an
        # item is never passed through); of all pairs, only the shortest tree is drawn - like a
        # pipe network (_network_trees), not every pair across every arrow
        routes = {}
        for a in ends:
            sub = h.subgraph(chain | {a})
            dist, path = nx.single_source_dijkstra(sub, a) if a in sub else ({}, {})
            for x in chain:
                if x not in dist:
                    continue
                for b in h.neighbors(x):
                    if b in ends and b != a:
                        cost = dist[x] + h[x][b]["weight"]
                        key = frozenset((a, b))
                        if key not in routes or cost < routes[key][0]:
                            routes[key] = (cost, path[x] + [b] if path[x][0] == a else [a, *path[x][1:], b])
        k = nx.Graph()
        for key, (cost, route) in routes.items():
            a, b = route[0], route[-1]
            k.add_edge(a, b, weight=cost, route=route)
        for a, b, data in nx.minimum_spanning_edges(k, data=True):
            route = data["route"]
            if frozenset((a, b)) in have:
                continue
            have.add(frozenset((a, b)))
            steps = [h[u][v]["line"] for u, v in zip(route, route[1:])]
            path = []
            for (u, v), c in zip(zip(route, route[1:]), steps):
                pts = list(c.path or [])
                path += pts if c.from_id == u else pts[::-1]
            traced = all(c.source == "picture tracing" for c in steps)
            nets = sorted({(c.attributes or {}).get("net") for c in steps} - {None})
            a, b = route[0], route[-1]
            flow = ""
            if (a in down or b in up) and not (a in up or b in down):
                a, b, path = b, a, path[::-1]
                flow = "drawn arrow"
            elif (a in up or b in down) and not (a in down or b in up):
                flow = "drawn arrow"
            review.connections.append(ReviewConnection(
                id=f"ac{len(review.connections)}", from_id=a, to_id=b,
                source="picture tracing" if traced else "image model",
                line_type="signal" if any(c.line_type == "signal" for c in steps) else "process",
                confidence="through a flow arrow", flow=flow, path=path or None,
                notes="" if traced else "predicted by the model, not traced on the picture - check",
                attributes={"net": nets[0], "nets": nets} if nets else {},   # one pipe through the arrow
            ))


def _imply_flow(review: ReviewDiagram) -> None:
    """Carry arrowhead directions to process lines without one (flow_direction rules:
    an item with two pipes passes the flow on)."""
    from pid_dexpi_aas.drawio.flow_direction import Leg, propagate

    by_id = {s.id: s for s in review.symbols}
    lines = [c for c in review.connections if c.line_type == "process" and c.from_id in by_id and c.to_id in by_id]
    legs = [Leg(by_id[c.from_id], by_id[c.to_id], 1 if c.flow == "drawn arrow" else 0,
                "both" if c.flow == FLOW_BOTH else "arrow" if c.flow == "drawn arrow" else "") for c in lines]
    instrument = lambda s: category_for_class(s.dexpi_class) == Category.INSTRUMENTATION
    propagate(legs, passes=lambda s: not instrument(s), junction=lambda s: False)
    for c, leg in zip(lines, legs):
        if c.flow or not leg.dir:
            continue
        c.flow = "implied"
        if leg.dir < 0:
            c.from_id, c.to_id = c.to_id, c.from_id
            c.path = (c.path or [])[::-1] or None


# ------------------------------------------------------------------------ cross-check ---
_COMPATIBLE = {
    Category.EQUIPMENT: {"pump", "tank", "general"},
    Category.PIPING: {"valve"},
    Category.INSTRUMENTATION: {"instrumentation"},
    Category.NOT_PROCESS_EQUIPMENT: {"arrow", "inlet/outlet", "general"},
}


def _norm_tag(t: str) -> str:
    return re.sub(r"[\s\-]", "", t.upper())


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def align_drawio_to_image(diagram: ParsedDiagram, ocr_tags: list[OcrTag]) -> dict | None:
    """Fit image = scale * drawio + offset (per axis) from tags present in both.

    draw.io's PNG export is a uniform scale plus an offset (diagram bounds + border), so
    two independent 1-D linear fits suffice. Pairs are draw.io text cells vs OCR tags
    with the same normalized text; a robust refit drops pairs > 3x the median residual.
    """
    by_text: dict[str, list[OcrTag]] = {}
    for t in ocr_tags:
        by_text.setdefault(_norm_tag(t.text), []).append(t)
    pairs = []
    for s in diagram.shapes:
        key = _norm_tag(s.text or "")
        if len(key) >= 3 and len(by_text.get(key, [])) == 1:  # unique on both sides only
            o = by_text[key][0]
            pairs.append(((s.x + s.width / 2, s.y + s.height / 2), o.center, key))
    if len(pairs) < 3:
        return None
    src = np.array([p[0] for p in pairs])
    dst = np.array([p[1] for p in pairs])
    keep = np.ones(len(pairs), bool)
    for _ in range(2):
        fx = np.polyfit(src[keep, 0], dst[keep, 0], 1)
        fy = np.polyfit(src[keep, 1], dst[keep, 1], 1)
        res = np.hypot(np.polyval(fx, src[:, 0]) - dst[:, 0], np.polyval(fy, src[:, 1]) - dst[:, 1])
        keep = res <= max(3 * np.median(res[keep]), 5.0)
        if keep.sum() < 3:
            return None
    return {"x": [float(v) for v in fx], "y": [float(v) for v in fy], "pairs_used": int(keep.sum()),
            "pairs_total": len(pairs), "median_error_px": float(np.median(res[keep]))}


def cross_check(drawio_review: ReviewDiagram, image_review: ReviewDiagram, infer: dict) -> ReviewDiagram:
    """draw.io stays the source of truth; annotate each row with what the image shows,
    and add image-only detections as excluded rows for the reviewer to consider."""
    review = copy.deepcopy(drawio_review)
    review.image_path, review.image_size = image_review.image_path, image_review.image_size
    review.unattached_tags = image_review.unattached_tags
    tags = assemble_tags(infer.get("ocr_tokens", []), known=[s.tag for s in drawio_review.symbols if s.tag])
    fit = align_drawio_to_image(review.drawio, tags)
    review.alignment = fit
    if fit is None:
        for s in review.symbols:
            s.image_check = "could not align draw.io to image (too few shared tags)"
        return review

    def to_px(box):
        x1, x2 = np.polyval(fit["x"], [box[0], box[2]])
        y1, y2 = np.polyval(fit["y"], [box[1], box[3]])
        return [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]

    model_syms = [s for s in image_review.symbols]
    used = set()
    match = {}
    review.picture = {"symbols": image_review.symbols, "connections": image_review.connections, "match": match}
    for s in review.symbols:
        s.box = to_px(s.box)
        best = max(model_syms, key=lambda m: _iou(s.box, m.box), default=None)
        if best is None or _iou(s.box, best.box) < 0.3:
            s.image_check = "not detected by model"
            continue
        used.add(best.id)
        match[s.id] = best.id
        ok = best.detected_as in _COMPATIBLE.get(category_for_class(s.dexpi_class), set())
        s.image_check = f"model agrees ({best.detected_as})" if ok else f"model sees '{best.detected_as}' here"
        if best.tag and s.tag and _norm_tag(best.tag) != _norm_tag(s.tag):
            s.image_check += f"; OCR reads '{best.tag}'"
    # A tag the drawing has at one place and the image at another: the two are different
    # versions of the sheet (LightUp's PNG predates moving V204 and P201). Say so on both
    # rows rather than leaving a "not detected" + "only in image" pair to puzzle over.
    missing = {_norm_tag(s.tag): s for s in review.symbols
               if s.image_check == "not detected by model" and s.tag and len(_norm_tag(s.tag)) >= 3}
    review.moved = []
    for m in model_syms:
        if m.id not in used:
            extra = copy.deepcopy(m)
            extra.include = False
            extra.image_check = "only in image - missing from draw.io?"
            twin = missing.get(_norm_tag(m.tag)) if m.tag else None
            if twin is not None:
                extra.image_check = f"only in image - the image draws {twin.tag} here, draw.io elsewhere (different versions?)"
                twin.image_check = f"not at this spot in the image - the image draws {twin.tag} elsewhere (different versions?)"
                review.moved.append(twin.tag)
            review.symbols.append(extra)
    return review


# ----------------------------------------------------------------------------- build ---
def validate_rows(review: ReviewDiagram) -> list[str]:
    """Problems that would make to_dexpi fail or silently drop things."""
    problems = []
    ids = [s.id for s in review.symbols]
    if len(ids) != len(set(ids)):
        problems.append("duplicate symbol ids")
    included = {s.id for s in review.symbols if s.include and s.dexpi_class != SKIP}
    for s in review.symbols:
        if s.include and s.dexpi_class not in DEXPI_CLASS_CHOICES:
            problems.append(f"{s.id}: unknown DEXPI class {s.dexpi_class!r}")
        if s.include and s.dexpi_class == OFF_PAGE and not s.tag:
            problems.append(f"{s.id}: an off-page connector needs a tag/label text")
    for c in review.connections:
        if c.include and (c.from_id not in included or c.to_id not in included):
            problems.append(f"connection {c.id}: endpoint {c.from_id} -> {c.to_id} is not an included symbol")
    return problems


@dataclass
class _TeeFlow:
    """build_model reads a pipe network's flow from diagram.flow.member_sense."""
    member_sense: dict


def _picture_tees(review: ReviewDiagram) -> tuple[list, dict, dict]:
    """Tees drawn in the picture editor (rows with a "tee" attribute: the branch's item
    joined to both ends of the line it meets) as draw.io pipe networks, so build_model makes
    a DEXPI PipeTee with one segment per leg - not the rows as separate pipes (the pair on
    the same side of the tee would become a pipe of its own). -> (network edges,
    {row id: its network edge ids}, member_sense {(network, item): (+1 feeds | -1 fed | 0, basis)})."""
    rows = {c.id: c for c in review.connections if c.include}
    edges, tee_rows, sense = [], {}, {}
    groups: dict = {}
    for c in rows.values():
        tee = (c.attributes or {}).get("tee")
        if tee and tee.get("line") in rows:
            groups.setdefault(tee["line"], []).append(c)
    for n, (line_id, legs) in enumerate(groups.items()):
        line = rows[line_id]
        net = f"net-tee{n}"
        item = legs[0].attributes["tee"].get("item") or legs[0].from_id
        members = list(dict.fromkeys([line.from_id, line.to_id, item]))
        flows = {}
        for c in [line, *legs]:
            if c.flow in FLOW_KNOWN:  # From feeds, To is fed
                flows.setdefault(c.from_id, 1)
                flows.setdefault(c.to_id, -1)
        for m in members:
            sense[(net, m)] = (flows.get(m, 0), "reviewer" if m in flows else "")
        ids = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                e = Edge(id=f"{net}:{a}:{b}", source=a, target=b, style="pipe network",
                         is_dashed=line.line_type == "signal")
                edges.append(e)
                ids.append(e.id)
        for c in [line, *legs]:
            tee_rows[c.id] = ids
    return edges, tee_rows, sense


def _traced_networks(review: ReviewDiagram, skip: dict) -> tuple[list, dict, dict]:
    """Pipe networks traced on the picture (rows with a "net": the tree of lines the review
    shows for one network, trace_lines.py; lines across a flow arrow join the networks on both
    sides) as draw.io pipe networks, so build_model makes them DEXPI PipeTee fittings with one
    segment per leg - not each shown line a pipe of its own. Process lines with three or more
    items only (two items: the line is the pipe). Members are listed along the drawn tree, so
    the tee chain follows the pipe. -> (network edges, {row id: its network's edge ids},
    member_sense {(network, item): (+1 feeds | -1 fed | 0, basis)})."""
    import networkx as nx
    rows = [c for c in review.connections if c.include and c.id not in skip and c.line_type != "signal"
            and (c.attributes or {}).get("net")]
    parent: dict = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for c in rows:
        nets = (c.attributes or {}).get("nets") or [c.attributes["net"]]
        for n in nets[1:]:
            parent[root(n)] = root(nets[0])
    groups: dict = {}
    for c in rows:
        groups.setdefault(root(c.attributes["net"]), []).append(c)
    edges, net_rows, sense = [], {}, {}
    for n, legs in enumerate(groups.values()):
        g = nx.Graph([(c.from_id, c.to_id) for c in legs])
        if g.number_of_nodes() < 3:
            continue
        # along the tree: from one end of its longest path, depth first
        start = next(iter(g.nodes))
        far = max(nx.single_source_shortest_path_length(g, start).items(), key=lambda kv: kv[1])[0]
        members = list(nx.dfs_preorder_nodes(g, far))
        net = f"net-trace{n}"
        flows = {}
        for c in legs:
            if c.flow in FLOW_KNOWN:  # From feeds, To is fed
                flows.setdefault(c.from_id, 1)
                flows.setdefault(c.to_id, -1)
        for m in members:
            sense[(net, m)] = (flows.get(m, 0), "arrow" if m in flows else "")
        ids = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                e = Edge(id=f"{net}:{a}:{b}", source=a, target=b, style="pipe network")
                edges.append(e)
                ids.append(e.id)
        for c in legs:
            net_rows[c.id] = ids
    return edges, net_rows, sense


def _mapping_for_row(s: ReviewSymbol) -> MappingResult:
    if not s.include or s.dexpi_class == SKIP:
        return MappingResult(Category.UNMAPPED, None, "excluded by reviewer" if not s.include else s.notes or "skipped")
    return MappingResult(category_for_class(s.dexpi_class), s.dexpi_class, s.notes)


def to_dexpi(review: ReviewDiagram) -> tuple[ParsedDiagram, BuildResult]:
    """Build the DEXPI model from the reviewed rows via build_model.

    Off-page rows become text-only shapes (build_model turns those into Flow In/Out
    off-page connectors when a pipe ends on them); everything else becomes a shape whose
    DEXPI class and tag come from the row (mapping_overrides / tag_overrides).
    """
    base = review.drawio
    shapes: list[Shape] = []
    row_ids = {s.id for s in review.symbols}
    if base is not None:
        # Keep draw.io cells that have no row (pure tag labels) so build_model's own tag
        # association and labels still work; rows replace their cells.
        shapes = [copy.copy(sh) for sh in base.shapes if sh.id not in row_ids]
    overrides, tag_overrides = {}, {}
    # a stream made from a pipe's loose end stays marked as one (the renderer draws its arrow)
    loose_ends = {sh.id for sh in base.shapes if sh.data.get("loose_end")} if base is not None else set()
    for s in review.symbols:
        box = s.box or [0, 0, 0, 0]
        off_page = s.include and s.dexpi_class == OFF_PAGE
        shapes.append(Shape(
            id=s.id, shape_token=None if off_page else f"review:{s.detected_as}", text=s.tag or "",
            x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1], style="",
            data={"loose_end": "1"} if off_page and s.id in loose_ends else {},
        ))
        if not off_page:
            overrides[s.id] = _mapping_for_row(s)
            tag_overrides[s.id] = s.tag or None

    # Reuse the original draw.io Edge where one exists (only its endpoints/line type may
    # have been edited), so build_model's checks that depend on raw geometry - e.g.
    # rejecting zero-length edges - apply exactly as in the plain draw.io pipeline.
    original_edges = {e.id: e for e in base.edges} if base is not None else {}
    edges = []
    tee_edges, tee_rows, member_sense = _picture_tees(review)
    edges += tee_edges
    net_edges, net_rows, net_sense = _traced_networks(review, tee_rows)
    edges += net_edges
    tee_rows = {**tee_rows, **net_rows}
    member_sense = {**member_sense, **net_sense}
    for c in review.connections:
        if not c.include or c.id in tee_rows:
            continue
        if c.id in original_edges:
            edge = copy.copy(original_edges[c.id])
            edge.source, edge.target, edge.is_dashed = c.from_id, c.to_id, c.line_type == "signal"
        else:
            edge = Edge(id=c.id, source=c.from_id, target=c.to_id, style="", is_dashed=c.line_type == "signal")
        # the row's From -> To is the flow direction when known (the reviewer may have swapped it)
        edge.flow = 1 if c.flow in FLOW_KNOWN else 0
        edge.flow_basis = ("both" if c.flow == FLOW_BOTH else "arrow" if c.flow == "drawn arrow"
                           else "implied" if edge.flow else "")
        edges.append(edge)
    if base is None:  # image mode: the redraw follows the pipes traced on the picture
        traced = {c.id: [tuple(p) for p in c.path] for c in review.connections if c.include and c.path}
    else:
        traced = {}
    # every other line without a drawn route (added, re-pointed, untraced) is routed around the
    # items like a draw.io connector, so the re-drawn picture has no straight lines across it
    from pid_dexpi_aas.render.routing import route
    drawn_routes = (base.routes or {}) if base is not None else {}
    original = {e.id: e for e in base.edges} if base is not None else {}
    item_box = {sh.id: (sh.x, sh.y, sh.x + sh.width, sh.y + sh.height) for sh in shapes if sh.width and sh.height}
    obstacles = list(item_box.values())
    for c in review.connections:
        if not c.include or c.id in traced or c.from_id not in item_box or c.to_id not in item_box:
            continue
        o = original.get(c.id)
        if c.id in drawn_routes and o is not None and {o.source, o.target} == {c.from_id, c.to_id}:
            continue  # drawn, unchanged
        if o is not None and o.merged_into is None and c.id in drawn_routes:
            pass  # re-pointed: the drawn route no longer fits
        elif c.id.startswith("net"):
            continue  # pipe networks are drawn piece by piece from their member lines
        traced[c.id] = route(item_box[c.from_id], item_box[c.to_id], obstacles)
    diagram = ParsedDiagram(source_file=review.source_name, shapes=shapes, edges=edges,
                            specs=dict(base.specs) if base is not None else {},
                            flow=base.flow if base is not None else (
                                _TeeFlow(member_sense) if member_sense else None),
                            routes=({**(base.routes or {}), **traced} if base is not None else (traced or None)),
                            drawn_edges=list(base.edges) if base is not None else None,
                            sections=list(base.sections) if base is not None else [])
    result = build_model(diagram, overrides, tag_overrides)
    for row, net_edges in tee_rows.items():  # a tee's rows map to its network's segments
        result.edge_segments[row] = list({id(x): x for e in net_edges for x in result.edge_segments.get(e, [])}.values())
    # specification fields filled in by the reviewer (dexpi/fields.py), over the drawing's notes
    from pid_dexpi_aas.dexpi.fields import apply_fields
    result.spec_problems = []
    # a line's specifications go onto what it became: its pipe segment(s) (a pipe network:
    # all of the network's segments), or its measuring / signal line
    from pydexpi.dexpi_classes.instrumentation import SignalConveyingFunction
    for c in review.connections:
        attrs = c.attributes or {}
        if not c.include or not (attrs.get("dexpi") or attrs.get("custom")):
            continue
        targets = list(result.edge_segments.get(c.id, [])) + [
            o for o in result.edge_signals.get(c.id, []) if isinstance(o, SignalConveyingFunction)]
        for obj in {id(o): o for o in targets}.values():
            result.spec_problems += [f"line {c.id}: {p}" for p in apply_fields(obj, attrs.get("dexpi"), attrs.get("custom"))]
    for s in review.symbols:
        obj = result.shape_objects.get(s.id)
        if obj is not None and s.section and "customAttributes" in type(obj).model_fields:
            # pyDEXPI 1.2 neither exports nor reads PlantSection structure items, so the plant
            # section travels as a named attribute on every item (equipment, instruments, valves)
            from pydexpi.dexpi_classes.pydantic_classes import CustomStringAttribute
            obj.customAttributes = [a for a in (obj.customAttributes or []) if a.attributeName != "PlantSectionName"] + [
                CustomStringAttribute(attributeName="PlantSectionName", attributeURI="urn:pid-dexpi-aas:PlantSectionName",
                                      value=s.section)]
        values, custom = (s.attributes or {}).get("dexpi"), (s.attributes or {}).get("custom")
        if obj is not None and (values or custom):
            result.spec_problems += [f"{s.tag or s.id}: {p}" for p in apply_fields(obj, values, custom)]
            if s.dexpi_class == OFF_PAGE:  # a stream is one connector per pipe in DEXPI: each gets them
                from pid_dexpi_aas.dexpi.build_model import pipe_twins
                for twin in pipe_twins(obj):
                    apply_fields(twin, values, custom)
        if obj is not None and s.dexpi_class == OFF_PAGE and custom:
            # a stream's fluid and temperature go onto its pipe too: DEXPI's own typed fields there
            fluid = custom.get("FluidCode") or custom.get("Fluid")
            temp = custom.get("OperatingTemperature")
            for c in review.connections:
                if s.id not in (c.from_id, c.to_id) or not c.include:
                    continue
                for seg in result.edge_segments.get(c.id, []):
                    if fluid and not getattr(seg, "fluidCode", None):
                        seg.fluidCode = str(fluid)
                    if temp and getattr(seg, "operatingTemperature", None) is None:
                        result.spec_problems += [f"{s.tag or s.id}: {p}"
                                                 for p in apply_fields(seg, {"operatingTemperature": temp}, {})]
    return diagram, result
