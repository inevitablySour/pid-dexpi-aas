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

from pid_dexpi_aas.build_model import BuildResult, _tag_name, build_model
from pid_dexpi_aas.mapping import Category, MappingResult, classify_detection, classify_instrument_tag, map_shape
from pid_dexpi_aas.ocr_tags import OcrTag, assemble_tags, attach_tags
from pid_dexpi_aas.parser_drawio import Edge, ParsedDiagram, Shape
from pid_dexpi_aas.tag_association import build_tag_lookup

SKIP = "(skip - not in DEXPI)"
OFF_PAGE = "(off-page connector)"
EQUIPMENT_CLASSES = sorted(
    n for n in dir(eq)
    if not n.startswith("_") and isinstance(getattr(eq, n), type)
    and issubclass(getattr(eq, n), eq.Equipment) and n != "Equipment"
)
VALVE_CLASSES = sorted(n for n in dir(piping) if not n.startswith("_") and ("Valve" in n or n.endswith("Fitting")))
INSTRUMENT_CLASSES = ["ProcessInstrumentationFunction", "SignalOffPageConnector"]
# One dropdown for the reviewer; the DEXPI category is implied by the class picked.
DEXPI_CLASS_CHOICES = [SKIP, OFF_PAGE] + INSTRUMENT_CLASSES + VALVE_CLASSES + EQUIPMENT_CLASSES


def category_for_class(dexpi_class: str | None) -> Category:
    if dexpi_class in EQUIPMENT_CLASSES:
        return Category.EQUIPMENT
    if dexpi_class in VALVE_CLASSES:
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
    include: bool = True


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
    # Process flow runs From -> To, and how that is known: "drawn arrow" | "implied" (from
    # arrows on neighbouring lines, flow_direction.py) | "reviewer"; "" = unknown;
    # "no flow between them" = two legs on the same side of a tee (both fed, or both feeding)
    flow: str = ""


FLOW_KNOWN = ("drawn arrow", "implied", "reviewer")


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
    tag_lookup = build_tag_lookup(diagram)
    endpoint_ids = {e.source for e in diagram.edges} | {e.target for e in diagram.edges}
    review = ReviewDiagram(source_name=diagram.source_file, drawio=diagram)

    for shape in diagram.shapes:
        if shape.is_labeled_only and not (shape.id in endpoint_ids and shape.text):
            continue
        tag = _tag_name(shape, tag_lookup) or shape.text
        mapping = map_shape(shape.shape_token, shape.text or tag or "")  # same rule as build_model
        if shape.is_labeled_only:
            dexpi_class, identified_by = OFF_PAGE, "text label a pipe ends on"
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
            notes=mapping.reason, include=dexpi_class != SKIP,
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
            flow={"arrow": "drawn arrow", "implied": "implied", "parallel": "no flow between them"}.get(
                edge.flow_basis, "") if edge.flow or edge.flow_basis == "parallel" else "",
            line_type="signal" if edge.is_dashed else "process", source="draw.io",
            confidence=resolution,
            notes="" if resolution in ("explicit", "adjacent", "junction") else
            ("pipe network joined at a junction; one end guessed - check" if resolution == "junction+guess"
             else "endpoint inferred from a loose line end - check"),
        ))
    _exclude_dangling(review)
    return review


def _exclude_dangling(review: ReviewDiagram) -> None:
    """Untick connections whose endpoint is not converted to DEXPI (build_model would skip
    them anyway); the reviewer can re-include after fixing the endpoint's class."""
    included = {s.id for s in review.symbols if s.include and s.dexpi_class != SKIP}
    for c in review.connections:
        if c.include and (c.from_id not in included or c.to_id not in included):
            c.include = False
            c.notes = "; ".join(x for x in (c.notes, "endpoint not converted to DEXPI") if x)


# ----------------------------------------------------------------------------- image ---
def from_image(infer: dict, source_name: str) -> ReviewDiagram:
    """Rows from ml/pid2graph/infer_sheet.py output: model symbols + OCR tags + name
    classifier, and the model's predicted connections."""
    tags = assemble_tags(infer.get("ocr_tokens", []))
    boxes = {s["id"]: tuple(s["box"]) for s in infer["symbols"]}
    attached = attach_tags(boxes, tags)
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
            notes="; ".join(x for x in (mapping.reason, f"tile votes {votes}" if votes else "") if x),
            include=dexpi_class != SKIP,
        ))
    review.unattached_tags = [
        {"tag": t.text, "box": list(t.box), "conf": round(t.conf, 3)} for t in tags if id(t) not in claimed
    ]
    for n, c in enumerate(infer.get("connections", [])):
        review.connections.append(ReviewConnection(
            id=f"mc{n}", from_id=c["from"], to_id=c["to"], source="image model",
            line_type=c.get("line_type", "process"),
            confidence=f"{c['prob']:.2f}", notes="model-predicted - unreliable in current model, check",
        ))
    _exclude_dangling(review)
    return review


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
    tags = assemble_tags(infer.get("ocr_tokens", []))
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
    for s in review.symbols:
        s.box = to_px(s.box)
        best = max(model_syms, key=lambda m: _iou(s.box, m.box), default=None)
        if best is None or _iou(s.box, best.box) < 0.3:
            s.image_check = "not detected by model"
            continue
        used.add(best.id)
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
    for s in review.symbols:
        box = s.box or [0, 0, 0, 0]
        off_page = s.include and s.dexpi_class == OFF_PAGE
        shapes.append(Shape(
            id=s.id, shape_token=None if off_page else f"review:{s.detected_as}", text=s.tag or "",
            x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1], style="",
        ))
        if not off_page:
            overrides[s.id] = _mapping_for_row(s)
            tag_overrides[s.id] = s.tag or None

    # Reuse the original draw.io Edge where one exists (only its endpoints/line type may
    # have been edited), so build_model's checks that depend on raw geometry - e.g.
    # rejecting zero-length edges - apply exactly as in the plain draw.io pipeline.
    original_edges = {e.id: e for e in base.edges} if base is not None else {}
    edges = []
    for c in review.connections:
        if not c.include:
            continue
        if c.id in original_edges:
            edge = copy.copy(original_edges[c.id])
            edge.source, edge.target, edge.is_dashed = c.from_id, c.to_id, c.line_type == "signal"
        else:
            edge = Edge(id=c.id, source=c.from_id, target=c.to_id, style="", is_dashed=c.line_type == "signal")
        # the row's From -> To is the flow direction when known (the reviewer may have swapped it)
        edge.flow = 1 if c.flow in FLOW_KNOWN else 0
        edge.flow_basis = "arrow" if c.flow == "drawn arrow" else "implied" if edge.flow else ""
        edges.append(edge)
    diagram = ParsedDiagram(source_file=review.source_name, shapes=shapes, edges=edges,
                            specs=dict(base.specs) if base is not None else {},
                            flow=base.flow if base is not None else None,
                            routes=base.routes if base is not None else None,
                            drawn_edges=list(base.edges) if base is not None else None)
    return diagram, build_model(diagram, overrides, tag_overrides)
