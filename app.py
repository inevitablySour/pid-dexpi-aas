"""P&ID -> DEXPI converter with a manual review step (no CLI needed).

Run with:
    .venv/bin/streamlit run app.py

Give it a draw.io file, a P&ID image, or both:
  - draw.io only: shapes/tags/connections come from the file (as before).
  - image only:   the trained model detects symbols and connections, OCR reads the tags,
                  and tag prefixes (PI, GV, PSV, P, V, ...) pick the DEXPI class.
  - both:         draw.io is the source of truth; the image is used to check it and to
                  flag symbols the model sees that the drawing doesn't have.
Every symbol and connection is then shown in editable tables - with how it was
identified - so an operator can fix classes, tags and connections before the DEXPI XML
is built and verified.

Beside the tables, an interactive map (pid_dexpi_aas.flow_map): for a draw.io file every
pipe along its drawn route with its flow direction; for an image the model's boxes and
connections. Clicking a line or box ticks its row's "show" box and moves the row to the top
of its table; ticking "show" in a table highlights that item (or a whole pipe network) on
the map and zooms to it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v2  # noqa: E402,F401  (st.components.v2: the clickable map)
from PIL import Image  # noqa: E402

from pid_dexpi_aas.check_sheet import build_check_sheet, number_items, numbered_overlay, render_after_png  # noqa: E402
from pid_dexpi_aas.flow_map import (  # noqa: E402
    MAP_CSS, MAP_JS, Box, Layers, build_layers, image_layers, map_markup, review_overrides,
)
from pid_dexpi_aas.decode_drawio import decode_drawio  # noqa: E402
from pid_dexpi_aas.mapping import classify_detection  # noqa: E402
from pid_dexpi_aas.parser_drawio import parse_drawio_xml_resolved  # noqa: E402
from pid_dexpi_aas.review import (  # noqa: E402
    DEXPI_CLASS_CHOICES, OFF_PAGE, SKIP, ReviewConnection, ReviewDiagram, ReviewSymbol,
    _instrument_attributes, cross_check, from_drawio, from_image, to_dexpi, validate_rows,
)
from pid_dexpi_aas.serialize import save_and_verify  # noqa: E402
from pid_dexpi_aas.validate_by_counting import report_for_diagram  # noqa: E402

_ML_VENV = ROOT / "ml" / "relationformer" / ".venv"
ML_PYTHON = next((p for p in (_ML_VENV / "bin" / "python", _ML_VENV / "Scripts" / "python.exe") if p.exists()),
                 _ML_VENV / "bin" / "python")
INFER_SCRIPT = ROOT / "ml" / "pid2graph" / "infer_sheet.py"
# The released model (ml/get_model.sh downloads it); PID_MODEL_CHECKPOINT points elsewhere.
DEFAULT_CHECKPOINT = os.environ.get("PID_MODEL_CHECKPOINT", str(ROOT / "ml" / "models" / "pid2graph_fold1.pt"))
HAS_GPU = shutil.which("nvidia-smi") is not None
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="P&ID to DEXPI converter", layout="wide")
st.title("P&ID to DEXPI converter")
st.caption("Upload a draw.io file, an image of a P&ID, or both. Review and correct what was "
           "detected, then build and download the DEXPI XML.")

# ------------------------------------------------------------------------------ inputs ---
with st.sidebar:
    st.header("Input")
    drawio_file = st.file_uploader("draw.io file (optional)", type=["drawio", "xml"])
    image_file = st.file_uploader("P&ID image (optional)", type=["png", "jpg", "jpeg"])
    with st.expander("Model settings"):
        checkpoint = st.text_input("Model checkpoint", DEFAULT_CHECKPOINT)
        device = st.selectbox("Device", ["cuda", "cpu"] if HAS_GPU else ["cpu", "cuda"],
                              help="CPU works (no NVIDIA GPU needed) but takes minutes per sheet.")
    st.divider()
    st.header("Saved review")
    review_upload = st.file_uploader("Resume a saved review (.json)", type=["json"])

if drawio_file is None and image_file is None and review_upload is None:
    st.info("Upload a draw.io file and/or a P&ID image in the sidebar to get started.")
    st.stop()

WORK = Path(tempfile.gettempdir()) / "pid_dexpi_app"
WORK.mkdir(exist_ok=True)


def _digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:16]


@st.cache_data(show_spinner=False)
def run_image_model(image_bytes: bytes, image_name: str, checkpoint: str, device: str) -> dict:
    key = _digest(image_bytes, checkpoint.encode(), device.encode())
    img_path = WORK / f"{key}_{Path(image_name).name}"
    out_path = WORK / f"{key}.json"
    img_path.write_bytes(image_bytes)
    if not out_path.exists():
        proc = subprocess.run(
            [str(ML_PYTHON), str(INFER_SCRIPT), "--image", str(img_path), "--checkpoint", checkpoint,
             "--out", str(out_path), "--device", device],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:])
    return json.loads(out_path.read_text())


def build_review() -> ReviewDiagram:
    drawio_review = image_review = infer = None
    if drawio_file is not None:
        xml_text, drawio_path = decode_drawio_bytes(drawio_file.getvalue(), drawio_file.name)
        # drawio_path lets the parser follow each line's drawn route (line_routes.py), so
        # lines ending on other lines (tees) connect - needs the draw.io app, else skipped.
        drawio_review = from_drawio(parse_drawio_xml_resolved(xml_text, source_file=drawio_file.name,
                                                              drawio_path=drawio_path))
    if image_file is not None:
        if not ML_PYTHON.exists():
            st.error(f"ML environment not found at {ML_PYTHON} - run ml/pid2graph/setup.sh first.")
            st.stop()
        if not Path(checkpoint).exists():
            st.error(f"Model checkpoint not found: {checkpoint}")
            st.stop()
        with st.spinner("Running the symbol model and OCR on the image (about 15 s on GPU)..."):
            try:
                infer = run_image_model(image_file.getvalue(), image_file.name, checkpoint, device)
            except RuntimeError as exc:
                st.error("Image model failed:")
                st.code(str(exc))
                st.stop()
        image_review = from_image(infer, image_file.name)
    if drawio_review and image_review:
        return cross_check(drawio_review, image_review, infer)
    return drawio_review or image_review


def decode_drawio_bytes(data: bytes, name: str) -> tuple[str, Path]:
    path = WORK / f"{_digest(data)}_{Path(name).name}"
    path.write_bytes(data)
    return decode_drawio(path)[0], path


def review_to_json(review: ReviewDiagram) -> str:
    return json.dumps({
        "source_name": review.source_name, "image_path": review.image_path, "image_size": review.image_size,
        "symbols": review.symbol_rows(), "connections": review.connection_rows(),
        "unattached_tags": review.unattached_tags, "alignment": review.alignment,
        "drawio_file": drawio_file.name if drawio_file else None,
    }, indent=1)


# Session state: rebuild the review only when the inputs change, so edits survive reruns.
input_key = _digest(
    drawio_file.getvalue() if drawio_file else b"", image_file.getvalue() if image_file else b"",
    review_upload.getvalue() if review_upload else b"", checkpoint.encode(),
)
if st.session_state.get("input_key") != input_key:
    if review_upload is not None:
        saved = json.loads(review_upload.getvalue())
        review = build_review() if (drawio_file or image_file) else ReviewDiagram(saved["source_name"])
        review.symbols = [ReviewSymbol(**s) for s in saved["symbols"]]
        review.connections = [ReviewConnection(**c) for c in saved["connections"]]
        review.unattached_tags = saved.get("unattached_tags", [])
    else:
        review = build_review()
    st.session_state.update(input_key=input_key, review=review, editor_version=0)
review: ReviewDiagram = st.session_state["review"]

mode = " + ".join(x for x, f in (("draw.io", drawio_file), ("image", image_file)) if f) or "saved review"
st.subheader(f"Review: {review.source_name}  ·  mode: {mode}")
if review.alignment:
    a = review.alignment
    st.caption(f"draw.io aligned to the image using {a['pairs_used']} shared tags "
               f"(median error {a['median_error_px']:.1f} px).")
if image_file is not None and drawio_file is None:
    st.warning("Image-only mode: connections come from the current model and are unreliable "
               "(known label issue, fix pending) - review them carefully or rebuild them by hand.")

# ------------------------------------------------------------------ needs-review logic ---
def needs_review(s: ReviewSymbol) -> bool:
    if s.source == "manual":
        return False
    return (
        s.dexpi_class == SKIP and s.source != "draw.io"
        or (s.confidence is not None and s.confidence < 0.7)
        or "overrides" in s.notes
        or (s.image_check and not s.image_check.startswith("model agrees"))
        or (s.source == "image model" and not s.tag and s.dexpi_class not in (SKIP,))
    )


# ------------------------------------------------------------------------- overlay ---
# the interactive map: clicking a line or box selects its table rows (flow_map.MAP_JS)
map_component = st.components.v2.component("pid_flow_map", html="", css=MAP_CSS, js=MAP_JS)


def symbol_color(s: ReviewSymbol, selected: set[str]) -> tuple[int, int, int]:
    if s.id in selected:
        return (220, 0, 220)
    if not s.include or s.dexpi_class == SKIP:
        return (150, 150, 150)
    if needs_review(s):
        return (220, 40, 40)
    return (0, 140, 70)


MODEL_ONLY = (245, 130, 49)
OCR_TAG = (120, 120, 120)


def _symbol_tip(s: ReviewSymbol) -> str:
    state = "excluded" if not s.include or s.dexpi_class == SKIP else "needs review" if needs_review(s) else "converted"
    tip = f"{s.tag or '(no tag)'} - {s.dexpi_class} ({state})"
    if s.tag_source.startswith("OCR"):
        tip += f"; tag read by OCR ({s.tag_source})"
    return tip + (f"; image: {s.image_check}" if s.image_check else "")


def _image_to_drawio(fit: dict):
    """Inverse of review.align_drawio_to_image's fit: image pixels -> draw.io units."""
    (ax, bx), (ay, by) = fit["x"], fit["y"]

    def f(box):
        x1, x2 = sorted(((box[0] - bx) / ax, (box[2] - bx) / ax))
        y1, y2 = sorted(((box[1] - by) / ay, (box[3] - by) / ay))
        return (x1, y1, x2, y2)
    return f


def _ocr_boxes(review: ReviewDiagram, to_map=lambda b: tuple(b)) -> list[Box]:
    """OCR-read tags no symbol claimed (possible missed symbols), as unselectable boxes."""
    return [Box("", to_map(t["box"]), OCR_TAG, t["tag"], 2,
                f"OCR read '{t['tag']}' (confidence {t.get('conf', 0):.2f}) - not attached to any symbol", True)
            for t in review.unattached_tags or [] if t.get("box")]


def drawio_layers(review: ReviewDiagram) -> Layers | None:
    """The draw.io flow map: every pipe along its drawn route with its flow direction
    (pid_dexpi_aas.flow_map) - automatic, reflects the table edits. In both-inputs mode it
    also shows what the image adds: symbols only the model found (dashed orange) and OCR
    tags nothing claimed (dashed grey), mapped onto the drawing via the tag alignment."""
    if drawio_file is None or review.drawio is None:
        return None
    _, path = decode_drawio_bytes(drawio_file.getvalue(), drawio_file.name)
    shapes = {sh.id: sh for sh in review.drawio.shapes}
    boxes = [Box(s.id, (sh.x, sh.y, sh.x + sh.width, sh.y + sh.height), symbol_color(s, set()),
                 s.tag or s.id, 3, _symbol_tip(s))
             for s in review.symbols if (sh := shapes.get(s.id)) is not None]
    extra_legend = []
    if review.alignment:
        to_drawio = _image_to_drawio(review.alignment)
        model_only = [s for s in review.symbols if s.id not in shapes and s.source == "image model" and s.box]
        boxes += [Box(s.id, to_drawio(s.box), MODEL_ONLY, f"model: {s.detected_as} {s.tag}".strip(), 3,
                      _symbol_tip(s), True) for s in model_only]
        ocr = _ocr_boxes(review, to_drawio)
        boxes += ocr
        extra_legend = [(f"found only by the model ({len(model_only)})", MODEL_ONLY),
                        (f"OCR tag not on any symbol ({len(ocr)})", OCR_TAG)]
    names = {s.id: s.tag or s.dexpi_class for s in review.symbols}
    layers = build_layers(review.drawio, path, boxes=boxes, overrides=review_overrides(review), names=names)
    if layers is not None:
        layers.legend += extra_legend
    return layers


def picture_layers(review: ReviewDiagram) -> Layers | None:
    """The uploaded picture with the symbol boxes and connections as rows say (straight
    lines between box centres - the model has no pipe routes), plus OCR tags no symbol
    claimed."""
    if not review.image_path or not Path(review.image_path).exists():
        return None
    boxes = [Box(s.id, tuple(s.box), MODEL_ONLY if s.image_check.startswith("only in image") else symbol_color(s, set()),
                 s.tag or (f"model: {s.detected_as}" if s.source == "image model" else s.dexpi_class), 3,
                 _symbol_tip(s), s.image_check.startswith("only in image"))
             for s in review.symbols if s.box]
    ocr = _ocr_boxes(review)
    name = {s.id: s.tag or s.id for s in review.symbols}
    conns = [(c.id, c.from_id, c.to_id, c.line_type, c.include,
              f"{name.get(c.from_id, c.from_id)} - {name.get(c.to_id, c.to_id)}: {c.line_type} ({c.source})")
             for c in review.connections]
    layers = image_layers(review.image_path, conns, boxes + ocr)
    layers.legend.append((f"OCR tag not on any symbol ({len(ocr)})", OCR_TAG))
    return layers


# --------------------------------------------------------------------------- tables ---
SYMBOL_COLS = ["show", "include", "id", "tag", "dexpi_class", "detected_as", "identified_by", "tag_source",
               "confidence", "image_check", "attributes", "notes", "source"]
def _keep_specs(old: dict, new: dict) -> dict:
    """Recomputed tag attributes, keeping the drawing's spec notes ("18 bar")."""
    return {**new, **({"specs": old["specs"]} if old and old.get("specs") else {})}


CONN_COLS = ["show", "include", "id", "from_id", "to_id", "flow", "line_type", "source", "confidence", "notes"]

left, right = st.columns([3, 2])
with right:
    only_flagged = st.toggle("Show only rows that need review", value=False)
    # "show" ticks highlight rows on the map. They are seeded only when the editors are
    # re-created (a map click), so the editors' input stays put between reruns.
    seed_syms, seed_conns = st.session_state.get("seed_show", (set(), set()))
    sym_df = pd.DataFrame([{**asdict(s), "show": s.id in seed_syms,
                            "attributes": json.dumps(s.attributes) if s.attributes else ""}
                           for s in review.symbols], columns=SYMBOL_COLS + ["box"])
    flagged_ids = {s.id for s in review.symbols if needs_review(s)}
    st.metric("Symbols", len(review.symbols), help="rows below")
    c1, c2, c3 = st.columns(3)
    c1.metric("Included", sum(s.include and s.dexpi_class != SKIP for s in review.symbols))
    c2.metric("Need review", len(flagged_ids))
    c3.metric("Connections", sum(c.include for c in review.connections))
    if st.button("Re-classify image symbols from their (edited) tags",
                 help="Re-runs the tag-prefix classifier for model-detected rows. Rows you set to a class "
                      "by hand are re-classified too - use before editing classes."):
        for s in review.symbols:
            if s.source == "image model":
                mapping, how = classify_detection(s.detected_as, s.tag or None)
                s.dexpi_class = OFF_PAGE if s.detected_as == "inlet/outlet" and not mapping.dexpi_class_name \
                    else (mapping.dexpi_class_name or SKIP)
                s.identified_by, s.notes = how, mapping.reason
                s.attributes = _keep_specs(s.attributes, _instrument_attributes(s.dexpi_class, s.tag))
        st.session_state["editor_version"] += 1
        st.rerun()

st.markdown("#### Symbols")
st.caption("Edit tag, DEXPI class, notes or include. To add a symbol the model missed, add a row "
           "at the bottom (id is generated if left empty; a position on the image is not needed for "
           "DEXPI). Untick include to drop a false detection.")
shown = sym_df[sym_df["id"].isin(flagged_ids)] if only_flagged else sym_df
edited_syms = st.data_editor(
    shown.drop(columns=["box"]),
    key=f"sym_editor_{st.session_state['editor_version']}_{only_flagged}",
    num_rows="dynamic", width="stretch", hide_index=True,
    column_config={
        "show": st.column_config.CheckboxColumn("show", width="small", help="Highlight on the map"),
        "include": st.column_config.CheckboxColumn("include", width="small"),
        "id": st.column_config.TextColumn("id", width="small"),
        "tag": st.column_config.TextColumn("tag"),
        "dexpi_class": st.column_config.SelectboxColumn("DEXPI class", options=DEXPI_CLASS_CHOICES, required=True),
        "detected_as": st.column_config.TextColumn("detected as", disabled=True),
        "identified_by": st.column_config.TextColumn("identified by", disabled=True),
        "tag_source": st.column_config.TextColumn("tag from", disabled=True),
        "confidence": st.column_config.NumberColumn("conf.", format="%.2f", disabled=True),
        "image_check": st.column_config.TextColumn("image check", disabled=True),
        "attributes": st.column_config.TextColumn("attributes (from tag)", disabled=True),
        "notes": st.column_config.TextColumn("notes"),
        "source": st.column_config.TextColumn("source", disabled=True),
    },
)

st.markdown("#### Connections")
symbol_ids = [s.id for s in review.symbols]
conn_df = pd.DataFrame([{**r, "show": r["id"] in seed_conns} for r in review.connection_rows()], columns=CONN_COLS)
edited_conns = st.data_editor(
    conn_df, key=f"conn_editor_{st.session_state['editor_version']}", num_rows="dynamic",
    width="stretch", hide_index=True,
    column_config={
        "show": st.column_config.CheckboxColumn("show", width="small", help="Highlight on the map"),
        "include": st.column_config.CheckboxColumn("include", width="small"),
        "id": st.column_config.TextColumn("id", width="small"),
        "from_id": st.column_config.SelectboxColumn("from", options=symbol_ids),
        "to_id": st.column_config.SelectboxColumn("to", options=symbol_ids),
        "flow": st.column_config.SelectboxColumn(
            "flow from -> to", options=["drawn arrow", "implied", "reviewer", "no flow between them", ""],
            help="How the direction from -> to is known; to reverse a pipe swap from and to and pick 'reviewer'"),
        "line_type": st.column_config.SelectboxColumn("line type", options=["process", "signal"]),
        "source": st.column_config.TextColumn("source", disabled=True),
        "confidence": st.column_config.TextColumn("confidence", disabled=True),
        "notes": st.column_config.TextColumn("notes"),
    },
)


def apply_edits() -> ReviewDiagram:
    """Fold the editor tables back into the ReviewDiagram (rows not shown when filtering
    are kept unchanged; new rows become manual symbols/connections)."""
    by_id = {s.id: s for s in review.symbols}
    shown_ids = set(shown["id"])
    new_symbols, n_manual = [], 0
    for row in edited_syms.to_dict("records"):
        sid = str(row.get("id") or "").strip()
        if not sid:
            n_manual += 1
            sid = f"manual{len(review.symbols) + n_manual}"
        base = by_id.get(sid) or ReviewSymbol(id=sid, source="manual", box=None, detected_as="manual",
                                               identified_by="manual", tag_source="manual")
        base.include = bool(row.get("include", True))
        base.tag = str(row.get("tag") or "").strip()
        base.dexpi_class = row.get("dexpi_class") or SKIP
        base.notes = str(row.get("notes") or "")
        base.attributes = _keep_specs(base.attributes, _instrument_attributes(base.dexpi_class, base.tag))
        new_symbols.append(base)
    kept = [s for s in review.symbols if s.id not in shown_ids]  # filtered-out rows, unchanged
    review.symbols = kept + new_symbols
    conns = []
    for n, row in enumerate(edited_conns.to_dict("records")):
        if not row.get("from_id") or not row.get("to_id"):
            continue
        conns.append(ReviewConnection(
            id=str(row.get("id") or f"manual_c{n}"), from_id=row["from_id"], to_id=row["to_id"],
            line_type=row.get("line_type") or "process", source=row.get("source") or "manual",
            confidence=str(row.get("confidence") or ""), notes=str(row.get("notes") or ""),
            include=bool(row.get("include", True)), flow=str(row.get("flow") or ""),
        ))
    review.connections = conns
    return review


def _shown(df) -> set[str]:
    return {str(r["id"]) for r in df.to_dict("records") if r.get("show") and r.get("id")}


def select_from_map(click: dict) -> None:
    """A click on the map: tick exactly the clicked line's / box's rows, move them to the top
    of their table and re-create the editors (the edits so far are already in `review`)."""
    syms = {click["sym"]} if click.get("sym") else set()
    conns = set(click.get("rows") or [])
    review.symbols.sort(key=lambda s: s.id not in syms)
    review.connections.sort(key=lambda c: c.id not in conns)
    st.session_state["seed_show"] = (syms, conns)
    st.session_state["map_click_zoom"] = False  # the user is looking right at it: keep the view
    st.session_state["editor_version"] += 1
    st.rerun()


with left:
    edited = apply_edits()
    show_syms, show_conns = _shown(edited_syms), _shown(edited_conns)
    zoom = st.session_state.pop("map_click_zoom", True)
    with st.spinner("Tracing pipes and flow directions..."):
        maps = [(name, layers) for name, layers in (("Flow map", drawio_layers(edited)),
                                                     ("Picture", picture_layers(edited))) if layers]
    containers = st.tabs([n for n, _ in maps]) if len(maps) > 1 else [st.container()] * len(maps)
    for tab, (name, layers) in zip(containers, maps):
        with tab:
            result = map_component(data=map_markup(layers, show_conns, show_syms, zoom_to_highlight=zoom),
                                   key=f"map_{name}", on_clicked_change=lambda: None)
            if result.clicked:
                select_from_map(json.loads(result.clicked))
            st.caption("Boxes: green = converted · red = needs review · grey = excluded. "
                       + ("Pipes follow their drawn route; colour and arrowheads give the flow direction and how "
                          "it is known. " if name == "Flow map" else "Lines join box centres (the model has no "
                          "pipe routes). ")
                       + "Click anything to select its row in the tables; tick 'show' in a table to find it here.")
    if not maps:
        st.info("No picture to show - upload an image, or a draw.io file with the draw.io app installed "
                "for the flow map.")
    if review.unattached_tags:
        with st.expander(f"OCR tags not attached to any symbol ({len(review.unattached_tags)}) - "
                         "possible missed symbols"):
            st.dataframe(pd.DataFrame(review.unattached_tags), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------- build ---
st.divider()
b1, b2 = st.columns([1, 1])
with b2:
    st.download_button("Save review (.json) to continue later", review_to_json(review),
                       file_name=f"{Path(review.source_name).stem}_review.json", mime="application/json")
with b1:
    build_clicked = st.button("Build & validate DEXPI", type="primary")

if build_clicked:
    problems = validate_rows(review)
    if problems:
        st.error("Fix these before building:")
        for p in problems:
            st.write(f"- {p}")
        st.stop()
    with tempfile.TemporaryDirectory() as tmp:
        diagram, result = to_dexpi(review)
        stem = Path(review.source_name).stem
        try:
            save_and_verify(result.model, Path(tmp), stem)
        except Exception as exc:  # export or round-trip failure - show, don't hide
            st.error(f"DEXPI export/verification failed: {exc}")
            st.stop()
        xml_bytes = (Path(tmp) / f"{stem}.xml").read_bytes()
    report = report_for_diagram(diagram, result)
    cm = result.model.conceptualModel
    st.success("DEXPI model built, exported, and verified by re-loading it.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equipment", len(cm.taggedPlantItems))
    m2.metric("Instrument functions", len(cm.processInstrumentationFunctions))
    m3.metric("Pipe segments", sum(len(s.segments) for s in cm.pipingNetworkSystems))
    m4.metric("Skipped elements", report.skipped_count)
    d1, d2 = st.columns(2)
    d1.download_button("Download DEXPI XML", xml_bytes, file_name=f"{stem}.xml", mime="application/xml")
    with st.spinner("Preparing the check sheet..."):
        sheet = build_check_sheet(review, result, mode, diagram)
    d2.download_button(
        "Download check sheet for review (.xlsx)", sheet, file_name=f"{stem}_check_sheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="For a domain expert: before/after drawings, every item and connection in plain words, "
             "with Yes/No columns to fill in and a verdict that updates itself.",
    )

    st.markdown("#### Before and after")
    before_col, after_col = st.columns(2)
    with before_col:
        st.caption("Before: the original drawing")
        if review.image_path and Path(review.image_path).exists():
            st.image(Image.open(io.BytesIO(numbered_overlay(review.image_path, number_items(review, result)))),
                     width="stretch")
        else:
            st.info("No picture uploaded - compare with your draw.io drawing.")
    with after_col:
        st.caption("After: re-drawn only from the DEXPI file")
        try:
            st.image(Image.open(io.BytesIO(render_after_png(diagram, result))), width="stretch")
        except Exception as exc:
            st.warning(f"Could not re-draw from the DEXPI file: {exc}")
    if report.skipped_details:
        with st.expander(f"Skipped elements ({len(report.skipped_details)})"):
            for detail in report.skipped_details:
                st.text(detail)
