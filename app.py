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

The page: (1) the drawing as an interactive map - for a draw.io file every pipe along its
drawn route with its flow direction, for an image the model's boxes and lines - with an
edit panel beside it: click an item or line to edit it there, or step through the items to
check; (2) all items and lines as tables for bulk edits; (3) build, verify and download.
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

from pid_dexpi_aas.render.drawio_editor import INSTALLED as EDITOR_INSTALLED, drawio_editor  # noqa: E402
from pid_dexpi_aas.render.check_sheet import build_check_sheet, number_items, numbered_overlay, render_after_png  # noqa: E402
from pid_dexpi_aas.render.flow_map import (  # noqa: E402
    BOX_CHECK, BOX_IN, BOX_LEGEND, BOX_OUT, BOX_PICTURE, MAP_CSS, MAP_JS, OCR_TAG, Box, Layers, add_edits,
    Line, build_layers, hide, image_layers, map_markup, review_overrides,
)
from pid_dexpi_aas.drawio.decode import decode_drawio  # noqa: E402
from pid_dexpi_aas.aas.export import DexpiModelInput, aasx_bytes, check_aasx, drawing_svg, mapping_entries  # noqa: E402
from pid_dexpi_aas.dexpi.fields import prefill, spec_fields  # noqa: E402
from pid_dexpi_aas.render.routing import route  # noqa: E402
from pid_dexpi_aas.dexpi.mapping import classify_detection  # noqa: E402
from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved  # noqa: E402
from pid_dexpi_aas.review import (  # noqa: E402
    DEXPI_CLASS_CHOICES, FLOW_KNOWN, INSTRUMENT_CLASSES, OFF_PAGE, SKIP, VALVE_CLASSES, ReviewConnection, ReviewDiagram, ReviewSymbol,
    _instrument_attributes, _short_token, cross_check, from_drawio, from_image, to_dexpi, validate_rows,
)
from pid_dexpi_aas.dexpi.serialize import save_and_verify  # noqa: E402
from pid_dexpi_aas.dexpi.validate_by_counting import report_for_diagram  # noqa: E402

_ML_VENV = ROOT / "ml" / "relationformer" / ".venv"
ML_PYTHON = next((p for p in (_ML_VENV / "bin" / "python", _ML_VENV / "Scripts" / "python.exe") if p.exists()),
                 _ML_VENV / "bin" / "python")
INFER_SCRIPT = ROOT / "ml" / "pid2graph" / "pipeline" / "infer_sheet.py"
# The released model (ml/get_model.sh downloads it): the final all-data model when present,
# else the fold-1 one; PID_MODEL_CHECKPOINT points elsewhere.
DEFAULT_CHECKPOINT = os.environ.get("PID_MODEL_CHECKPOINT", str(next(
    (p for p in (ROOT / "ml" / "models" / "pid2graph_final.pt", ROOT / "ml" / "models" / "pid2graph_fold1.pt")
     if p.exists()), ROOT / "ml" / "models" / "pid2graph_final.pt")))
HAS_GPU = shutil.which("nvidia-smi") is not None
# Served to others (scripts/serve_public.sh): a password screen, and no model settings -
# visitors cannot point the app at files on this PC.
PUBLIC = os.environ.get("APP_PUBLIC") == "1"
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
CPU_MINUTES = "about 3 to 8 minutes"  # measured: ~5 min for an OPEN100 sheet with OCR on this PC's CPU
def cpu_reason(choice: str) -> str:
    if choice == "cpu":
        return "Pictures are set to be processed on the processor (Model settings)."
    if not HAS_GPU:
        return "This computer has no NVIDIA graphics card, so pictures are processed on the processor."
    return ("The graphics card is busy (a model is being trained on it), so pictures are processed on the "
            "processor instead.")
GPU_BUSY_MB = 1500  # another process holding this much GPU memory is a training run (desktop apps hold < 100)


@st.cache_data(ttl=20, show_spinner=False)
def gpu_busy() -> bool:
    """Is something else computing on the GPU? Free memory alone is no guide: a training
    run's use swings between ~2 and ~9 GB, so a moment with room says nothing."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return any(int(v) >= GPU_BUSY_MB for v in out.split() if v.strip().isdigit())
    except Exception:
        return True


def pick_device(choice: str) -> str:
    """"auto": the GPU unless something else (a training run) is computing on it - the app
    must not push that out of memory - else the CPU (minutes per sheet instead of seconds)."""
    if choice != "auto":
        return choice
    return "cuda" if HAS_GPU and not gpu_busy() else "cpu"
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="P&ID to DEXPI converter", layout="wide")

# ------------------------------------------------------------------------------- login ---
# Password screen when shared (scripts/serve_public.sh). Logged in stays logged in across
# reloads through a cookie: an expiry time signed with the password (so it cannot be made
# up, and a new password ends every login), moved 48 h ahead on every visit and every hour
# of use - after 48 h without use the password is asked again. Streamlit reads cookies but
# cannot set them; a tiny page script (AUTH_COOKIE_JS) does.
AUTH_COOKIE = "pid_dexpi_auth"
AUTH_IDLE_S = 48 * 3600
AUTH_RENEW_S = 3600
AUTH_COOKIE_JS = """
export default function ({ data }) {
  if (!data || !data.cookie) return;
  document.cookie = data.cookie + (location.protocol === 'https:' ? '; Secure' : '');
}
"""


def _auth_token(expiry: int) -> str:
    import hmac
    sig = hmac.new(APP_PASSWORD.encode(), f"pid-dexpi|{expiry}".encode(), "sha256").hexdigest()[:40]
    return f"{expiry}.{sig}"


def _auth_cookie_ok(value: str | None) -> bool:
    import hmac
    import time
    try:
        expiry = int((value or "").split(".", 1)[0])
    except ValueError:
        return False
    return expiry > time.time() and hmac.compare_digest(value, _auth_token(expiry))


if APP_PASSWORD:
    import time

    auth_cookie = st.components.v2.component("pid_auth_cookie", html="", js=AUTH_COOKIE_JS)
    if not st.session_state.get("authed") and _auth_cookie_ok(st.context.cookies.get(AUTH_COOKIE)):
        st.session_state["authed"] = True  # still within 48 h of the last visit
    if not st.session_state.get("authed"):
        import hmac
        st.markdown("### P&ID to DEXPI converter")
        with st.form("login"):
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Open"):
                if hmac.compare_digest(pw.encode(), APP_PASSWORD.encode()):
                    st.session_state["authed"] = True
                    st.rerun()
                st.error("Wrong password.")
        st.stop()
    # every visit, and every hour of use: 48 h more
    if time.time() - st.session_state.get("auth_renewed", 0) > AUTH_RENEW_S:
        expiry = int(time.time()) + AUTH_IDLE_S
        auth_cookie(data={"cookie": f"{AUTH_COOKIE}={_auth_token(expiry)}; Max-Age={AUTH_IDLE_S}; Path=/; SameSite=Strict"},
                    key=f"auth_cookie_{expiry}")
        st.session_state["auth_renewed"] = time.time()
st.markdown("### P&ID to DEXPI converter")
st.html("""<style>
/* the edit panel stays in view while scrolling the page */
div[data-testid="stColumn"]:has(.edit-panel) {position: sticky; top: 3.75rem; align-self: flex-start; z-index: 2;}
/* the panel's tabs fit its width */
div[data-testid="stColumn"]:has(.edit-panel) [role="tablist"] {gap: 0.75rem; flex-wrap: nowrap; overflow-x: auto;
  overflow-y: hidden; scrollbar-width: thin;}
div[data-testid="stColumn"]:has(.edit-panel) [role="tab"] {flex: 0 0 auto;}
div[data-testid="stColumn"]:has(.edit-panel) [role="tab"] p {font-size: 0.85rem; white-space: nowrap;}
div[data-testid="stMetric"] {padding: 0;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>""")

# ------------------------------------------------------------------------------ inputs ---
with st.sidebar:
    if APP_PASSWORD and st.button("Log out", type="tertiary", help="Forget this browser's login"):
        auth_cookie(data={"cookie": f"{AUTH_COOKIE}=; Max-Age=0; Path=/; SameSite=Strict"}, key="auth_cookie_logout")
        st.session_state["authed"] = False
        st.session_state["auth_renewed"] = 0
        st.info("Logged out - reload the page.")
        st.stop()
    st.header("Input")
    in_drawio, in_picture, in_editor = st.tabs(["draw.io file", "Picture", "Editor"])
    with in_drawio:
        uploaded_drawio = st.file_uploader("draw.io file", type=["drawio", "xml"])
    with in_picture:
        image_file = st.file_uploader("P&ID image", type=["png", "jpg", "jpeg"])
        st.caption("With a draw.io file as well, the drawing is checked against the picture.")
    with in_editor:
        if not EDITOR_INSTALLED:
            st.caption("The editor is not installed on this server (scripts/get_drawio_editor.py).")
        else:
            st.caption("Draw a P&ID, or change the uploaded draw.io file - or a picture's result - in draw.io "
                       "right here. **Save** (Ctrl+S) in the editor loads it into the app.")
            st.toggle("Show the editor", key="editor_open")
            if st.session_state.get("editor_xml"):
                st.download_button("Download the edited drawing", st.session_state["editor_xml"].encode(),
                                   file_name=st.session_state["editor_name"], mime="application/xml",
                                   width="stretch")
                if st.button("Discard the edits", width="stretch",
                             help="Back to the uploaded draw.io file (or nothing)"):
                    st.session_state.pop("editor_xml")
                    st.rerun()
    if PUBLIC:
        checkpoint, device = DEFAULT_CHECKPOINT, "auto"
    else:
        with st.expander("Model settings"):
            checkpoint = st.text_input("Model checkpoint", DEFAULT_CHECKPOINT)
            device = st.selectbox("Device", ["auto", "cuda", "cpu"] if HAS_GPU else ["cpu", "cuda"],
                                  help="auto: the GPU unless a training run is using it, else the CPU. CPU works (no NVIDIA GPU "
                                       "needed) but takes minutes per sheet.")
    if pick_device(device) == "cpu":
        st.warning(f"**Pictures are slow right now.** {cpu_reason(device)} "
                   f"A picture takes **{CPU_MINUTES}**. draw.io files are not affected.")
    st.divider()
    st.header("Saved review")
    review_upload = st.file_uploader("Resume a saved review (.json)", type=["json"])
    st.divider()
    st.header("Training data")
    with st.expander("Add a drawing labelled in draw.io"):
        st.caption("1. Upload a picture above, check it, and use 'Download for labelling (.drawio)' at the bottom. "
                   "2. Correct boxes, lines and dashed / solid in draw.io (app.diagrams.net works). "
                   "3. Upload the corrected file here to add it to the model's training data.")
        label_upload = st.file_uploader("Corrected .drawio", type=["drawio", "xml"], key="label_upload")
        show_training = st.checkbox("Show the drawings added so far", key="show_training")



class EditedDrawing:
    """A drawing saved in the Editor tab, standing in for an uploaded .drawio file."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name, self._data = name, data

    def getvalue(self) -> bytes:
        return self._data


# The Editor tab: draw.io on the page. It starts from the uploaded draw.io file (or a blank
# page); each Save replaces the drawing the app works on, until a new file is uploaded.
upload_id = hashlib.sha256(uploaded_drawio.getvalue()).hexdigest() if uploaded_drawio else ""
if st.session_state.get("editor_base") != upload_id:
    st.session_state.update(editor_base=upload_id)
    st.session_state.pop("editor_xml", None)
if st.session_state.get("editor_open") and EDITOR_INSTALLED:
    from pid_dexpi_aas.drawio.editor_data import editor_patch
    from pid_dexpi_aas.render.drawio_editor import write_schema
    st.markdown("### draw.io editor")
    st.caption("Draw with the P&ID libraries on the left; set each item's and line's DEXPI details in the panel on "
               "the right. **Save** (Ctrl+S) loads the drawing into the check below. The side panel below and this "
               "editor stay in step: details changed in either show up in both.")
    write_schema()
    # each cell's details from the review as the last run left it (side-panel edits included);
    # a Save rebuilds the review further down, which reruns once to bring them here
    prev = st.session_state.get("review")
    if prev is not None and prev.drawio is None and prev.image_path and not uploaded_drawio:
        # a picture review: the model's result drawn over the picture, rebuilt from the review each
        # time the editor is switched on (so map edits made meanwhile are in it); a Save updates the rows
        from pid_dexpi_aas.drawio.picture_editor import apply_editor_drawing, drawing_ids, review_to_editor_xml
        from pid_dexpi_aas.render.drawio_editor import picture_url
        doc = st.session_state.get("editor_pic_doc")
        if (doc is None or doc["key"] != st.session_state.get("input_key")
                or not st.session_state.get("editor_was_open")):
            xml = review_to_editor_xml(prev, picture_url(prev.image_path))
            doc = {"key": st.session_state.get("input_key"), "xml": xml, "ids": drawing_ids(xml)}
            st.session_state["editor_pic_doc"] = doc
        st.session_state["editor_was_open"] = True
        saved = drawio_editor(doc["xml"], editor_patch(prev), title=prev.source_name, mode="picture")
        if saved and saved.get("t") != st.session_state.get("editor_t"):  # a new Save
            # the rows the editor was showing (it adds the side panel's new ones as they come)
            changes = apply_editor_drawing(prev, saved["xml"], set(saved.get("ids") or doc["ids"]))
            doc.update(xml=saved["xml"], ids=drawing_ids(saved["xml"]))
            st.session_state["editor_t"] = saved["t"]
            st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
            done = ", ".join(f"{v} {k}" for k, v in changes.items() if v)
            st.toast(f"Loaded from the editor: {done or 'details updated'}.")
            st.rerun()
    else:
        saved = drawio_editor(st.session_state.get("editor_xml") or (uploaded_drawio.getvalue().decode("utf-8", "replace")
                                                                   if uploaded_drawio else ""),
                              editor_patch(prev) if prev is not None and prev.drawio is not None else {},
                              title=uploaded_drawio.name if uploaded_drawio else "new drawing")
        if saved and saved.get("t") != st.session_state.get("editor_t"):  # a new Save
            stem = Path(uploaded_drawio.name).stem if uploaded_drawio else "drawing"
            xml = saved["xml"]
            if not xml.lstrip().startswith("<mxfile"):  # embed mode saves the bare graph: make it a .drawio file
                xml = f'<mxfile host="pid-dexpi-aas"><diagram id="page-1" name="Page-1">{xml}</diagram></mxfile>'
            st.session_state.update(editor_t=saved["t"], editor_xml=xml, editor_name=f"{stem} (edited).drawio")
            st.rerun()
else:
    st.session_state["editor_was_open"] = False
drawio_file = (EditedDrawing(st.session_state["editor_name"], st.session_state["editor_xml"].encode())
               if st.session_state.get("editor_xml") else uploaded_drawio)

DATA_ROOT = ROOT / "data" / "PID2Graph"
if label_upload is not None or show_training:
    from pid_dexpi_aas.image.drawio_labels import drawio_to_labels, preview_png, safe_name
    from pid_dexpi_aas.image.training_export import (dir_writer, list_reviewed, normalise_scale, remove_reviewed,
                                                     write_sheet)
    st.markdown("### Training data from labelled drawings")
    if label_upload is not None:
        try:
            picture, t_syms, t_edges, t_notes = drawio_to_labels(label_upload.getvalue().decode("utf-8"))
        except Exception as exc:  # not a labelling file, a damaged file, a picture too large
            st.error(f"This file can't be read as a labelled drawing: {exc}. Use a file downloaded with "
                     "'Download for labelling (.drawio)' and saved from draw.io.")
            st.stop()
        name = safe_name(label_upload.name.removesuffix(".drawio").removesuffix("_labels"))
        classes = {c: sum(1 for _, k, _ in t_syms if k == c) for c in sorted({k for _, k, _ in t_syms})}
        st.markdown(f"**{name}** - {len(t_syms)} items ({', '.join(f'{v} {k}' for k, v in classes.items())}), "
                    f"{len(t_edges)} lines ({sum(1 for e in t_edges if e[2] == 'non-solid')} dashed)")
        for note in t_notes:
            st.warning(note)
        st.image(preview_png(picture, t_syms, t_edges), caption="The labels as they will be learned: boxes coloured "
                 "by class (green valve, blue instrument, orange tank, purple pump, grey general, brown inlet/outlet, "
                 "pink arrow), red lines along their paths (dashed = signal). Fix anything wrong in draw.io first.")
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        use = c1.radio("Use it for", ["Training", "Held-out test (never trained on)"], horizontal=True, key="t_use",
                       help="Held-out: a fixed set of unseen drawings every model is scored on. Keep a drawing in "
                            "one of the two, never both.")
        subset = "Reviewed" if use == "Training" else "Reviewed test"
        existing = {(d["name"], d["subset"]) for d in list_reviewed(DATA_ROOT)}
        other = "Reviewed test" if subset == "Reviewed" else "Reviewed"
        if (name, other) in existing:
            st.error(f"'{name}' is already added as {'held-out test' if other.endswith('test') else 'training'} data - "
                     "remove it there first (a drawing must not be in both).")
        elif c2.button("Replace it" if (name, subset) in existing else "Add to the training data", type="primary"):
            pic, sy, ed, scale = normalise_scale(picture, t_syms, t_edges)
            tiles = write_sheet(dir_writer(DATA_ROOT), pic, name, subset, sy, ed)
            st.success(f"Added '{name}' as {'held-out test' if subset.endswith('test') else 'training'} data: "
                       f"{tiles} tiles (scaled x{scale:.2f} to the model's symbol size).")
    added = list_reviewed(DATA_ROOT)
    st.markdown(f"**Added so far** ({len(added)})")
    if not added:
        st.caption("None yet.")
    for d in added:
        r1, r2, r3 = st.columns([4, 2, 1], vertical_alignment="center")
        r1.write(d["name"])
        r2.caption(f"{'held-out test' if d['subset'].endswith('test') else 'training'} · {d['tiles']} tiles")
        if r3.button("Remove", key=f"rm_{d['subset']}_{d['name']}"):
            remove_reviewed(DATA_ROOT, d["subset"], d["name"])
            st.rerun()
    st.caption("Training uses these after 'Patched/Reviewed' is added to the training config "
               "(DATA.EXTRA_TRAIN_SUBSETS); held-out drawings are scored with eval_sheets.py --set reviewed.")
    st.stop()

if drawio_file is None and image_file is None and review_upload is None:
    st.info("Upload a draw.io file and/or a P&ID image in the sidebar - or draw one in the Editor tab - "
            "to get started.")
    st.stop()

WORK = Path(tempfile.gettempdir()) / "pid_dexpi_app"
WORK.mkdir(exist_ok=True)


def _digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:16]


# Bump when the model's input handling changes, so results cached by an older version
# are redone (v2: dark-theme exports keep their light lines - sheet_image.py; v3: traced pipes).
INFER_VERSION = b"v4"


def _image_done(image_bytes: bytes, checkpoint: str) -> bool:
    """Already processed (no wait)?"""
    return (WORK / f"{_digest(image_bytes, checkpoint.encode(), INFER_VERSION)}.json").exists()


@st.cache_data(show_spinner=False)
def run_image_model(image_bytes: bytes, image_name: str, checkpoint: str, device: str) -> dict:
    # the result does not depend on the device, so it is not part of the key
    key = _digest(image_bytes, checkpoint.encode(), INFER_VERSION)
    img_path = WORK / f"{key}_{Path(image_name).name}"
    out_path = WORK / f"{key}.json"
    img_path.write_bytes(image_bytes)
    if not out_path.exists():
        try:
            import fcntl
        except ImportError:  # Windows: no lock (a single local user there)
            fcntl = None
        # one image at a time: two users at once would hold the model twice
        with open(WORK / "infer.lock", "w") as lock:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            if not out_path.exists():
                proc = subprocess.run(
                    [str(ML_PYTHON), str(INFER_SCRIPT), "--image", str(img_path), "--checkpoint", checkpoint,
                     "--out", str(out_path), "--device", pick_device(device)],
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
        on = pick_device(device)
        note = st.empty()
        if on == "cpu" and not _image_done(image_file.getvalue(), checkpoint):
            note.warning(f"**This will take {CPU_MINUTES}** - please keep this tab open. "
                         f"{cpu_reason(device)} If someone else's picture is being "
                         "processed, yours starts right after it. The page fills in when it is done.")
        with st.spinner("Finding symbols, lines and tags in the picture "
                        + ("(about 15 seconds)..." if on == "cuda" else f"on the processor ({CPU_MINUTES})..."),
                        show_time=True):
            try:
                infer = run_image_model(image_file.getvalue(), image_file.name, checkpoint, on)
            except RuntimeError as exc:
                st.error("Image model failed:")
                st.code(str(exc))
                st.stop()
        note.empty()
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
    if st.session_state.get("editor_open"):
        st.rerun()  # the editor at the top takes each cell's details from this new review
review: ReviewDiagram = st.session_state["review"]

mode = " + ".join(x for x, f in (("draw.io", drawio_file), ("image", image_file)) if f) or "saved review"
st.caption(f"**{review.source_name}** · input: {mode}" + (
    f" · drawing and picture matched on {review.alignment['pairs_used']} shared tags "
    f"(off by {review.alignment['median_error_px']:.0f} px on average)" if review.alignment else ""))
if review.moved:
    st.warning(f"The image and the draw.io file look like different versions of the drawing: "
               f"{', '.join(review.moved)} {'is' if len(review.moved) == 1 else 'are'} drawn at a different place in "
               f"each. Boxes there won't line up - use matching files, or treat the image checks around them "
               f"with care.")
if image_file is not None and drawio_file is None:
    st.info("Image only: symbols from the model, tags from OCR, pipes traced along the drawn lines "
            "(flow from arrowheads). Pipes the tracer could not follow come from the model's predictions - "
            "dashed orange on the map; check those first.")

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
        return BOX_OUT
    if needs_review(s):
        return BOX_CHECK
    return BOX_IN


def _left_out(s: ReviewSymbol) -> bool:
    return not s.include or s.dexpi_class == SKIP


def _box_kind(s: ReviewSymbol) -> str:
    """Toolbar layer of an item box (BOX_LEGEND's chips)."""
    return "box-out" if _left_out(s) else "box-check" if needs_review(s) else "box-in"


MODEL_ONLY = BOX_PICTURE


def _symbol_tip(s: ReviewSymbol) -> str:
    tip = f"{item_name(s.id)} ({refs.get(s.id, '')}): {friendly_class(s.dexpi_class)}, {status(s)}"
    if s.tag_source.startswith("OCR"):
        tip += f"; tag read in the picture ({s.tag_source})"
    return tip + (f"; picture check: {s.image_check}" if s.image_check else "")


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
                f"Text read in the picture: '{t['tag']}' (confidence {t.get('conf', 0):.0%}) - not on any item", True, "ocr")
            for t in review.unattached_tags or [] if t.get("box")]


def drawio_layers(review: ReviewDiagram) -> Layers | None:
    """The draw.io flow map: every pipe along its drawn route with its flow direction
    (pid_dexpi_aas.render.flow_map) - automatic, reflects the table edits. In both-inputs mode it
    also shows what the image adds: symbols only the model found (dashed orange) and OCR
    tags nothing claimed (dashed grey), mapped onto the drawing via the tag alignment."""
    if drawio_file is None or review.drawio is None:
        return None
    _, path = decode_drawio_bytes(drawio_file.getvalue(), drawio_file.name)
    shapes = {sh.id: sh for sh in review.drawio.shapes}
    # names only on items in (or to check for) DEXPI; left-out shapes and picture-only finds
    # keep theirs in the hover text, so labels don't pile up
    boxes = [Box(s.id, (sh.x, sh.y, sh.x + sh.width, sh.y + sh.height), symbol_color(s, set()),
                 item_name(s.id) if status(s) != "left out" else "", 3, _symbol_tip(s), _left_out(s), _box_kind(s))
             for s in review.symbols if (sh := shapes.get(s.id)) is not None]
    extra_legend = []
    drawn = [s for s in review.symbols if s.source == "manual" and s.box and s.id not in shapes]
    to_d = _image_to_drawio(review.alignment) if review.alignment and review.image_path else (lambda bx: tuple(bx))
    boxes += [Box(s.id, to_d(s.box), symbol_color(s, set()), item_name(s.id), 3, _symbol_tip(s) + " (drawn by you)",
                  _left_out(s), _box_kind(s)) for s in drawn]
    if review.alignment:
        to_drawio = _image_to_drawio(review.alignment)
        model_only = [s for s in review.symbols if s.id not in shapes and s.source == "image model" and s.box]
        boxes += [Box(s.id, to_drawio(s.box), MODEL_ONLY, "", 3,
                      f"Found only in the picture: {s.tag or s.detected_as} - {_symbol_tip(s)}", True, "model")
                  for s in model_only]
        ocr = _ocr_boxes(review, to_drawio)
        boxes += ocr
        extra_legend = [(f"item found only in the picture ({len(model_only)})", MODEL_ONLY, "model", "box"),
                        (f"text read in the picture, not on any item ({len(ocr)})", OCR_TAG, "ocr", "box")]
    names = {s.id: item_name(s.id) for s in review.symbols}
    layers = build_layers(review.drawio, path, boxes=boxes, overrides=review_overrides(review), names=names)
    if layers is not None:
        layers.legend = BOX_LEGEND + layers.legend + extra_legend
        # lines added or re-pointed in the panel/tables: drawn between their new ends
        original = {e.id: e for e in review.drawio.edges}
        edits, replaced = [], set()
        for c in review.connections:
            e = original.get(c.id)
            if not c.include or (e is not None and {c.from_id, c.to_id} == {e.source, e.target}):
                continue
            if e is not None:
                replaced.add(c.id)
            edits.append((c.id, c.from_id, c.to_id, 1 if c.flow in FLOW_KNOWN else 0,
                          f"{item_name(c.from_id)} -> {item_name(c.to_id)}: line you "
                          f"{'changed' if e is not None else 'added'}"))
        add_edits(layers, edits, replaced)
    return layers


AGREE = (0, 140, 70)
ADDED_BY_YOU = (200, 0, 160)
DRAWING_ONLY = (40, 100, 220)
PICTURE_ONLY = BOX_PICTURE
TYPE_DIFFERS = BOX_CHECK
FLOW_BASIS = {"drawn arrow": "arrow", "implied": "implied", "reviewer": "reviewer"}


def _to_px(review: ReviewDiagram):
    """draw.io units -> picture pixels (review.align_drawio_to_image's fit)."""
    (ax, bx), (ay, by) = review.alignment["x"], review.alignment["y"]
    return lambda p: (ax * p[0] + bx, ay * p[1] + by)


def _picture_rows(review: ReviewDiagram):
    """Picture item id -> the review row it stands for (its matched draw.io item, else
    itself when it is a picture-only row), and draw.io pair -> draw.io line row."""
    pic = review.picture
    row_of = {m: d for d, m in pic["match"].items()}
    rows = {s.id for s in review.symbols}
    item_row = {s.id: row_of.get(s.id, s.id if s.id in rows else "") for s in pic["symbols"]}
    drawing_pairs = {frozenset((c.from_id, c.to_id)): c for c in review.connections
                     if c.source == "draw.io" and c.include}
    return item_row, drawing_pairs


def hybrid_picture_layers(review: ReviewDiagram) -> Layers:
    """Both inputs: what the PICTURE alone shows - the model's items and the pipes traced on
    it - nothing from the drawing. Clicking selects the matching review row."""
    pic = review.picture
    item_row, drawing_pairs = _picture_rows(review)
    boxes = [Box(item_row[s.id], tuple(s.box), BOX_IN, s.tag or s.detected_as, 3,
                 f"Found in the picture: {s.tag or '(no tag)'} - {s.detected_as}"
                 + (f"; matches {item_name(item_row[s.id])} in the drawing" if item_row[s.id] in pic["match"] else
                    "; not in the drawing"), False, "pic-item")
             for s in pic["symbols"] if s.box]
    conns = []
    for c in pic["connections"]:
        a, b = item_row.get(c.from_id) or c.from_id, item_row.get(c.to_id) or c.to_id
        row = drawing_pairs.get(frozenset((a, b)))
        conns.append((row.id if row else c.id, c.from_id, c.to_id, c.line_type, True,
                      f"{c.line_type} {('traced on the picture' if c.path else 'predicted by the model')}"
                      + ("; also in the drawing" if row else "; not in the drawing"),
                      c.path, FLOW_BASIS.get(c.flow, "")))
    boxes += [Box(s.id, tuple(s.box), symbol_color(s, set()), item_name(s.id), 3, _symbol_tip(s) + " (drawn by you)",
                  False, _box_kind(s)) for s in review.symbols if s.source == "manual" and s.box]
    raw = [Box(s.id, tuple(s.box), BOX_IN, "", 3) for s in pic["symbols"] if s.box]  # centres by picture id
    layers = image_layers(review.image_path, conns, raw)
    for ln in layers.lines:  # clicking a picture line selects its drawing row, when there is one
        ln.rows = {r for r in ln.rows if r in {c.id for c in review.connections}}
    layers.boxes = boxes
    layers.legend = [("item found in the picture", BOX_IN, "pic-item", "box")] + layers.legend
    return layers


def _densify(pts, step: float = 8.0) -> list:
    """Points every `step` px along a polyline."""
    out = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        n = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // step))
        out += [(x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n) for k in range(n)]
    return out + list(pts[-1:])


def _near_index(points, cell: float = 12.0) -> set:
    """Grid cells holding a point, with their neighbours: 'within ~12 px of a line'."""
    cells = set()
    for x, y in points:
        cx, cy = int(x // cell), int(y // cell)
        cells.update((cx + i, cy + j) for i in (-1, 0, 1) for j in (-1, 0, 1))
    return cells


def _covered(pts, index, cell: float = 12.0) -> float:
    """Share of a polyline's length lying near the other source's lines."""
    dense = _densify(pts)
    return sum((int(x // cell), int(y // cell)) in index for x, y in dense) / max(len(dense), 1)


def compare_layers(review: ReviewDiagram) -> tuple[Layers, dict] | None:
    """Both inputs, laid over each other on the picture and coloured by agreement:
    green both agree, blue only in the drawing, brown dashed only in the picture, amber
    in both but the picture sees another type. Lines are compared pair by pair."""
    if not review.picture or not review.alignment or not review.image_path:
        return None
    pic = review.picture
    item_row, drawing_pairs = _picture_rows(review)
    px = _to_px(review)
    centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    counts = {"items": {"agree": 0, "drawing": 0, "picture": 0, "type": 0, "added": 0},
              "lines": {"agree": 0, "drawing": 0, "picture": 0}}
    boxes = []
    for s in review.symbols:
        if not s.box or s.dexpi_class == SKIP:
            continue
        if s.source == "manual":
            kind, color, dashed = "added", ADDED_BY_YOU, False
            what = "drawn by you"
        elif s.source == "image model":
            kind, color, dashed = "picture", PICTURE_ONLY, True
            what = "only in the picture - missing from the drawing?"
        elif s.id not in pic["match"]:
            kind, color, dashed = "drawing", DRAWING_ONLY, False
            what = "only in the drawing - the picture shows nothing here"
        elif s.image_check.startswith("model agrees"):
            kind, color, dashed = "agree", AGREE, False
            what = "in the drawing and the picture"
        else:
            kind, color, dashed = "type", TYPE_DIFFERS, False
            what = f"in both, but {s.image_check}"
        counts["items"][kind] += 1
        boxes.append(Box(s.id, tuple(s.box), color, item_name(s.id) if kind != "agree" else "", 3,
                         f"{item_name(s.id)}: {what}", dashed, f"cmp-{kind}"))
    centres = {b.id: centre(b.box) for b in boxes}
    lines = []
    # Lines are compared by GEOMETRY, piece by piece: a drawn pipe piece is "in both" when
    # the picture traced a line along most of its length (and the other way round). Pair by
    # pair, a pipe network's every pair of ends would be drawn as a straight line - a fan
    # of diagonals over the sheet.
    traced = [(c, [tuple(p) for p in c.path]) for c in pic["connections"] if c.path]
    drawn = []  # (edge, points in picture px, review rows)
    net_rows: dict = {}
    for c in review.connections:
        if c.source == "draw.io" and c.include and c.id.startswith("net"):
            net_rows.setdefault(c.id.split(":", 1)[0], set()).add(c.id)
    row_ids = {c.id for c in review.connections if c.include}
    for e in (review.drawio.edges if review.drawio else []):
        r = (review.drawio.routes or {}).get(e.id)
        if not r or e.is_decorative or e.style in ("pipe network", "adjacent"):
            continue
        rows = net_rows.get(e.merged_into, set()) if e.merged_into else ({e.id} & row_ids)
        if rows:
            drawn.append((e, [px(p) for p in r], rows))
    near_traced = _near_index([p for _, pts in traced for p in _densify(pts)])
    near_drawn = _near_index([p for _, pts, _ in drawn for p in _densify(pts)])
    for e, pts, rows in drawn:
        both = _covered(pts, near_traced) >= 0.6
        counts["lines"]["agree" if both else "drawing"] += 1
        lines.append(Line(pts, AGREE if both else DRAWING_ONLY, 4 if both else 3, rows,
                          f"{'in the drawing and the picture' if both else 'only in the drawing - not traced on the picture'}",
                          dashed=e.is_dashed, kind="cmp-agree" if both else "cmp-drawing"))
    for c, pts in traced:
        if _covered(pts, near_drawn) >= 0.6:
            continue  # already shown green from the drawing's side
        a, b = item_row.get(c.from_id) or c.from_id, item_row.get(c.to_id) or c.to_id
        counts["lines"]["picture"] += 1
        lines.append(Line(pts, PICTURE_ONLY, 3, set(), f"{item_name(a)} - {item_name(b)}: traced on the picture, "
                          "not in the drawing - missing from it?", dashed=True, ends=(a, b), kind="cmp-picture"))
    for c in pic["connections"]:  # the model's untraced predictions: only where the drawing has no such pair
        if c.path:
            continue
        a, b = item_row.get(c.from_id) or c.from_id, item_row.get(c.to_id) or c.to_id
        if frozenset((a, b)) in drawing_pairs or a not in centres or b not in centres:
            continue
        counts["lines"]["picture"] += 1
        box_of = {bx.id: bx.box for bx in boxes}
        lines.append(Line(route(box_of[a], box_of[b], list(box_of.values())), PICTURE_ONLY, 2, set(),
                          f"{item_name(a)} - {item_name(b)}: "
                          "predicted by the model only (not traced), not in the drawing", dashed=True, ends=(a, b),
                          kind="cmp-picture"))
    ci, cl = counts["items"], counts["lines"]
    # chips toggle per agreement class (items and lines together): hide "in both" to see only the differences
    legend = [(f"in both: {ci['agree']} items, {cl['agree']} lines", AGREE, "cmp-agree", "box"),
              (f"only in the drawing: {ci['drawing']} items, {cl['drawing']} lines", DRAWING_ONLY, "cmp-drawing", "box"),
              (f"only in the picture: {ci['picture']} items, {cl['picture']} lines", PICTURE_ONLY, "cmp-picture", "box"),
              (f"in both, other type in the picture: {ci['type']} items", TYPE_DIFFERS, "cmp-type", "box")]
    if ci["added"]:
        legend.append((f"drawn by you: {ci['added']} items", ADDED_BY_YOU, "cmp-added", "box"))
    with Image.open(review.image_path) as im:
        size = im.size
    return Layers(Path(review.image_path), size, lines, boxes, counts, legend), counts


def picture_layers(review: ReviewDiagram) -> Layers | None:
    """The uploaded picture with the symbol boxes and connections as rows say (straight
    lines between box centres - the model has no pipe routes), plus OCR tags no symbol
    claimed."""
    if not review.image_path or not Path(review.image_path).exists():
        return None
    if review.picture:  # both inputs: the picture's own findings only
        return hybrid_picture_layers(review)
    only = lambda s: s.image_check.startswith("only in image")
    boxes = [Box(s.id, tuple(s.box), MODEL_ONLY if only(s) else symbol_color(s, set()),
                 "" if only(s) or status(s) == "left out" else item_name(s.id), 3,
                 _symbol_tip(s), only(s) or _left_out(s), "model" if only(s) else _box_kind(s))
             for s in review.symbols if s.box]
    ocr = _ocr_boxes(review)
    name = {s.id: item_name(s.id) for s in review.symbols}
    basis = {"drawn arrow": "arrow", "implied": "implied", "reviewer": "reviewer"}
    conns = [(c.id, c.from_id, c.to_id, c.line_type, c.include,
              f"{name.get(c.from_id, c.from_id)} -> {name.get(c.to_id, c.to_id)}: {LINE_LABELS.get(c.line_type, c.line_type)}"
              f", {found_by(c)}" + (f", flow: {FLOW_LABELS.get(c.flow, c.flow)}" if c.flow else ""),
              c.path, basis.get(c.flow, "")) for c in review.connections]
    layers = image_layers(review.image_path, conns, boxes + ocr)
    layers.legend = BOX_LEGEND + layers.legend + [
        (f"text read in the picture, not on any item ({len(ocr)})", OCR_TAG, "ocr", "box")]
    return layers


# ------------------------------------------------------------------- names & refs ---
# Short, stable references (#12 for items, L7 for lines) instead of draw.io's internal ids,
# so the map, the edit panel and the tables can all say which thing they mean.
refs: dict = st.session_state.setdefault("refs", {})
for s in review.symbols:
    refs.setdefault(s.id, f"#{sum(1 for r in refs.values() if r.startswith('#')) + 1}")
for c in review.connections:
    refs.setdefault(c.id, f"L{sum(1 for r in refs.values() if r.startswith('L')) + 1}")
by_sym = {s.id: s for s in review.symbols}


def friendly_class(cls: str) -> str:
    return {SKIP: "not in DEXPI", OFF_PAGE: "stream in / out (off-page)"}.get(cls, cls)


def item_name(sid: str) -> str:
    s = by_sym.get(sid)
    if s is None:
        return sid
    return s.tag or f"{friendly_class(s.dexpi_class)} {refs.get(sid, '')}".strip()


def item_option(sid: str) -> str:
    """A unique, readable label for every item picker: "#12 V101", "#55 CustomEquipment" -
    always the ref first, so items with the same name (or a new one with no tag yet) can be
    told apart, and the map can outline the entry being browsed (MAP_JS reads the "#12")."""
    s = by_sym.get(sid)
    if s is None:
        return sid
    return f"{refs.get(sid, '?')} {s.tag or friendly_class(s.dexpi_class)}"


def status(s: ReviewSymbol) -> str:
    if not s.include or s.dexpi_class == SKIP:
        return "left out"
    return "needs review" if needs_review(s) else "in DEXPI"


def found_as(s: ReviewSymbol) -> str:
    if s.source == "manual":
        return "added by you"
    if s.source == "image model":
        conf = f" ({s.confidence:.0%})" if s.confidence is not None else ""
        return f"model: {s.detected_as}{conf}"
    return f"draw.io: {_short_token(s.detected_as)}"


FLOW_LABELS = {"drawn arrow": "arrow on the drawing", "implied": "implied by other arrows",
               "reviewer": "set by you", "no flow between them": "no flow (same side of a tee)", "": "unknown"}
FLOW_VALUES = {v: k for k, v in FLOW_LABELS.items()}
LINE_LABELS = {"process": "process pipe", "signal": "signal line"}


def found_by(c: ReviewConnection) -> str:
    if c.source == "image model":
        return f"model prediction ({c.confidence})"
    if c.source == "picture tracing":
        return "traced on the picture"
    if c.source == "manual":
        return "added by you"
    return {"explicit": "draw.io: line snapped to both items", "junction": "draw.io: joined at a tee",
            "adjacent": "draw.io: compartments touching",
            "junction+guess": "draw.io: tee, one end guessed"}.get(c.confidence, "draw.io: one end guessed")


def _keep_specs(old: dict, new: dict) -> dict:
    """Recomputed tag attributes, keeping the drawing's spec notes ("18 bar") and the
    specification fields filled in by the reviewer."""
    return {**new, **{k: old[k] for k in ("specs", "dexpi", "custom") if old and old.get(k)}}


# ------------------------------------------------------------------------- selection ---
# The selection is what is ticked "selected" in the tables; a map click or the edit panel
# sets it by re-creating the tables with those rows ticked and moved to the top.
def select(syms=(), conns=(), zoom=False) -> None:
    syms, conns = set(syms), set(conns)
    review.symbols.sort(key=lambda s: s.id not in syms)
    review.connections.sort(key=lambda c: c.id not in conns)
    st.session_state["seed_show"] = (syms, conns)
    st.session_state["map_click_zoom"] = zoom
    st.session_state["editor_version"] += 1


def refresh_tables() -> None:
    """After an edit in the panel: rebuild the tables from `review`, keeping the selection."""
    st.session_state["seed_show"] = st.session_state.get("current_show", (set(), set()))
    st.session_state["editor_version"] += 1


def add_drawn_item(map_name: str, layers: Layers, box) -> None:
    """A box drawn on a map ('Add item') becomes a new item: its box in the review's
    coordinates (picture pixels when there is a picture, else draw.io units), a tag from text
    the OCR read inside it and no item claimed, and it opens in the panel to set its type."""
    x1, y1, x2, y2 = box
    if map_name.startswith("Drawing"):
        ox, oy = layers.origin
        b = [x1 - ox, y1 - oy, x2 - ox, y2 - oy]  # draw.io units
        if review.alignment and review.image_path:  # both inputs: rows live in picture pixels
            px = _to_px(review)
            (a1, b1), (a2, b2) = px((b[0], b[1])), px((b[2], b[3]))
            b = [min(a1, a2), min(b1, b2), max(a1, a2), max(b1, b2)]
    else:
        b = [x1, y1, x2, y2]  # picture pixels
    tag, tag_source = "", "manual"
    if review.image_path and review.unattached_tags:
        inside = [t for t in review.unattached_tags if t.get("box") and
                  b[0] <= (t["box"][0] + t["box"][2]) / 2 <= b[2] and b[1] <= (t["box"][1] + t["box"][3]) / 2 <= b[3]]
        if inside:
            best = max(inside, key=lambda t: t.get("conf", 0))
            tag, tag_source = best["tag"], "OCR inside"
            review.unattached_tags = [t for t in review.unattached_tags if t is not best]
    n = 1 + sum(1 for s in review.symbols if s.id.startswith("drawn"))
    item = ReviewSymbol(id=f"drawn{n}", source="manual", box=b, detected_as="drawn by you", tag=tag,
                        tag_source=tag_source, dexpi_class="CustomEquipment", identified_by="box drawn by you",
                        notes="added by drawing a box on the map - set its DEXPI type")
    item.attributes = _keep_specs({}, _instrument_attributes(item.dexpi_class, item.tag))
    review.symbols.append(item)
    select({item.id}, (), zoom=False)


seed_syms, seed_conns = st.session_state.get("seed_show", (set(), set()))
version = st.session_state["editor_version"]

# ---------------------------------------------------------------------------- layout ---
n_in = sum(status(s) == "in DEXPI" for s in review.symbols)
n_check = sum(status(s) == "needs review" for s in review.symbols)
n_out = sum(status(s) == "left out" for s in review.symbols)
n_conn = sum(c.include for c in review.connections)
fs = review.drawio.flow if review.drawio is not None and review.drawio.flow else None
n_unknown = fs.unknown if fs else sum(c.include and c.line_type == "process" and not c.flow for c in review.connections)
m = st.columns(5)
m[0].metric("Items in DEXPI", n_in)
m[1].metric("Items to check", n_check, help="Red on the map. Use 'Next to check' in the panel.")
m[2].metric("Items left out", n_out, help="Decorative shapes, labels, or excluded by you")
m[3].metric("Lines in DEXPI", n_conn)
m[4].metric("Pipes with unknown flow", n_unknown, help="Pipes (not instrument taps) with no arrow on them or "
            "their neighbours - grey on the map")

st.markdown("##### 1 · Check the drawing")
map_col, panel_col = st.columns([7, 3], gap="medium")
st.markdown("##### 2 · All items and lines (bulk edits)")
items_tab, lines_tab = st.tabs([f"Items ({len(review.symbols)})", f"Lines ({len(review.connections)})"])

# ---------------------------------------------------------------------------- tables ---
ITEM_FILTERS = ["All", "Selected", "To check", "In DEXPI", "Left out"]
with items_tab:
    f1, f2, f3 = st.columns([3, 2, 2])
    item_filter = f1.segmented_control("Show", ITEM_FILTERS, default="All", key="item_filter")
    section_names = sorted({s.section for s in review.symbols if s.section} |
                           {sec.name for sec in (review.drawio.sections if review.drawio else [])})
    section_filter = f3.selectbox("Plant section", ["All sections"] + section_names + (["(no section)"] if section_names else []),
                                  key="section_filter", disabled=not section_names,
                                  help="Frames drawn around groups of items on the drawing")
    if any(s.source == "image model" for s in review.symbols) and f2.button(
            "Re-classify model items from their tags",
            help="Re-runs the tag-prefix classifier (PI -> instrument, GV -> valve, ...) for items the model found."):
        for s in review.symbols:
            if s.source == "image model":
                mapping, how = classify_detection(s.detected_as, s.tag or None)
                s.dexpi_class = OFF_PAGE if s.detected_as == "inlet/outlet" and not mapping.dexpi_class_name \
                    else (mapping.dexpi_class_name or SKIP)
                s.identified_by, s.notes = how, mapping.reason
                s.attributes = _keep_specs(s.attributes, _instrument_attributes(s.dexpi_class, s.tag))
        refresh_tables()
        st.rerun()
    keep = {"All": lambda s: True, "Selected": lambda s: s.id in seed_syms,
            "To check": lambda s: status(s) == "needs review", "In DEXPI": lambda s: status(s) == "in DEXPI",
            "Left out": lambda s: status(s) == "left out"}[item_filter or "All"]
    in_section = (lambda s: True) if section_filter in (None, "All sections") else \
        (lambda s: not s.section) if section_filter == "(no section)" else (lambda s: s.section == section_filter)
    shown_syms = [s for s in review.symbols if keep(s) and in_section(s)]
    hybrid = any(s.image_check for s in review.symbols)
    item_df = pd.DataFrame([{
        "selected": s.id in seed_syms, "id": s.id, "ref": refs[s.id], "tag": s.tag, "dexpi_class": s.dexpi_class,
        "include": s.include, "status": ("● " if s.id in seed_syms else "") + status(s), "found_as": found_as(s),
        "section": s.section, "why": s.identified_by, "image_check": s.image_check,
        "specs": (s.attributes or {}).get("specs", ""), "notes": s.notes} for s in shown_syms],
        columns=["selected", "id", "ref", "tag", "dexpi_class", "include", "status", "found_as", "section", "why",
                 "image_check", "specs", "notes"])
    edited_items = st.data_editor(
        item_df, key=f"items_{version}_{item_filter}", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "selected": st.column_config.CheckboxColumn("Select", width="small",
                                                        help="Tick to select: highlighted on the map and opened in the panel"),
            "id": None,
            "ref": st.column_config.TextColumn("Ref", width="small", disabled=True,
                                               help="Short reference, also shown on the map"),
            "tag": st.column_config.TextColumn("Tag", help="The item's name on the drawing, e.g. GV201"),
            "dexpi_class": st.column_config.SelectboxColumn("DEXPI type", options=DEXPI_CLASS_CHOICES, required=True),
            "include": st.column_config.CheckboxColumn("In DEXPI", width="small",
                                                       help="Untick to leave this item out of the DEXPI file"),
            "status": st.column_config.TextColumn("Status", disabled=True),
            "found_as": st.column_config.TextColumn("Found as", disabled=True),
            "section": (st.column_config.SelectboxColumn("Plant section", options=[""] + section_names,
                                                         help="The frame it was drawn in; change it here")
                        if section_names else None),
            "why": st.column_config.TextColumn("How the type was chosen", disabled=True),
            "image_check": (st.column_config.TextColumn("Picture check", disabled=True) if hybrid else None),
            "specs": st.column_config.TextColumn("Specs from the drawing", disabled=True),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )

LINE_FILTERS = ["All", "Selected", "Unknown flow", "Guessed end", "Left out"]
with lines_tab:
    line_filter = st.segmented_control("Show", LINE_FILTERS, default="All", key="line_filter")
    keep_c = {"All": lambda c: True, "Selected": lambda c: c.id in seed_conns,
              "Unknown flow": lambda c: c.line_type == "process" and not c.flow,
              "Guessed end": lambda c: "guess" in found_by(c), "Left out": lambda c: not c.include}[line_filter or "All"]
    shown_conns = [c for c in review.connections if keep_c(c)]
    options = [item_option(s.id) for s in review.symbols]
    option_id = {item_option(s.id): s.id for s in review.symbols}
    line_df = pd.DataFrame([{
        "selected": c.id in seed_conns, "id": c.id, "ref": refs.get(c.id, ""),
        "from": item_option(c.from_id), "to": item_option(c.to_id), "flow": FLOW_LABELS.get(c.flow, "unknown"),
        "line_type": LINE_LABELS.get(c.line_type, c.line_type), "include": c.include, "found_by": found_by(c),
        "notes": c.notes} for c in shown_conns],
        columns=["selected", "id", "ref", "from", "to", "flow", "line_type", "include", "found_by", "notes"])
    edited_lines = st.data_editor(
        line_df, key=f"lines_{version}_{line_filter}", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "selected": st.column_config.CheckboxColumn("Select", width="small",
                                                        help="Tick to select: highlighted on the map and opened in the panel"),
            "id": None,
            "ref": st.column_config.TextColumn("Ref", width="small", disabled=True),
            "from": st.column_config.SelectboxColumn("From", options=options, help="Upstream end when the flow is known"),
            "to": st.column_config.SelectboxColumn("To", options=options),
            "flow": st.column_config.SelectboxColumn("Flow From -> To", options=list(FLOW_VALUES),
                                                     help="How the direction is known. To reverse a pipe, swap From "
                                                          "and To and pick 'set by you' (or use the panel's Reverse)."),
            "line_type": st.column_config.SelectboxColumn("Line", options=list(LINE_LABELS.values())),
            "include": st.column_config.CheckboxColumn("In DEXPI", width="small"),
            "found_by": st.column_config.TextColumn("Found by", disabled=True),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )


def apply_edits() -> ReviewDiagram:
    """Fold the tables back into the review: shown rows are updated (or dropped if deleted),
    rows hidden by a filter stay as they are, new rows become manual items/lines."""
    shown_ids = {s.id for s in shown_syms}
    rows = {str(r["id"]): r for r in edited_items.to_dict("records") if r.get("id")}
    out = []
    for s in review.symbols:
        if s.id not in shown_ids:
            out.append(s)
        elif s.id in rows:
            r = rows[s.id]
            s.include = bool(r.get("include", True))
            s.tag = str(r.get("tag") or "").strip()
            s.dexpi_class = r.get("dexpi_class") or SKIP
            s.notes = str(r.get("notes") or "")
            if "section" in r and r.get("section") is not None:
                s.section = str(r.get("section") or "")
            s.attributes = _keep_specs(s.attributes, _instrument_attributes(s.dexpi_class, s.tag))
            out.append(s)
    for n, r in enumerate(x for x in edited_items.to_dict("records") if not x.get("id")):
        sid = f"manual{len(review.symbols) + n + 1}"
        out.append(ReviewSymbol(id=sid, source="manual", box=None, detected_as="manual", identified_by="added by you",
                                tag_source="manual", tag=str(r.get("tag") or "").strip(),
                                dexpi_class=r.get("dexpi_class") or SKIP, include=bool(r.get("include", True)),
                                notes=str(r.get("notes") or "")))
    review.symbols = out
    shown_cids = {c.id for c in shown_conns}
    crows = {str(r["id"]): r for r in edited_lines.to_dict("records") if r.get("id")}
    outc = []
    for c in review.connections:
        if c.id not in shown_cids:
            outc.append(c)
        elif c.id in crows:
            r = crows[c.id]
            c.from_id = option_id.get(r.get("from"), c.from_id)
            c.to_id = option_id.get(r.get("to"), c.to_id)
            c.flow = FLOW_VALUES.get(r.get("flow"), "")
            c.line_type = {v: k for k, v in LINE_LABELS.items()}.get(r.get("line_type"), c.line_type)
            c.include = bool(r.get("include", True))
            c.notes = str(r.get("notes") or "")
            outc.append(c)
    for n, r in enumerate(x for x in edited_lines.to_dict("records") if not x.get("id")):
        a, b = option_id.get(r.get("from")), option_id.get(r.get("to"))
        if a and b:
            outc.append(ReviewConnection(id=f"manual_c{len(review.connections) + n + 1}", from_id=a, to_id=b,
                                         line_type={v: k for k, v in LINE_LABELS.items()}.get(r.get("line_type"), "process"),
                                         source="manual", flow=FLOW_VALUES.get(r.get("flow"), ""),
                                         include=bool(r.get("include", True)), notes=str(r.get("notes") or "")))
    review.connections = outc
    return review


edited = apply_edits()
by_sym = {s.id: s for s in review.symbols}
show_syms = {str(r["id"]) for r in edited_items.to_dict("records") if r.get("selected") and r.get("id")}
show_conns = {str(r["id"]) for r in edited_lines.to_dict("records") if r.get("selected") and r.get("id")}
# rows a filter hides keep their selection
show_syms |= {sid for sid in seed_syms if sid not in {s.id for s in shown_syms}}
show_conns |= {cid for cid in seed_conns if cid not in {c.id for c in shown_conns}}
st.session_state["current_show"] = (show_syms, show_conns)

# ------------------------------------------------------------------------------- map ---
with map_col:
    zoom = st.session_state.pop("map_click_zoom", True)
    with st.spinner("Tracing pipes and flow directions..."):
        cmp = compare_layers(edited)
        maps = [(name, layers) for name, layers in (("Compare drawing ↔ picture", cmp[0] if cmp else None),
                                                     ("Drawing (draw.io)", drawio_layers(edited)),
                                                     ("Picture (image)", picture_layers(edited))) if layers]
    containers = st.tabs([n for n, _ in maps]) if len(maps) > 1 else [st.container()] * len(maps)
    for tab, (name, layers) in zip(containers, maps):
        with tab:
            hidden_syms, hidden_rows = st.session_state.setdefault("hidden", (set(), set()))
            layers = hide(layers, hidden_syms, hidden_rows)
            result = map_component(data=map_markup(layers, show_conns, show_syms, zoom_to_highlight=zoom,
                                                   key=name, height=620,
                                                   names="off" if name.startswith("Drawing") else "auto",
                                                   refs={sid: r for sid, r in refs.items() if r.startswith("#")}),
                                   key=f"map_{name}", on_clicked_change=lambda: None, on_drawn_change=lambda: None)
            if result.drawn:
                add_drawn_item(name, layers, json.loads(result.drawn)["box"])
                st.rerun()
            if result.clicked:
                click = json.loads(result.clicked)
                select({click["sym"]} if click.get("sym") else (), click.get("rows") or ())
                st.rerun()
            st.caption({
                "Compare drawing ↔ picture": "Drawing and picture laid over each other: green = both agree, blue = "
                    "only in the drawing, brown dashed = only in the picture, amber = other type in the picture.",
                "Drawing (draw.io)": "What the draw.io file says: pipes along their drawn routes, coloured by how "
                    "their flow direction is known.",
                "Picture (image)": "What the picture shows: items the model found and pipes traced on it.",
            }.get(name, ""))
    if not maps:
        st.info("No picture to show - upload an image, or a draw.io file (with the draw.io app installed) "
                "for the flow map.")

# ---------------------------------------------------------------------- edit panel ---
def _set_item(sid: str, field: str, key: str) -> None:
    s = by_sym[sid]
    setattr(s, field, st.session_state[key])
    if field in ("tag", "dexpi_class"):
        s.attributes = _keep_specs(s.attributes, _instrument_attributes(s.dexpi_class, s.tag))
    refresh_tables()


def _set_line(cid: str, field: str, key: str) -> None:
    c = next(x for x in review.connections if x.id == cid)
    v = st.session_state[key]
    if field in ("from_id", "to_id"):
        v = option_id.get(v, getattr(c, field))
    elif field == "flow":
        v = FLOW_VALUES.get(v, "")
    elif field == "line_type":
        v = {b: a for a, b in LINE_LABELS.items()}.get(v, c.line_type)
    setattr(c, field, v)
    refresh_tables()


def _to_stream(cid: str, end: str, key: str) -> None:
    """One end of a line is a stream leaving or entering the drawing (product, vent, feed):
    a new off-page row named by the reviewer, placed where the line ends."""
    c = next(x for x in review.connections if x.id == cid)
    name = (st.session_state.get(key) or "").strip() or ("Product" if end == "to" else "Feed")
    keep = by_sym.get(c.from_id if end == "to" else c.to_id)
    route = (review.drawio.routes or {}).get(cid) if review.drawio else None
    if route and not c.id.startswith("net"):
        p = route[-1] if end == "to" else route[0]
        if review.alignment and review.image_path:
            p = _to_px(review)(p)
    elif keep is not None and keep.box:
        b = keep.box
        p = (b[2] + (b[2] - b[0]), (b[1] + b[3]) / 2)  # beside the item it comes from
    else:
        p = (0.0, 0.0)
    w = 60.0
    n = 1 + sum(1 for x in review.symbols if x.id.startswith("stream"))
    review.symbols.append(ReviewSymbol(
        id=f"stream{n}", source="manual", box=[p[0] - w / 2, p[1] - w / 4, p[0] + w / 2, p[1] + w / 4],
        detected_as="stream in/out", tag=name, tag_source="manual", dexpi_class=OFF_PAGE,
        identified_by="marked by you as a stream in / out", notes="set in the line panel"))
    if end == "to":
        c.to_id = f"stream{n}"
    else:
        c.from_id = f"stream{n}"
    c.flow = c.flow if c.flow in FLOW_KNOWN else "reviewer"  # From -> To: into / out of the drawing
    c.notes = "; ".join(x for x in (c.notes, f"{'to' if end == 'to' else 'from'} stream '{name}'") if x)
    select((), {cid}, zoom=True)


def _new_stream_id() -> str:
    n = 1 + sum(1 for x in review.symbols if x.id.startswith("stream"))
    while f"stream{n}" in by_sym:
        n += 1
    return f"stream{n}"


_OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
_ACROSS = {"left": ("top", "bottom"), "right": ("bottom", "top"), "top": ("left", "right"), "bottom": ("right", "left")}


def _side_towards(box, point) -> str:
    """Which side of `box` faces `point` (taking the box's shape into account)."""
    x1, y1, x2, y2 = box
    dx, dy = point[0] - (x1 + x2) / 2, point[1] - (y1 + y2) / 2
    if abs(dx) * max(y2 - y1, 1) >= abs(dy) * max(x2 - x1, 1):
        return "right" if dx > 0 else "left"
    return "bottom" if dy > 0 else "top"


def _stream_side(sid: str, out: bool) -> str:
    """Where a new stream goes on item `sid`: opposite its existing lines of the other
    direction (an inlet opposite the outlets, an outlet opposite the inlets); if that side
    is taken, a free side; with every side taken, the least used one. No lines of the other
    direction: outlets right, inlets left (the usual flow on a P&ID)."""
    s = by_sym[sid]
    used: dict = {}
    ins, outs = {}, {}
    for c in review.connections:
        if not c.include or sid not in (c.from_id, c.to_id) or c.from_id == c.to_id:
            continue
        other = by_sym.get(c.to_id if c.from_id == sid else c.from_id)
        if other is None or not other.box:
            continue
        side = _side_towards(s.box, ((other.box[0] + other.box[2]) / 2, (other.box[1] + other.box[3]) / 2))
        used[side] = used.get(side, 0) + 1
        if c.flow in FLOW_KNOWN:
            bucket = outs if c.from_id == sid else ins
            bucket[side] = bucket.get(side, 0) + 1
    against = ins if out else outs
    first = _OPPOSITE[max(against, key=against.get)] if against else ("right" if out else "left")
    order = [first, *_ACROSS[first], _OPPOSITE[first]]
    free = [x for x in order if not used.get(x)]
    return free[0] if free else min(order, key=lambda x: used.get(x, 0))


def _add_stream(sid: str, direction: str, key: str) -> None:
    """A stream into or out of the drawing at an item (feed, product, vent): a new off-page
    row beside it, on the side _stream_side picks (stacked next to anything already there),
    and a line with its direction set (DEXPI: FlowIn / FlowOut connector)."""
    s = by_sym[sid]
    out = direction == "out"
    name = (st.session_state.get(key) or "").strip() or ("Product" if out else "Feed")
    stream_id = _new_stream_id()
    box = None
    if s.box:
        x1, y1, x2, y2 = s.box
        w = max(60.0, 0.6 * (x2 - x1))
        h = w / 2
        side = _stream_side(sid, out)
        # streams already on that side of this item: stack the new one beside them
        here = [o for c in review.connections if c.include and sid in (c.from_id, c.to_id)
                for o in [by_sym.get(c.to_id if c.from_id == sid else c.from_id)]
                if o is not None and o.box and o.dexpi_class == OFF_PAGE
                and _side_towards(s.box, ((o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2)) == side]
        gap = max(40.0, 0.5 * max(x2 - x1, y2 - y1))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if side in ("left", "right"):
            bx = x2 + gap if side == "right" else x1 - gap - w
            by = cy - h / 2 + len(here) * 1.6 * h
        else:
            bx = cx - w / 2 + len(here) * 1.3 * w
            by = y2 + gap if side == "bottom" else y1 - gap - h
        box = [bx, by, bx + w, by + h]
    review.symbols.append(ReviewSymbol(
        id=stream_id, source="manual", box=box, detected_as="stream in/out", tag=name, tag_source="manual",
        dexpi_class=OFF_PAGE, identified_by=f"added by you as a stream {'out of' if out else 'into'} the drawing",
        notes=f"stream {'out of' if out else 'into'} {item_name(sid)}"))
    a, b = (sid, stream_id) if out else (stream_id, sid)
    cid = f"manual_c{len(review.connections) + 1}_{a[-4:]}_{b[-4:]}"
    review.connections.append(ReviewConnection(id=cid, from_id=a, to_id=b, line_type="process", source="manual",
                                               flow="reviewer", notes=f"stream {name} added by you"))
    select((), {cid}, zoom=True)  # the new line: its panel shows both ends and the direction


def _reverse(cid: str) -> None:
    c = next(x for x in review.connections if x.id == cid)
    c.from_id, c.to_id, c.flow = c.to_id, c.from_id, "reviewer"
    refresh_tables()


def _spec_key(sid: str, name: str, part: str, k: str) -> str:
    return f"spec_{sid}_{name}_{part}_{k}"


def _spec_owner(oid: str):
    """An item or a line - both carry specification fields."""
    return by_sym.get(oid) or next((c for c in review.connections if c.id == oid), None)


def _set_spec(sid: str, f, k: str) -> None:
    s = _spec_owner(sid)
    store = s.attributes.setdefault("custom" if f.custom else "dexpi", {})
    if f.kind == "quantity":
        v = st.session_state.get(_spec_key(sid, f.name, "v", k))
        u = st.session_state.get(_spec_key(sid, f.name, "u", k)) or f.units[0]
        store[f.name] = {"value": v, "unit": u} if v is not None else None
    else:
        store[f.name] = st.session_state.get(_spec_key(sid, f.name, "v", k)) or None
    if store.get(f.name) is None:
        store.pop(f.name, None)


def _spec_widget(s, f, cur, k: str) -> None:
    label = f.label + (" (custom)" if f.custom else "")
    if f.kind == "quantity":
        c1, c2 = st.columns([3, 2])  # room for the unit (mm, kW/(m2·K), ...)
        c1.number_input(label, value=(cur or {}).get("value"), format="%g", key=_spec_key(s.id, f.name, "v", k),
                        on_change=_set_spec, args=(s.id, f, k))
        unit = (cur or {}).get("unit")
        c2.selectbox("unit", f.units, index=f.units.index(unit) if unit in f.units else 0,
                     key=_spec_key(s.id, f.name, "u", k), on_change=_set_spec, args=(s.id, f, k))
    elif f.kind == "choice":
        opts = [""] + f.options
        st.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0, key=_spec_key(s.id, f.name, "v", k),
                     on_change=_set_spec, args=(s.id, f, k))
    elif f.kind == "number":
        st.number_input(label, value=cur, key=_spec_key(s.id, f.name, "v", k), on_change=_set_spec, args=(s.id, f, k))
    else:
        st.text_input(label, cur or "", key=_spec_key(s.id, f.name, "v", k), on_change=_set_spec, args=(s.id, f, k))


def _neighbours(sid: str) -> list:
    """The items a process item connects to, upstream ones first (from known flow)."""
    ups, downs, other = [], [], []
    for c in review.connections:
        if not c.include or c.line_type != "process" or sid not in (c.from_id, c.to_id):
            continue
        o = c.to_id if c.from_id == sid else c.from_id
        known = c.flow in FLOW_KNOWN
        (ups if known and c.to_id == sid else downs if known else other).append(o)
    out = []
    for o in ups + other + downs:
        if o not in out and o != sid:
            out.append(o)
    return out


def _add_line(a: str, b: str, flow: str = "", note: str = "") -> str:
    cid = f"manual_c{len(review.connections) + 1}_{a[-4:]}_{b[-4:]}"
    review.connections.append(ReviewConnection(id=cid, from_id=a, to_id=b, line_type="process", source="manual",
                                               flow=flow, notes=note or "added by you"))
    select((), {cid}, zoom=True)
    return cid


def _bypass(sid: str, up: str, down: str) -> None:
    """Take an in-line item out: its lines to `up` and `down` go, one line up -> down comes in."""
    for c in review.connections:
        if c.include and sid in (c.from_id, c.to_id) and {c.from_id, c.to_id} & {up, down}:
            c.include = False
            c.notes = "; ".join(x for x in (c.notes, f"replaced by the bypass of {item_name(sid)}") if x)
    by_sym[sid].include = False
    known = any(c.flow in FLOW_KNOWN and c.to_id == sid and c.from_id == up for c in review.connections)
    _add_line(up, down, "reviewer" if known else "", f"bypass of {item_name(sid)}")


def _overlap(a, b) -> float:
    """Share of the smaller box covered by the other (0 when apart)."""
    if not a or not b:
        return 0.0
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1.0
    return w * h / small


def _gap(a, b) -> float:
    return max(0.0, max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))


def _merge_candidates(s: ReviewSymbol) -> list:
    """Every other item, most likely duplicates first: the most overlap, then the closest,
    then items with no box. The picker is searchable, so any item can be merged."""
    def rank(o):
        if not (s.box and o.box):
            return (1, 0.0, float("inf"))
        return (0, -_overlap(s.box, o.box), _gap(s.box, o.box))
    return sorted((o for o in review.symbols if o.id != s.id), key=rank)


def _merge_label(s: ReviewSymbol, o: ReviewSymbol) -> str:
    where = ""
    if s.box and o.box:
        ov = _overlap(s.box, o.box)
        where = f", overlaps {ov:.0%}" if ov > 0.05 else f", {_gap(s.box, o.box):.0f} px away"
    return f"{item_option(o.id)} ({friendly_class(o.dexpi_class)}{where})"


def _merge(keep_id: str, other_ids: list) -> None:
    """Fold duplicate boxes of one item into `keep_id`: their lines move to it (dropping
    lines that would join it to itself or repeat an existing line), a missing tag / type /
    specifications are taken from them, the box grows to cover them all, their rows go."""
    keep = by_sym[keep_id]
    others = [by_sym[i] for i in other_ids if i in by_sym and i != keep_id]
    for o in others:
        if not keep.tag and o.tag:
            keep.tag, keep.tag_source = o.tag, o.tag_source
        if keep.dexpi_class == SKIP and o.dexpi_class != SKIP:
            keep.dexpi_class = o.dexpi_class
        for key in ("dexpi", "custom"):
            if o.attributes.get(key):
                keep.attributes.setdefault(key, {})
                for k2, v in o.attributes[key].items():
                    keep.attributes[key].setdefault(k2, v)
        if o.attributes.get("specs") and not keep.attributes.get("specs"):
            keep.attributes["specs"] = o.attributes["specs"]
        if keep.box and o.box:
            keep.box = [min(keep.box[0], o.box[0]), min(keep.box[1], o.box[1]),
                        max(keep.box[2], o.box[2]), max(keep.box[3], o.box[3])]
        keep.include = keep.include or o.include
        for c in review.connections:
            c.from_id = keep_id if c.from_id == o.id else c.from_id
            c.to_id = keep_id if c.to_id == o.id else c.to_id
    gone = {o.id for o in others}
    review.merged = {k: (keep_id if v in gone else v) for k, v in review.merged.items()} | {g: keep_id for g in gone}
    review.symbols = [x for x in review.symbols if x.id not in gone]
    if not keep.tag and keep.box and review.unattached_tags:
        # the merged box may now hold a tag no single box claimed ("V101" printed inside)
        inside = [t for t in review.unattached_tags if t.get("box") and
                  keep.box[0] <= (t["box"][0] + t["box"][2]) / 2 <= keep.box[2] and
                  keep.box[1] <= (t["box"][1] + t["box"][3]) / 2 <= keep.box[3]]
        if inside:
            best = max(inside, key=lambda t: t.get("conf", 0))
            keep.tag, keep.tag_source = best["tag"], "OCR inside (after merge)"
            review.unattached_tags = [t for t in review.unattached_tags if t is not best]
            keep.attributes = _keep_specs(keep.attributes, _instrument_attributes(keep.dexpi_class, keep.tag))
    kept, pairs = [], set()
    for c in review.connections:
        key = (frozenset((c.from_id, c.to_id)), c.line_type)
        if c.from_id == c.to_id or (key in pairs and keep_id in (c.from_id, c.to_id)):
            continue  # the merge made it a self-loop or a repeat
        pairs.add(key)
        kept.append(c)
    review.connections = kept
    keep.notes = "; ".join(x for x in (keep.notes, f"merged with {', '.join(refs.get(i, i) for i in gone)}") if x)
    select({keep_id}, (), zoom=True)


def _rename_section(sec_id: str, key: str) -> None:
    new = " ".join((st.session_state.get(key) or "").split())
    sec = next((x for x in review.drawio.sections if x.id == sec_id), None)
    if not sec or not new or new == sec.name:
        return
    for s in review.symbols:
        if s.section == sec.name:
            s.section = new
    sec.name = new
    refresh_tables()


def _remove_section(sec_id: str) -> None:
    sec = next((x for x in review.drawio.sections if x.id == sec_id), None)
    if not sec:
        return
    for s in review.symbols:
        if s.section == sec.name:
            s.section = ""
    review.drawio.sections = [x for x in review.drawio.sections if x.id != sec_id]
    select()


def _hide(syms=(), rows=()) -> None:
    """Hide items/lines from the map (not from DEXPI) and drop them from the selection."""
    hs, hr = st.session_state.setdefault("hidden", (set(), set()))
    st.session_state["hidden"] = (hs | set(syms), hr | set(rows))
    select()


def _unhide() -> None:
    st.session_state["hidden"] = (set(), set())


def _set_lines(cids: list, include: bool) -> None:
    for c in review.connections:
        if c.id in cids:
            c.include = include
    refresh_tables()


PANEL_HEIGHT = 750  # the map's toolbar, chips and drawing: the panel never runs past them


def _spec_list(owner, fields, vals, cust, pre, k) -> None:
    for f in fields:
        _spec_widget(owner, f, (cust if f.custom else vals).get(f.name) or pre.get(f.name), k)


def _line_kind(c):
    """What the line becomes in DEXPI (build_model._wire_instruments), for its fields: a line
    touching an instrument is a measuring or signal line, never a pipe; a line between two
    non-instruments is a pipe, even drawn dashed. -> (kind, DEXPI class, why)."""
    inst_ends = [x for x in (c.from_id, c.to_id) if x in by_sym and by_sym[x].dexpi_class in INSTRUMENT_CLASSES]
    other = next((x for x in (c.from_id, c.to_id) if x not in inst_ends), None)
    if len(inst_ends) == 2:
        return "Signal line", "SignalLineFunction", ""
    if inst_ends and c.line_type == "signal" and other in by_sym and by_sym[other].dexpi_class in VALVE_CLASSES:
        return "Signal line", "SignalLineFunction", f"{item_name(inst_ends[0])} drives {item_name(other)}'s actuator over this line."
    if inst_ends:
        return "Measuring line", "MeasuringLineFunction", (
            f"This line connects to instrument {item_name(inst_ends[0])}, so DEXPI records it as a measuring line "
            "(where the instrument measures), not as a pipe.")
    return "Pipe", "PipingNetworkSegment", (
        "Drawn as a signal line, but neither end is an instrument, so DEXPI builds it as a pipe."
        if c.line_type == "signal" else "")


with panel_col:
    st.html('<span class="edit-panel"></span>')
    k = f"{version}"
    sel_items = [by_sym[i] for i in show_syms if i in by_sym]
    sel_lines = [c for c in review.connections if c.id in show_conns]
    sel_sections = [sec for sec in (review.drawio.sections if review.drawio else []) if f"section:{sec.id}" in show_conns]
    # one panel as tall as the map, scrolling inside itself: nothing below the drawing to scroll to
    with st.container(border=True, height=PANEL_HEIGHT):
        b1, b2 = st.columns(2)
        to_check = [x for x in review.symbols if status(x) == "needs review"]
        current = next(iter(show_syms), None)
        if b1.button(f"Next to check ▸ ({len(to_check)})", disabled=not to_check, width="stretch",
                     help="Selects the next item marked red, and zooms to it"):
            ids = [x.id for x in to_check]
            select({ids[(ids.index(current) + 1) % len(ids)] if current in ids else ids[0]}, (), zoom=True)
            st.rerun()
        if b2.button("Clear selection", disabled=not (show_syms or show_conns), width="stretch"):
            select()
            st.rerun()
        hidden_syms, hidden_rows = st.session_state.get("hidden", (set(), set()))
        if hidden_syms or hidden_rows:
            h1, h2 = st.columns([3, 2], vertical_alignment="center")
            h1.caption(f"{len(hidden_syms)} item(s), {len(hidden_rows)} line(s) hidden on the map")
            h2.button("Show all", key=f"unhide_{k}", on_click=_unhide, width="stretch")

        if sel_sections and not sel_items and not sel_lines:
            sec = sel_sections[0]
            members = [x for x in review.symbols if x.section == sec.name]
            st.markdown(f"**Plant section: {sec.name}**")
            st.caption(f"Frame drawn around {len(members)} items on the drawing. Every item in it carries this name "
                       "into DEXPI (PlantSectionName) and so into the AAS.")
            st.text_input("Name", sec.name, key=f"secname_{sec.id}_{k}", on_change=_rename_section,
                          args=(sec.id, f"secname_{sec.id}_{k}"))
            st.caption(", ".join(item_name(x.id) for x in members[:30]) + (" ..." if len(members) > 30 else ""))
            c1, c2 = st.columns(2)
            if c1.button("Select all its items", key=f"secsel_{sec.id}_{k}", width="stretch",
                         help="Highlights them on the map and ticks them in the table - e.g. to merge or check them"):
                select({x.id for x in members}, (), zoom=True)
                st.rerun()
            c2.button("Remove section", key=f"secdel_{sec.id}_{k}", on_click=_remove_section, args=(sec.id,),
                      width="stretch", help="Not a plant section after all: its items get no section")

        elif not sel_items and not sel_lines:
            st.markdown("**Nothing selected**")
            st.caption("Click an item or a line on the map to see and edit it here. "
                       "Hover anything on the map for a quick description.")

        elif len(sel_items) == 1 and not sel_lines:
            s = sel_items[0]
            st.markdown(f"**{item_name(s.id)}** &nbsp; `{refs[s.id]}`")
            st.caption(f"{found_as(s)} · {status(s)}" + (f" · {s.section}" if s.section else ""))
            fields = spec_fields(s.dexpi_class) if s.dexpi_class not in (SKIP, OFF_PAGE) else []
            vals, cust = (s.attributes or {}).get("dexpi", {}), (s.attributes or {}).get("custom", {})
            pre = prefill(s.dexpi_class, (s.attributes or {}).get("specs", "")) if fields else {}
            filled = sum(1 for f in fields if (cust if f.custom else vals).get(f.name) or pre.get(f.name))
            mine = [c for c in review.connections if s.id in (c.from_id, c.to_id) and c.include]
            can_stream = s.dexpi_class not in INSTRUMENT_CLASSES and s.dexpi_class not in (OFF_PAGE, SKIP)
            names = ["Edit", f"Lines {len(mine)}", "Connect"] + (["Streams"] if can_stream else []) \
                + ([f"Specs {filled}/{len(fields)}"] if fields else [])
            tabs = dict(zip(names, st.tabs(names)))
            with tabs["Edit"]:
                st.text_input("Tag", s.tag, key=f"tag_{s.id}_{k}", on_change=_set_item, args=(s.id, "tag", f"tag_{s.id}_{k}"))
                st.selectbox("DEXPI type", DEXPI_CLASS_CHOICES,
                             index=DEXPI_CLASS_CHOICES.index(s.dexpi_class) if s.dexpi_class in DEXPI_CLASS_CHOICES else 0,
                             key=f"cls_{s.id}_{k}", on_change=_set_item, args=(s.id, "dexpi_class", f"cls_{s.id}_{k}"))
                st.toggle("In DEXPI", s.include, key=f"inc_{s.id}_{k}", on_change=_set_item,
                          args=(s.id, "include", f"inc_{s.id}_{k}"))
                if section_names:
                    secs = [""] + section_names
                    st.selectbox("Plant section", secs, index=secs.index(s.section) if s.section in secs else 0,
                                 key=f"sec_{s.id}_{k}", on_change=_set_item, args=(s.id, "section", f"sec_{s.id}_{k}"),
                                 help="The frame it was drawn in on the drawing")
                st.text_input("Notes", s.notes, key=f"note_{s.id}_{k}", on_change=_set_item,
                              args=(s.id, "notes", f"note_{s.id}_{k}"))
                for label, value in (("How the type was chosen", s.identified_by), ("Picture check", s.image_check),
                                     ("Specs from the drawing", (s.attributes or {}).get("specs", ""))):
                    if value:
                        st.caption(f"**{label}:** {value}")
                st.button("Hide this item on the map", key=f"hide_{k}", width="stretch", on_click=_hide,
                          args=([s.id], []), help="Takes it off the map so you can see what is behind it. It stays in "
                          "the tables and in the DEXPI file - use 'In DEXPI' to leave it out of the file.")
            with tabs[names[1]]:
                if mine:
                    st.caption("Click a line to open it.")
                    for c in mine:
                        other = c.to_id if c.from_id == s.id else c.from_id
                        arrow = ("→" if c.from_id == s.id else "←") if c.flow in FLOW_KNOWN else "–"
                        if st.button(f"{arrow} {item_option(other)} · {LINE_LABELS.get(c.line_type, c.line_type)}",
                                     key=f"peekline__{c.id}__{k}", type="tertiary"):  # key: the map outlines it on hover
                            select((), {c.id}, zoom=True)
                            st.rerun()
                else:
                    st.caption("No lines yet - add one under Connect, or add a stream.")
            with tabs["Connect"]:
                others = [o for o in review.symbols if o.id != s.id and o.include and o.dexpi_class != SKIP]
                target = st.selectbox("Connect to", [""] + [item_option(o.id) for o in others], key=f"conn_to_{s.id}_{k}",
                                      help="Adds a process pipe from this item to the one you pick")
                if target and st.button("Add this line", key=f"add_line_{s.id}_{k}", width="stretch"):
                    _add_line(s.id, option_id[target])
                    st.rerun()
                nb = _neighbours(s.id)
                if len(nb) == 2:
                    st.button(f"Bypass: {item_name(nb[0])} → {item_name(nb[1])} directly", key=f"bypass_{s.id}_{k}",
                              on_click=_bypass, args=(s.id, nb[0], nb[1]), width="stretch",
                              help="Replaces this item's two lines by one line between its neighbours and takes the item "
                                   "out of DEXPI. Undo: switch 'In DEXPI' back on for the item and its old lines.")
                elif len(nb) > 2:
                    a_ = st.selectbox("Bypass from", [item_option(x) for x in nb], key=f"bpa_{s.id}_{k}")
                    b_ = st.selectbox("to", [item_option(x) for x in nb], index=1, key=f"bpb_{s.id}_{k}")
                    ids = {item_option(x): x for x in nb}
                    if a_ != b_ and st.button("Bypass this item", key=f"bypass_{s.id}_{k}", width="stretch"):
                        _bypass(s.id, ids[a_], ids[b_])
                        st.rerun()
                near = _merge_candidates(s)
                if near:
                    labels = [_merge_label(s, o) for o in near]
                    pick = st.selectbox("Same item drawn twice? Merge with", [""] + labels, key=f"merge_{s.id}_{k}",
                                        help="Folds the other box into this item: its lines move here and its row goes. "
                                             "Type to search; the most overlapping come first.")
                    if pick and st.button(f"Merge into {item_name(s.id)}", key=f"do_merge_{s.id}_{k}", width="stretch"):
                        _merge(s.id, [near[labels.index(pick)].id])
                        st.rerun()
            if can_stream:
                with tabs["Streams"]:
                    st.caption("A feed, product or vent stream at this item: a named stream beside it - opposite its "
                               "existing lines - joined by a line with its direction set. In DEXPI: a Flow In / Flow "
                               "Out off-page connector.")
                    st.text_input("Stream name", "", placeholder="e.g. Product, N2, Vent", key=f"stream_name_{s.id}_{k}")
                    c1, c2 = st.columns(2)
                    c1.button("Stream out →", key=f"stream_out_{s.id}_{k}", width="stretch", on_click=_add_stream,
                              args=(s.id, "out", f"stream_name_{s.id}_{k}"),
                              help=f"Something leaves the drawing from {item_name(s.id)} (default name: Product)")
                    c2.button("→ Stream in", key=f"stream_in_{s.id}_{k}", width="stretch", on_click=_add_stream,
                              args=(s.id, "in", f"stream_name_{s.id}_{k}"),
                              help=f"Something enters the drawing into {item_name(s.id)} (default name: Feed)")
            if fields:
                with tabs[names[-1]]:
                    st.caption(f"Fields DEXPI defines for a {s.dexpi_class}" + (", pre-filled from the drawing's notes"
                               if pre else "") + ". Written into the DEXPI file when you build.")
                    _spec_list(s, fields, vals, cust, pre, k)

        elif len(sel_lines) == 1 and not sel_items:
            c = sel_lines[0]
            st.markdown(f"**{item_name(c.from_id)} → {item_name(c.to_id)}** &nbsp; `{refs.get(c.id, '')}`")
            st.caption(f"{LINE_LABELS.get(c.line_type, c.line_type)} · {found_by(c)}")
            loop = c.from_id == c.to_id
            if loop:
                st.warning(f"Both ends are on {item_name(c.from_id)}. A line like this usually leaves the drawing - "
                           "a product, vent or feed stream. Mark it under Stream.")
            kind, cls, why = _line_kind(c)
            fields = spec_fields(cls)
            vals, cust = (c.attributes or {}).get("dexpi", {}), (c.attributes or {}).get("custom", {})
            filled = sum(1 for f in fields if (cust if f.custom else vals).get(f.name))
            spec_tab = f"Specs {filled}/{len(fields)}"
            names = (["Stream", "Edit"] if loop else ["Edit", "Stream"]) + ["Ends", spec_tab]
            tabs = dict(zip(names, st.tabs(names)))
            with tabs["Edit"]:
                st.selectbox("From", options, index=options.index(item_option(c.from_id)) if item_option(c.from_id) in options else 0,
                             key=f"from_{c.id}_{k}", on_change=_set_line, args=(c.id, "from_id", f"from_{c.id}_{k}"))
                st.selectbox("To", options, index=options.index(item_option(c.to_id)) if item_option(c.to_id) in options else 0,
                             key=f"to_{c.id}_{k}", on_change=_set_line, args=(c.id, "to_id", f"to_{c.id}_{k}"))
                flows = list(FLOW_VALUES)
                g1, g2 = st.columns([4, 1], vertical_alignment="bottom")
                g1.selectbox("Flow From → To", flows, index=flows.index(FLOW_LABELS.get(c.flow, "unknown")),
                             key=f"flow_{c.id}_{k}", on_change=_set_line, args=(c.id, "flow", f"flow_{c.id}_{k}"))
                g2.button("⇄", key=f"rev_{c.id}_{k}", on_click=_reverse, args=(c.id,), width="stretch",
                          help="Reverse the direction: swap From and To")
                types = list(LINE_LABELS.values())
                t1, t2 = st.columns([3, 2], vertical_alignment="bottom")
                t1.selectbox("Line", types, index=types.index(LINE_LABELS.get(c.line_type, "process pipe")),
                             key=f"lt_{c.id}_{k}", on_change=_set_line, args=(c.id, "line_type", f"lt_{c.id}_{k}"))
                t2.toggle("In DEXPI", c.include, key=f"cinc_{c.id}_{k}", on_change=_set_line,
                          args=(c.id, "include", f"cinc_{c.id}_{k}"))
                st.text_input("Notes", c.notes, key=f"cnote_{c.id}_{k}", on_change=_set_line,
                              args=(c.id, "notes", f"cnote_{c.id}_{k}"))
                st.button("Hide this line on the map", key=f"hide_{k}", width="stretch", on_click=_hide,
                          args=([], [c.id]), help="Takes it off the map so you can see what is behind it. It stays in "
                          "the tables and in the DEXPI file - use 'In DEXPI' to leave it out of the file.")
            with tabs["Stream"]:
                st.caption("For a line that ends at a product, vent or feed stream instead of an item: that end "
                           "becomes a named stream - in DEXPI a Flow In / Flow Out off-page connector.")
                st.text_input("Stream name", "Product", key=f"sname_{c.id}_{k}")
                s1, s2 = st.columns(2)
                s1.button("Out: To end", key=f"sout_{c.id}_{k}", on_click=_to_stream, args=(c.id, "to", f"sname_{c.id}_{k}"),
                          width="stretch", help=f"Replace the To end ({item_name(c.to_id)}) by a stream leaving the drawing")
                s2.button("In: From end", key=f"sin_{c.id}_{k}", on_click=_to_stream,
                          args=(c.id, "from", f"sname_{c.id}_{k}"), width="stretch",
                          help=f"Replace the From end ({item_name(c.from_id)}) by a stream entering the drawing")
            with tabs["Ends"]:
                st.caption("Click an end to select it.")
                for label, end in (("From", c.from_id), ("To", c.to_id)):
                    if st.button(f"{label}: {item_option(end)} ›", key=f"end_{c.id}_{label}_{k}", type="tertiary"):
                        select({end}, (), zoom=True)
                        st.rerun()
            with tabs[spec_tab]:
                st.caption(" ".join(x for x in (
                    f"**{kind}.**", why,
                    "DEXPI's pipe segment fields, plus design pressure, rating and material as custom attributes."
                    if cls == "PipingNetworkSegment" else f"DEXPI's {kind.lower()} fields.",
                    "Written into the DEXPI file when you build.",
                    "Part of a pipe network: the values go on all of its pipe segments."
                    if cls == "PipingNetworkSegment" and c.id.startswith("net") else "") if x))
                _spec_list(c, fields, vals, cust, {}, k)

        else:
            ends = sorted({e for c in sel_lines for e in (c.from_id, c.to_id)} | {x.id for x in sel_items})
            if sel_lines and not sel_items and all(c.id.startswith("net") for c in sel_lines):
                st.markdown(f"**Pipe network joining {len(ends)} items**")
                st.caption("Lines that meet at tees. Every pair of its ends is a line in the table.")
            else:
                st.markdown(f"**{len(sel_items)} items and {len(sel_lines)} lines selected**")
                if len(sel_items) >= 2 and not sel_lines:
                    best = max(sel_items, key=lambda x: (bool(x.tag), (x.box[2] - x.box[0]) * (x.box[3] - x.box[1])
                                                          if x.box else 0))  # keep the tagged / biggest one
                    names = [item_option(x.id) for x in sel_items]
                    keep_name = st.selectbox("Merge these into one - keep", names, index=names.index(item_option(best.id)),
                                             key=f"mkeep_{k}")
                    if st.button(f"Merge {len(sel_items)} items into one", key=f"mmerge_{k}", width="stretch"):
                        keep_id = sel_items[names.index(keep_name)].id
                        _merge(keep_id, [x.id for x in sel_items if x.id != keep_id])
                        st.rerun()
            st.caption(", ".join(item_name(e) for e in ends[:20]))
            if sel_lines:
                inc = all(c.include for c in sel_lines)
                st.toggle("In DEXPI", inc, key=f"ninc_{k}", on_change=lambda ids=[c.id for c in sel_lines]:
                          _set_lines(ids, st.session_state[f"ninc_{k}"]))
            st.button("Hide the selection on the map", key=f"hide_{k}", width="stretch", on_click=_hide,
                      args=([x.id for x in sel_items], [c.id for c in sel_lines]))
            for e in ends[:12]:
                if st.button(f"Go to {item_name(e)} ›", key=f"nend_{e}_{k}", type="tertiary"):
                    select({e}, (), zoom=True)
                    st.rerun()

    if review.unattached_tags:
        with st.expander(f"OCR tags not on any item ({len(review.unattached_tags)})"):
            st.caption("Text the OCR read that no item claimed - possibly a missed symbol.")
            st.dataframe(pd.DataFrame(review.unattached_tags)[["tag", "conf"]], width="stretch", hide_index=True)


# ---------------------------------------------------------------------------- build ---
st.markdown("##### 3 · Build the DEXPI file and the AAS")
with st.expander("AAS details (Asset Administration Shell, IDTA 02012-1-0) - optional, used for the .aasx"):
    st.caption("Describes the plant this P&ID belongs to. The AAS standard requires none of these fields - the parts "
               "it does require (the DEXPI file, its model entry) are made for you. Filled-in fields go into the AAS's "
               "PlantMetadata and ModelMetadata.")
    stem_default = Path(review.source_name).stem.strip()
    st.markdown("**Recommended**")
    r1, r2 = st.columns(2)
    plant_name = r1.text_input("Plant name", stem_default, key="aas_plant",
                               help="The AAS's ids and every item's id ('<plant>:asset:<tag>') are built from it: give "
                                    "all drawings of one plant the same plant name, so the same valve or tank gets the "
                                    "same id in each.")
    drawing_number = r2.text_input("Drawing number", stem_default, key="aas_drawing",
                                   help="Tells the drawings of one plant apart in its AAS")
    if plant_name.strip() == stem_default:
        st.caption("The plant name is still the file name. Set the real plant name, so drawings of the same plant "
                   "share item ids.")
    st.markdown("**Optional**")
    o1, o2, o3 = st.columns(3)
    aas_plant = {
        "ProcessPlantName": plant_name,
        "PlantSectionName": o1.text_input("Plant section", "", key="aas_section"),
        "ProjectName": o2.text_input("Project", "", key="aas_project"),
        "EnterpriseName": o3.text_input("Organisation", "", key="aas_enterprise"),
        "SiteName": o1.text_input("Site", "", key="aas_site"),
    }
    aas_model_meta = {
        "DrawingNumber": drawing_number,
        "CreatorName": o2.text_input("Drawn by", "", key="aas_creator"),
        "CheckerName": o3.text_input("Checked by", "", key="aas_checker"),
    }
    st.markdown("**Package**")
    aas_version = st.radio("AAS version", ["3.0", "3.1"], horizontal=True, key="aas_version",
                           help="3.0: what IDTA 02012-1-0 was published for, read by most AAS software, checked with the "
                                "IDTA's own test engine. 3.1: the newest metamodel - check that your AAS software reads it. "
                                "Both are verified against their official metamodel before download.")
    aas_base = st.text_input("Id prefix for the AAS, its Submodel and the items (leave empty for an automatic one)",
                             "", key="aas_base", help="e.g. https://your-lab.example/plants/lightup - the twins look "
                             "items up by '<prefix>:asset:<tag>'")
if review.image_path:
    with st.expander("Use this picture as training data (label it in draw.io)"):
        st.caption("Training labels need exact boxes, pipe paths and solid / dashed lines - edit those in draw.io: "
                   "download the picture with your checked items and lines as a draw.io file, correct it there "
                   "(the file explains how; app.diagrams.net works), then add it in the sidebar under "
                   "Training data.")
        from pid_dexpi_aas.image.drawio_labels import review_to_drawio
        # one click: the file is made when the button is pressed, and the page does not rerun
        st.download_button("Download for labelling (.drawio)", lambda r=review: review_to_drawio(r).encode(),
                           file_name=f"{Path(review.source_name).stem}_labels.drawio", mime="application/xml",
                           on_click="ignore", key="label_dl")

b1, b2 = st.columns([1, 1])
with b2:
    st.download_button("Save review (.json) to continue later", review_to_json(review),
                       file_name=f"{Path(review.source_name).stem}_review.json", mime="application/json")
with b1:
    build_clicked = st.button("Build & validate DEXPI", type="primary")

if build_clicked:
    problems = validate_rows(review)
    if problems:
        st.session_state["build_problems"] = problems
    else:
        st.session_state.pop("build_problems", None)
problems = st.session_state.get("build_problems")
if problems:
    st.error("Fix these before building (or untick 'In DEXPI' for the item):")
    for n, p in enumerate(problems):
        sid = next((s.id for s in review.symbols if p.startswith(f"{s.id}:")), None)
        c1, c2 = st.columns([5, 1])
        c1.write(f"- {item_option(sid)}: {p.split(':', 1)[1].strip()}" if sid else f"- {p}")
        if sid and c2.button("Show on map", key=f"prob_{n}"):
            select({sid}, (), zoom=True)
            st.rerun()
    build_clicked = False
if build_clicked:
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
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download DEXPI XML", xml_bytes, file_name=f"{stem}.xml", mime="application/xml")
    with st.spinner("Packaging the AAS..."):
        aas_in = DexpiModelInput(stem, xml_bytes, drawing_svg(diagram, result),
                                 {k: v for k, v in aas_model_meta.items() if v})
        aas_data = aasx_bytes([aas_in], {k: v for k, v in aas_plant.items() if v}, version=aas_version,
                              base_iri=aas_base or None)
        aas_ok, aas_problems = check_aasx(aas_data)
    d3.download_button(f"Download AAS {aas_version} (.aasx)", aas_data, file_name=f"{stem}.aasx", mime="application/asset-administration-shell-package",
                       help="Asset Administration Shell with the IDTA 02012-1-0 DEXPI Submodel: the DEXPI file, the "
                            "drawing as SVG, and every tagged item mapped to a global id.")
    n_mapped = len(mapping_entries(xml_bytes))
    if aas_ok:
        d3.caption(f"✓ Valid AAS {aas_version} ({'IDTA test engine + ' if aas_version == '3.0' else ''}aas-core "
                   f"metamodel check) · {n_mapped} items mapped")
    else:
        d3.warning(f"The AAS {aas_version} package did not pass the compliance check")
        with st.expander("Compliance check details"):
            st.text("\n".join(aas_problems))
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
        st.caption("After: what the DEXPI file contains, drawn at the drawing's positions and pipe routes. "
                   "Signal lines and instrument taps are not in the DEXPI file yet, so they are not drawn.")
        try:
            st.image(Image.open(io.BytesIO(render_after_png(diagram, result))), width="stretch")
        except Exception as exc:
            st.warning(f"Could not re-draw from the DEXPI file: {exc}")
    if report.skipped_details:
        with st.expander(f"Skipped elements ({len(report.skipped_details)})"):
            for detail in report.skipped_details:
                st.text(detail)
