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
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v2  # noqa: E402,F401  (st.components.v2: the clickable map)
from PIL import Image  # noqa: E402

from pid_dexpi_aas.render.drawio_editor import INSTALLED as EDITOR_INSTALLED, drawio_editor  # noqa: E402
from pid_dexpi_aas.render.check_sheet import (CAIRO_MISSING, build_check_sheet, number_items,  # noqa: E402
                                              numbered_overlay, render_after_png)
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
from pid_dexpi_aas.dexpi.spec_sheet import apply_spec_sheet, read_xlsx, template_xlsx
from pid_dexpi_aas.review import duplicate_tags, is_flow_arrow, look_reasons
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
# The models to choose from (ml/models): name and what sets each apart. Scores are the
# 4-fold cross-validation on the 12 real OPEN100 sheets (items / lines), where measured.
MODEL_INFO = {
    "pid2graph_final.pt": ("v3 (default)",
                           "The tested default, and the best measured: OPEN100 items 0.719, lines 0.343 (4-fold "
                           "cross-validation, tiles merged border-aware); text read with docTR."),
    "pid2graph_final_v5.pt": ("v5 - flow arrows (not recommended)",
                              "v3 + the team drawings' flow arrows labelled (boxed at OPEN100's arrow size). OPEN100 "
                              "items 0.673, lines 0.268 - slightly below v3. Not recommended: on drawings unlike the "
                              "team's (MEM4CHEM Zaiput) it reads many valves as arrows (28 vs v3's 9) and links "
                              "everything around them."),
    "pid2graph_final_v4.pt": ("v4 - flow arrows, small boxes (superseded)",
                              "Arrows boxed at their drawn size - too small to learn. OPEN100 items 0.680, lines "
                              "0.251. Kept for comparison; use v5 instead."),
    "pid2graph_final_v6.pt": ("v6 - flow arrows + bends and tees",
                              "v5 + each pipe's corners and tee joints learned as points (folded back into "
                              "connections). Final model only - no cross-validation score."),
    "pid2graph_fold1.pt": ("Fold 1 (old release)",
                           "The first released model (one cross-validation fold, before the team-data fixes). "
                           "For reference only."),
}


# superseded - kept on disk, not offered: worse links than v3 on all 4 OPEN100 folds
# (with the new tile merge: links mAP v3 0.343, v4 0.317, v5 0.327); v5 also reads valves as
# arrows on drawings unlike the team's
HIDDEN_MODELS = {"pid2graph_final_v4.pt", "pid2graph_final_v5.pt", "pid2graph_final_v6.pt"}


def available_models() -> dict[str, tuple[str, str]]:
    """{path: (name, description)} for the model files present, the default first."""
    found = sorted((p for p in (ROOT / "ml" / "models").glob("*.pt") if p.name not in HIDDEN_MODELS),
                   key=lambda p: (list(MODEL_INFO).index(p.name) if p.name in MODEL_INFO else len(MODEL_INFO), p.name))
    out = {str(p): MODEL_INFO.get(p.name, (p.stem, "No description.")) for p in found}
    if DEFAULT_CHECKPOINT not in out:  # PID_MODEL_CHECKPOINT pointing elsewhere
        out = {DEFAULT_CHECKPOINT: (Path(DEFAULT_CHECKPOINT).stem, "Set by PID_MODEL_CHECKPOINT."), **out}
    return out


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


@st.cache_data(show_spinner=False)
def blank_spec_sheet() -> bytes:
    return template_xlsx()


def sheet_name() -> str:
    """The loaded spec sheet, for the editor's Sync buttons ("" = none)."""
    sheet = st.session_state.get("spec_sheet")
    return f"{sheet['name']} ({len(sheet['rows'])} tags)" if sheet else ""


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
class UploadedPicture:
    """A picture made in the app (a PDF's page) where an uploaded file is expected: name + getvalue()."""

    def __init__(self, name: str, data: bytes):
        self.name, self._data = name, data

    def getvalue(self) -> bytes:
        return self._data


@st.cache_data(show_spinner="Reading the AutoCAD file...", max_entries=4)
def cad_to_drawio(data: bytes, name: str) -> tuple[str, dict]:
    """An AutoCAD P&ID (.dwg via LibreDWG, or .dxf) as a draw.io drawing (cad/to_drawio.py).
    In memory: the converter's temporary folder is deleted straight after."""
    from pid_dexpi_aas.cad.dwg_read import read_cad
    from pid_dexpi_aas.cad.to_drawio import dxf_to_drawio
    return dxf_to_drawio(read_cad(data, name), Path(name).stem)


@st.cache_data(show_spinner=False, max_entries=8)
def pdf_page_count(data: bytes) -> int:
    import pypdfium2 as pdfium
    return len(pdfium.PdfDocument(data))


@st.cache_data(show_spinner="Reading the PDF...", max_entries=4)
def pdf_page_png(data: bytes, index: int) -> bytes:
    """One PDF page as a PNG, its longer side about 7000 px (100-300 dpi): an A0 sheet at
    150 dpi, the scale the model was checked at. In memory only (uploads are not kept on disk)."""
    import pypdfium2 as pdfium
    page = pdfium.PdfDocument(data)[index]
    w, h = page.get_size()   # points, 1/72 inch
    dpi = min(300.0, max(100.0, 7000.0 / (max(w, h) / 72.0)))
    image = page.render(scale=dpi / 72.0).to_pil().convert("RGB")
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


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
        uploaded_drawio = st.file_uploader("draw.io or AutoCAD file", type=["drawio", "xml", "dwg", "dxf"],
                                           help="An AutoCAD P&ID (.dwg or .dxf) is turned into a draw.io drawing: "
                                                "symbols with their tags, equipment outlines, pipes and streams. "
                                                "Check it in the editor - unknown symbols are flagged.")
        if uploaded_drawio is not None and uploaded_drawio.name.lower().endswith((".dwg", ".dxf")):
            try:
                cad_xml, cad_report = cad_to_drawio(uploaded_drawio.getvalue(), uploaded_drawio.name)
                n_sym = sum(cad_report["symbols"].values())
                st.caption(f"Read from AutoCAD: {n_sym} symbols, {cad_report['pipes']} pipes, "
                           f"{cad_report['signal_lines']} signal lines"
                           + (f"; {len(cad_report['unrecognised'])} symbol types not recognised (flagged in the editor)"
                              if cad_report["unrecognised"] else "") + ".")
                uploaded_drawio = UploadedPicture(Path(uploaded_drawio.name).stem + ".drawio", cad_xml.encode())
            except Exception as exc:
                st.error(f"This CAD file can't be read: {exc}")
                uploaded_drawio = None
        frames_as_sections = st.checkbox(
            "Frames are plant sections", False, key="frames_as_sections",
            help="Coloured frames drawn around groups of items (e.g. 'Autoclave') are not part of a P&ID and are "
                 "left out. Tick to use each as a plant section instead: its name goes onto the items inside it "
                 "(DEXPI PlantSectionName).")
    with in_picture:
        image_file = st.file_uploader("P&ID image or PDF", type=["png", "jpg", "jpeg", "pdf"])
        if image_file is not None and image_file.name.lower().endswith(".pdf"):
            # a PDF (e.g. plotted from AutoCAD): one page as the picture the model reads
            try:
                n_pages = pdf_page_count(image_file.getvalue())
                page = st.number_input("Page", 1, n_pages, 1, key="pdf_page") if n_pages > 1 else 1
                png = pdf_page_png(image_file.getvalue(), page - 1)
                stem = Path(image_file.name).stem + (f" (page {page})" if n_pages > 1 else "")
                image_file = UploadedPicture(f"{stem}.png", png)
            except Exception as exc:  # damaged or password-protected
                st.error(f"This PDF can't be read: {exc}")
                image_file = None
        st.caption("With a draw.io file as well, the drawing is checked against the picture.")
        if image_file is not None:
            with st.popover("Re-run the model on this picture", width="stretch"):
                st.caption("A picture is processed once and the result kept. Re-run it to use the current model "
                           "and clean-up rules. **This starts the review of this picture again: changes made to it "
                           "are replaced** - save the review first to keep them.")
                if st.button("Re-run now", type="primary", width="stretch", key="rerun_model_btn"):
                    st.session_state["rerun_model"] = True
                    st.session_state["model_run"] = st.session_state.get("model_run", 0) + 1
    with in_editor:
        if not EDITOR_INSTALLED:
            st.caption("The editor is not installed on this server (scripts/get_drawio_editor.py).")
        else:
            st.caption("Draw a P&ID, or change the uploaded draw.io file - or a picture's result - in draw.io "
                       "right here. **Save** (Ctrl+S) in the editor loads it into the app.")
            st.session_state.setdefault("editor_open", True)  # the main screen; the map is the old view
            st.toggle("Show the editor", key="editor_open")
            if st.session_state.get("editor_xml"):
                st.download_button("Download the edited drawing", lambda x=st.session_state["editor_xml"]: x.encode(),
                                   file_name=st.session_state["editor_name"], mime="application/xml",
                                   width="stretch")
                if st.button("Discard the edits", width="stretch",
                             help="Back to the uploaded draw.io file (or nothing)"):
                    st.session_state.pop("editor_xml")
                    st.rerun()
    models = available_models()
    checkpoint = st.selectbox("Model (pictures)", list(models), format_func=lambda p: models[p][0],
                              index=list(models).index(DEFAULT_CHECKPOINT), key="model_pick",
                              help="Which trained model reads uploaded pictures. Each model's result is kept, "
                                   "so switching back is instant.")
    with st.expander("About this model"):
        st.caption(models[checkpoint][1])
        st.caption("All models: " + " · ".join(f"**{n}**" for n, _ in models.values()))
    if PUBLIC:
        device = "auto"
    else:
        with st.expander("Model settings"):
            other = st.text_input("Other model file", "", help="A checkpoint path on this PC; overrides the list.")
            if other.strip():
                checkpoint = other.strip()
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
    st.header("Specifications")
    st.caption("An Excel sheet with a row per item: its tag (as on the P&ID), its DEXPI type from a dropdown, "
               "and then the fields of that type to fill in - they change with the type picked. Filled in once "
               "and used for every P&ID: items whose tags match get its values (empty fields only) and its type "
               "(where the drawing gave none); the editor's 'Sync from the spec sheet' puts them over what is there.")
    st.download_button("Download a blank spec sheet (.xlsx)", blank_spec_sheet, file_name="spec_sheet.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    spec_slot = st.container()  # the sheet with this drawing's tags, once there is a drawing
    spec_upload = st.file_uploader("Filled-in spec sheet (.xlsx)", type=["xlsx"], key="spec_upload")
    if spec_upload is not None:
        digest = hashlib.sha256(spec_upload.getvalue()).hexdigest()[:16]
        if (st.session_state.get("spec_sheet") or {}).get("digest") != digest:
            try:
                rows, sheet_notes = read_xlsx(spec_upload.getvalue())
                st.session_state["spec_sheet"] = {"digest": digest, "name": spec_upload.name, "rows": rows,
                                                  "notes": sheet_notes}
            except Exception as exc:  # not the template, a damaged file
                st.session_state.pop("spec_sheet", None)
                st.error(f"This file can't be read as a spec sheet: {exc}. Start from 'Download a blank spec sheet'.")
        sheet = st.session_state.get("spec_sheet")
        if sheet:
            st.caption(f"{len(sheet['rows'])} tags in {sheet['name']}.")
            for note in sheet["notes"][:5]:
                st.warning(note)
    else:
        st.session_state.pop("spec_sheet", None)
    st.divider()
    st.header("Training data")
    with st.expander("Add a drawing labelled in draw.io"):
        st.caption("Checked sheets are added from 'Add this to the training data' at the bottom of the page. "
                   "Here: a picture's drawing ('Download the drawing (.drawio)' there) corrected in desktop "
                   "draw.io, and the list of sheets added so far.")
        label_upload = st.file_uploader("Corrected .drawio", type=["drawio", "xml"], key="label_upload")
        show_training = st.checkbox("Show the drawings added so far", key="show_training")



def drawn(xml: str | None) -> bool:
    """Something is drawn in it: an empty page (a blank editor's autosave) is no drawing."""
    return bool(xml) and ('vertex="1"' in xml or 'edge="1"' in xml)


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
def model_key(checkpoint: str) -> bytes:
    """A model in cache keys: its path and its file's time - a retrained model deployed under
    the same name (ml/pid2graph/auto_retrain.py) is a new model, not the cached one."""
    try:
        return f"{checkpoint}|{os.path.getmtime(checkpoint)}".encode()
    except OSError:
        return checkpoint.encode()


def _digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:16]


# Uploads and the model's results are on disk only while they are in use (security, 2026-11):
# in a private folder per browser session (mkdtemp: mode 0700, random name), deleted as soon as
# its picture or drawing is removed or replaced, and swept once idle for SESSION_TTL (a closed
# tab sends nothing). The shared folder keeps only the model lock and its timings.
WORK = Path(tempfile.gettempdir()) / "pid_dexpi_app"
SESSIONS = WORK / "sessions"
SESSION_TTL = 3 * 3600
KEEP_SHARED = {"infer.lock", "infer_timings.json"}


def session_dir() -> Path:
    """This browser session's private folder (made on first use; marked in use on each run)."""
    d = st.session_state.get("work_dir")
    if d and Path(d).is_dir():
        os.utime(d)
        return Path(d)
    SESSIONS.mkdir(parents=True, exist_ok=True)
    os.chmod(WORK, 0o700)
    d = tempfile.mkdtemp(prefix="s_", dir=SESSIONS)
    st.session_state["work_dir"] = d
    return Path(d)


def forget_upload(key: str) -> None:
    """Delete everything kept for one upload: its copy, the model's result, the editor's copy."""
    d = st.session_state.get("work_dir")
    if key and d:
        for p in Path(d).glob(f"{key}*"):
            p.unlink(missing_ok=True)
    from pid_dexpi_aas.render.drawio_editor import FILES as EDITOR_FILES
    for name in st.session_state.setdefault("editor_pictures", {}).pop(key, []):
        (EDITOR_FILES / "pictures" / name).unlink(missing_ok=True)
    # the editor's drawing made over that picture points at the copy just deleted: made again
    # from scratch if the same file comes back (else its boxes showed with no picture behind)
    st.session_state.pop("editor_pic_doc", None)
    # and the review made from it is out of date: the same file uploaded again is read again
    # (its picture and result were just deleted), with the progress bar, not shown from memory
    if st.session_state.get("review_picture") == key:
        st.session_state.pop("review_picture", None)
        st.session_state["model_run"] = st.session_state.get("model_run", 0) + 1


def sweep_old_files() -> None:
    """Sessions idle for SESSION_TTL, the editor's copies as old, and every upload the app kept
    before 2026-11 (one shared folder of all of them). At most once a minute."""
    global _LAST_SWEEP
    now = time.time()
    if now - _LAST_SWEEP < 60:
        return
    _LAST_SWEEP = now
    from pid_dexpi_aas.render.drawio_editor import FILES as EDITOR_FILES
    WORK.mkdir(exist_ok=True)
    for p in WORK.iterdir():
        if p.is_file() and p.name not in KEEP_SHARED:
            p.unlink(missing_ok=True)
    for d in SESSIONS.iterdir() if SESSIONS.exists() else []:
        if now - d.stat().st_mtime > SESSION_TTL:
            shutil.rmtree(d, ignore_errors=True)
    for p in (EDITOR_FILES / "pictures").glob("*.png"):
        if now - p.stat().st_mtime > SESSION_TTL:
            p.unlink(missing_ok=True)


_LAST_SWEEP = 0.0
sweep_old_files()


# Bump when the model's input handling changes, so results cached by an older version
# are redone (v2: dark-theme exports keep their light lines - sheet_image.py; v3: traced pipes;
# v5: tiles merged border-aware with weighted box fusion - stitch.py; v6: docTR reads the text;
# v7: the pipe tracer keeps thin lines and joins lines ending just short of a symbol).
INFER_VERSION = b"v13"


# The picture the review on screen was made from, with which model: a new picture, a re-run, or
# a newly trained model deployed make it stale - the editor then shows the progress of reading
# the picture in its place (PROGRESS_SLOT), not the old drawing, and opens with the new result.
PICTURE_KEY = _digest(image_file.getvalue(), model_key(checkpoint), INFER_VERSION) if image_file else ""
PICTURE_STALE = bool(image_file) and not uploaded_drawio and (
    bool(st.session_state.get("rerun_model")) or st.session_state.get("review_picture") != PICTURE_KEY)
PROGRESS_SLOT = None
# a removed or replaced picture: its files go at once (the review on screen goes with it)
if st.session_state.get("kept_picture", "") not in ("", PICTURE_KEY):
    forget_upload(st.session_state["kept_picture"])
st.session_state["kept_picture"] = PICTURE_KEY
if st.session_state.get("editor_open") and EDITOR_INSTALLED and PICTURE_STALE:
    st.markdown("### draw.io editor")
    PROGRESS_SLOT = st.empty()
    st.info("Reading the picture - the editor opens with the result.")
elif st.session_state.get("editor_open") and EDITOR_INSTALLED:
    from pid_dexpi_aas.drawio.editor_data import editor_patch
    from pid_dexpi_aas.render.drawio_editor import write_schema
    st.markdown("### draw.io editor")
    st.caption("Draw with the P&ID libraries on the left; set each item's and line's DEXPI details in the panel on "
               "the right (Find: search, and what still needs a look). Changes save themselves after a few seconds "
               "(or Ctrl+S) and go into the checks and exports below.")
    write_schema()
    # each cell's details from the review as the last run left it (side-panel edits included);
    # a Save rebuilds the review further down, which reruns once to bring them here
    prev = st.session_state.get("review")
    # a picture review only while its picture is still uploaded: removed, the editor clears at once
    if (prev is not None and prev.drawio is None and prev.image_path and not uploaded_drawio and image_file
            and not Path(prev.image_path).exists()):
        # its picture is gone (removed and uploaded again): the review is being made again
        st.info("Reading the picture again - the editor opens with the result.")
    elif prev is not None and prev.drawio is None and prev.image_path and not uploaded_drawio and image_file:
        # a picture review: the model's result drawn over the picture, rebuilt from the review each
        # time the editor is switched on (so map edits made meanwhile are in it); a Save updates the rows
        from pid_dexpi_aas.drawio.picture_editor import apply_editor_drawing, drawing_ids, review_to_editor_xml
        from pid_dexpi_aas.render.drawio_editor import picture_url
        doc = st.session_state.get("editor_pic_doc")
        if (doc is None or doc["key"] != st.session_state.get("input_key")
                or not st.session_state.get("editor_was_open")):
            url = picture_url(prev.image_path, tag=session_dir().name)  # this session's own copy
            st.session_state.setdefault("editor_pictures", {}).setdefault(PICTURE_KEY, []).append(url.rsplit("/", 1)[-1])
            xml = review_to_editor_xml(prev, url)
            doc = {"key": st.session_state.get("input_key"), "xml": xml, "ids": drawing_ids(xml)}
            st.session_state["editor_pic_doc"] = doc
        st.session_state["editor_was_open"] = True
        saved = drawio_editor(doc["xml"], editor_patch(prev), title=prev.source_name, mode="picture", sheet=sheet_name(),
                              focus=st.session_state.get("editor_focus", ""))
        if saved and saved.get("t") != st.session_state.get("editor_t"):  # a new Save
            if saved.get("sync"):
                st.session_state["spec_sync"] = saved["sync"]
            # the rows the editor was showing (it adds the side panel's new ones as they come)
            changes = apply_editor_drawing(prev, saved["xml"], set(saved.get("ids") or doc["ids"]), saved.get("routes"))
            doc.update(xml=saved["xml"], ids=drawing_ids(saved["xml"]))
            st.session_state["editor_t"] = saved["t"]
            st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
            done = ", ".join(f"{v} {k}" for k, v in changes.items() if v)
            st.toast(f"Loaded from the editor: {done or 'details updated'}.")
            st.rerun()
    elif image_file and not uploaded_drawio and not drawn(st.session_state.get("editor_xml")):
        # a picture being read: the editor opens with the model's result once it is there (a
        # blank editor here autosaved its empty page, which then stood in for a draw.io file)
        st.info("The editor opens with the model's result once the picture has been read.")
    else:
        saved = drawio_editor(st.session_state.get("editor_xml") or (uploaded_drawio.getvalue().decode("utf-8", "replace")
                                                                   if uploaded_drawio else ""),
                              editor_patch(prev) if prev is not None and prev.drawio is not None else {},
                              title=uploaded_drawio.name if uploaded_drawio else "new drawing", sheet=sheet_name(),
                              focus=st.session_state.get("editor_focus", ""))
        if saved and saved.get("t") != st.session_state.get("editor_t"):  # a new Save
            if saved.get("sync"):
                st.session_state["spec_sync"] = saved["sync"]
            stem = Path(uploaded_drawio.name).stem if uploaded_drawio else "drawing"
            xml = saved["xml"]
            if not xml.lstrip().startswith("<mxfile"):  # embed mode saves the bare graph: make it a .drawio file
                xml = f'<mxfile host="pid-dexpi-aas"><diagram id="page-1" name="Page-1">{xml}</diagram></mxfile>'
            st.session_state["editor_t"] = saved["t"]
            if drawn(xml) or uploaded_drawio:  # an empty page saved from a blank editor is not a drawing
                st.session_state.update(editor_xml=xml, editor_name=f"{stem} (edited).drawio")
                st.rerun()
else:
    st.session_state["editor_was_open"] = False
drawio_file = (EditedDrawing(st.session_state["editor_name"], st.session_state["editor_xml"].encode())
               if drawn(st.session_state.get("editor_xml")) else uploaded_drawio)

# the same for a draw.io drawing (each saved edit is a new one: the previous copy goes)
_drawio_key = "d" + _digest(drawio_file.getvalue()) if drawio_file else ""
if st.session_state.get("kept_drawio", "") not in ("", _drawio_key):
    forget_upload(st.session_state["kept_drawio"])
st.session_state["kept_drawio"] = _drawio_key


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

def _plant_add(stem: str, xml: bytes, svg, meta: dict) -> None:
    """A built drawing into the plant AAS (st.session_state["plant_models"])."""
    st.session_state.setdefault("plant_models", {})[stem] = {"xml": xml, "svg": svg, "meta": meta}
    st.toast(f"{stem} is in the plant AAS ({len(st.session_state['plant_models'])} drawings).")


def _slug_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "plant"




def _image_done(image_bytes: bytes, checkpoint: str) -> bool:
    """Already processed (no wait)?"""
    return (session_dir() / f"{_digest(image_bytes, model_key(checkpoint), INFER_VERSION)}.json").exists()


def infer_with_progress(image_bytes: bytes, image_name: str, checkpoint: str, device: str) -> None:
    """Run the model on a picture (infer_sheet.py, in the ML environment) with a progress bar:
    the script writes its stage and how far it is (infer_sheet.Progress - weighted by this
    sheet's size and how fast this machine has been); the bar shows it, with the time left.
    One picture at a time: another one being read is waited for, and the bar says so."""
    key = _digest(image_bytes, model_key(checkpoint), INFER_VERSION)
    work = session_dir()
    img_path = work / f"{key}_{Path(image_name).name}"
    out_path = work / f"{key}.json"
    if out_path.exists():
        return
    img_path.write_bytes(image_bytes)
    bar = (PROGRESS_SLOT or st).progress(0.0, text="Starting...")   # above the editor, when it is open
    try:
        import fcntl
    except ImportError:  # Windows: no lock (a single local user there)
        fcntl = None
    # one image at a time: two users at once would hold the model twice
    with open(WORK / "infer.lock", "w") as lock:
        if fcntl is not None:
            t0 = time.time()
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    bar.progress(0.0, text=f"Waiting for another picture to finish ({time.time() - t0:.0f} s) - "
                                           "yours starts right after it")
                    time.sleep(1.0)
        if out_path.exists():  # read while we waited (the same picture)
            bar.empty()
            return
        progress_path = work / f"{key}.progress.json"
        progress_path.unlink(missing_ok=True)
        err_path = work / f"{key}.err"
        timings_path = WORK / "infer_timings.json"
        try:  # how long start-up and a whole run took before (infer_sheet.Progress)
            past = json.loads(timings_path.read_text()).get("cuda" if pick_device(device) != "cpu" else "cpu", {})
        except (OSError, ValueError):
            past = {}
        startup_s, total_s = float(past.get("startup", 4.0)), float(past.get("_total", 12.0))
        with open(err_path, "w") as err:
            t0 = time.time()
            proc = subprocess.Popen(
                [str(ML_PYTHON), str(INFER_SCRIPT), "--image", str(img_path), "--checkpoint", checkpoint,
                 "--out", str(out_path), "--device", pick_device(device), "--progress", str(progress_path),
                 "--timings", str(timings_path), "--started", str(t0)],
                stdout=subprocess.DEVNULL, stderr=err, text=True)
            shown, anchor, last = 0.0, None, -1.0   # anchor: (bar, report) when the reports took over / fell back
            while proc.poll() is None:
                now = time.time()
                try:
                    p = json.loads(progress_path.read_text())
                except (OSError, ValueError):  # not written yet: Python and the libraries starting up
                    p = None
                if p is None:
                    # start-up: along its usual share of a run, easing off (never quite to its end)
                    span = startup_s / max(startup_s, total_s)
                    target = span * (1 - math.exp(-1.5 * (now - t0) / max(0.5, startup_s)))
                    step, total = "Starting the model", max(startup_s, total_s)
                else:
                    # the last report, moved on within its stage at the stage's expected pace -
                    # easing off as it overruns (a slow stage keeps creeping, never stops)
                    a, b = float(p.get("stage_start", 0.0)), float(p.get("stage_end", p.get("done", 0.0)))
                    pace = (now - float(p.get("stage_t0", now))) / max(0.2, float(p.get("stage_est", 1.0)))
                    target = max(float(p.get("done", 0.0)), a + (b - a) * (1 - math.exp(-1.5 * pace)))
                    step, total = p.get("step", ""), float(p.get("total", total_s))
                    # the reports' own scale, fitted into what is left of the bar: no stall when the
                    # start-up guess ran ahead of them, or when they fall back (a stage turned out
                    # bigger - how much text needs a second look is known only then)
                    if anchor is None or target < last - 1e-4:
                        anchor = (shown, target)
                    last = target
                    target = anchor[0] + (1 - anchor[0]) * (target - anchor[1]) / max(1e-6, 1 - anchor[1])
                shown = max(shown, min(0.99, target))   # never backwards
                elapsed = now - t0
                left = total * (1 - shown)
                if shown > 0.3:  # well under way: this run's own pace counts too
                    left = 0.5 * left + 0.5 * elapsed * (1 - shown) / shown
                left = f" - about {left:.0f} s left" if left >= 1 else ""
                bar.progress(shown, text=f"{step}{left} ({elapsed:.0f} s so far)")
                time.sleep(0.1)
        progress_path.unlink(missing_ok=True)
        message = err_path.read_text()[-3000:]
        err_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            bar.empty()
            raise RuntimeError(message)
    bar.progress(1.0, text="Done")
    bar.empty()


def run_image_model(image_bytes: bytes, image_name: str, checkpoint: str, device: str, model_version: bytes = b"") -> dict:
    # the result is read from this session's folder (no cache shared between sessions); the key
    # holds the model's file time (model_key), so a model retrained under the same name runs again.
    # infer_with_progress has written the result already; this only reads it (and runs it, with
    # no bar, if it is missing)
    key = _digest(image_bytes, model_key(checkpoint), INFER_VERSION)
    work = session_dir()
    out_path = work / f"{key}.json"
    if not out_path.exists():
        img_path = work / f"{key}_{Path(image_name).name}"
        img_path.write_bytes(image_bytes)
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
        try:
            infer_with_progress(image_file.getvalue(), image_file.name, checkpoint, on)
            infer = run_image_model(image_file.getvalue(), image_file.name, checkpoint, on, model_key(checkpoint))
        except RuntimeError as exc:
            st.error("Image model failed:")
            st.code(str(exc))
            st.stop()
        note.empty()
        # the drawing's tags, where known (a spec sheet; a draw.io file given with the picture):
        # OCR reads a look-alike off one of them become it (ocr_tags.snap_to_known)
        known = [row["tag"] for row in (st.session_state.get("spec_sheet") or {}).get("rows", {}).values()]
        if drawio_review is not None:
            known += [s.tag for s in drawio_review.symbols if s.tag]
        image_review = from_image(infer, image_file.name, known_tags=known or None)
    review = cross_check(drawio_review, image_review, infer) if drawio_review and image_review \
        else drawio_review or image_review
    if review is not None and not frames_as_sections:
        # Frames drawn around groups of items are not part of a P&ID (team review, 2026-10): no
        # plant sections from them. Their lines are still known as frames, so never pipes.
        for s in review.symbols:
            s.section = ""
        if review.drawio is not None:
            review.drawio.sections = []
    return review


def decode_drawio_bytes(data: bytes, name: str) -> tuple[str, Path]:
    path = session_dir() / f"d{_digest(data)}_{Path(name).name}"
    if not path.exists():
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
if st.session_state.pop("rerun_model", False) and image_file is not None:
    # "Re-run the model": drop the kept result for this picture, so it is processed again
    (session_dir() / f"{_digest(image_file.getvalue(), model_key(checkpoint), INFER_VERSION)}.json").unlink(missing_ok=True)
input_key = _digest(
    drawio_file.getvalue() if drawio_file else b"", image_file.getvalue() if image_file else b"",
    review_upload.getvalue() if review_upload else b"", model_key(checkpoint),
    str(st.session_state.get("model_run", 0)).encode(),  # a re-run starts the review again
    b"sections" if frames_as_sections else b"",
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
    st.session_state.update(input_key=input_key, review=review, editor_version=0,
                            review_picture=PICTURE_KEY if image_file and not uploaded_drawio else "")
    if st.session_state.get("editor_open"):
        st.rerun()  # the editor at the top takes each cell's details from this new review
review: ReviewDiagram = st.session_state["review"]

# The spec sheet: every item whose tag is in it gets its values - automatically into empty
# fields, once per drawing (and after each picture Save, for items added in the editor);
# over what is there when the editor asks ("Sync from the spec sheet").
sheet = st.session_state.get("spec_sheet")
if sheet:
    mark = (st.session_state.get("input_key"), sheet["digest"], st.session_state.get("editor_version", 0))
    rerun = False
    if st.session_state.get("spec_applied") != mark:
        res = apply_spec_sheet(review, sheet["rows"])
        st.session_state["spec_applied"] = mark
        if res["fields"] or res.get("types"):
            typed = f", {res['types']} DEXPI types set" if res.get("types") else ""
            st.toast(f"Spec sheet: {res['fields']} values filled in on {res['items']} items{typed}.")
            rerun = True
        st.session_state["spec_result"] = res
    sync = st.session_state.pop("spec_sync", None)
    if sync:
        res = apply_spec_sheet(review, sheet["rows"], overwrite=True, ids=None if sync == "all" else set(sync))
        typed = f", {res['types']} DEXPI types set" if res.get("types") else ""
        st.toast(f"Synced from the spec sheet: {res['items']} items matched, {res['fields']} values written{typed}."
                 if res["items"] else "No item's tag is in the spec sheet.")
        st.session_state["spec_result"] = {**res, "unused": st.session_state.get("spec_result", {}).get("unused", [])}
        rerun = bool(res["fields"] or res.get("types"))
    if rerun and st.session_state.get("editor_open"):
        st.rerun()  # the editor at the top shows the new specifications
    res = st.session_state.get("spec_result") or {}
    with spec_slot:
        if res.get("problems"):
            with st.expander(f"Spec sheet: {len(res['problems'])} values not used"):
                st.markdown("\n".join(f"- {p}" for p in res["problems"][:50]))
        if res.get("unused"):
            st.caption(f"Not on this drawing: {', '.join(res['unused'][:12])}" + (" ..." if len(res["unused"]) > 12 else ""))
elif st.session_state.pop("spec_sync", None):
    st.toast("Upload a spec sheet in the sidebar first.")
with spec_slot:
    st.download_button("Download a spec sheet with this drawing's tags", lambda r=review: template_xlsx(r),
                       file_name=f"{Path(review.source_name).stem}_specs.xlsx", width="stretch",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       help="The template with every tagged item of this drawing on its tab (and the specifications "
                            "already set) - fill in the rest and upload it.")

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
    # review.look_reasons: the same rule flags items in the editor (a tag used twice included)
    rv = st.session_state.get("review")
    return bool(look_reasons(s, duplicate_tags(rv) if rv is not None else None))


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
FLOW_BASIS = {"drawn arrow": "arrow", "implied": "implied", "reviewer": "reviewer", "both ways": "both"}


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
    # flow arrows are not items (is_flow_arrow): not drawn - only their pipes' flow shows them
    arrows = {s.id for s in review.symbols if is_flow_arrow(s)}
    boxes = [Box(s.id, tuple(s.box), MODEL_ONLY if only(s) else symbol_color(s, set()),
                 "" if only(s) or status(s) == "left out" else item_name(s.id), 3,
                 _symbol_tip(s), only(s) or _left_out(s), "model" if only(s) else _box_kind(s))
             for s in review.symbols if s.box and s.id not in arrows]
    ocr = _ocr_boxes(review)
    name = {s.id: item_name(s.id) for s in review.symbols}
    basis = {"drawn arrow": "arrow", "implied": "implied", "reviewer": "reviewer"}
    conns = [(c.id, c.from_id, c.to_id, c.line_type, c.include,
              f"{name.get(c.from_id, c.from_id)} -> {name.get(c.to_id, c.to_id)}: {LINE_LABELS.get(c.line_type, c.line_type)}"
              f", {found_by(c)}" + (f", flow: {FLOW_LABELS.get(c.flow, c.flow)}" if c.flow else ""),
              c.path, basis.get(c.flow, "")) for c in review.connections
             if c.from_id not in arrows and c.to_id not in arrows]
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
               "reviewer": "set by you", "both ways": "both ways (two-way pipe)",
               "no flow between them": "no flow (same side of a tee)", "": "unknown"}
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
n_out = sum(status(s) == "left out" and not is_flow_arrow(s) for s in review.symbols)
n_conn = sum(c.include for c in review.connections)
fs = review.drawio.flow if review.drawio is not None and review.drawio.flow else None
n_unknown = fs.unknown if fs else sum(c.include and c.line_type == "process" and not c.flow for c in review.connections)
m = st.columns(5)
m[0].metric("Items in DEXPI", n_in)
m[1].metric("Items to check", n_check, help="Marked ! in the editor ('Needs a look' in its Find list); the guided "
            "check takes them first.")
m[2].metric("Items left out", n_out, help="Decorative shapes, labels, or excluded by you")
m[3].metric("Lines in DEXPI", n_conn)
m[4].metric("Pipes with unknown flow", n_unknown, help="Pipes (not instrument taps) with no arrow on them or "
            "their neighbours - grey on the map")

# With the editor open (the main screen) the map, its side panel and the tables are the old
# view: folded away, still there for bulk edits in the tables.
_old_view = (st.expander("The map, side panel and tables (the old view - everything is in the editor above)",
                         expanded=False)
             if st.session_state.get("editor_open") and EDITOR_INSTALLED else st.container())
with _old_view:
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
    shown_syms = [s for s in review.symbols if keep(s) and in_section(s) and not is_flow_arrow(s)]
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


STREAM_IN, STREAM_OUT = "In - a feed entering the drawing", "Out - a product leaving the drawing"


def stream_pipes(sid: str) -> list:
    return [c for c in review.connections if sid in (c.from_id, c.to_id) and c.line_type == "process"]


def stream_way(sid: str) -> str | None:
    """In or out, from the flow on its pipe(s): DEXPI's FlowIn/FlowOut connector follows it."""
    pipes = [c for c in stream_pipes(sid) if c.flow in FLOW_KNOWN]
    if pipes and all(c.from_id == sid for c in pipes):
        return STREAM_IN
    if pipes and all(c.to_id == sid for c in pipes):
        return STREAM_OUT
    return None


def _set_stream(sid: str, key: str) -> None:
    """In: its pipe(s) flow away from it; out: towards it (set by the reviewer)."""
    out = st.session_state[key] == STREAM_OUT
    for c in stream_pipes(sid):
        other = c.to_id if c.from_id == sid else c.from_id
        want = (other, sid) if out else (sid, other)
        if (c.from_id, c.to_id) != want:
            c.from_id, c.to_id = want
            c.path = (c.path or [])[::-1] or None
        c.flow = "reviewer"
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


def _merge(keep_id: str, other_ids: list, cover: bool = False) -> None:
    """Fold duplicate boxes of one item into `keep_id`: their lines move to it (dropping
    lines that would join it to itself or repeat an existing line), a missing tag / type /
    specifications are taken from them, their rows go. The kept item keeps its own box -
    pick the tight one to keep - unless `cover`: then its box grows to cover them all (the
    model split one symbol into parts)."""
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
        if cover and keep.box and o.box:
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
            fields = spec_fields(s.dexpi_class) if s.dexpi_class != SKIP else []
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
                if s.dexpi_class == OFF_PAGE:
                    if stream_pipes(s.id):
                        way = stream_way(s.id)
                        st.radio("Stream", [STREAM_IN, STREAM_OUT], index=[STREAM_IN, STREAM_OUT].index(way) if way else None,
                                 key=f"way_{s.id}_{k}", on_change=_set_stream, args=(s.id, f"way_{s.id}_{k}"),
                                 help="Sets the flow on its pipe: away from it (in) or towards it (out). DEXPI: a "
                                      "FlowIn- or FlowOutPipeOffPageConnector.")
                        if not way:
                            st.caption("In or out is not known yet: pick one.")
                    else:
                        st.caption("A stream needs its pipe: connect it to an item first (Connect tab).")
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
                        arrow = (("→" if c.from_id == s.id else "←") if c.flow in FLOW_KNOWN
                                 else "↔" if c.flow == "both ways" else "–")
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
                    cover = st.checkbox("Grow its box to cover both", key=f"merge_cover_{s.id}_{k}",
                                        help=f"Off: {item_name(s.id)} keeps its own box - keep the tight one. On: "
                                             "the box grows to cover both (one symbol the model found in parts).")
                    if pick and st.button(f"Merge into {item_name(s.id)}", key=f"do_merge_{s.id}_{k}", width="stretch"):
                        _merge(s.id, [near[labels.index(pick)].id], cover=cover)
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
with st.expander("Add this drawing to the training data"):
    from pid_dexpi_aas.image.drawio_labels import review_to_drawio, safe_name
    from pid_dexpi_aas.image.training_export import add_review_to_training, list_reviewed, retrain_status
    # what it is worth, and how ready it is: the labels teach the model as they are
    t_items = [s for s in review.symbols if s.include and not is_flow_arrow(s)]
    t_lines = [c for c in review.connections if c.include]
    unchecked = (sum(not s.checked for s in t_items), sum(not c.checked for c in t_lines))
    st.markdown(f"This drawing adds **{len(t_items) + len(t_lines)} units** of training data "
                f"({len(t_items)} items + {len(t_lines)} connections).")
    if any(unchecked):
        st.warning(f"Not checked yet: {unchecked[0]} items and {unchecked[1]} connections. The model learns these "
                   "labels as they are - go through the editor's guided check first.")
    from_drawing = review.drawio is not None and drawio_file is not None
    st.caption("Teach the model from this sheet once it is checked: "
               + ("the drawing is rendered by draw.io and labelled like the team's drawings (items, lines, "
                  "dashed = signal)." if from_drawing else
                  "the boxes, lines, line types and paths as you checked them here - on the map, in the side "
                  "panel or in the editor - are the labels. Only add a sheet whose boxes fit their symbols."))
    t1, t2 = st.columns([3, 2])
    t_name = safe_name(t1.text_input("Name", safe_name(Path(review.source_name).stem.removesuffix(".drawio")),
                                     key="train_name"))
    t_use = t2.radio("Use it for", ["Training", "Held-out test"], horizontal=True, key="train_use",
                     help="Held-out: a fixed set of unseen sheets every model is scored on. Keep a sheet in one of "
                          "the two, never both.")
    t_subset = "Reviewed" if t_use == "Training" else "Reviewed test"
    t_other = "Reviewed test" if t_subset == "Reviewed" else "Reviewed"
    t_existing = {(d["name"], d["subset"]) for d in list_reviewed(DATA_ROOT)}
    a1, a2 = st.columns(2)
    if (t_name, t_other) in t_existing:
        a1.error(f"'{t_name}' is already {'held-out test' if t_other.endswith('test') else 'training'} data - "
                 "remove it there first (Training data, in the sidebar).")
    elif a1.button("Replace it in the training data" if (t_name, t_subset) in t_existing else "Add to the training data",
                   type="primary", width="stretch", key="train_add"):
        with st.spinner("Writing the training tiles..."):
            try:
                if from_drawing:
                    sys.path.insert(0, str(ROOT / "ml" / "pid2graph" / "data"))
                    from make_team_data import process as drawing_to_training
                    with tempfile.TemporaryDirectory() as tmp:
                        f = Path(tmp) / f"{t_name}.drawio"
                        f.write_bytes(drawio_file.getvalue())
                        # labelled like the live model's own team data (Team v3: no team arrow labels -
                        # v4/v5's arrow labels made links worse and read valves as arrows)
                        info = drawing_to_training(f, DATA_ROOT, t_subset, 1500, 500, arrows=False)
                    done = f"{info['symbols']} items, {info['edges']} lines, {info['patches']} tiles"
                else:
                    info = add_review_to_training(review, DATA_ROOT, t_subset, t_name)
                    done = (f"{info['symbols']} items, {info['lines']} lines, {info['tiles']} tiles "
                            f"(scaled x{info['scale']:.2f} to the model's symbol size)")
                st.success(f"Added '{t_name}' as {'held-out test' if t_subset.endswith('test') else 'training'} "
                           f"data: {done}. It is listed under Training data in the sidebar.")
            except Exception as exc:
                st.error(f"Could not add it: {exc}")
    # the finished sheet as a .drawio file: the edited drawing, or the picture with its checked labels
    if from_drawing:
        a2.download_button("Download the drawing (.drawio)", drawio_file.getvalue(), file_name=f"{t_name}.drawio",
                           mime="application/xml", width="stretch", key="train_dl")
    elif review.image_path:
        a2.download_button("Download the drawing (.drawio)", lambda r=review: review_to_drawio(r).encode(),
                           file_name=f"{t_name}.drawio", mime="application/xml", on_click="ignore",
                           width="stretch", key="train_dl")
    st.caption("Or: download the drawing, correct it in draw.io, and add it under 'Add a drawing labelled in "
               "draw.io' in the sidebar.")
    # automatic retraining (ml/pid2graph/auto_retrain.py, nightly): how far the next one is
    rs = retrain_status(DATA_ROOT)
    st.markdown("**Automatic retraining**")
    st.progress(min(1.0, rs["new_units"] / rs["threshold"]),
                text=f"New since the last retraining: {rs['new_units']} of {rs['threshold']} units "
                     f"({len(rs['new'])} sheets) - 10 % of the {rs['base_units']} hand-labelled units the model has")
    st.caption("When enough has been added, a new model is trained at night (~4 h) and replaces the live one only "
               f"if it scores no worse on the held-out sheets ({rs['held_out']} so far - add some as 'Held-out "
               "test', or new models wait for approval). Value = labelled items + connections, so a large P&ID "
               "counts for more than a small one.")
    if rs["last"]:
        st.caption(f"Last retraining ({rs['last']['when']}): {rs['last'].get('result', '')}")
    if not PUBLIC:
        r1, r2 = st.columns(2)
        if r1.button("Retrain now", width="stretch", help="Trains a candidate now (~4 h on the GPU; pictures run on "
                     "the processor meanwhile), checked against the held-out sheets as at night"):
            subprocess.Popen(["systemd-inhibit", "--what=sleep:idle", "--why=pid-dexpi retrain", str(ROOT / ".venv" / "bin" / "python"),
                              str(ROOT / "ml" / "pid2graph" / "auto_retrain.py"), "--force"],
                             stdout=open(ROOT / "ml" / "logs" / "auto_retrain.log", "a"), stderr=subprocess.STDOUT,
                             cwd=ROOT, start_new_session=True)
            st.success("Retraining started - see ml/logs/auto_retrain.log.")
        if rs["pending"] and r2.button(f"Deploy the held-back model ({rs['pending']['when']})", width="stretch",
                                       help="Trained without held-out sheets to check it on: deploy it as the live model"):
            subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "ml" / "pid2graph" / "auto_retrain.py"),
                            "--approve"], cwd=ROOT)
            st.success("Deployed - the old model is in ml/models/archive/.")

b1, b2 = st.columns([1, 1])
with b2:
    st.download_button("Save review (.json) to continue later", lambda r=review: review_to_json(r),
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
        if sid and c2.button("Show it", key=f"prob_{n}", help="Selects it in the editor (and on the map)"):
            select({sid}, (), zoom=True)
            st.session_state["editor_focus"] = f"{sid}|{time.time()}"
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
    d1.download_button("Download DEXPI XML", lambda b=xml_bytes: b, file_name=f"{stem}.xml", mime="application/xml")
    with st.spinner("Packaging the AAS..."):
        aas_in = DexpiModelInput(stem, xml_bytes, drawing_svg(diagram, result),
                                 {k: v for k, v in aas_model_meta.items() if v})
        aas_data = aasx_bytes([aas_in], {k: v for k, v in aas_plant.items() if v}, version=aas_version,
                              base_iri=aas_base or None)
        aas_ok, aas_problems = check_aasx(aas_data)
    d3.download_button(f"Download AAS {aas_version} (.aasx)", lambda b=aas_data: b, file_name=f"{stem}.aasx", mime="application/asset-administration-shell-package",
                       help="Asset Administration Shell with the IDTA 02012-1-0 DEXPI Submodel: the DEXPI file, the "
                            "drawing as SVG, and every tagged item mapped to a global id.")
    d3.button("Add this drawing to the plant AAS", key="plant_add", width="stretch", on_click=_plant_add,
              args=(stem, xml_bytes, drawing_svg(diagram, result), {k: v for k, v in aas_model_meta.items() if v}),
              help="Several drawings of one plant in one AAS (below: 'Plant AAS')")
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
        "Download check sheet for review (.xlsx)", lambda b=sheet: b, file_name=f"{stem}_check_sheet.xlsx",
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
                   "Instruments are drawn with their line to where they measure; signal lines (dashed) are in the DEXPI file but not drawn.")
        try:
            after = render_after_png(diagram, result)
            if after is None:
                st.info(CAIRO_MISSING)
            else:
                st.image(Image.open(io.BytesIO(after)), width="stretch")
        except Exception as exc:
            st.warning(f"Could not re-draw from the DEXPI file: {exc}")
    if report.skipped_details:
        with st.expander(f"Skipped elements ({len(report.skipped_details)})"):
            for detail in report.skipped_details:
                st.text(detail)


# ------------------------------------------------------------------------ plant AAS ---
# Several drawings of one plant as one AAS: one shell for the plant, one DEXPI submodel with a
# Model01, Model02 ... per drawing (each its DEXPI file, its SVG and its tag mapping). A tag on
# several drawings gets the same global asset id in each, so the plant's twins find one asset.
plant = st.session_state.setdefault("plant_models", {})
with st.expander(f"Plant AAS - several drawings in one AAS ({len(plant)} drawings)", expanded=bool(plant)):
    st.caption("Build each drawing's DEXPI file and press 'Add this drawing to the plant AAS' (or add DEXPI files "
               "made before), then download one AAS for the whole plant. The plant details are the ones under "
               "'AAS details' above. A tag that appears on several drawings (a line crossing between sheets, an "
               "instrument shown twice) is one asset in the AAS.")
    more = st.file_uploader("Add DEXPI files (.xml) made before", type=["xml"], accept_multiple_files=True,
                            key="plant_upload")
    for f in more or []:
        if Path(f.name).stem not in plant:
            plant[Path(f.name).stem] = {"xml": f.getvalue(), "svg": None, "meta": {}}
    if plant:
        tags = {name: {e.tag for e in mapping_entries(m["xml"]) if e.kind == "TagMapping"} for name, m in plant.items()}
        shared = {t for t in set().union(*tags.values()) if sum(t in v for v in tags.values()) > 1}
        for name in list(plant):
            c1, c2 = st.columns([5, 1])
            c1.write(f"**{name}** - {len(tags[name])} tagged items")
            if c2.button("Remove", key=f"plant_rm_{name}"):
                plant.pop(name)
                st.rerun()
        if len(plant) > 1:
            st.caption(f"On more than one drawing: {len(shared)} tags" + (f" ({', '.join(sorted(shared)[:15])}"
                       + (" ..." if len(shared) > 15 else "") + ")" if shared else ""))
        models = [DexpiModelInput(name, m["xml"], m["svg"], m["meta"]) for name, m in plant.items()]
        plant_meta = {k: v for k, v in aas_plant.items() if v}
        plant_data = aasx_bytes(models, plant_meta, version=aas_version, base_iri=aas_base or None)
        ok, problems = check_aasx(plant_data)
        plant_stem = _slug_name(plant_meta.get("ProcessPlantName") or "plant")
        st.download_button(f"Download the plant AAS {aas_version} (.aasx)", lambda b=plant_data: b, file_name=f"{plant_stem}.aasx",
                           mime="application/asset-administration-shell-package", type="primary")
        if ok:
            st.caption(f"✓ Valid AAS {aas_version} · {len(models)} drawings · "
                       f"{sum(len(v) for v in tags.values())} tagged items mapped")
        else:
            st.warning("The plant AAS did not pass the compliance check")
            st.text("\n".join(problems))
