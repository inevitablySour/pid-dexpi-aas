"""The draw.io editor inside the app (the Editor input tab).

draw.io's own web editor (Apache 2.0; installed by scripts/get_drawio_editor.py into
drawio_editor_files/webapp) runs in embed mode inside a Streamlit component, served by
the app itself - nothing is loaded from or sent to diagrams.net. Next to it, the DEXPI
panel (drawio_editor_files/index.html) sets each item's type, tag, specifications and each
line's type and flow, kept in the drawing (pid_dexpi_aas/drawio/editor_data.py). Save (or
Ctrl+S) returns the drawing to the app, which reads it like an uploaded .drawio file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit.components.v1 as components

FILES = Path(__file__).with_name("drawio_editor_files")
INSTALLED = (FILES / "webapp" / "index.html").exists()

# embed mode, JSON messages, our configuration; no exit button, page tabs, styling panel
# or external services
PARAMS = "&".join([
    "embed=1", "proto=json", "spin=1", "configure=1", "libraries=1", "noExitBtn=1", "saveAndExit=0",
    "stealth=1", "lang=en", "ui=kennedy", "pages=0", "format=0",
])
CONFIG = {
    # only the P&ID shapes the converter understands (plus the panel's lines and labels)
    "defaultLibraries": "pid",
    "enabledLibraries": ["pid"],
    # new lines are right-angled pipes
    "defaultEdgeStyle": {"edgeStyle": "orthogonalEdgeStyle", "rounded": 0, "orthogonalLoop": 1,
                         "jettySize": "auto", "html": 1},
    "compressXml": False,
}

_component = components.declare_component("drawio_editor", path=str(FILES))


def write_schema() -> None:
    """The panel's lists - DEXPI types, specification fields per class - as a file next to it
    (fetched once, instead of resent on every page update)."""
    from pid_dexpi_aas.drawio.editor_data import spec_schema
    from pid_dexpi_aas.review import DEXPI_CLASS_CHOICES, INSTRUMENT_CLASSES, OFF_PAGE, SKIP, VALVE_CLASSES
    classes = [c for c in DEXPI_CLASS_CHOICES if c not in (SKIP, OFF_PAGE)]
    data = json.dumps({"choices": DEXPI_CLASS_CHOICES, "skip": SKIP, "off_page": OFF_PAGE,
                       "instruments": INSTRUMENT_CLASSES, "valves": VALVE_CLASSES,
                       "classes": spec_schema(classes + ["PipingNetworkSegment", "SignalLineFunction",
                                                         "MeasuringLineFunction"])}, sort_keys=True)
    path = FILES / "schema.json"
    if not path.exists() or path.read_text() != data:
        path.write_text(data)


def drawio_editor(xml: str, patch: dict | None = None, *, title: str = "", height: int = 820,
                  key: str = "drawio_editor") -> dict | None:
    """Show the editor with `xml` (a draw.io file's text; "" = a blank page) and each cell's
    DEXPI details (`patch`, editor_data.editor_patch). Returns {"xml", "t"} after each Save,
    else the last value (None before the first)."""
    patch = patch or {}
    return _component(xml=xml, patch=patch, patch_id=hashlib.sha1(json.dumps(patch, sort_keys=True).encode()).hexdigest(),
                      title=title, params=PARAMS, config=CONFIG, height=height, key=key, default=None)
