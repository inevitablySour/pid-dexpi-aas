"""The draw.io editor inside the app (the Editor input tab).

draw.io's own web editor (Apache 2.0; installed by scripts/get_drawio_editor.py into
drawio_editor_files/webapp) runs in embed mode inside a Streamlit component, served by
the app itself - nothing is loaded from or sent to diagrams.net. Save (or Ctrl+S) in the
editor returns the drawing's XML to the app, which reads it like an uploaded .drawio file.
"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

FILES = Path(__file__).with_name("drawio_editor_files")
INSTALLED = (FILES / "webapp" / "index.html").exists()

# embed mode, JSON messages, our configuration; no exit button, no external services
PARAMS = "&".join([
    "embed=1", "proto=json", "spin=1", "configure=1", "libraries=1", "noExitBtn=1", "saveAndExit=0",
    "stealth=1", "lang=en", "ui=kennedy",
])
CONFIG = {
    # the P&ID shapes the converter understands, open next to the general ones
    "defaultLibraries": "general;pid",
    # new lines are right-angled pipes
    "defaultEdgeStyle": {"edgeStyle": "orthogonalEdgeStyle", "rounded": 0, "orthogonalLoop": 1,
                         "jettySize": "auto", "html": 1},
    "compressXml": False,
}

_component = components.declare_component("drawio_editor", path=str(FILES))


def drawio_editor(xml: str, *, title: str = "", height: int = 820, key: str = "drawio_editor") -> dict | None:
    """Show the editor with `xml` (a draw.io file's text; "" = a blank page). Returns
    {"xml", "t"} after each Save, else the last value (None before the first)."""
    return _component(xml=xml, title=title, params=PARAMS, config=CONFIG, height=height, key=key, default=None)
