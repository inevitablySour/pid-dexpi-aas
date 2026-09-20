"""Decode draw.io's compressed <diagram> payload into plain mxGraph XML.

draw.io stores each diagram tab as base64(deflate(urlencode(xml))) inside
<diagram>...</diagram>. Copied from handoff/decode.py (confirmed working on all 8 team
files during the handoff) into the package so run_pipeline.py can import it directly.
"""
from __future__ import annotations

import base64
import re
import urllib.parse
import zlib
from pathlib import Path


def decode_drawio(path: str | Path) -> list[str]:
    """Return the plain mxGraph XML for each <diagram> tab in a .drawio file."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    diagrams = re.findall(r"<diagram[^>]*>(.*?)</diagram>", content, re.S)
    decoded = []
    for payload in diagrams:
        payload = payload.strip()
        if not payload:
            decoded.append("")
            continue
        try:
            raw = base64.b64decode(payload)
            inflated = zlib.decompress(raw, -15)
            xml = urllib.parse.unquote(inflated.decode("utf-8"))
        except Exception:
            xml = payload  # already plain, uncompressed XML
        decoded.append(xml)

    return decoded
