"""Parse decoded mxGraph (draw.io) XML into plain shape/edge records.

Every mxCell with vertex="1" is a shape (equipment, valve, instrument bubble, or a
floating text label with no `shape=` style token). Every mxCell with edge="1" is a
piping/signal connector between two shape ids.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class Shape:
    id: str
    shape_token: str | None  # e.g. "mxgraph.pid.vessels.reactor", or None for plain rects/text
    text: str  # HTML-stripped label/value text
    x: float
    y: float
    width: float
    height: float
    style: str

    @property
    def is_labeled_only(self) -> bool:
        """True for plain text/rect cells that carry a tag rather than an equipment symbol."""
        return self.shape_token is None


@dataclass
class Edge:
    id: str
    source: str | None
    target: str | None
    style: str


@dataclass
class ParsedDiagram:
    source_file: str
    shapes: list[Shape] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


_TAG_RE = re.compile(r"<[^>]+>")
_SHAPE_TOKEN_RE = re.compile(r"(?:^|;)shape=([^;]+)")


def _strip_html(value: str) -> str:
    """mxCell value attrs are HTML-escaped fragments like '&lt;span&gt;BPR&lt;/span&gt;'."""
    unescaped = html.unescape(value or "")
    no_tags = _TAG_RE.sub(" ", unescaped)
    return re.sub(r"\s+", " ", no_tags).strip()


def _extract_shape_token(style: str) -> str | None:
    match = _SHAPE_TOKEN_RE.search(style or "")
    if not match:
        return None
    token = match.group(1)
    # "text;html=1;..." styles use shape=text or no shape= at all; treat "text" as no shape.
    return None if token in ("text",) else token


def parse_drawio_xml(xml_text: str, source_file: str) -> ParsedDiagram:
    root = ET.fromstring(xml_text)
    diagram = ParsedDiagram(source_file=source_file)

    for cell in root.iter("mxCell"):
        cell_id = cell.get("id")
        if cell_id in (None, "0", "1"):
            continue  # the two mxGraph root cells carry no shape data

        style = cell.get("style", "")

        if cell.get("edge") == "1":
            diagram.edges.append(
                Edge(
                    id=cell_id,
                    source=cell.get("source"),
                    target=cell.get("target"),
                    style=style,
                )
            )
            continue

        if cell.get("vertex") == "1":
            geometry = cell.find("mxGeometry")
            x = float(geometry.get("x", 0)) if geometry is not None else 0.0
            y = float(geometry.get("y", 0)) if geometry is not None else 0.0
            width = float(geometry.get("width", 0)) if geometry is not None else 0.0
            height = float(geometry.get("height", 0)) if geometry is not None else 0.0

            diagram.shapes.append(
                Shape(
                    id=cell_id,
                    shape_token=_extract_shape_token(style),
                    text=_strip_html(cell.get("value", "")),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    style=style,
                )
            )

    return diagram


def parse_drawio_file(path: str | Path) -> ParsedDiagram:
    path = Path(path)
    return parse_drawio_xml(path.read_text(encoding="utf-8"), source_file=path.name)
