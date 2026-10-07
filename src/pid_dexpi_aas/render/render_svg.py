"""Render a built Diagram to SVG, correcting a confirmed pyDEXPI 1.2.0 viewBox bug.

pyDEXPI's DrawDiagram.draw_svg() calls SvgRenderer.reset_bounds() as its first line,
discarding the min/max X/Y that DrawDiagram.__init__ seeds from Diagram.minX/minY/maxX/
maxY. The viewBox actually written is then rebuilt from SvgRenderer.update_bounds() calls
triggered while converting each primitive - but those calls use each primitive's raw LOCAL
coordinates (our symbol_library.py's normalized 0-100 box), never accounting for the
per-shape <g transform="translate(...) scale(...)"> that places it on the diagram. So the
computed viewBox only ever spans roughly 0-100 regardless of where symbols are actually
positioned - confirmed by direct testing (a diagram spanning ~650x500 drawio units
produced viewBox="-20 -20 40 40"). transform_content_group() does compute transformed
bounds correctly, but it is only invoked for a single shape's OWN isolated SVG (e.g.
DrawRepresentationGroup used standalone), not accumulated across a whole Diagram's
multiple placed shapes by DrawDiagram itself.

Rather than patch pyDEXPI's renderer internals for the viewBox specifically (the local
ShapeUsage placement/scale/rotation math needed to reproduce bounds correctly is exactly
what DrawShapeUsage._create_content_group already does internally - not worth
re-deriving), this module renders normally via DrawDiagram, then overwrites the root
<svg>'s width/height/viewBox attributes using the ALREADY-CORRECT global bounds
build_diagram.py computed from the original drawio shape geometry.

Importing pid_dexpi_aas.render.svg_renderer_patch (for its side effect) fixes a separate,
unrelated pyDEXPI rendering bug: dashed lines and filled polygons (needed for arrowheads)
never rendering correctly. See that module's docstring for why a class-level monkeypatch
was needed instead of subclassing DrawDiagram.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from pydexpi.dexpi_classes.graphics import Diagram
from pydexpi.loaders.svg_loader import DrawDiagram

import pid_dexpi_aas.render.svg_renderer_patch  # noqa: F401 - import applies the monkeypatch


def render_diagram_svg(diagram: Diagram, output_path: str | Path, padding: float = 20.0) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drawer = DrawDiagram(diagram, padding=padding, pretty=True)
    svg_root = drawer.draw_svg(return_element=True)

    # DEXPI Y increases upward; SVG Y increases downward (same flip as
    # SvgRenderer.to_svg_y, applied here to our own already-correct bounds).
    width = diagram.maxX - diagram.minX + 2 * padding
    height = diagram.maxY - diagram.minY + 2 * padding
    viewbox_x = diagram.minX - padding
    viewbox_y = -diagram.maxY - padding

    svg_root.set("width", str(width))
    svg_root.set("height", str(height))
    svg_root.set("viewBox", f"{viewbox_x} {viewbox_y} {width} {height}")

    # The background rect DrawDiagram already drew used its own (wrong) bounds - fix it
    # to match, since a stale small background rect would otherwise sit visibly wrong.
    background_rect = svg_root.find("rect")
    if background_rect is not None:
        background_rect.set("x", str(viewbox_x))
        background_rect.set("y", str(viewbox_y))
        background_rect.set("width", str(width))
        background_rect.set("height", str(height))

    ET.indent(svg_root, space="  ")
    xml_bytes = ET.tostring(svg_root, encoding="unicode", xml_declaration=True)
    output_path.write_text(xml_bytes, encoding="utf-8")
    return output_path
