"""A DEXPI file's own drawing as SVG: the symbols in its <ShapeCatalogue> placed where its items
are (<Position>, ComponentName), its pipes and signal lines along their <CenterLine>s, and its
tag <Label>s - the Proteus 4.1 graphics P&ID tools write (and dexpi/graphics.py adds to ours).

So the picture shows what is in the file, drawn with the drawing's own symbols (a CAD file's
symbols come through as drawn), not stand-in shapes. Units are the file's (mm), Y upwards.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from html import escape


def _num(el, name, default=0.0) -> float:
    try:
        return float(el.get(name, default))
    except (TypeError, ValueError):
        return default


def _xy(el) -> tuple[float, float]:
    return _num(el, "X"), _num(el, "Y")


def has_drawing(xml: bytes) -> bool:
    root = ET.fromstring(xml)
    return root.find("ShapeCatalogue") is not None or any(True for _ in root.iter("CenterLine"))


def dexpi_svg(xml: bytes, margin: float = 5.0) -> str | None:
    """SVG text of the drawing in DEXPI file `xml`; None when the file has no drawing."""
    root = ET.fromstring(xml)
    catalogue = {}
    cat = root.find("ShapeCatalogue")
    if cat is not None:
        for sym in cat:
            catalogue[sym.get("ComponentName") or sym.get("ID")] = sym
    in_catalogue = {id(e) for sym in catalogue.values() for e in sym.iter()}
    xs, ys, out = [], [], []

    def stroke(el):
        pr = el.find("Presentation")
        w = _num(pr, "LineWeight", 0.35) if pr is not None else 0.35
        dashed = pr is not None and pr.get("LineType") not in (None, "0", "")
        colour = "#000"
        if pr is not None and pr.get("R") is not None:
            r, g, b = (_num(pr, c) for c in "RGB")
            if max(r, g, b) <= 1:
                r, g, b = r * 255, g * 255, b * 255
            colour = f"rgb({int(r)},{int(g)},{int(b)})" if (r, g, b) != (255, 255, 255) else "#000"
        return (f'stroke="{colour}" stroke-width="{max(w, 0.18):.2f}"'
                + (' stroke-dasharray="2,1.2"' if dashed else ""))

    def place(T):
        def P(x, y):
            X, Y = T(x, y)
            xs.append(X)
            ys.append(Y)
            return f"{X:.2f},{-Y:.2f}"
        return P

    def primitives(el, T, scale=1.0):
        P = place(T)
        for g in el:
            if g.tag in ("PolyLine", "Line", "Shape", "CenterLine"):
                pts = [P(*_xy(c)) for c in g.findall("Coordinate")]
                if len(pts) < 2:
                    continue
                if g.tag == "Shape":
                    fill = "#000" if g.get("Filled") == "Solid" else "none"
                    out.append(f'<polygon points="{" ".join(pts)}" fill="{fill}" {stroke(g)}/>')
                else:
                    out.append(f'<polyline points="{" ".join(pts)}" fill="none" {stroke(g)}/>')
            elif g.tag in ("Circle", "Ellipse"):
                loc = g.find("Position/Location")
                cx, cy = _xy(loc) if loc is not None else (0.0, 0.0)
                rx = _num(g, "Radius", _num(g, "PrimaryAxis", 1)) * scale
                ry = _num(g, "Radius", _num(g, "SecondaryAxis", rx)) * scale
                c = P(cx, cy).split(",")
                xs.extend([float(c[0]) - rx, float(c[0]) + rx])
                ys.extend([-float(c[1]) - ry, -float(c[1]) + ry])
                fill = "#000" if g.get("Filled") == "Solid" else "none"
                out.append(f'<ellipse cx="{c[0]}" cy="{c[1]}" rx="{rx:.2f}" ry="{ry:.2f}" fill="{fill}" {stroke(g)}/>')
            elif g.tag == "TrimmedCurve":
                circ = g.find("Circle")
                if circ is None:
                    continue
                loc = circ.find("Position/Location")
                cx, cy = _xy(loc) if loc is not None else (0.0, 0.0)
                r = _num(circ, "Radius", 1)
                a0, a1 = math.radians(_num(g, "StartAngle")), math.radians(_num(g, "EndAngle"))
                if a1 <= a0:
                    a1 += 2 * math.pi
                n = max(6, int((a1 - a0) / (math.pi / 18)))
                pts = [P(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n))
                       for k in range(n + 1)]
                out.append(f'<polyline points="{" ".join(pts)}" fill="none" {stroke(circ)}/>')

    # items placed on the sheet with a catalogue symbol
    for el in root.iter():
        if id(el) in in_catalogue:
            continue
        sym = catalogue.get(el.get("ComponentName"))
        pos = el.find("Position")
        if sym is None or pos is None:
            continue
        loc, ref = pos.find("Location"), pos.find("Reference")
        ox, oy = _xy(loc) if loc is not None else (0.0, 0.0)
        c, s = (_xy(ref) if ref is not None else (1.0, 0.0))
        n = math.hypot(c, s) or 1.0
        c, s = c / n, s / n
        sc = el.find("Scale")
        kx, ky = (_num(sc, "X", 1), _num(sc, "Y", 1)) if sc is not None else (1.0, 1.0)
        primitives(sym, lambda x, y, ox=ox, oy=oy, c=c, s=s, kx=kx, ky=ky:
                   (ox + kx * x * c - ky * y * s, oy + kx * x * s + ky * y * c), abs(kx))

    # pipes and signal lines: their centre lines (absolute coordinates); a flow arrow at the end
    # of a pipe whose segment says it runs one way
    for seg in root.iter():
        lines = seg.findall("CenterLine")
        if not lines:
            continue
        holder = ET.Element("x")
        holder.extend(lines)
        primitives(holder, lambda x, y: (x, y))

    # tags
    for text in root.iter("Text"):
        if id(text) in in_catalogue:
            continue
        loc = text.find("Position/Location")
        if loc is None or not text.get("String"):
            continue
        x, y = _xy(loc)
        h = _num(text, "Height", 2.5) or 2.5
        h = h if h > 0.5 else 2.5
        just = text.get("Justification", "CenterCenter")
        anchor = "start" if just.startswith("Left") else "end" if just.startswith("Right") else "middle"
        base = ("hanging" if "Top" in just else "auto" if "Bottom" in just else "central")
        xs.append(x)
        ys.append(y)
        out.append(f'<text x="{x:.2f}" y="{-y:.2f}" font-size="{h:.2f}" font-family="Arial, sans-serif" '
                   f'text-anchor="{anchor}" dominant-baseline="{base}" fill="#1a1a1a">{escape(text.get("String"))}</text>')

    if not out:
        return None
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    w, h = x1 - x0, y1 - y0
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {-y1:.2f} {w:.2f} {h:.2f}" '
            f'width="{w:.1f}mm" height="{h:.1f}mm" stroke-linecap="round" stroke-linejoin="round">'
            f'<rect x="{x0:.2f}" y="{-y1:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#fff"/>'
            + "".join(out) + "</svg>")


def drawing_of(diagram, result, source=None) -> bytes | None:
    """The DEXPI file (with its drawing) for a built model, made the way the app's download is:
    for pictures of a model that was not exported yet (the check sheet, the AAS)."""
    import tempfile
    from pathlib import Path

    from pid_dexpi_aas.dexpi.graphics import add_connector_references, add_drawing
    from pid_dexpi_aas.dexpi.proteus_patch import PatchedProteusSerializer

    with tempfile.TemporaryDirectory() as tmp:
        PatchedProteusSerializer().save(result.model, Path(tmp), "m")
        xml = (Path(tmp) / "m.xml").read_bytes()
    return add_drawing(add_connector_references(xml, diagram, result), diagram, result, source)
