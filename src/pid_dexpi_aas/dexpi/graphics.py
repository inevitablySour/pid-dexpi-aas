"""The drawing in a DEXPI file: Proteus 4.1 graphics added to the exported XML.

A DEXPI 1.3 file can carry its P&ID drawing beside the plant data, as commercial P&ID tools
write it (see the DEXPI TrainingTestCases, e.g. E06V01-AUD.EX01.xml):

- a <Drawing> with the sheet's <Extent> and a <Label> per tag (a <Text> naming its item);
- a <ShapeCatalogue> holding each symbol's geometry once (<PolyLine>, <Shape>, <Circle>), in
  its own coordinates around its origin;
- on each item: <Extent>, <Position> (its origin on the sheet), <Scale> and ComponentName,
  naming its symbol in the catalogue;
- on each pipe segment and signal line: a <CenterLine> per drawn stretch.

pyDEXPI's exporter writes the plant data only, so this module adds the graphics afterwards,
to the elements by their IDs ("<Class>-<object id>"). Geometry comes from the reviewed drawing:
a CAD symbol's own picture (cad/to_drawio.py writes it as SVG) or else the app's ISO 10628
symbol (render/symbol_library.py), pipes as drawn (line routes). Units are mm, Y upwards.
"""
from __future__ import annotations

import base64
import hashlib
import math
import re
import xml.etree.ElementTree as ET

MM = 0.25   # mm per draw.io unit (cad/to_drawio.py draws 4 units per mm)
_SVG_IMAGE = re.compile(r"image=data:image/svg\+xml,([A-Za-z0-9+/=]+)")
_PATH = re.compile(r'<path d="([^"]+)"(?: fill="([^"]+)")?')


def _f(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") if v else "0"


def _xyz(tag: str, x: float, y: float) -> ET.Element:
    return ET.Element(tag, X=_f(x), Y=_f(y), Z="0")


def _extent(xs, ys) -> ET.Element:
    e = ET.Element("Extent")
    e.append(_xyz("Min", min(xs), min(ys)))
    e.append(_xyz("Max", max(xs), max(ys)))
    return e


def _position(x: float, y: float, angle: float = 0.0) -> ET.Element:
    p = ET.Element("Position")
    p.append(_xyz("Location", x, y))
    p.append(ET.Element("Axis", X="0", Y="0", Z="1"))
    p.append(ET.Element("Reference", X=_f(math.cos(angle)), Y=_f(math.sin(angle)), Z="0"))
    return p


def _presentation(weight: float = 0.35, dashed: bool = False) -> ET.Element:
    # Proteus colours are 0-1; LineType 0 solid, 2 dashed
    return ET.Element("Presentation", Layer="0", LineType="2" if dashed else "0",
                      LineWeight=_f(weight), R="0", G="0", B="0")


def _polyline(points, closed=False, filled=False, dashed=False) -> ET.Element:
    """A drawn stroke: <PolyLine> (open), or <Shape> (closed; Filled="Solid" when filled)."""
    if closed and points[0] != points[-1]:
        points = list(points) + [points[0]]
    el = ET.Element("Shape" if closed else "PolyLine", NumPoints=str(len(points)))
    if closed and filled:
        el.set("Filled", "Solid")
    el.append(_presentation(dashed=dashed))
    el.append(_extent([p[0] for p in points], [p[1] for p in points]))
    for x, y in points:
        el.append(_xyz("Coordinate", x, y))
    return el


def _circle(cx: float, cy: float, r: float) -> ET.Element:
    el = ET.Element("Circle", Radius=_f(r))
    el.append(_presentation())
    el.append(_extent([cx - r, cx + r], [cy - r, cy + r]))
    el.append(_position(cx, cy))
    return el


def _svg_strokes(svg: str, w: float, h: float) -> list:
    """A CAD symbol's picture (our own SVG: one path per stroke, "M x,y L x,y ... [Z]") as
    strokes in symbol coordinates: mm, around its centre, Y upwards, scaled to w x h mm."""
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not vb:
        return []
    vw, vh = float(vb.group(1)), float(vb.group(2))
    kx, ky = (w / vw if vw else 1.0), (h / vh if vh else 1.0)
    dashed = "stroke-dasharray" in svg
    out = []
    for d, fill in _PATH.findall(svg):
        closed = d.rstrip().endswith("Z")
        pts = [(float(a), float(b)) for a, b in re.findall(r"(-?[\d.]+),(-?[\d.]+)", d)]
        pts = [((x - vw / 2) * kx, (vh / 2 - y) * ky) for x, y in pts]
        if len(pts) >= 2:
            out.append(_polyline(pts, closed=closed, filled=closed and fill not in (None, "", "none"), dashed=dashed))
    return out


def _library_strokes(class_name: str, w: float, h: float) -> list:
    """The app's ISO 10628 symbol for a DEXPI class (a 0-100 box, Y down), at w x h mm."""
    from pydexpi.dexpi_classes.graphics import Ellipse, PolyLine, Polygon
    from pid_dexpi_aas.render.symbol_library import build_primitives_for_class
    T = lambda x, y: ((x - 50) * w / 100, (50 - y) * h / 100)
    out = []
    for prim in build_primitives_for_class(class_name):
        if isinstance(prim, (PolyLine, Polygon)):
            pts = [T(p.x, p.y) for p in prim.points]
            if len(pts) >= 2:
                filled = isinstance(prim, Polygon) and str(getattr(prim, "fillStyle", "")).lower().endswith("solid")
                out.append(_polyline(pts, closed=isinstance(prim, Polygon), filled=filled))
        elif isinstance(prim, Ellipse):
            cx, cy = T(prim.center.x, prim.center.y)
            rx, ry = prim.horizontalSemiAxis * w / 100, prim.verticalSemiAxis * h / 100
            if abs(rx - ry) < 0.01 * max(rx, ry, 1e-9):
                out.append(_circle(cx, cy, rx))
            else:
                out.append(_polyline([(cx + rx * math.cos(t * math.pi / 18), cy + ry * math.sin(t * math.pi / 18))
                                      for t in range(36)], closed=True))
    return out


def _rotation(style: str) -> float:
    m = re.search(r"rotation=(-?[\d.]+)", style or "")
    return -math.radians(float(m.group(1))) if m else 0.0   # draw.io turns clockwise, Y down


_CONNECTORS = ("FlowInPipeOffPageConnector", "FlowOutPipeOffPageConnector")


def element_map(root: ET.Element, diagram, result) -> dict:
    """{draw.io shape id: its XML element}. By the object's ID; an off-page connector the
    build replaced (its flow class decided late) is found in the segment of its pipe instead."""
    by_id = {}
    for el in root.iter():
        i = el.get("ID")
        if i and "-" in i:
            by_id[i.split("-", 1)[1]] = el
    out = {}
    for s in diagram.shapes:
        obj = result.shape_objects.get(s.id)
        el = by_id.get(getattr(obj, "id", None)) if obj is not None else None
        if el is None and obj is not None and type(obj).__name__ in _CONNECTORS:
            for e in diagram.edges:
                if s.id not in (e.source, e.target):
                    continue
                for seg in result.edge_segments.get(e.id, []) or []:
                    ends = [i for i in (getattr(seg, "items", None) or []) if type(i).__name__ in _CONNECTORS]
                    if not ends:
                        continue
                    taken = {id(x) for x in out.values()}
                    ends = [i for i in ends if id(by_id.get(i.id)) not in taken] or ends
                    pick = ends[0] if len(ends) == 1 else (ends[0] if e.source == s.id else ends[-1])
                    el = by_id.get(pick.id)
                    if el is not None and any(x is el for x in out.values()):
                        el = None
                    if el is not None:
                        break
                if el is not None:
                    break
        if el is not None:
            out[s.id] = el
    return out


def add_connector_references(xml: bytes, diagram, result) -> bytes:
    """An off-page connector's name and the drawing it continues on, written as DEXPI has
    them: a PipeOffPageConnectorReferenceByNumber (ReferencedDrawingNumber) and its name (the
    text on the drawing, e.g. "MEDIUM 01") as a custom attribute - pyDEXPI writes neither."""
    root = ET.fromstring(xml)
    els = element_map(root, diagram, result)
    n = 0
    for s in diagram.shapes:
        el = els.get(s.id)
        if el is None or el.tag != "PipeOffPageConnector":
            continue
        drawing = next((g.get("Value") for g in el.iter("GenericAttribute") if g.get("Name") == "ConnectedDrawing"), None)
        name = (s.text or "").replace("\n", " ").strip()
        if name and not name.startswith("<"):
            ga = el.find("GenericAttributes[@Set='CustomAttributes']")
            if ga is None:
                ga = ET.Element("GenericAttributes", Set="CustomAttributes", Number="0")
                at = next((k for k, c in enumerate(el) if c.tag not in ("PersistentID", "Presentation", "Extent", "Position",
                                                                         "Scale", "ConnectionPoints", "GenericAttributes")), len(el))
                el.insert(at, ga)
            ET.SubElement(ga, "GenericAttribute", Name="ConnectorName", AttributeURI="urn:pid-dexpi-aas:spec:ConnectorName",
                          Format="string", Value=name)
            ga.set("Number", str(len(ga)))
        if drawing and el.find("PipeOffPageConnectorReference") is None:
            ref = ET.SubElement(el, "PipeOffPageConnectorReference", ID=f"PipeOffPageConnectorReferenceByNumber-{el.get('ID')}",
                                ComponentClass="PipeOffPageConnectorReferenceByNumber",
                                ComponentClassURI="http://sandbox.dexpi.org/rdl/PipeOffPageConnectorReferenceByNumber")
            ga = ET.SubElement(ref, "GenericAttributes", Set="DexpiAttributes", Number="1")
            ET.SubElement(ga, "GenericAttribute", Name="ReferencedDrawingNumberAssignmentClass",
                          AttributeURI="http://sandbox.dexpi.org/rdl/ReferencedDrawingNumberAssignmentClass",
                          Format="string", Value=drawing)
        n += 1
    add_connector_references.last = n
    ET.indent(root, "  ")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")


def _pipe_end(routes: dict, p) -> tuple | None:
    """The pipe end nearest point p (draw.io coordinates), within 30 units: (end point, unit
    direction of the flow there, whether the pipe leaves the sheet there)."""
    best = None
    for r in routes.values():
        if len(r) < 2:
            continue
        for end, prev, leaving in ((r[-1], r[-2], True), (r[0], r[1], False)):
            d = math.dist(end, p)
            if best is None or d < best[0]:
                v = (end[0] - prev[0], end[1] - prev[1]) if leaving else (prev[0] - end[0], prev[1] - end[1])
                n = math.hypot(*v) or 1.0
                best = (d, end, (v[0] / n, v[1] / n), leaving)
    return best[1:] if best and best[0] <= 30 else None


def add_drawing(xml: bytes, diagram, result, source=None) -> bytes:
    """`xml` (a DEXPI file from dexpi/serialize.py) with the drawing of `diagram` (the
    reviewed draw.io diagram it was built from) and `result` (its BuildResult) added.
    `source`: the diagram as uploaded (review.drawio), whose shapes keep their own pictures
    (a CAD symbol) where the reviewed rows carry only their DEXPI class."""
    root = ET.fromstring(xml)
    drawn = {s.id: s.style for s in (source.shapes if source is not None else [])}
    drawio_symbols = getattr(source, "symbols", None) or getattr(diagram, "symbols", None) or {}
    els = element_map(root, diagram, result)
    by_id = {}
    for el in root.iter():
        i = el.get("ID")
        if i and "-" in i:
            by_id[i.split("-", 1)[1]] = el

    shapes = [s for s in diagram.shapes if result.shape_objects.get(s.id) is not None]
    routes = diagram.routes or {}
    ys = [s.y + (s.height or 0) for s in shapes] + [p[1] for r in routes.values() for p in r] or [0]
    top = max(ys)
    X = lambda x: x * MM
    Y = lambda y: (top - y) * MM

    catalogue = ET.Element("ShapeCatalogue", Name="Symbols")
    drawing = ET.Element("Drawing", Name=diagram.source_file or "drawing", Type="PID",
                         Title=(getattr(diagram, "metadata", {}) or {}).get("drawingName", diagram.source_file or ""))
    drawing.append(_presentation(0))
    sheet_x, sheet_y = [], []
    symbols: dict[str, str] = {}
    placed = 0
    for s in shapes:
        obj = result.shape_objects[s.id]
        el = els.get(s.id)
        if el is None:
            continue
        w, h = X(s.width or 40), (s.height or 40) * MM
        cx, cy = X(s.x + (s.width or 40) / 2), Y(s.y + (s.height or 40) / 2)
        m = _SVG_IMAGE.search(s.style or "") or _SVG_IMAGE.search(drawn.get(s.id) or "")
        own = None if m else drawio_symbols.get(s.id)
        angle = _rotation(s.style)
        stream = type(obj).__name__ in _CONNECTORS and not m and not own
        if stream:   # a stream with no symbol drawn (a loose pipe end, a text label): the usual
            # stream arrow at the pipe's end, pointing the way it flows
            end = _pipe_end(routes, (s.x + (s.width or 0) / 2, s.y + (s.height or 0) / 2))
            w, h = 10.0, 5.0
            if end:
                (px, py), (ux, uy), leaving = end
                ex, ey = X(px), Y(py)
                ux, uy = ux, -uy   # draw.io Y down -> DEXPI Y up
                k = 1 if leaving else -1   # past the end of a pipe that leaves; before one that enters
                cx, cy = ex + k * ux * w / 2, ey + k * uy * w / 2
                angle = math.atan2(uy, ux)
            key = "STREAM_ARROW"
        if stream:
            pass
        elif m:   # the CAD symbol itself
            key = "CAD_" + hashlib.sha1(f"{m.group(1)}|{w:.2f}|{h:.2f}".encode()).hexdigest()[:10]
        elif own:   # the symbol as draw.io draws it on this drawing
            local = [([(round(X(x) - cx, 3), round(Y(y) - cy, 3)) for x, y in pts], closed, filled)
                     for pts, closed, filled in own]
            key = "DRAWIO_" + hashlib.sha1(repr(local).encode()).hexdigest()[:10]
        else:
            key = f"{type(obj).__name__}_{w:.1f}x{h:.1f}".replace(".", "_")
        if key not in symbols and stream:
            strokes = [_polyline([(-5, -2.5), (2.5, -2.5), (5, 0), (2.5, 2.5), (-5, 2.5)], closed=True)]
            sym = ET.SubElement(catalogue, "PipeOffPageConnector", ID=key, ComponentName=key)
            sym.extend(strokes)
            symbols[key] = key
        if key not in symbols:
            strokes = (_svg_strokes(base64.b64decode(m.group(1)).decode("utf-8", "replace"), w, h) if m
                       else [_polyline(pts, closed=closed, filled=filled) for pts, closed, filled in local if len(pts) > 1]
                       if own else _library_strokes(type(obj).__name__, w, h))
            sym = ET.SubElement(catalogue, el.tag, ID=key, ComponentName=key)
            sym.extend(strokes)
            symbols[key] = key
        el.set("ComponentName", key)
        graphics = [_extent([cx - w / 2, cx + w / 2], [cy - h / 2, cy + h / 2]),
                    _position(cx, cy, angle), ET.Element("Scale", X="1", Y="1", Z="1")]
        at = 1 if len(el) and el[0].tag == "PersistentID" else 0
        for k, g in enumerate(graphics):
            el.insert(at + k, g)
        sheet_x += [cx - w / 2, cx + w / 2]
        sheet_y += [cy - h / 2, cy + h / 2]
        placed += 1
        # its tag, as a label on the drawing: in an instrument bubble, else below the symbol
        tag = getattr(obj, "tagName", None) or (s.text or "").replace("\n", " ").strip()
        if tag and not tag.startswith("<"):
            inside = type(obj).__name__ == "ProcessInstrumentationFunction"
            lx, ly = cx, (cy if inside else cy - (max(w, h) if stream else h) / 2 - 2.5)
            label = ET.SubElement(drawing, "Label", ID=f"Label-{obj.id}", ComponentClass="Label", ComponentName="Tag")
            label.append(_extent([lx], [ly]))
            label.append(_position(lx, ly))
            if inside and " " in tag:   # an instrument bubble: letters above, number below
                letters, number = tag.split(" ", 1)
                size = min(3.0, max(1.5, h * 0.25))
                for part, dy in ((letters, size * 0.6), (number, -size * 0.6)):
                    t = ET.SubElement(label, "Text", Justification="CenterCenter", Width="0", Height=_f(size),
                                      Font="Arial", String=part, ItemID=el.get("ID"))
                    t.append(_presentation(0))
                    t.append(_extent([lx], [ly + dy]))
                    t.append(_position(lx, ly + dy))
                continue
            text = ET.SubElement(label, "Text", Justification="CenterCenter" if inside else "CenterTop",
                                 Width="0", Height=_f(min(3.0, max(1.5, h * 0.3)) if inside else 2.5),
                                 Font="Arial", String=tag, ItemID=el.get("ID"))
            text.append(_presentation(0))
            text.append(_extent([lx], [ly]))
            text.append(_position(lx, ly))

    def centre_lines(edge_ids, dashed=False):
        out = []
        for eid in edge_ids:
            r = routes.get(eid)
            if r and len(r) >= 2:
                pts = [(X(x), Y(y)) for x, y in r]
                cl = ET.Element("CenterLine", NumPoints=str(len(pts)))
                cl.append(_presentation(0.5 if not dashed else 0.25, dashed))
                cl.append(_extent([p[0] for p in pts], [p[1] for p in pts]))
                for x, y in pts:
                    cl.append(_xyz("Coordinate", x, y))
                sheet_x.extend(p[0] for p in pts)
                sheet_y.extend(p[1] for p in pts)
                out.append(cl)
        return out

    def attach(el, lines):
        if el is None or not lines:
            return 0
        # before the segment's <Connection>, as in the DEXPI 1.3 reference files
        conn = el.find("Connection")
        at = list(el).index(conn) if conn is not None else len(el)
        for k, cl in enumerate(lines):
            el.insert(at + k, cl)
        return len(lines)

    # pipes: a direct line onto its segment; a network's drawn pieces onto the network's
    # segments, each piece onto the segment between the items it runs from or to
    edges = list(diagram.drawn_edges or diagram.edges)
    members: dict[str, list] = {}
    for e in edges:
        if e.merged_into:
            members.setdefault(e.merged_into, []).append(e)
    lines = 0
    for key, segs in result.edge_segments.items():
        segs = [sg for sg in segs if sg is not None]
        if not segs:
            continue
        if key.startswith("net"):
            continue
        lines += attach(by_id.get(segs[0].id), centre_lines([key]))
    for net, pieces in members.items():
        keys = [k for k in result.edge_segments if k.split(":", 1)[0] == net]
        if not keys:
            continue
        for e in pieces:
            ends = {e.source, e.target}
            k = next((k for k in keys if ends & set(k.split(":")[1:])), keys[0])
            segs = [sg for sg in result.edge_segments[k] if sg is not None]
            if segs:
                lines += attach(by_id.get(segs[0].id), centre_lines([e.id]))
    signals = 0
    for key, flows in result.edge_signals.items():
        for f in (flows if isinstance(flows, list) else [flows]):
            signals += attach(by_id.get(getattr(f, "id", None)), centre_lines([key], dashed=True))

    if sheet_x:
        pad = 10.0
        drawing.insert(1, _extent([0, max(sheet_x) + pad], [0, max(sheet_y) + pad]))
    info = root.find("PlantInformation")
    at = list(root).index(info) + 1 if info is not None else 0
    if sheet_x:
        root.insert(at, _extent([0, max(sheet_x) + pad], [0, max(sheet_y) + pad]))
        at += 1
    root.insert(at, drawing)
    root.insert(at + 1, catalogue)
    ET.indent(root, "  ")
    add_drawing.last = {"symbols": placed, "shapes_in_catalogue": len(symbols), "pipe_lines": lines,
                        "signal_lines": signals, "labels": len(drawing.findall("Label"))}
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")
