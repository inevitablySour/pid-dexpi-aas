"""The P&ID as a PDF that carries its plant data, from a DEXPI file with its drawing:

- the drawing in vector form (render/dexpi_svg.py), its tags real text - searchable, copyable;
- on each item an invisible note (a PDF Square annotation) with its tag, DEXPI type and
  specifications - PDF viewers show it when the item is hovered or clicked;
- the DEXPI file itself attached (an embedded file, as e-invoices carry their XML), so the one
  PDF is both the drawing and the machine-readable model;
- the drawing's title and number as the document's information.

Written with pypdfium2 (already the app's PDF reader) on a PDF drawn by cairosvg.
"""
from __future__ import annotations

import ctypes
import io
import re
import xml.etree.ElementTree as ET

MM_TO_PT = 72 / 25.4


def _wide(text: str):
    """A PDFium wide string (UTF-16LE, zero-terminated) as a ctypes buffer."""
    data = (text + "\0").encode("utf-16-le")
    buf = ctypes.create_string_buffer(data, len(data))
    _keep.append(buf)   # (kept alive until the PDF is saved)
    return ctypes.cast(buf, ctypes.POINTER(ctypes.c_ushort))


_keep: list = []


def _label(name: str) -> str:
    """"nominalCapacityVolume" -> "Nominal capacity volume"."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).split()
    return " ".join([words[0].capitalize()] + [w.lower() if not w.isupper() else w for w in words[1:]]) if words else name


_SKIP = re.compile(r"^(TagName|PipingComponentName|PipingComponentNumber|ProcessInstrumentationFunction"
                   r"(Category|Number|Modifier)|FlowDirectionBasis|FileName)", re.I)


def _attributes(el) -> list[str]:
    """"Label: value unit" for an element's own attributes (DEXPI and custom), read from the XML -
    pyDEXPI's reader leaves several out (custom attributes, some quantities)."""
    out = []
    for ga in el.findall("GenericAttributes"):
        for g in ga.findall("GenericAttribute"):
            name = re.sub(r"AssignmentClass$", "", g.get("Name", ""))
            value = (g.get("Value") or "").strip()
            if not value or _SKIP.match(name) or value.startswith("review:"):
                continue
            unit = g.get("Units") or ""
            unit = {"MetreCubed": "m³", "Bar": "bar", "BarGauge": "barg", "DegreeCelsius": "°C",
                    "Millimetre": "mm", "Metre": "m"}.get(unit, unit)
            out.append(f"{_label(name)}: {value} {unit}".strip())
    return out


def item_notes(xml: bytes) -> list[dict]:
    """[{"box": (x0, y0, x1, y1) in the file's mm, "title": tag, "text": the note}] for every item
    the drawing places, from the plant data in the same file."""
    from pid_dexpi_aas.dexpi.to_drawio import _class, _tag, read_plant
    plant = read_plant(xml)
    by_pid = {getattr(o, "proteusId", None): o for o in plant["items"].values()}
    root = ET.fromstring(xml)
    out = []
    for el in root.iter():
        ext, obj = el.find("Extent"), by_pid.get(el.get("ID"))
        if obj is None or ext is None or el.find("Position") is None or type(obj).__name__ == "PipeTee":
            continue   # (a tee is where pipes meet, nothing to say about)
        mn, mx = ext.find("Min"), ext.find("Max")
        if mn is None or mx is None:
            continue
        box = tuple(float(v.get(k, 0)) for v, k in ((mn, "X"), (mn, "Y"), (mx, "X"), (mx, "Y")))
        if box[2] - box[0] < 0.5 or box[3] - box[1] < 0.5:
            continue
        cls = _class(obj)
        attrs = _attributes(el)
        tag = _tag(obj) or next((a.split(": ", 1)[1] for a in attrs if a.startswith("Connector name")), "")
        out.append({"box": box, "title": tag or cls, "text": "\n".join([f"{tag or '(no tag)'} - {cls}"] + attrs)})
    return out


def dexpi_pdf(xml: bytes, filename: str = "model.xml") -> bytes | None:
    """A PDF of the drawing in DEXPI file `xml` (one with its drawing) with each item's data on
    it and the file attached; None when the file has no drawing."""
    import cairosvg
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    from pid_dexpi_aas.render.dexpi_svg import dexpi_svg
    svg = dexpi_svg(xml)
    if not svg:
        return None
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    vx, vy, vw, vh = vb   # SVG: y down = the file's -Y; the file's Y range is -vy - vh .. -vy
    pdf_bytes = _draw_pdf(svg, _title(xml))
    doc = pdfium.PdfDocument(pdf_bytes)
    page = doc[0]
    pw, ph = page.get_size()
    kx, ky = pw / vw, ph / vh
    y_low = -vy - vh   # the file's lowest Y: the page's bottom edge
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%SZ")
    for note in item_notes(xml):
        x0, y0, x1, y1 = note["box"]
        annot = raw.FPDFPage_CreateAnnot(page.raw, raw.FPDF_ANNOT_SQUARE)
        if not annot:
            continue
        rect = raw.FS_RECTF((x0 - vx) * kx, (y1 - y_low) * ky, (x1 - vx) * kx, (y0 - y_low) * ky)
        raw.FPDFAnnot_SetRect(annot, ctypes.byref(rect))
        raw.FPDFAnnot_SetBorder(annot, 0.0, 0.0, 0.0)   # invisible: the symbol itself shows
        raw.FPDFAnnot_SetStringValue(annot, b"Contents", _wide(note["text"]))
        raw.FPDFAnnot_SetStringValue(annot, b"T", _wide(note["title"]))
        raw.FPDFAnnot_SetStringValue(annot, b"M", _wide(stamp))   # (else viewers show 1970)
        raw.FPDFPage_CloseAnnot(annot)
    # the DEXPI file itself, attached
    att = raw.FPDFDoc_AddAttachment(doc.raw, _wide(filename))
    if att:
        buf = ctypes.create_string_buffer(xml, len(xml))
        raw.FPDFAttachment_SetFile(att, doc.raw, buf, len(xml))
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()


def _title(xml: bytes) -> str:
    """"<drawing number> - <drawing name>" from the file's metadata, else "P&ID"."""
    root = ET.fromstring(xml)
    meta = {re.sub(r"AssignmentClass$", "", g.get("Name", "")): g.get("Value", "")
            for m in root.findall("MetaData") for g in m.iter("GenericAttribute")}
    return " - ".join(v for v in (meta.get("DrawingNumber"), meta.get("DrawingName")) if v) or "P&ID"


def _draw_pdf(svg: str, title: str) -> bytes:
    """The SVG as a vector PDF (cairosvg), its title and subject in the document information."""
    import cairocffi
    from cairosvg.parser import Tree
    from cairosvg.surface import PDFSurface

    class Titled(PDFSurface):
        def finish(self):
            try:
                self.cairo.set_metadata(cairocffi.PDF_METADATA_TITLE, title)
                self.cairo.set_metadata(cairocffi.PDF_METADATA_SUBJECT, "P&ID with its DEXPI plant data attached")
                self.cairo.set_metadata(cairocffi.PDF_METADATA_CREATOR, "pid-dexpi-aas")
            except Exception:   # (cairo older than 1.16: no metadata)
                pass
            super().finish()
    out = io.BytesIO()
    Titled.convert(bytestring=svg.encode(), write_to=out)
    return out.getvalue()
