"""The P&ID as a PDF carrying its data (render/pdf_export.py): vector drawing with searchable
tags, a note with each item's data, the DEXPI file attached, the drawing's title."""
import ctypes
import tempfile
from pathlib import Path

import pytest

from pid_dexpi_aas.dexpi.graphics import add_drawing
from pid_dexpi_aas.dexpi.serialize import save_and_verify
from pid_dexpi_aas.drawio import line_routes
from pid_dexpi_aas.drawio.decode import decode_drawio
from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved
from pid_dexpi_aas.render.pdf_export import dexpi_pdf, item_notes
from pid_dexpi_aas.review import from_drawio, to_dexpi

DRAWING = Path(__file__).parent / "data" / "reviewer_simple_pid.drawio"
pytestmark = pytest.mark.skipif(not line_routes.available(), reason="needs the draw.io app (line routes)")


@pytest.fixture(scope="module")
def drawn():
    review = from_drawio(parse_drawio_xml_resolved(decode_drawio(DRAWING)[0], source_file=DRAWING.name,
                                                   drawio_path=DRAWING))
    diagram, result = to_dexpi(review)
    with tempfile.TemporaryDirectory() as tmp:
        save_and_verify(result.model, Path(tmp), "m")
        return add_drawing((Path(tmp) / "m.xml").read_bytes(), diagram, result, review.drawio)


def test_each_item_gets_a_note_with_its_data(drawn):
    notes = item_notes(drawn)
    assert len(notes) >= 4
    assert all(n["text"] and n["box"][2] > n["box"][0] for n in notes)
    assert any("ProcessInstrumentationFunction" in n["text"] for n in notes)


def test_the_pdf_has_the_drawing_the_notes_and_the_file(drawn):
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    pdf = dexpi_pdf(drawn, "m.xml")
    doc = pdfium.PdfDocument(pdf)
    page = doc[0]
    assert raw.FPDFPage_GetAnnotCount(page.raw) == len(item_notes(drawn))
    assert raw.FPDFDoc_GetAttachmentCount(doc.raw) == 1   # the DEXPI file inside
    att = raw.FPDFDoc_GetAttachment(doc.raw, 0)
    n = ctypes.c_ulong(0)
    raw.FPDFAttachment_GetFile(att, None, 0, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    raw.FPDFAttachment_GetFile(att, buf, n.value, ctypes.byref(n))
    assert buf.raw[:n.value] == drawn
    assert "TI" in page.get_textpage().get_text_range()   # the tags are text: searchable
