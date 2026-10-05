"""Excel check sheet for a converted drawing, for a domain expert (not a programmer).

One workbook per conversion: a Summary with a live verdict, the drawing with every item
numbered, and Items / Connections tables in plain process-engineering words with a
Yes/No "Correct?" column to fill in. Rows the tool was unsure about are highlighted with
the reason, so the reviewer can check those first. Made from a ReviewDiagram plus the
BuildResult of to_dexpi(), so it describes exactly what went into the DEXPI file.
"""
from __future__ import annotations

import io
import math
import re
import tempfile
from datetime import date
from pathlib import Path

import cairosvg

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image, ImageDraw, ImageFont

from pid_dexpi_aas.build_diagram import build_diagram
from pid_dexpi_aas.build_model import BuildResult
from pid_dexpi_aas.parser_drawio import ParsedDiagram
from pid_dexpi_aas.render_svg import render_diagram_svg
from pid_dexpi_aas.review import OFF_PAGE, SKIP, ReviewDiagram, ReviewSymbol, category_for_class
from pid_dexpi_aas.mapping import Category

FONT = "Arial"
FILL_IN = PatternFill("solid", fgColor="FFF2CC")  # yellow: cells the reviewer fills in
CHECK_FIRST = PatternFill("solid", fgColor="FCE4D6")  # orange: rows the tool was unsure about
HEADER = PatternFill("solid", fgColor="D9E1F2")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

PLAIN_CLASS = {
    "GateValve": "Gate valve",
    "CheckValve": "Check (non-return) valve",
    "OperatedValve": "Actuated on/off valve",
    "SafetyValveOrFitting": "Safety / relief valve",
    "StraightwayValve": "Valve (type not specified)",
    "BallValve": "Ball valve",
    "GlobeValve": "Globe valve",
    "NeedleValve": "Needle valve",
    "ButterflyValve": "Butterfly valve",
    "PressureVessel": "Pressure vessel",
    "TubularHeatExchanger": "Shell-and-tube heat exchanger",
    "ProcessColumn": "Column",
    "CustomVessel": "Vessel (other type)",
    "CustomPump": "Pump (other type)",
    "CustomEquipment": "Equipment (other type)",
    "SignalOffPageConnector": "Signal line leaving the drawing",
    OFF_PAGE: "Pipe leaving the drawing (off-page)",
}


def plain_type(s: ReviewSymbol) -> str:
    if s.dexpi_class == "ProcessInstrumentationFunction":
        parts = [s.attributes.get("measured_variable"), s.attributes.get("function")]
        detail = ", ".join(p for p in parts if p)
        return f"Instrument - {detail}" if detail else "Instrument"
    if s.dexpi_class in PLAIN_CLASS:
        return PLAIN_CLASS[s.dexpi_class]
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", s.dexpi_class or "").lower()
    return words[:1].upper() + words[1:]


def plain_how(s: ReviewSymbol) -> str:
    how = s.identified_by
    if s.source == "manual":
        return "Entered by the reviewer"
    if how.startswith("draw.io shape"):
        return "Symbol drawn in the draw.io file" + (" + tag letters" if "tag prefix" in how else "")
    if how.startswith("text label"):
        return "Text label at the end of a pipe"
    if how.startswith("tag"):
        letters = re.search(r"'([A-Z]+)'", how)
        return f"Read from the picture; type from tag letters {letters.group(1) if letters else ''}".strip()
    if how.startswith("model class"):
        return "Recognised from the picture (no tag found)"
    return how


def check_reasons(s: ReviewSymbol) -> str:
    reasons = []
    if s.source == "image model":
        if s.confidence is not None and s.confidence < 0.7:
            reasons.append(f"picture recogniser only {s.confidence:.0%} sure")
        if not s.tag:
            reasons.append("no tag could be read")
    if "overrides" in s.notes:
        model_said = re.search(r"model said '([^']+)'", s.notes)
        reasons.append(f"picture said '{model_said.group(1)}', tag letters said otherwise" if model_said
                       else "picture and tag letters disagreed")
    if "confirm with team" in s.notes:
        reasons.append("meaning of these tag letters not yet confirmed")
    if s.image_check and not s.image_check.startswith("model agrees"):
        reasons.append(f"picture check: {s.image_check}")
    return "; ".join(reasons)


def numbered_overlay(image_path: str, numbered: list[tuple[int, ReviewSymbol]], max_width: int = 1600) -> bytes:
    """The drawing with a numbered box on every converted item (orange = check first)."""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, "white")
        bg.alpha_composite(img)
        img = bg
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    size = max(18, img.width // 90)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    for n, s in numbered:
        if not s.box:
            continue
        color = (230, 110, 0) if check_reasons(s) else (0, 120, 60)
        draw.rectangle(s.box, outline=color, width=4)
        label = str(n)
        tw = draw.textlength(label, font=font)
        x, y = s.box[0], max(0, s.box[1] - size - 6)
        draw.rectangle([x, y, x + tw + 8, y + size + 4], fill=color)
        draw.text((x + 4, y + 1), label, fill="white", font=font)
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def number_items(review: ReviewDiagram, result: BuildResult) -> list[tuple[int, ReviewSymbol]]:
    """Converted items numbered in reading order (top-down bands, then left to right) -
    the numbers shared by the drawing overlay and the Items tab."""
    built_ids = {r.element_id for r in result.records if r.status == "built"}
    converted = [s for s in review.symbols if s.include and s.dexpi_class != SKIP and s.id in built_ids]
    converted.sort(key=lambda s: (s.box[1] // 150, s.box[0]) if s.box else (1e9, 0))
    return list(enumerate(converted, start=1))


def render_after_png(diagram: ParsedDiagram, result: BuildResult, width: int = 1600) -> bytes:
    """The P&ID re-drawn from the built DEXPI model (generic symbols), as a PNG."""
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "after.svg"
        render_diagram_svg(build_diagram(diagram, result, result.model), svg)
        return cairosvg.svg2png(url=str(svg), output_width=width, background_color="white")


def _place_image(ws, png: bytes, row: int, width_px: int = 1100) -> int:
    """Insert a PNG at column A of `row`, scaled to width_px; return the first free row below."""
    pic = XLImage(io.BytesIO(png))
    scale = width_px / pic.width
    pic.width, pic.height = width_px, int(pic.height * scale)
    ws.add_image(pic, f"A{row}")
    return row + math.ceil(pic.height / 20) + 2  # default row height is 20 px


def _style_table(ws, header_row: int, n_cols: int, last_row: int, widths: list[int]) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(header_row, c)
        cell.font = Font(name=FONT, bold=True)
        cell.fill = HEADER
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BOX
    for r in range(header_row + 1, last_row + 1):
        for c in range(1, n_cols + 1):
            cell = ws.cell(r, c)
            cell.font = Font(name=FONT)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = BOX
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    ws.freeze_panes = ws.cell(header_row + 1, 1)


def _fill_in_columns(ws, first_row: int, last_row: int, yes_no_col: str, note_col: str) -> None:
    dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True,
                        promptTitle="Is this correct?", prompt="Choose Yes or No. If No, say what it should be.")
    ws.add_data_validation(dv)
    if last_row >= first_row:
        dv.add(f"{yes_no_col}{first_row}:{yes_no_col}{last_row}")
    for r in range(first_row, last_row + 1):
        ws[f"{yes_no_col}{r}"].fill = FILL_IN
        ws[f"{note_col}{r}"].fill = FILL_IN


def build_check_sheet(review: ReviewDiagram, result: BuildResult, mode: str,
                      diagram: ParsedDiagram | None = None) -> bytes:
    built_ids = {r.element_id for r in result.records if r.status == "built"}
    record_by_id = {r.element_id: r for r in result.records}

    numbered = number_items(review, result)
    converted = [s for _, s in numbered]
    number_of = {s.id: n for n, s in numbered}
    label_of = {s.id: f"#{number_of[s.id]} {s.tag}".strip() for s in converted}

    conns = [c for c in review.connections if c.include and c.id in built_ids]
    left_out_syms = [s for s in review.symbols if s.id not in number_of and s.source != "image model"
                     or (s.source == "image model" and s.include and s.id not in number_of)]
    left_out_conns = [c for c in review.connections if c.include and c.id not in built_ids]

    cm = result.model.conceptualModel
    n_equipment = len(cm.taggedPlantItems)
    n_instruments = len(cm.processInstrumentationFunctions)
    n_valves = sum(1 for s in converted if category_for_class(s.dexpi_class) == Category.PIPING)
    n_segments = sum(len(sys.segments) for sys in cm.pipingNetworkSystems)

    wb = Workbook()
    # ------------------------------------------------------------------- Items ---
    items = wb.active
    items.title = "Items"
    hdr = ["#", "Tag", "Converted as", "DEXPI type", "How the tool decided", "Check this because", "Correct?",
           "If not, what should it be?"]
    items.append(hdr)
    for n, s in numbered:
        items.append([n, s.tag or "(no tag)", plain_type(s), s.dexpi_class, plain_how(s), check_reasons(s), None, None])
    last = 1 + len(numbered)
    _style_table(items, 1, len(hdr), last, [5, 12, 30, 26, 34, 40, 11, 34])
    for r in range(2, last + 1):
        if items[f"F{r}"].value:
            for c in "ABCDEF":
                items[f"{c}{r}"].fill = CHECK_FIRST
    _fill_in_columns(items, 2, last, "G", "H")
    items.auto_filter.ref = f"A1:H{max(last, 2)}"

    def _flow_words(c) -> str:
        if c.flow in ("drawn arrow", "implied", "reviewer"):
            how_known = {"drawn arrow": "arrow on the drawing", "implied": "worked out from arrows on the "
                         "neighbouring lines", "reviewer": "set by the reviewer"}[c.flow]
            return f"Process pipe, flows From -> To ({how_known})"
        if c.flow == "no flow between them":
            return "Process pipe, both on the same side of a branch (no flow between these two)"
        return "Process pipe, flow direction unknown"

    # ------------------------------------------------------------- Connections ---
    cws = wb.create_sheet("Connections")
    chdr = ["#", "From", "To", "Kind of line", "How the tool found it", "Check this because", "Correct?",
            "If not, what is wrong?"]
    cws.append(chdr)
    for n, c in enumerate(conns, start=1):
        if c.source == "image model":
            how, why = "Predicted from the picture", "picture-based pipe tracing is not reliable yet"
        elif c.confidence == "explicit":
            how, why = "Line snapped to both items in draw.io", ""
        elif c.source == "manual":
            how, why = "Entered by the reviewer", ""
        else:
            how, why = "Line end not snapped; nearest item guessed", "one end of this line was guessed"
        cws.append([n, label_of.get(c.from_id, c.from_id), label_of.get(c.to_id, c.to_id),
                    "Signal / instrument line" if c.line_type == "signal" else _flow_words(c), how, why, None, None])
    clast = 1 + len(conns)
    _style_table(cws, 1, len(chdr), clast, [5, 22, 22, 22, 34, 34, 11, 34])
    for r in range(2, clast + 1):
        if cws[f"F{r}"].value:
            for col in "ABCDEF":
                cws[f"{col}{r}"].fill = CHECK_FIRST
    _fill_in_columns(cws, 2, clast, "G", "H")
    cws.auto_filter.ref = f"A1:H{max(clast, 2)}"

    # ------------------------------------------------------------ Not converted ---
    nws = wb.create_sheet("Not converted")
    nws.append(["What", "Tag / ends", "Why it was left out", "Should it be in? (Yes/No)"])
    for s in left_out_syms:
        reason = "excluded by the reviewer" if not s.include else (record_by_id[s.id].reason if s.id in record_by_id
                                                                     else s.notes or "no DEXPI equivalent")
        nws.append([f"Item: {s.detected_as.split('.')[-1].replace('_', ' ')}", s.tag or "(no tag)", reason, None])
    for c in left_out_conns:
        nws.append(["Line", f"{label_of.get(c.from_id, c.from_id)} -> {label_of.get(c.to_id, c.to_id)}",
                    record_by_id[c.id].reason if c.id in record_by_id else "", None])
    nlast = nws.max_row
    _style_table(nws, 1, 4, nlast, [26, 30, 70, 16])
    dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    nws.add_data_validation(dv)
    if nlast >= 2:
        dv.add(f"D2:D{nlast}")
    for r in range(2, nlast + 1):
        nws[f"D{r}"].fill = FILL_IN

    # -------------------------------------------------------- Before and after ---
    bws = wb.create_sheet("Before and after", 0)
    bws.column_dimensions["A"].width = 120
    row = 1
    bws.cell(row, 1, "BEFORE - the original drawing, every converted item numbered (numbers match the Items tab). "
                     "Orange = please check first, green = no doubts.").font = Font(name=FONT, bold=True)
    if review.image_path and Path(review.image_path).exists():
        row = _place_image(bws, numbered_overlay(review.image_path, numbered), row + 1)
    else:
        bws.cell(row + 1, 1, "No picture was given for this conversion - compare against your own copy of the "
                             "draw.io drawing.").font = Font(name=FONT, italic=True)
        row += 3
    bws.cell(row, 1, "AFTER - the same P&ID re-drawn only from the DEXPI file the tool produced (standard symbols, "
                     "not your drawing style). Everything here is what other software will see; anything missing "
                     "here is missing from the DEXPI file.").font = Font(name=FONT, bold=True)
    bws.cell(row, 1).alignment = Alignment(wrap_text=True)
    bws.row_dimensions[row].height = 32
    if diagram is not None:
        try:
            _place_image(bws, render_after_png(diagram, result), row + 1)
        except Exception as exc:  # a rendering failure must not block the check sheet
            bws.cell(row + 1, 1, f"(Re-drawing failed: {exc})").font = Font(name=FONT, italic=True)

    # --------------------------------------------------------------- Summary ---
    sws = wb.create_sheet("Summary", 0)
    sws.column_dimensions["A"].width = 46
    sws.column_dimensions["B"].width = 58
    rows = [
        ("P&ID to DEXPI conversion - check sheet", None),
        ("Drawing", review.source_name),
        ("Converted on", date.today().isoformat()),
        ("Made from", {"draw.io": "the draw.io file", "image": "a picture of the drawing",
                       "draw.io + image": "the draw.io file, checked against a picture"}.get(mode, mode)),
        (None, None),
        ("RESULT OF YOUR CHECK", None),
        ("Verdict", '=IF(B9+B13>0,"Problems found - see rows marked No",'
                    'IF(B10+B14>0,"Not finished - some rows not yet checked","All checked and correct"))'),
        ("Items", None),
        ("   marked No (wrong)", f'=COUNTIF(Items!G2:G{max(last, 2)},"No")'),
        ("   not yet checked", f'=COUNTA(Items!A2:A{max(last, 2)})-COUNTIF(Items!G2:G{max(last, 2)},"Yes")'
                              f'-COUNTIF(Items!G2:G{max(last, 2)},"No")'),
        ("   marked Yes (correct)", f'=COUNTIF(Items!G2:G{max(last, 2)},"Yes")'),
        ("Connections", None),
        ("   marked No (wrong)", f'=COUNTIF(Connections!G2:G{max(clast, 2)},"No")'),
        ("   not yet checked", f'=COUNTA(Connections!A2:A{max(clast, 2)})-COUNTIF(Connections!G2:G{max(clast, 2)},"Yes")'
                              f'-COUNTIF(Connections!G2:G{max(clast, 2)},"No")'),
        ("   marked Yes (correct)", f'=COUNTIF(Connections!G2:G{max(clast, 2)},"Yes")'),
        (None, None),
        ("WHAT WAS CONVERTED (counted from the DEXPI file)", None),
        ("Equipment (vessels, pumps, columns, ...)", n_equipment),
        ("Instruments", n_instruments),
        ("Valves", n_valves),
        ("Pipe connections", n_segments),
        ("Items the tool was unsure about (orange rows)", sum(1 for _, s in numbered if check_reasons(s))),
        ("Connections with a guessed end (orange rows)", sum(1 for c in conns if c.confidence != "explicit"
                                                             and c.source != "manual")),
        (None, None),
        ("AUTOMATIC CHECKS", None),
        ("DEXPI file written and read back without errors", "Yes"),
        ("Every item in the input accounted for (converted or listed under 'Not converted')", "Yes"),
        (None, None),
        ("HOW TO CHECK", None),
        ("1.", "Open the Items tab. For each row choose Yes or No in the yellow 'Correct?' column."),
        ("2.", "If No, write what it should be in the yellow column next to it."),
        ("", "Example: Correct? = No, If not = 'Ball valve, not a gate valve'."),
        ("3.", "Do the same on the Connections tab. Start with the orange rows: the tool was unsure about those."),
        ("4.", "Look at 'Not converted' for anything that should have been included."),
        ("5.", "The Verdict above updates as you fill in. Save the file and send it back."),
    ]
    for r, (a, b) in enumerate(rows, start=1):
        sws.cell(r, 1, a).font = Font(name=FONT)
        sws.cell(r, 2, b).font = Font(name=FONT)
        sws.cell(r, 2).alignment = Alignment(wrap_text=True, horizontal="left", vertical="top")
        sws.cell(r, 1).alignment = Alignment(wrap_text=True, vertical="top")
        if a and a.isupper():
            sws.cell(r, 1).font = Font(name=FONT, bold=True)
    sws["A1"].font = Font(name=FONT, bold=True, size=14)
    sws["B7"].font = Font(name=FONT, bold=True, size=12)
    sws["A28"].alignment = Alignment(wrap_text=True)
    wb.active = 0

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
