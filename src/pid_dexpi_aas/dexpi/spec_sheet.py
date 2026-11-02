"""The specification sheet: an Excel workbook of tags and their specifications, filled in once
and used for every P&ID the items appear on.

  template_xlsx(review=None)  the workbook to download: one tab per kind of item (vessels and
                              tanks, pumps, valves, instruments, ...) with that kind's DEXPI
                              specification fields as columns; with a review, its items' tags
                              (and any specifications already set) filled in
  read_xlsx(data)             a filled-in workbook -> {normalised tag: row}
  apply_spec_sheet(...)       each item whose tag is in the sheet gets the sheet's values -
                              into empty fields only (the automatic layer after reading a drawing
                              or a picture), or over what is there (the editor's "Sync")

A cell holds one value: a number in the column's first unit ("5"), or a number and a unit
("5 m3", "16 bar"); choices from the column's dropdown; text as text. Tags match the way they
are written on the P&ID, ignoring case, spaces and hyphens ("PSV-201" = "PSV 201" = "psv201").
"""
from __future__ import annotations

import io
import re

from pid_dexpi_aas.dexpi.fields import SpecField, spec_fields

TAG, DESCRIPTION = "tag", "Description"
# (tab, the classes whose fields are its columns)
SHEETS = [
    ("Vessels and tanks", ["Tank", "PressureVessel", "Vessel", "ProcessColumn"]),
    ("Pumps", ["CentrifugalPump", "Pump"]),
    ("Compressors, fans", ["Compressor", "Fan", "Blower"]),
    ("Heat exchangers", ["HeatExchanger", "Heater"]),
    ("Motors, drives", ["Motor", "ElectricGenerator"]),
    ("Other equipment", ["CustomEquipment"]),
    ("Valves and piping", ["GateValve", "Strainer"]),
    ("Safety valves", ["SpringLoadedGlobeSafetyValve"]),
    ("Instruments", ["ProcessInstrumentationFunction"]),
    ("Streams", ["(off-page connector)"]),   # by stream name: N2, Product, Feed ...
]
# filled in from the tag, or structure: not asked for
_NOT_ASKED = {"pipingComponentName", "pipingComponentNumber", "processInstrumentationFunctionNumber",
              "processInstrumentationFunctionCategory", "typeURI"}


def norm_tag(tag: str) -> str:
    return re.sub(r"[\s\-_]", "", str(tag or "")).upper()


def _columns(classes: list[str]) -> list[SpecField]:
    seen, out = set(), []
    for cls in classes:
        for f in spec_fields(cls):
            if f.name not in seen and f.name not in _NOT_ASKED:
                seen.add(f.name)
                out.append(f)
    return out


def sheet_for(dexpi_class: str) -> str:
    """The tab an item of this DEXPI class goes on."""
    from pydexpi.dexpi_classes import equipment as eq, instrumentation as ins, piping as pp
    if dexpi_class == "(off-page connector)":
        return "Streams"
    cls = getattr(eq, dexpi_class, None) or getattr(pp, dexpi_class, None) or getattr(ins, dexpi_class, None)
    if not isinstance(cls, type):
        return "Other equipment"
    groups = [("Instruments", (ins.ProcessInstrumentationFunction,)),
              ("Safety valves", (pp.SafetyValveOrFitting,)),
              ("Valves and piping", (pp.PipingComponent,)),
              ("Vessels and tanks", (eq.Vessel, eq.Tank, eq.PressureVessel, eq.Silo, eq.ProcessColumn)),
              ("Pumps", (eq.Pump,)),
              ("Compressors, fans", (eq.Compressor, eq.Fan, eq.Blower)),
              ("Heat exchangers", (eq.HeatExchanger, eq.Heater)),
              ("Motors, drives", (eq.Motor, eq.ElectricGenerator, eq.Turbine))]
    return next((tab for tab, bases in groups if issubclass(cls, bases)), "Other equipment")


def _shown(value) -> object:
    """A stored specification as a cell: {"value": 5, "unit": "m3"} -> "5 m3"."""
    if isinstance(value, dict) and "value" in value:
        return f"{value['value']:g} {value.get('unit', '')}".strip()
    return value


STREAM_TYPE = "Stream (feed or product)"   # the sheet's name for an off-page connector
ITEMS, TYPES, CHOICES = "Items", "Types", "Choices"
N_FIELDS = 14          # field/value pairs on the Items tab: the most any DEXPI type has
N_ROWS = 100           # rows ready (formatted, with formulas and dropdowns) for items


def asked_fields(dexpi_class: str) -> list[SpecField]:
    """The fields the sheet asks for an item of this DEXPI class, in the order of its
    Field 1, Field 2 ... columns (the order the reader maps values back by)."""
    return [f for f in spec_fields(dexpi_class) if f.name not in _NOT_ASKED][:N_FIELDS]


def _types() -> list[tuple[str, str]]:
    """(name in the sheet, DEXPI class) for every type the review can give an item."""
    from pid_dexpi_aas.review import DEXPI_CLASS_CHOICES, OFF_PAGE, SKIP
    out = [(STREAM_TYPE, OFF_PAGE)]
    out += sorted((c, c) for c in DEXPI_CLASS_CHOICES if c not in (SKIP, OFF_PAGE))
    return out


def _plain(name: str) -> str:
    """CentrifugalPump -> Centrifugal pump (a run of capitals stays: PIDTank -> PID tank)."""
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", name)
    text = " ".join(w if w.isupper() and len(w) > 1 else w.lower() for w in words) or name
    return text[0].upper() + text[1:]


def _kind(dexpi_class: str) -> str:
    from pydexpi.dexpi_classes import equipment as eq, instrumentation as ins, piping as pp
    from pid_dexpi_aas.review import OFF_PAGE
    if dexpi_class == OFF_PAGE:
        return "Stream"
    for module, kind in ((eq, "Equipment"), (pp, "Piping"), (ins, "Instrumentation")):
        if isinstance(getattr(module, dexpi_class, None), type):
            return kind
    return "Other"


def _label(f: SpecField) -> str:
    if f.kind == "quantity" and f.units:
        return f"{f.label} ({f.units[0]})"
    if f.kind == "choice":
        return f"{f.label} (pick)"
    return f.label


def template_xlsx(review=None) -> bytes:
    """The specification sheet: one Items tab - a row per item: its tag, its DEXPI type (a
    dropdown), a description, and then Field 1 / Value 1 ... pairs whose field names follow
    the type picked (formulas reading the hidden Types tab; a field with fixed choices offers
    them as a dropdown in its value cell, from the hidden Choices tab). With a review, its
    tagged items are filled in: tag, type, and any specifications already set."""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    from pid_dexpi_aas.review import SKIP

    arial = lambda **k: Font(name="Arial", **{"size": 10, **k})
    types = _types()
    wb = Workbook()

    # --- How to use ------------------------------------------------------------------
    intro = wb.active
    intro.title = "How to use"
    lines = [
        ("Specification sheet", "title"),
        ("One row per item on the Items tab. Fill in what you know; leave the rest empty.", ""),
        ("1. Tag: exactly as it appears on the P&ID (PSV-201, TI 101, V-201). Case, spaces and hyphens do not matter.", ""),
        ("2. DEXPI type: pick it from the dropdown (CentrifugalPump, Tank, GateValve, ...). Not sure which? The "
         "'Type list' tab has a search box: type pump, valve, heat ... and the matching types light up, with "
         "their plain names and the fields they ask for. (In Microsoft 365 you can also type in the dropdown cell "
         f"to filter it.) Streams entering or leaving the drawing: pick '{STREAM_TYPE}' and use the name written "
         "at the pipe's end as the tag (N2, Feed).", ""),
        ("3. The Field columns then show what to fill in for that type; type each value in the Value column "
         "to its right. Greyed pairs are not used by that type.", ""),
        ("   Values with a unit: a number in the unit shown in brackets (5), or a number and a unit (5 m3, 16 bar).", ""),
        ("   Fields marked (pick): choose from the value cell's dropdown.", ""),
        ("4. Upload the sheet in the app (sidebar, 'Specification sheet'). Every P&ID read afterwards gets these "
         "values for the items whose tags match - only fields still empty, and the type only where the drawing "
         "gave the item none. The editor's 'Sync from the spec sheet' puts the sheet's values and types over what is there.", ""),
        ("Do not change the header row or the tab names; the Types and Choices tabs are hidden and used by the "
         "formulas. Rows can be in any order; one sheet can serve many P&IDs.", ""),
        ("", ""),
        ("Example (what a filled row looks like):", "bold"),
        ("Tag: V-201  |  DEXPI type: Tank  |  Description: Feed tank  |  Field 1 shows the first Tank field, "
         "e.g. 'Nominal capacity (volume) (m3)', Value 1: 5", ""),
    ]
    for r, (text, style) in enumerate(lines, 1):
        c = intro.cell(row=r, column=1, value=text)
        c.font = arial(bold=bool(style), size=14 if style == "title" else 10)
    intro.column_dimensions["A"].width = 130

    # --- Type list: every type, searchable --------------------------------------------------
    tl = wb.create_sheet("Type list")
    tl["A1"] = "Search:"
    tl["A1"].font = arial(bold=True, size=11)
    tl["B1"].fill = PatternFill("solid", fgColor="FFF6CC")
    tl["B1"].border = Border(*(Side(style="medium", color="7F93AD"),) * 4)
    tl["B1"].font = arial(size=11)
    tl["C1"] = ("Type part of a name (pump, valve, heat, tank ...): matching types turn yellow and get 'yes' in "
                "the Match column - filter on it. Then pick the type in column B of the Items tab.")
    tl["C1"].font = arial(color="555555", italic=True)
    tl["C1"].alignment = Alignment(wrap_text=True, vertical="center")
    tl.row_dimensions[1].height = 42
    for col, text in enumerate(["DEXPI type (as in the dropdown)", "Plain name", "Kind", "Fields to fill in", "Match"], 1):
        c = tl.cell(row=3, column=col, value=text)
        c.font, c.fill = arial(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3A5F")
        c.alignment = Alignment(vertical="center", wrap_text=True)
    for r, (name, cls) in enumerate(types, 4):
        fields = ", ".join(f.label for f in asked_fields(cls))
        for col, value in enumerate([name, _plain(name) if name != STREAM_TYPE else name, _kind(cls), fields], 1):
            c = tl.cell(row=r, column=col, value=value)
            c.font = arial(bold=col == 1)
            c.alignment = Alignment(vertical="top", wrap_text=col == 4)
            c.border = Border(*(Side(style="thin", color="C9D1DB"),) * 4)
        m = tl.cell(row=r, column=5, value=f'=IF($B$1="","",IF(ISNUMBER(SEARCH($B$1,A{r}&" "&B{r}&" "&C{r})),"yes",""))')
        m.font, m.alignment = arial(bold=True), Alignment(horizontal="center", vertical="top")
        m.border = Border(*(Side(style="thin", color="C9D1DB"),) * 4)
    last_list = len(types) + 3
    tl.conditional_formatting.add(f"A4:E{last_list}", FormulaRule(formula=['$E4="yes"'],
                                                                   fill=PatternFill("solid", fgColor="FFF6CC")))
    tl.auto_filter.ref = f"A3:E{last_list}"
    tl.freeze_panes = "A4"
    for col, width in zip("ABCDE", (34, 34, 16, 90, 9)):
        tl.column_dimensions[col].width = width

    # --- Types: one row per type, its field labels across --------------------------------
    tt = wb.create_sheet(TYPES)
    tt.append(["Type (as in the sheet)", "DEXPI class", *[f"Field {k}" for k in range(1, N_FIELDS + 1)]])
    for name, cls in types:
        tt.append([name, cls, *[_label(f) for f in asked_fields(cls)]])
    tt.sheet_state = "hidden"
    last_type = len(types) + 1

    # --- Choices: one row per (type, field number) with fixed options -------------------
    ch = wb.create_sheet(CHOICES)
    ch.append(["", "", ""])               # row 1 empty: what a field without choices points at
    for name, cls in types:
        for k, f in enumerate(asked_fields(cls), 1):
            if f.kind == "choice" and f.options:
                ch.append([f"{name}|{k}", len(f.options), *f.options])
    ch.sheet_state = "hidden"
    last_choice = ch.max_row

    # --- Items ------------------------------------------------------------------------
    ws = wb.create_sheet(ITEMS, 1)
    item_head = PatternFill("solid", fgColor="1F3A5F")      # Tag, type, description
    pair_head = PatternFill("solid", fgColor="3A5A80")      # the field / value pairs
    band = PatternFill("solid", fgColor="F3F6FA")           # every other row of the first three columns
    field_fill = PatternFill("solid", fgColor="E8EEF5")
    input_fill = PatternFill("solid", fgColor="FFF6CC")
    grey_fill = PatternFill("solid", fgColor="EDEDED")
    thin = Side(style="thin", color="C9D1DB")
    pair_edge = Side(style="medium", color="7F93AD")         # between one pair and the next
    headers = ["Tag (as on the P&ID)", "DEXPI type", "Description"]
    for k in range(1, N_FIELDS + 1):
        headers += [f"Field {k}", f"Value {k}"]
    for col, text in enumerate(headers, 1):
        c = ws.cell(row=1, column=col, value=text)
        c.font = arial(bold=True, color="FFFFFF")
        c.fill = item_head if col <= 3 else pair_head
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = Border(left=pair_edge if col >= 4 and (col - 4) % 2 == 0 else thin, right=thin, top=thin,
                          bottom=Side(style="medium", color="1F3A5F"))
        ws.column_dimensions[get_column_letter(col)].width = 18 if col <= 3 else (32 if (col - 3) % 2 == 1 else 16)
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 26
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "D2"              # tag, type and description stay in view while scrolling right
    ws.auto_filter.ref = f"A1:C{N_ROWS + 1}"
    ws.sheet_view.zoomScale = 100
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "1:1"

    last = N_ROWS + 1
    pick = DataValidation(type="list", formula1=f"{TYPES}!$A$2:$A${last_type}", allow_blank=True,
                          showErrorMessage=True, errorTitle="DEXPI type",
                          error="Pick a type from the list (the Types tab lists them all).")
    ws.add_data_validation(pick)
    pick.add(f"B2:B{last}")
    for k in range(1, N_FIELDS + 1):
        fcol = get_column_letter(4 + 2 * (k - 1))
        vcol = get_column_letter(5 + 2 * (k - 1))
        for r in range(2, last + 1):
            # the k-th field of the row's type, or nothing ("" & ... turns an empty cell into "")
            c = ws.cell(row=r, column=4 + 2 * (k - 1),
                        value=f'=IFERROR(""&INDEX({TYPES}!$C$2:${get_column_letter(2 + N_FIELDS)}${last_type},'
                              f'MATCH($B{r},{TYPES}!$A$2:$A${last_type},0),{k}),"")')
            c.font, c.fill = arial(color="333333", italic=True), field_fill
            c.border = Border(left=pair_edge, right=thin, top=thin, bottom=thin)
            c.alignment = Alignment(vertical="center")
            v = ws.cell(row=r, column=5 + 2 * (k - 1))
            v.font, v.border = arial(), Border(left=thin, right=thin, top=thin, bottom=thin)
            v.alignment = Alignment(vertical="center")
        # a field with fixed choices: its options as the value cell's dropdown (typing stays free)
        key = f'$B2&"|{k}"'
        found = f"MATCH({key},{CHOICES}!$A$1:$A${last_choice},0)"
        dv = DataValidation(
            type="list", allow_blank=True, showErrorMessage=False,
            formula1=f"OFFSET({CHOICES}!$C$1,IFERROR({found}-1,0),0,1,"
                     f"IFERROR(INDEX({CHOICES}!$B$1:$B${last_choice},{found}),1))")
        ws.add_data_validation(dv)
        dv.add(f"{vcol}2:{vcol}{last}")
        # unused pairs grey, the ones to fill in light yellow
        ws.conditional_formatting.add(f"{fcol}2:{vcol}{last}", FormulaRule(formula=[f'${fcol}2=""'], fill=grey_fill))
        ws.conditional_formatting.add(f"{vcol}2:{vcol}{last}", FormulaRule(formula=[f'${fcol}2<>""'], fill=input_fill))
    for r in range(2, last + 1):
        ws.row_dimensions[r].height = 18
        for col in (1, 2, 3):
            c = ws.cell(row=r, column=col)
            c.font = arial(bold=col == 1)
            c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
            c.alignment = Alignment(vertical="center")
            if r % 2 == 1:
                c.fill = band

    # --- this drawing's items -------------------------------------------------------------
    if review is not None:
        name_of = {cls: name for name, cls in types}
        seen, rows = set(), []
        for s in review.symbols:
            if s.tag and s.include and s.dexpi_class != SKIP and s.dexpi_class in name_of and norm_tag(s.tag) not in seen:
                seen.add(norm_tag(s.tag))  # one row per tag, though the drawing shows it twice
                rows.append(s)
        for r, s in enumerate(sorted(rows, key=lambda s: (name_of[s.dexpi_class], s.tag)), 2):
            ws.cell(row=r, column=1, value=s.tag)
            ws.cell(row=r, column=2, value=name_of[s.dexpi_class])
            have = {**((s.attributes or {}).get("custom") or {}), **((s.attributes or {}).get("dexpi") or {})}
            if have.get(DESCRIPTION):
                ws.cell(row=r, column=3, value=_shown(have[DESCRIPTION]))
            for k, f in enumerate(asked_fields(s.dexpi_class), 1):
                if f.name in have:
                    ws.cell(row=r, column=5 + 2 * (k - 1), value=_shown(have[f.name]))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _read_items(ws, out: dict, notes: list) -> None:
    """The Items tab: per row its tag, type and values - each value's field from the type's
    field list and the pair it is in (the Field cells' formulas are only for the person)."""
    by_name = dict(_types())
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] in (None, ""):
            continue
        tag = str(row[0]).strip()
        type_name = str(row[1]).strip() if len(row) > 1 and row[1] not in (None, "") else ""
        cls = by_name.get(type_name)
        if type_name and cls is None:
            notes.append(f"{tag}: '{type_name}' is not a DEXPI type the app knows - its values are not used")
            continue
        values = {}
        if len(row) > 2 and row[2] not in (None, ""):
            values[DESCRIPTION] = row[2]
        fields = asked_fields(cls) if cls else []
        for k in range(N_FIELDS):
            col = 4 + 2 * k   # the k-th Value column (0-based: E = 4)
            raw = row[col] if len(row) > col else None
            if raw in (None, ""):
                continue
            if k >= len(fields):
                notes.append(f"{tag}: Value {k + 1} is filled in, but a {type_name or 'row without a type'} "
                             f"has no field {k + 1} - not used")
                continue
            values[fields[k].name] = raw
        key = norm_tag(tag)
        if key in out:
            notes.append(f"{tag} is on more than one row - the later one is used")
        out[key] = {"tag": tag, "sheet": ws.title, "type": cls, "values": values}


def read_xlsx(data: bytes) -> tuple[dict, list[str]]:
    """{normalised tag: {"tag", "sheet", "type" (DEXPI class or None), "values": {field name: cell}}},
    and notes (a tag on two rows: the later row wins; an unknown type; a value with no field).
    Reads the Items tab, and the older layout (a tab per kind of item) too."""
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    out, notes = {}, []
    for ws in wb.worksheets:
        if ws.title == ITEMS:
            _read_items(ws, out, notes)
            continue
        if ws.title in (TYPES, CHOICES):
            continue
        # an older sheet: a tab per kind of item, field names in its hidden second row
        rows = ws.iter_rows(values_only=True)
        try:
            next(rows)
            names = [str(n).strip() if n is not None else "" for n in next(rows)]
        except StopIteration:
            continue
        if not names or names[0] != TAG:
            continue  # not a specification tab (the "How to use" one)
        for row in rows:
            if not row or row[0] in (None, ""):
                continue
            tag = str(row[0]).strip()
            values = {n: v for n, v in zip(names[1:], row[1:]) if n and v not in (None, "")}
            key = norm_tag(tag)
            if key in out:
                notes.append(f"{tag} is on more than one row - the one on '{ws.title}' is used")
            out[key] = {"tag": tag, "sheet": ws.title, "type": None, "values": values}
    return out, notes


_UNIT_ALIASES = {"L": "l", "LITRE": "l", "LITER": "l", "C": "°C", "DEGC": "°C", "M³": "m3", "M²": "m2",
                 "RPM": "min-1", "1/MIN": "min-1"}


def to_value(f: SpecField | None, raw):
    """A cell as a stored specification for field f (None: a field the item's class does not
    have - kept as text). -> (value, problem or "")."""
    if f is None or f.kind == "text":
        return (f"{raw:g}" if isinstance(raw, float) else str(raw)).strip(), ""
    if f.kind == "number":
        try:
            return float(str(raw).replace(",", ".")), ""
        except ValueError:
            return None, f"{f.label}: '{raw}' is not a number"
    if f.kind == "choice":
        hit = next((o for o in f.options if o.lower() == str(raw).strip().lower()), None)
        return (hit, "") if hit else (None, f"{f.label}: '{raw}' is not one of the choices")
    # quantity: a number, or a number and a unit
    if isinstance(raw, (int, float)):
        return {"value": float(raw), "unit": f.units[0]}, ""
    m = re.match(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*(.*?)\s*$", str(raw))
    if not m:
        return None, f"{f.label}: '{raw}' is not a number"
    number, unit = float(m.group(1).replace(",", ".")), m.group(2)
    if not unit:
        return {"value": number, "unit": f.units[0]}, ""
    unit = _UNIT_ALIASES.get(unit.upper(), unit)
    hit = next((u for u in f.units if u == unit), None) or next((u for u in f.units if u.lower() == unit.lower()), None)
    return ({"value": number, "unit": hit}, "") if hit else (None, f"{f.label}: unit '{unit}' not one of {', '.join(f.units)}")


def apply_spec_sheet(review, sheet: dict, overwrite: bool = False, ids=None) -> dict:
    """Give the items whose tags are in the sheet its values. overwrite False: only fields still
    empty (a person's value wins); True: the sheet's value over what is there. ids: only these
    items (None: all). The sheet's DEXPI type (Items tab) goes onto an item the drawing gave none,
    and over the drawing's on Sync. -> {"items", "fields", "types", "problems", "unused"}: items
    matched, values written, types set, values that could not be read, sheet tags on no item."""
    from pid_dexpi_aas.review import OFF_PAGE, SKIP

    matched, written, typed, problems, used = 0, 0, 0, [], set()
    known = {g.name for tab in SHEETS for g in _columns(tab[1])}
    label = {g.name: g.label for tab in SHEETS for g in _columns(tab[1])}
    for s in review.symbols:
        if ids is not None and s.id not in ids:
            continue
        row = sheet.get(norm_tag(s.tag)) if s.tag else None
        if row is None or (s.dexpi_class == SKIP and not row.get("type")):
            continue
        used.add(norm_tag(s.tag))
        matched += 1
        # the sheet's DEXPI type: where the drawing gave the item none, or on Sync (overwrite)
        if row.get("type") and row["type"] != s.dexpi_class and (overwrite or s.dexpi_class == SKIP):
            s.dexpi_class = row["type"]
            s.include = True
            s.identified_by = "spec sheet"
            typed += 1
        fields = {f.name: f for f in spec_fields(s.dexpi_class)}
        attrs = {k: dict(v) for k, v in (s.attributes or {}).items() if isinstance(v, dict)}
        attrs.update({k: v for k, v in (s.attributes or {}).items() if not isinstance(v, dict)})
        for name, raw in row["values"].items():
            f = fields.get(name)
            if f is None and (name in known or row.get("type")):
                why = f"the sheet says {row['type']}" if row.get("type") else "on the wrong tab?"
                problems.append(f"{s.tag}: no '{label.get(name, name)}' on a {s.dexpi_class} ({why})")
                continue
            value, problem = to_value(f, raw)
            if problem:
                problems.append(f"{s.tag}: {problem}")
                continue
            part = "custom" if f is None or f.custom else "dexpi"
            have = attrs.setdefault(part, {})
            if have.get(name) == value or (not overwrite and have.get(name) not in (None, "", {})):
                continue
            have[name] = value
            written += 1
        s.attributes = attrs
    if typed:  # lines unticked only because an end had no DEXPI type: back in, now both ends have one
        included = {x.id for x in review.symbols if x.include and x.dexpi_class != SKIP}
        for c in review.connections:
            if not c.include and "endpoint not converted to DEXPI" in (c.notes or "") \
                    and c.from_id in included and c.to_id in included:
                c.include = True
                c.notes = "; ".join(x for x in c.notes.split("; ") if x != "endpoint not converted to DEXPI")
    unused = sorted(row["tag"] for key, row in sheet.items() if key not in used) if ids is None else []
    return {"items": matched, "fields": written, "types": typed, "problems": problems, "unused": unused}
