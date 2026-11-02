"""The spec sheet: specifications filled in once in Excel, by tag, put onto every P&ID."""
import io

from openpyxl import load_workbook

from pid_dexpi_aas.dexpi.spec_sheet import apply_spec_sheet, norm_tag, read_xlsx, template_xlsx, to_value
from pid_dexpi_aas.dexpi.fields import spec_fields
from pid_dexpi_aas.review import ReviewDiagram, ReviewSymbol


def _review():
    r = ReviewDiagram(source_name="t")
    r.symbols = [ReviewSymbol(id="v", source="draw.io", box=None, detected_as="", tag="V-201", dexpi_class="Tank"),
                 ReviewSymbol(id="p", source="draw.io", box=None, detected_as="", tag="PSV 201",
                              dexpi_class="SpringLoadedGlobeSafetyValve",
                              attributes={"dexpi": {"setPressureHigh": {"value": 10.0, "unit": "bar"}}})]
    return r


def _put(ws, row, field_name, value):
    """Type a value into the Value cell of the pair that holds this field for the row's type."""
    from pid_dexpi_aas.dexpi.spec_sheet import STREAM_TYPE, asked_fields
    from pid_dexpi_aas.review import OFF_PAGE
    t = ws.cell(row=row, column=2).value
    names = [f.name for f in asked_fields(OFF_PAGE if t == STREAM_TYPE else t)]
    ws.cell(row=row, column=5 + 2 * names.index(field_name), value=value)


def _filled(review):
    wb = load_workbook(io.BytesIO(template_xlsx(review)))
    ws = wb["Items"]
    rows = {ws.cell(row=r, column=1).value: r for r in range(2, 10) if ws.cell(row=r, column=1).value}
    assert ws.cell(row=rows["V-201"], column=2).value == "Tank"            # this drawing's tags and types
    _put(ws, rows["V-201"], "nominalCapacityVolume", 5)
    _put(ws, rows["V-201"], "DesignPressure", "16 barg")
    _put(ws, rows["V-201"], "Material", "SS316")
    r = rows["PSV 201"]
    names = [f.name for f in __import__("pid_dexpi_aas.dexpi.spec_sheet", fromlist=["x"]).asked_fields(
        "SpringLoadedGlobeSafetyValve")]
    assert ws.cell(row=r, column=5 + 2 * names.index("setPressureHigh")).value == "10 bar"   # specs already set too
    ws.cell(row=r, column=1, value="psv-201")                               # written differently: still the same tag
    _put(ws, r, "setPressureHigh", "18 bar")
    buf = io.BytesIO()
    wb.save(buf)
    return read_xlsx(buf.getvalue())[0]


def test_sheet_fills_empty_fields_and_sync_overwrites():
    r = _review()
    sheet = _filled(r)
    res = apply_spec_sheet(r, sheet)
    v, p = r.symbols
    assert v.attributes["dexpi"]["nominalCapacityVolume"] == {"value": 5.0, "unit": "l"}  # the column's unit
    assert v.attributes["custom"] == {"DesignPressure": {"value": 16.0, "unit": "barg"}, "Material": "SS316"}
    assert p.attributes["dexpi"]["setPressureHigh"] == {"value": 10.0, "unit": "bar"}     # a person's value stays
    assert res["items"] == 2 and not res["problems"]
    apply_spec_sheet(r, sheet, overwrite=True, ids={"p"})                                # the editor's Sync
    assert p.attributes["dexpi"]["setPressureHigh"] == {"value": 18.0, "unit": "bar"}


def test_values_that_cannot_be_used_are_reported():
    f = {x.name: x for x in spec_fields("Tank")}["nominalCapacityVolume"]
    assert to_value(f, "2.5 m3") == ({"value": 2.5, "unit": "m3"}, "")
    assert to_value(f, "5 L")[0] == {"value": 5.0, "unit": "l"}
    assert to_value(f, "12 parsecs")[0] is None and "unit" in to_value(f, "12 parsecs")[1]
    r = _review()
    sheet = {norm_tag("PSV201"): {"tag": "PSV201", "sheet": "x", "type": None, "values": {"nominalCapacityVolume": 3}}}
    assert "wrong tab" in apply_spec_sheet(r, sheet)["problems"][0]


def test_the_fields_follow_the_type_picked():
    """The Field cells are formulas on the row's DEXPI type: the k-th field of that type."""
    from pid_dexpi_aas.dexpi.spec_sheet import N_FIELDS, asked_fields
    wb = load_workbook(io.BytesIO(template_xlsx()))
    ws, types = wb["Items"], wb["Types"]
    assert ws.cell(row=1, column=4).value == "Field 1" and ws.cell(row=1, column=5).value == "Value 1"
    assert "MATCH($B2" in ws.cell(row=2, column=4).value and ",1)" in ws.cell(row=2, column=4).value
    tank = next(r for r in types.iter_rows(min_row=2, values_only=True) if r[0] == "Tank")
    assert list(tank[2:2 + len(asked_fields("Tank"))]) == [
        f.label + (f" ({f.units[0]})" if f.kind == "quantity" else " (pick)" if f.kind == "choice" else "")
        for f in asked_fields("Tank")]
    assert types.sheet_state == "hidden" and wb["Choices"].sheet_state == "hidden"
    assert ws.max_column == 3 + 2 * N_FIELDS
    # a choice field's value cell offers its options: the Choices tab holds them by type and field number
    gate = [f.name for f in asked_fields("GateValve")]
    k = gate.index("numberOfPorts") + 1
    row = next(r for r in wb["Choices"].iter_rows(values_only=True) if r[0] == f"GateValve|{k}")
    options = [x for x in row[2:] if x is not None]
    assert row[1] == len(options) >= 2 and "3 port valve" in options


def test_the_sheets_type_goes_onto_items_without_one():
    """An item the drawing gave no DEXPI type gets the sheet's (its lines ticked again); one with a
    type keeps it unless the editor syncs."""
    from pid_dexpi_aas.review import SKIP, ReviewConnection
    r = _review()
    r.symbols.append(ReviewSymbol(id="x", source="image model", box=None, detected_as="general", tag="P-101",
                                  dexpi_class=SKIP, include=False))
    r.connections = [ReviewConnection(id="c", from_id="x", to_id="v", include=False,
                                      notes="endpoint not converted to DEXPI")]
    wb = load_workbook(io.BytesIO(template_xlsx(r)))
    ws = wb["Items"]
    ws.cell(row=9, column=1, value="P-101")
    ws.cell(row=9, column=2, value="CentrifugalPump")
    _put(ws, 9, "DesignPressure", "6 bar")
    vrow = next(i for i in range(2, 9) if ws.cell(row=i, column=1).value == "V-201")
    ws.cell(row=vrow, column=2, value="PressureVessel")                     # the sheet says otherwise
    buf = io.BytesIO()
    wb.save(buf)
    sheet, notes = read_xlsx(buf.getvalue())
    assert not notes and sheet["P101"]["type"] == "CentrifugalPump"
    res = apply_spec_sheet(r, sheet)
    x, v = r.symbols[2], r.symbols[0]
    assert x.dexpi_class == "CentrifugalPump" and x.include and x.identified_by == "spec sheet"
    assert x.attributes["custom"]["DesignPressure"] == {"value": 6.0, "unit": "bar"}
    assert r.connections[0].include and "endpoint" not in r.connections[0].notes
    assert v.dexpi_class == "Tank" and res["types"] == 1                    # a drawing's type stays ...
    apply_spec_sheet(r, sheet, overwrite=True)
    assert v.dexpi_class == "PressureVessel"                                # ... until Sync


def test_unknown_types_and_extra_values_are_noted():
    wb = load_workbook(io.BytesIO(template_xlsx()))
    ws = wb["Items"]
    ws.append(["X-1", "Spaceship", "", "", 5])
    ws.cell(row=3, column=1, value="V-9")
    ws.cell(row=3, column=2, value="Tank")
    ws.cell(row=3, column=5 + 2 * 12, value=1)                              # Value 13: a Tank has fewer fields
    buf = io.BytesIO()
    wb.save(buf)
    sheet, notes = read_xlsx(buf.getvalue())
    assert any("Spaceship" in n for n in notes) and any("Value 13" in n for n in notes)
    assert "X1" not in sheet and sheet["V9"]["values"] == {}


def test_an_older_sheet_with_a_tab_per_kind_is_still_read():
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Vessels and tanks"
    ws.append(["Tag (as on the P&ID)", "Nominal capacity (volume) (l)"])
    ws.append(["tag", "nominalCapacityVolume"])
    ws.append(["V-201", 7])
    buf = io.BytesIO()
    wb.save(buf)
    sheet, _ = read_xlsx(buf.getvalue())
    r = _review()
    apply_spec_sheet(r, sheet)
    assert r.symbols[0].attributes["dexpi"]["nominalCapacityVolume"] == {"value": 7.0, "unit": "l"}


def test_the_type_list_can_be_searched():
    """A visible Type list tab: every type with a plain name, its kind and its fields, and a
    search box whose matches the Match column marks (SEARCH: part of a word, any case)."""
    wb = load_workbook(io.BytesIO(template_xlsx()))
    tl = wb["Type list"]
    assert tl.sheet_state == "visible" and tl["A1"].value == "Search:"
    rows = {tl.cell(row=r, column=1).value: r for r in range(4, tl.max_row + 1)}
    r = rows["CentrifugalPump"]
    assert tl.cell(row=r, column=2).value == "Centrifugal pump" and tl.cell(row=r, column=3).value == "Equipment"
    assert "SEARCH($B$1" in tl.cell(row=r, column=5).value
    assert tl.cell(row=rows["GateValve"], column=3).value == "Piping"
    assert len(rows) == wb["Types"].max_row - 1      # every type in the dropdown is listed
