"""Review layer: OCR tag assembly, tag classifier, and edits reaching the DEXPI model."""
from __future__ import annotations

import collections
from pathlib import Path

import pytest

from pid_dexpi_aas.dexpi.build_model import build_model
from pid_dexpi_aas.drawio.decode import decode_drawio
from pid_dexpi_aas.dexpi.mapping import Category, classify_detection, classify_tag
from pid_dexpi_aas.image.ocr_tags import assemble_tags, attach_tags, normalize_token
from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved
from pid_dexpi_aas.review import (
    SKIP, ReviewConnection, ReviewDiagram, ReviewSymbol, from_drawio, from_image, to_dexpi, validate_rows,
)

ROOT = Path(__file__).resolve().parent.parent
DRAWIO_FILES = sorted((ROOT / "subprojects").glob("*/*.drawio"))
LIGHTUP = ROOT / "subprojects" / "LightUp" / "P&ID gramflow reactor V3.drawio"


def tok(text, x1, y1, x2, y2, conf=0.95):
    return {"text": text, "conf": conf, "box": [x1, y1, x2, y2]}


# ------------------------------------------------------------------------- OCR tags ---
def test_stacked_instrument_letters_and_number_are_joined():
    tags = assemble_tags([tok("PI", 100, 100, 130, 120), tok("102", 98, 124, 134, 144)])
    assert [t.text for t in tags] == ["PI 102"]


def test_medium_under_letters_joins_instead_of_standing_alone():
    tags = assemble_tags([tok("PC", 100, 100, 130, 120), tok("N2", 102, 124, 128, 144)])
    assert [t.text for t in tags] == ["PC N2"]


def test_one_line_tags_kept_and_far_number_not_joined():
    tags = assemble_tags([tok("GV204", 0, 0, 60, 20), tok("PI", 500, 500, 530, 520), tok("101", 900, 900, 940, 920)])
    assert sorted(t.text for t in tags) == ["GV204", "PI"]


@pytest.mark.parametrize("raw,expected", [("Fl", "FI"), ("PSV-1O1", "PSV-101"), ("gv2O4", "GV204"), ("1O2", "102")])
def test_normalize_fixes_letter_digit_confusions(raw, expected):
    assert normalize_token(raw) == expected


def test_attach_prefers_tag_inside_symbol_and_caps_distance():
    tags = assemble_tags([tok("PI", 110, 110, 130, 125), tok("201", 110, 128, 135, 140), tok("GV101", 1000, 1000, 1060, 1020)])
    attached = attach_tags({"bubble": (100, 100, 150, 150)}, tags)
    tag, how = attached["bubble"]
    assert (tag.text, how) == ("PI 201", "inside")
    assert attach_tags({"far": (0, 0, 50, 50)}, [t for t in tags if t.text == "GV101"]) == {}


# ---------------------------------------------------------------------- classifier ---
@pytest.mark.parametrize("tag,kind,cls", [
    ("PSV101", "valve", "SafetyValveOrFitting"),
    ("GV204", "valve", "GateValve"),
    ("PI 102", "instrument", "ProcessInstrumentationFunction"),
    ("TIC 203", "instrument", "ProcessInstrumentationFunction"),
    ("P101", "equipment", "Pump"),
    ("V201", "equipment", "Vessel"),
    ("X101", "equipment", "ProcessColumn"),
])
def test_classify_tag(tag, kind, cls):
    assert classify_tag(tag)[:2] == (kind, cls)


def test_tag_overrides_wrong_model_class_and_says_so():
    mapping, how = classify_detection("pump", "PI N2")
    assert mapping.category == Category.INSTRUMENTATION
    assert "model said 'pump'" in mapping.reason and "PI" in how


def test_v_prefix_on_a_valve_symbol_stays_a_valve():
    mapping, _ = classify_detection("valve", "V12")
    assert mapping.category == Category.PIPING


def test_untagged_general_symbol_needs_manual_classification():
    mapping, _ = classify_detection("general", None)
    assert mapping.category == Category.UNMAPPED and mapping.dexpi_class_name is None


# ------------------------------------------------------------------- review -> DEXPI ---
def _summary(result):
    cm = result.model.conceptualModel
    return (
        sum(len(s.segments) for s in cm.pipingNetworkSystems),
        collections.Counter(type(o).__name__ for o in cm.taggedPlantItems),
        len(cm.processInstrumentationFunctions),
        sorted(str(getattr(o, "tagName", None)) for o in cm.taggedPlantItems),
    )


def _parse(path):
    return parse_drawio_xml_resolved(decode_drawio(path)[0], source_file=path.name)


@pytest.mark.parametrize("path", DRAWIO_FILES, ids=lambda p: p.name)
def test_unedited_drawio_review_builds_same_dexpi_as_plain_pipeline(path):
    review = from_drawio(_parse(path))
    assert validate_rows(review) == []
    assert _summary(to_dexpi(review)[1]) == _summary(build_model(_parse(path)))


def test_reviewer_class_and_tag_edits_reach_the_dexpi_model():
    review = from_drawio(_parse(LIGHTUP))
    pump = next(s for s in review.symbols if s.dexpi_class == "Pump")
    pump.dexpi_class, pump.tag = "CustomPump", "P999"
    cm = to_dexpi(review)[1].model.conceptualModel
    edited = [o for o in cm.taggedPlantItems if getattr(o, "tagName", None) == "P999"]
    assert len(edited) == 1 and type(edited[0]).__name__ == "CustomPump"


def test_excluded_symbol_is_left_out_and_its_connections_flagged():
    review = from_drawio(_parse(LIGHTUP))
    target = next(s for s in review.symbols if s.dexpi_class == "Pump")
    target.include = False
    assert any(target.id in p for p in validate_rows(review))  # its pipes now dangle
    for c in review.connections:
        if target.id in (c.from_id, c.to_id):
            c.include = False
    assert validate_rows(review) == []
    names = [type(o).__name__ for o in to_dexpi(review)[1].model.conceptualModel.taggedPlantItems]
    assert names.count("Pump") == collections.Counter(
        type(o).__name__ for o in build_model(_parse(LIGHTUP)).model.conceptualModel.taggedPlantItems)["Pump"] - 1


def test_image_review_from_model_output():
    infer = {
        "image": "x.png", "image_size": [1000, 1000],
        "symbols": [
            {"id": "m0", "box": [100, 100, 150, 150], "class": "pump", "score": 0.9},
            {"id": "m1", "box": [400, 100, 450, 150], "class": "valve", "score": 0.95},
            {"id": "m2", "box": [700, 700, 750, 750], "class": "general", "score": 0.6},
        ],
        "connections": [{"from": "m0", "to": "m1", "prob": 0.98}, {"from": "m1", "to": "m2", "prob": 0.95}],
        "ocr_tokens": [tok("PI", 110, 110, 140, 125), tok("N2", 112, 128, 138, 142), tok("XV101", 400, 155, 450, 170)],
    }
    review = from_image(infer, "x.png")
    by_id = {s.id: s for s in review.symbols}
    assert (by_id["m0"].tag, by_id["m0"].dexpi_class) == ("PI N2", "ProcessInstrumentationFunction")
    assert (by_id["m1"].tag, by_id["m1"].dexpi_class) == ("XV101", "OperatedValve")
    assert by_id["m2"].dexpi_class == SKIP and not by_id["m2"].include
    # m1 -> m2 dangles (m2 not converted) so it is unticked rather than failing the build
    assert [c.include for c in review.connections] == [True, False]
    assert validate_rows(review) == []
    to_dexpi(review)  # builds without error


def test_manual_symbol_and_connection_build():
    review = ReviewDiagram("manual.png")
    review.symbols = [
        ReviewSymbol(id="a", source="manual", box=None, detected_as="manual", tag="V101", dexpi_class="Vessel"),
        ReviewSymbol(id="b", source="manual", box=None, detected_as="manual", tag="GV1", dexpi_class="GateValve"),
    ]
    review.connections = [ReviewConnection(id="c", from_id="a", to_id="b", source="manual")]
    _, result = to_dexpi(review)
    cm = result.model.conceptualModel
    assert [o.tagName for o in cm.taggedPlantItems] == ["V101"]
    assert sum(len(s.segments) for s in cm.pipingNetworkSystems) == 1


def test_drawio_valves_get_their_type_from_the_separate_tag_label():
    # Regression: valve symbols carry no text of their own; the tag is a separate label.
    # Classifying from the symbol's own text made all 123 valves generic StraightwayValves.
    built = collections.Counter(
        r.dexpi_class_name for r in build_model(_parse(LIGHTUP)).records if r.element_kind == "shape"
    )
    assert built["GateValve"] > 0 and built["CheckValve"] > 0 and built["OperatedValve"] > 0
    assert built["SafetyValveOrFitting"] > 0


def test_model_only_connections_need_to_be_sure_and_no_skip():
    """The tracer's connections are kept; a model prediction it did not confirm only when the
    model is sure enough (MODEL_ONLY_MIN) and it does not jump over an item the tracer put between them."""
    from pid_dexpi_aas.review import from_image
    box = lambda x: [x, 0, x + 40, 40]
    infer = {"image": "x.png", "image_size": [1000, 200], "ocr_tokens": [],
             "symbols": [{"id": f"m{i}", "box": box(100 * i), "class": "valve", "score": 0.99} for i in range(5)],
             "traced": [{"from": "m0", "to": "m1"}, {"from": "m1", "to": "m2"}],
             "connections": [{"from": "m0", "to": "m2", "prob": 0.99},   # skips m1: dropped
                             {"from": "m2", "to": "m3", "prob": 0.95},   # sure: kept
                             {"from": "m3", "to": "m4", "prob": 0.4}]}   # not sure: dropped
    pairs = {frozenset((c.from_id, c.to_id)) for c in from_image(infer, "x.png").connections}
    assert pairs == {frozenset(p) for p in (("m0", "m1"), ("m1", "m2"), ("m2", "m3"))}


def test_possible_missed_item_is_a_suggestion():
    """A symbol the model drew no box for, on a traced pipe: a flagged item for the guided
    check, left out of DEXPI, the pipe kept as traced (splitting it made the labelled sets worse)."""
    from pid_dexpi_aas.review import MISSED, from_image, look_reasons
    infer = {"image": "x.png", "image_size": [1000, 200], "ocr_tokens": [], "connections": [],
             "symbols": [{"id": "m0", "box": [0, 0, 40, 40], "class": "pump", "score": 0.99},
                         {"id": "m1", "box": [400, 0, 440, 40], "class": "tank", "score": 0.99}],
             "traced": [{"from": "m0", "to": "m1", "flow": 1, "path": [[40 + 10 * i, 20] for i in range(37)]}],
             "missed": [{"box": [190, 5, 230, 35], "from": "m0", "to": "m1", "class": "", "score": None,
                         "how": "pipe traced through a drawn symbol"}]}
    review = from_image(infer, "x.png")
    item = next(s for s in review.symbols if s.detected_as == MISSED)
    assert item.dexpi_class and not item.include
    assert any("possible missed item" in w for w in look_reasons(item))
    assert [(c.from_id, c.to_id, c.flow) for c in review.connections] == [("m0", "m1", "drawn arrow")]


def test_flow_arrow_direction_carries_to_the_bridged_line():
    """pump - arrow - tank: the tracer read the arrow pointing on from the pump, so the line
    joining pump and tank across the arrow flows pump -> tank."""
    from pid_dexpi_aas.review import from_image
    box = lambda x: [x, 0, x + 40, 40]
    infer = {"image": "x.png", "image_size": [1000, 200], "ocr_tokens": [], "connections": [],
             "symbols": [{"id": "m0", "box": box(300), "class": "tank", "score": 0.99},
                         {"id": "m1", "box": box(150), "class": "arrow", "score": 0.99},
                         {"id": "m2", "box": box(0), "class": "pump", "score": 0.99}],
             "traced": [{"from": "m2", "to": "m1", "flow": 1}, {"from": "m1", "to": "m0", "flow": 0}]}
    review = from_image(infer, "x.png")
    bridged = [c for c in review.connections if c.confidence == "through a flow arrow"]
    assert [(c.from_id, c.to_id, c.flow) for c in bridged] == [("m2", "m0", "drawn arrow")]


def test_arrow_bridge_is_traced_when_both_its_lines_are():
    """pump - arrow - tank traced, plus a model guess arrow - valve: the pump - tank line
    across the arrow is traced; the one line on to the valve (a tree, not every pair) a guess."""
    from pid_dexpi_aas.review import from_image
    box = lambda x: [x, 0, x + 40, 40]
    infer = {"image": "x.png", "image_size": [1000, 200], "ocr_tokens": [],
             "symbols": [{"id": "m0", "box": box(300), "class": "tank", "score": 0.99},
                         {"id": "m1", "box": box(150), "class": "arrow", "score": 0.99},
                         {"id": "m2", "box": box(0), "class": "pump", "score": 0.99},
                         {"id": "m3", "box": box(600), "class": "valve", "score": 0.99}],
             "traced": [{"from": "m2", "to": "m1"}, {"from": "m1", "to": "m0"}],
             "connections": [{"from": "m1", "to": "m3", "prob": 0.95}]}
    review = from_image(infer, "x.png")
    bridged = {frozenset((c.from_id, c.to_id)): c.source for c in review.connections if c.confidence == "through a flow arrow"}
    assert bridged[frozenset(("m0", "m2"))] == "picture tracing"
    to_valve = [src for pair, src in bridged.items() if "m3" in pair]
    assert len(bridged) == 2 and to_valve == ["image model"]   # a tree: one line on to the valve


def test_traced_pipe_network_is_built_with_tees():
    """A traced pipe network of four items (shown as a tree of three lines) is one DEXPI pipe
    network with two PipeTee fittings - not three separate pipes."""
    from pydexpi.dexpi_classes import piping

    from pid_dexpi_aas.review import from_image, to_dexpi
    box = lambda x: [x, 0, x + 40, 40]
    infer = {"image": "x.png", "image_size": [1000, 300], "ocr_tokens": [], "connections": [],
             "symbols": [{"id": f"m{i}", "box": box(200 * i), "class": c, "score": 0.99}
                         for i, c in enumerate(["pump", "valve", "valve", "tank"])],
             # every pair of the network, as the tracer reports it (net 7)
             "traced": [{"from": f"m{i}", "to": f"m{j}", "net": 7, "length": 200.0 * (j - i),
                         "path": [[200 * i + 40, 20], [200 * j, 20]]} for i in range(4) for j in range(i + 1, 4)]}
    review = from_image(infer, "x.png")
    shown = [c for c in review.connections if c.include]
    assert len(shown) == 3 and all(c.attributes.get("net") == "t7" for c in shown)
    _, result = to_dexpi(review)
    segs = [s for sys_ in result.model.conceptualModel.pipingNetworkSystems for s in sys_.segments]
    tees = {id(i) for s in segs for i in s.items if isinstance(i, piping.PipeTee)}
    assert len(tees) == 2
