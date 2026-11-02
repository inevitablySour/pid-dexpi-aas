"""Every shape in draw.io's P&ID library gets a DEXPI type: the Editor tab offers all of it."""
import pytest

from pid_dexpi_aas.dexpi.mapping import Category, map_shape, shape_rules
from pid_dexpi_aas.review import DEXPI_CLASS_CHOICES, category_for_class


@pytest.mark.parametrize("token, cls, category", [
    ("mxgraph.pid.engines.electric_motor", "Motor", Category.EQUIPMENT),
    ("mxgraph.pid.engines.generator_(ac)", "AlternatingCurrentGenerator", Category.EQUIPMENT),
    ("mxgraph.pid.centrifuges.centrifuge_(pusher)", "FilteringCentrifuge", Category.EQUIPMENT),
    ("mxgraph.pid.crushers_grinding.mill_(vibration)", "Mill", Category.EQUIPMENT),
    ("mxgraph.pid.misc.conveyor", "Conveyor", Category.EQUIPMENT),
    ("mxgraph.pid.misc.aerator_with_sparger", "CustomEquipment", Category.EQUIPMENT),
    ("mxgraph.pid.piping.concentric_reducer", "PipeReducer", Category.PIPING),
    ("mxgraph.pid.piping.basket_strainer", "Strainer", Category.PIPING),
    ("mxgraph.pid.fittings.flame_arrestor", "FlameArrestor", Category.PIPING),
    ("mxgraph.pid.piping.closed_figure_8_blind", "LineBlind", Category.PIPING),
])
def test_library_shapes_map_to_dexpi(token, cls, category):
    r = map_shape(token, "")
    assert (r.dexpi_class_name, r.category) == (cls, category)
    assert category_for_class(cls) == category  # and the review agrees when a person picks it


def test_vessel_parts_are_left_to_be_grouped_with_their_vessel():
    assert map_shape("mxgraph.pid.apparatus_elements.support_leg", "").dexpi_class_name is None


def test_every_class_the_rules_give_can_be_chosen_in_the_app():
    r = shape_rules()
    named = {c for c in r["exact"].values() if c} | set(r["valve_types"].values()) | {r["valve_default"],
             r["instrument_class"], r["piping_default"]} | {c for _, c in r["valve_words"] + r["piping_words"]}
    for _, default, words in r["equipment_families"]:
        named |= {default} | {c for _, c in words}
    assert named - set(DEXPI_CLASS_CHOICES) == set()
