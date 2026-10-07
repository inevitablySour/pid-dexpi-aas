"""Regression test for a confirmed data-corruption bug: tag_association.py's nearest-
label search had no distance cap, so a label with no genuinely nearby shape got
force-matched to whatever real shape happened to be least-far-away, however distant.

Found by hand-verifying the gramflow reactor file's DEXPI output against the original
P&ID image: a tank shape showed tagName "V201, XV202" (two DISTINCT valve/tank labels
merged onto one shape), because "V201" (758 units away) and "XV202" (535 units away)
had no real nearby match and got force-attached to that tank anyway - creating the false
impression that a piping connection pointed at the wrong equipment, when the connection
itself was correct and only the printed tag was corrupted.
"""
from pathlib import Path

from pid_dexpi_aas.drawio.parser import parse_drawio_file
from pid_dexpi_aas.drawio.tag_association import associate_tags, build_tag_lookup

GRAMFLOW = (
    Path(__file__).parent
    / "fixtures" / "decoded_xml"
    / "P&ID gramflow reactor V3__diagram0.xml"
)


def test_far_away_labels_are_unresolved_not_force_matched():
    diagram = parse_drawio_file(GRAMFLOW)
    associations = {a.label_text: a for a in associate_tags(diagram)}

    for label_text in ("V201", "XV202"):
        assoc = associations[label_text]
        assert assoc.target_shape_id is None, (
            f"{label_text} should be unresolved (its real nearest shape is >150 units "
            f"away), not force-matched at distance {assoc.distance}"
        )


def test_tank_shape_does_not_have_the_wrong_merged_tag():
    """The specific tank that incorrectly picked up "V201, XV202" before this fix."""
    diagram = parse_drawio_file(GRAMFLOW)
    tag_lookup = build_tag_lookup(diagram)
    tags = tag_lookup.get("h17LKkp3DZ7R27TqlmPq-121")
    assert tags is None or ("V201" not in tags and "XV202" not in tags)
