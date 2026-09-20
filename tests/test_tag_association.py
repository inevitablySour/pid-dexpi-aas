from pathlib import Path

from pid_dexpi_aas.parser_drawio import parse_drawio_file
from pid_dexpi_aas.tag_association import associate_tags, build_tag_lookup

FIXTURE = (
    Path(__file__).parent.parent
    / "handoff"
    / "decoded_xml"
    / "PFD combined set up 301122__diagram0.xml"
)


def test_reactor_label_attaches_to_reactor_shape():
    diagram = parse_drawio_file(FIXTURE)
    lookup = build_tag_lookup(diagram)
    reactor_shape = next(
        s for s in diagram.shapes if s.shape_token == "mxgraph.pid.vessels.reactor"
    )
    assert "Reactor" in lookup[reactor_shape.id]


def test_every_labeled_text_is_associated_when_shapes_exist():
    diagram = parse_drawio_file(FIXTURE)
    associations = associate_tags(diagram)
    label_only_with_text = [s for s in diagram.shapes if s.is_labeled_only and s.text]
    assert len(associations) == len(label_only_with_text)
    assert all(a.target_shape_id is not None for a in associations)


def test_empty_text_labels_are_skipped():
    diagram = parse_drawio_file(FIXTURE)
    associations = associate_tags(diagram)
    associated_ids = {a.label_shape_id for a in associations}
    empty_labels = [s for s in diagram.shapes if s.is_labeled_only and not s.text]
    assert not (associated_ids & {s.id for s in empty_labels})
