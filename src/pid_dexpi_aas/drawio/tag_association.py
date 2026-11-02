"""Associate floating tag/label text to the nearest real equipment/valve/instrument shape.

Roughly half of all vertices in the team's drawio files have no `shape=` token — they are
plain-rectangle text labels (e.g. "V101", "Reactor", "150 bar") sitting near, but not the
same mxCell as, the shape they describe. This module resolves each such label to the
nearest shaped vertex by bounding-box center distance, so tags land on the correct DEXPI
object's tagName at model-build time rather than being patched on afterwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pid_dexpi_aas.drawio.parser import ParsedDiagram, Shape


@dataclass
class TagAssociation:
    label_shape_id: str
    label_text: str
    target_shape_id: str | None  # None if no shaped vertex exists at all in the diagram
    distance: float


def _center(shape: Shape) -> tuple[float, float]:
    return (shape.x + shape.width / 2, shape.y + shape.height / 2)


def _distance(a: Shape, b: Shape) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


# A label further than this from every real shape is not associated with anything,
# rather than force-matched to whatever happens to be least-far-away. Confirmed as a
# real bug on the gramflow reactor file: with no cap, "XV202" (a valve tag) was
# incorrectly attached to a tank shape 535 units away, and "V201" to the same tank at
# 758 units - both nowhere near their real, intended shapes - producing a corrupted
# tagName and creating the false impression of a wrong piping connection downstream
# (the connection itself was correct; the printed tag was not). 150 units is generous
# relative to typical shape sizes in these files (~80-180 units) and typical genuine
# label-to-shape distances seen elsewhere (0-110 units in the PFD smoke-test fixture).
_MAX_TAG_ASSOCIATION_DISTANCE = 150.0


def associate_tags(diagram: ParsedDiagram) -> list[TagAssociation]:
    """Return one TagAssociation per label-only shape that has non-empty text.

    Label-only shapes with empty text are dropped (nothing to attach). Ties in distance
    are broken by shape id order, deterministically. A label with no real shape within
    _MAX_TAG_ASSOCIATION_DISTANCE gets target_shape_id=None (unresolved) rather than
    being force-matched to the nearest shape regardless of how far away it actually is.

    Texts a line ends on that are not tags are stream names (off-page connectors) and are
    never associated. KNOWN LIMITATION (for the rest): nearest-distance-only association conflates true equipment tags
    (e.g. "V101") with nearby stream/state annotations (e.g. "Product", "Reaction
    mixture") when both sit closest to the same shape and no edge/parent-child data
    disambiguates them - confirmed on the PFD smoke-test file, where a tank picks up
    both its own tag and two unrelated stream labels. validate_by_counting.py should
    report shapes with more than one associated tag as needing manual review, not
    silently trust every association.
    """
    from pid_dexpi_aas.dexpi.mapping import classify_tag

    labeled_shapes = [s for s in diagram.shapes if not s.is_labeled_only]
    associations: list[TagAssociation] = []
    # A text a line ends on that is not a tag ("Product", "Fume hood [atm]") names the
    # stream leaving there (an off-page connector), never the tag of the symbol beside it.
    line_ends = {x for e in diagram.edges for x in (e.source, e.target) if x}
    # a tag label made in the editor ("Add a tag label") is linked to its item (the item's
    # tag_cell): it is that item's tag wherever it has been moved to
    shape_ids = {s.id for s in diagram.shapes}
    linked = {s.data["tag_cell"]: s.id for s in diagram.shapes
              if (s.data or {}).get("tag_cell") and s.data["tag_cell"] in shape_ids and s.data["tag_cell"] != s.id}

    for label in diagram.shapes:
        if not label.is_labeled_only or not label.text:
            continue
        if label.id in linked:
            associations.append(TagAssociation(label.id, label.text, linked[label.id], 0.0))
            continue

        if not labeled_shapes:
            associations.append(TagAssociation(label.id, label.text, None, math.inf))
            continue

        if label.id in line_ends and classify_tag(label.text)[0] is None:
            associations.append(TagAssociation(label.id, label.text, None, math.inf))
            continue

        nearest = min(labeled_shapes, key=lambda s: (_distance(label, s), s.id))
        nearest_distance = _distance(label, nearest)
        target_id = nearest.id if nearest_distance <= _MAX_TAG_ASSOCIATION_DISTANCE else None
        associations.append(TagAssociation(label.id, label.text, target_id, nearest_distance))

    return associations


def build_tag_lookup(diagram: ParsedDiagram) -> dict[str, list[str]]:
    """shape_id -> list of tag texts attached to it, for convenient lookup in build_model."""
    lookup: dict[str, list[str]] = {}
    for association in associate_tags(diagram):
        if association.target_shape_id is None:
            continue
        lookup.setdefault(association.target_shape_id, []).append(association.label_text)
    return lookup
