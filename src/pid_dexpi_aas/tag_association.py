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

from pid_dexpi_aas.parser_drawio import ParsedDiagram, Shape


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


def associate_tags(diagram: ParsedDiagram) -> list[TagAssociation]:
    """Return one TagAssociation per label-only shape that has non-empty text.

    Label-only shapes with empty text are dropped (nothing to attach). Ties in distance
    are broken by shape id order, deterministically.

    KNOWN LIMITATION: nearest-distance-only association conflates true equipment tags
    (e.g. "V101") with nearby stream/state annotations (e.g. "Product", "Reaction
    mixture") when both sit closest to the same shape and no edge/parent-child data
    disambiguates them - confirmed on the PFD smoke-test file, where a tank picks up
    both its own tag and two unrelated stream labels. validate_by_counting.py should
    report shapes with more than one associated tag as needing manual review, not
    silently trust every association.
    """
    labeled_shapes = [s for s in diagram.shapes if not s.is_labeled_only]
    associations: list[TagAssociation] = []

    for label in diagram.shapes:
        if not label.is_labeled_only or not label.text:
            continue

        if not labeled_shapes:
            associations.append(TagAssociation(label.id, label.text, None, math.inf))
            continue

        nearest = min(labeled_shapes, key=lambda s: (_distance(label, s), s.id))
        associations.append(
            TagAssociation(label.id, label.text, nearest.id, _distance(label, nearest))
        )

    return associations


def build_tag_lookup(diagram: ParsedDiagram) -> dict[str, list[str]]:
    """shape_id -> list of tag texts attached to it, for convenient lookup in build_model."""
    lookup: dict[str, list[str]] = {}
    for association in associate_tags(diagram):
        if association.target_shape_id is None:
            continue
        lookup.setdefault(association.target_shape_id, []).append(association.label_text)
    return lookup
