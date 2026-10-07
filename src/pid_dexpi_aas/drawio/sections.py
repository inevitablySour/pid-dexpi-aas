"""Plant sections drawn as frames around groups of equipment.

The Lumen drawings outline parts of the setup with thick coloured closed lines - "N2 - H2
Containers and safety measures", "Reactor, lamp and safety measures", "Autoclave", ... - each
with a title written inside. Two things depend on recognising them:

  - they are not pipes: drawn as draw.io lines, a frame was joined into pipe networks (net9,
    net10) and attached to symbols, i.e. read as fake connections;
  - they say which part of the plant every item belongs to (ISO 10209 / DEXPI plant section).

A frame = a closed draw.io line or plain box that holds items, with no line connecting to its
outline itself (see find_sections) - never decided by colour or line weight. Its
name = the title text inside it (a few words, not a tag, not a spec note, not a line end).
Every symbol whose centre lies inside belongs to that section (the smallest one when frames
nest). Needs the line routes (line_routes.py); without them frames stay as before.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from pid_dexpi_aas.drawio.parser import ParsedDiagram

_STROKE = re.compile(r"(?:^|;)strokeColor=([^;]+)")


@dataclass
class Section:
    id: str  # the frame line's cell id
    name: str
    polygon: list  # draw.io coordinates
    color: str = "#888888"
    members: list = field(default_factory=list)  # shape ids
    title_cell: str | None = None


def _area(poly) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]))) / 2


def inside(p, poly) -> bool:
    """Ray casting point-in-polygon."""
    x, y = p
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-9):
            hit = not hit
    return hit


def _is_title(text: str) -> bool:
    from pid_dexpi_aas.dexpi.mapping import classify_tag
    from pid_dexpi_aas.dexpi.spec_labels import parse_specs

    t = " ".join(text.split())
    if len(t) < 6 or len(t.split()) < 1:
        return False
    if parse_specs(t)[0] or classify_tag(t)[0]:
        return False
    return not re.fullmatch(r"[A-Z]{1,4}\s*-?\s*\d[\w/.-]*", t)


def _route_dist(p, route) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in zip(route, route[1:]):
        vx, vy = x2 - x1, y2 - y1
        L = vx * vx + vy * vy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - x1) * vx + (p[1] - y1) * vy) / L))
        best = min(best, math.hypot(p[0] - (x1 + t * vx), p[1] - (y1 + t * vy)))
    return best


def find_sections(diagram: ParsedDiagram, routes: dict) -> list[Section]:
    """Detect section frames - closed draw.io lines and plain boxes drawn around groups -,
    take them out of the pipe work, name them and assign members (in place: frame lines
    become decorative and loose; title cells are blanked so tag association cannot hang a
    section title on an item). Sets diagram.sections and returns them.

    A frame is recognised by what it holds and what connects to it, never by colour or
    line weight: it encloses at least two items (or one item with lines, or one item and a
    title - "Injection point" around V201), and no line connects to the outline itself -
    pipes go to the items inside."""
    from pid_dexpi_aas.drawio.plain_shapes import find_frames

    centre = lambda s: (s.x + s.width / 2, s.y + s.height / 2)
    symbols = [s for s in diagram.shapes if s.shape_token]
    loose_ends = [(e.id, pt) for e in diagram.edges for pt, att in ((e.source_point, e.source), (e.target_point, e.target))
                  if pt and att is None]
    sections = []
    for e in diagram.edges:
        r = routes.get(e.id)
        if not r or len(r) < 4 or e.style in ("pipe network", "adjacent") or math.dist(r[0], r[-1]) > 8:
            continue  # not a closed line
        poly = [tuple(p) for p in r[:-1]]
        holds = [s for s in symbols if inside(centre(s), poly)]
        lines_inside = sum(1 for f in diagram.edges if f.id != e.id and (r2 := routes.get(f.id))
                           and inside(r2[len(r2) // 2], poly))
        titled = any(not t.shape_token and t.text.strip() and _is_title(t.text) and inside(centre(t), poly)
                     for t in diagram.shapes)
        if not holds or (len(holds) < 2 and not lines_inside and not titled):
            continue  # holds nothing, or one item with no lines and no title: not a grouping
        if any(other != e.id and _route_dist(pt, r) <= 6 for other, pt in loose_ends):
            continue  # a line ends on it: a pipe loop, not a frame
        m = _STROKE.search(e.style or "")
        sections.append(Section(e.id, "", poly, m.group(1) if m else "#888888"))
        e.is_decorative, e.is_section_frame = True, True
        e.source = e.target = None
        e.merged_into = None
    for box in find_frames(diagram):
        m = _STROKE.search(box.style or "")
        sections.append(Section(box.id, box.text.strip(), [(box.x, box.y), (box.x + box.width, box.y),
                                                           (box.x + box.width, box.y + box.height), (box.x, box.y + box.height)],
                                m.group(1) if m else "#888888"))
    if not sections:
        diagram.sections = []
        return []

    endpoints = {x for e in diagram.edges for x in (e.source, e.target) if x}
    by_area = sorted(sections, key=lambda s: _area(s.polygon))
    smaller = lambda sec: [o for o in by_area if o is not sec and _area(o.polygon) < _area(sec.polygon)]
    # titles: free text inside the frame, closest to its outline (titles sit along the frame,
    # an item's own label sits by the item) - the smallest frame claims text first
    used = set()
    for sec in by_area:
        if sec.name:
            continue
        ring = sec.polygon + sec.polygon[:1]
        cands = [t for t in diagram.shapes if not t.shape_token and t.text.strip() and t.id not in used
                 and t.id not in endpoints and _is_title(t.text) and inside(centre(t), sec.polygon)
                 and not any(inside(centre(t), o.polygon) for o in smaller(sec))]
        if cands:
            best = min(cands, key=lambda t: (_route_dist(centre(t), ring), -len(t.text)))
            sec.name, sec.title_cell = " ".join(best.text.split()), best.id
            used.add(best.id)
    for n, sec in enumerate(sections, start=1):
        sec.name = sec.name or f"Section {n}"
    for t in diagram.shapes:
        if t.id in used:
            t.text = ""
    # members: every symbol inside, in the smallest frame that holds it
    for s in symbols:
        holder = next((sec for sec in by_area if inside(centre(s), sec.polygon) and s.id != sec.id), None)
        if holder:
            holder.members.append(s.id)
    diagram.sections = sections
    return sections


def section_of(diagram: ParsedDiagram) -> dict:
    """shape id -> section name."""
    return {m: sec.name for sec in (getattr(diagram, "sections", None) or []) for m in sec.members}
