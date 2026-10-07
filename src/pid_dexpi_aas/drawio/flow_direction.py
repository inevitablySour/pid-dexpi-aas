"""Which way does the process fluid flow along each pipe?

Read from the arrowheads the drawing puts on its lines, then carried to lines without one:

  1. A line with an arrowhead at one end only flows towards that end ("arrow").
  2. A non-instrument item with exactly two pipes (a valve, a pump, a vessel in a line)
     passes the flow on: what comes in through one leaves through the other ("implied").
  3. A pipe junction (a tee; lines joined into one network) conserves flow: if every leg
     whose direction is known flows IN and one leg is unknown, that leg flows OUT - and
     the other way round.

Rules 2 and 3 repeat until nothing changes. A leg that would be forced both ways keeps
what it had (arrows win). Instruments, signal lines and compartment walls carry no
process flow and are left out. Lines with arrows at both ends, or none and no implied
direction, stay unknown (0) - they are shown as such for the reviewer, never guessed.

Used at two levels: on the parsed drawing (Edge.flow / flow_basis, shown in the review
table and check sheet) and on build_model's DEXPI links, which include the tee fittings
of pipe networks (segments are oriented source -> target from it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pid_dexpi_aas.drawio.parser import Edge, ParsedDiagram


@dataclass
class Leg:
    u: object
    v: object
    dir: int = 0  # +1 u -> v, -1 v -> u, 0 unknown
    basis: str = ""  # "arrow" | "implied" | ""


def arrow_dir(e: Edge) -> int:
    """+1 source -> target, -1 target -> source, 0 none / both ends."""
    if e.has_end_arrow and not e.has_start_arrow:
        return 1
    if e.has_start_arrow and not e.has_end_arrow:
        return -1
    return 0


def _sense(leg: Leg, node) -> int:
    """+1 if the leg flows INTO node, -1 out of it, 0 unknown."""
    if leg.dir == 0:
        return 0
    into_v = leg.dir == 1
    return (1 if into_v else -1) if leg.v is node else (-1 if into_v else 1)


def _set_sense(leg: Leg, node, sense: int) -> None:
    """Orient leg so it flows into node (sense +1) or out of it (-1)."""
    leg.dir = sense if leg.v is node else -sense
    leg.basis = "implied"


def propagate(legs: list[Leg], passes, junction) -> None:
    """Apply rules 2 and 3 in place. passes(node) / junction(node) -> bool. Nodes are
    matched by identity (`is`), so pass the same object for the same node."""
    at: dict[int, tuple[object, list[Leg]]] = {}
    for leg in legs:
        if leg.u is leg.v:
            continue
        for n in (leg.u, leg.v):
            at.setdefault(id(n), (n, []))[1].append(leg)
    changed = True
    while changed:
        changed = False
        for node, ls in at.values():
            senses = [_sense(l, node) for l in ls]
            unknown = [l for l, s in zip(ls, senses) if s == 0]
            known = [s for s in senses if s]
            if len(unknown) != 1 or not known or len(set(known)) != 1:
                continue
            if (len(ls) == 2 and passes(node)) or junction(node):
                _set_sense(unknown[0], node, -known[0])
                changed = True


@dataclass
class FlowSummary:
    by_arrow: int = 0
    implied: int = 0
    unknown: int = 0
    unknown_ids: list = field(default_factory=list)
    # (network id, shape id) -> (+1 the shape feeds the network | -1 is fed by it | 0, basis)
    member_sense: dict = field(default_factory=dict)


def infer_flow(diagram: ParsedDiagram) -> FlowSummary:
    """Set Edge.flow / flow_basis on the drawing's process connections, in place: direct
    lines and the derived "net:" edges of pipe networks."""
    from pid_dexpi_aas.dexpi.mapping import Category, map_shape

    shapes = {s.id: s for s in diagram.shapes}
    instruments = {sid for sid, s in shapes.items() if s.shape_token
                   and map_shape(s.shape_token, s.text).category == Category.INSTRUMENTATION}

    def process_line(e: Edge) -> bool:
        return not (e.is_decorative or e.is_dashed or e.style in ("adjacent", "pipe network"))

    hubs: dict[str, str] = {}  # network id -> hub node name (interned string)
    legs: list[Leg] = []
    direct: dict[str, Leg] = {}
    member_legs: dict[tuple[str, str], list[Leg]] = {}  # (network, shape) -> legs
    line_legs: dict[str, list[Leg]] = {}  # network member line -> its legs (same sense as the line)
    for e in diagram.edges:
        if not process_line(e):
            continue
        a = arrow_dir(e)
        if e.merged_into:
            hub = hubs.setdefault(e.merged_into, f"hub:{e.merged_into}")
            for end, sid in ((0, e.source), (1, e.target)):
                if sid is None or sid not in shapes:
                    continue
                # attached at the source end: flow shape -> hub when the line runs source -> target
                # attached at the target end: the leg runs hub -> shape, same sense as the line
                leg = Leg(shapes[sid], hub, a, "arrow" if a else "") if end == 0 else \
                    Leg(hub, shapes[sid], a, "arrow" if a else "")
                legs.append(leg)
                member_legs.setdefault((e.merged_into, sid), []).append(leg)
                line_legs.setdefault(e.id, []).append(leg)
        elif e.source in shapes and e.target in shapes:
            leg = Leg(shapes[e.source], shapes[e.target], a, "arrow" if a else "")
            legs.append(leg)
            direct[e.id] = leg

    hub_names = set(hubs.values())
    propagate(legs,
              passes=lambda n: not isinstance(n, str) and n.id not in instruments,
              junction=lambda n: isinstance(n, str) and n in hub_names)

    # Count pipe LEGS (a direct line, or one line end on a network), not symbol pairs: a
    # network's pairwise edges between two of its outlets have no flow between them at all.
    summary = FlowSummary()
    for leg in legs:
        if leg.u is leg.v or any(not isinstance(n, str) and n.id in instruments for n in (leg.u, leg.v)):
            continue  # an instrument tap carries no process flow
        if not leg.dir:
            summary.unknown += 1
            summary.unknown_ids.append(next((eid for eid, l in direct.items() if l is leg), None))
        elif leg.basis == "arrow":
            summary.by_arrow += 1
        else:
            summary.implied += 1

    for e in diagram.edges:
        if e.id in direct:
            e.flow, e.flow_basis = direct[e.id].dir, direct[e.id].basis
        elif e.id in line_legs:  # a drawn line inside a pipe network (for drawing it)
            known = {(l.dir, l.basis) for l in line_legs[e.id] if l.dir}
            dirs = {d for d, _ in known}
            e.flow = dirs.pop() if len(dirs) == 1 else arrow_dir(e)
            e.flow_basis = ("arrow" if arrow_dir(e) else "implied") if e.flow else ""
        elif e.style == "pipe network" and not e.is_dashed:
            net = e.id.split(":", 1)[0]

            def side(sid):
                ls = [l for l in member_legs.get((net, sid), []) if l.dir]
                # +1: this shape feeds the network, -1: it is fed by it
                senses = {-_sense(l, shapes[sid]) for l in ls}
                return (senses.pop() if len(senses) == 1 else 0), all(l.basis == "arrow" for l in ls)

            (sa, arrow_a), (sb, arrow_b) = side(e.source), side(e.target)
            for sid, sense, arrow in ((e.source, sa, arrow_a), (e.target, sb, arrow_b)):
                summary.member_sense[(net, sid)] = (sense, ("arrow" if arrow else "implied") if sense else "")
            if sa and sa == sb:
                e.flow, e.flow_basis = 0, "parallel"  # both feed it, or both are fed by it
            else:
                e.flow = 1 if (sa, sb) == (1, -1) else -1 if (sa, sb) == (-1, 1) else 0
                e.flow_basis = ("arrow" if arrow_a and arrow_b else "implied") if e.flow else ""
    return summary
