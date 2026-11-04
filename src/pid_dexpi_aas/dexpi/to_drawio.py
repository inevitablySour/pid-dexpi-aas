"""A DEXPI file back to a P&ID: a draw.io drawing the app's editor opens like any other.

A file with its drawing (Proteus graphics: dexpi/graphics.py writes them, as P&ID tools do) is
drawn as it was drawn - each item where its <Position> puts it, with its own symbol from the
<ShapeCatalogue>, pipes along their <CenterLine>s. A file with the plant data only is laid out:

- process items (equipment, valves and fittings, off-page connectors) left to right in flow
  order: a layer per step along the pipes (longest path), ordered within a layer to keep lines
  from crossing (barycentre sweeps), each placed level with what feeds it where it can be;
- instruments above what they measure, an instrument fed by another one above that one;
- pipes routed at right angles by draw.io itself (orthogonal edges), with their flow arrow.

Every item carries its DEXPI class, tag and specifications as editor data, so building DEXPI
from the drawing gives the same model back (the class is never guessed from the shape).
"""
from __future__ import annotations

import html
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import quoteattr

DY = 190                   # row pitch (draw.io units)
GAP = 46                   # room between neighbouring layers (for the tag under a valve)
INST_DY = 72               # instrument stack pitch above its item


# ------------------------------------------------------------------ shapes per class ---
def _shape(cls: str) -> tuple[str, int, int]:
    """(draw.io style, width, height) for a DEXPI class - the draw.io P&ID stencils the app's
    editor and the draw.io route use (dexpi/mapping.py reads them back)."""
    base = "html=1;verticalLabelPosition=bottom;verticalAlign=top;fontSize=10;"
    valve = "shape=mxgraph.pid2valves.valve;" + base
    table = {
        "Pump": ("shape=mxgraph.pid.pumps_-_iso.pump_(liquid);" + base, 40, 40),
        "CentrifugalPump": ("shape=mxgraph.pid.pumps_-_iso.pump_(liquid);" + base, 40, 40),
        "Tank": ("shape=mxgraph.pid.vessels.container,_tank,_cistern;" + base, 70, 70),
        "Vessel": ("shape=mxgraph.pid.vessels.pressurized_vessel;" + base, 50, 90),
        "PressureVessel": ("shape=mxgraph.pid.vessels.pressurized_vessel;" + base, 50, 90),
        "CustomVessel": ("shape=mxgraph.pid.vessels.pressurized_vessel;" + base, 50, 90),
        "ProcessColumn": ("shape=mxgraph.pid2misc.column;" + base, 40, 120),
        "GateValve": (valve + "valveType=gate;actuator=none;", 32, 16),
        "BallValve": (valve + "valveType=ball;actuator=none;", 32, 16),
        "GlobeValve": (valve + "valveType=globe;actuator=none;", 32, 16),
        "ButterflyValve": (valve + "valveType=butterfly;actuator=none;", 32, 16),
        "NeedleValve": (valve + "valveType=needle;actuator=none;", 32, 16),
        "CheckValve": (valve + "valveType=check;actuator=none;", 32, 16),
        "StraightwayValve": (valve + "valveType=gate;actuator=none;", 32, 16),
        "OperatedValve": (valve + "valveType=globe;actuator=diaph;", 32, 28),
        "SafetyValveOrFitting": (valve + "valveType=angleGlobe;actuator=spring;", 28, 28),
        "SpringLoadedGlobeSafetyValve": (valve + "valveType=angleGlobe;actuator=spring;", 28, 28),
        "PipeReducer": ("shape=trapezoid;perimeter=trapezoidPerimeter;direction=south;fixedSize=1;size=4;" + base, 16, 12),
        "PipeTee": ("shape=ellipse;perimeter=ellipsePerimeter;fillColor=#000000;strokeColor=#000000;html=1;", 6, 6),
        "(off-page connector)": ("shape=offPageConnector;direction=east;whiteSpace=wrap;" + base, 46, 22),
        "ProcessInstrumentationFunction": ("shape=mxgraph.pid2inst.discInst;mounting=field;html=1;fontSize=10;whiteSpace=wrap;", 36, 36),
    }
    if cls in table:
        return table[cls]
    if cls.endswith("Valve"):
        return valve + "valveType=gate;actuator=none;", 32, 16
    if any(w in cls for w in ("Vessel", "Tank", "Silo", "Separator", "Column")):
        return table["Vessel"]
    if "Pump" in cls or "Compressor" in cls or "Blower" in cls:
        return table["Pump"]
    if "Equipment" in cls or "Exchanger" in cls or "Mixer" in cls or "Filter" in cls:
        return "rounded=1;whiteSpace=wrap;" + base, 60, 50
    return "rounded=0;whiteSpace=wrap;html=1;fontSize=9;", 18, 14   # a fitting / piping component


# ------------------------------------------------------------------ reading the model ---
def _value(v):
    if v is None:
        return None
    if hasattr(v, "value") and hasattr(v, "unit"):   # a quantity
        u = v.unit.value if hasattr(v.unit, "value") else v.unit
        return {"value": v.value, "unit": u} if v.value is not None else None
    if hasattr(v, "value"):   # an enumeration
        return v.value
    return v if isinstance(v, (str, int, float)) else None


def _specs(obj) -> dict:
    """The editor's specifications ({"dexpi": {...}, "custom": {...}}) of a DEXPI object."""
    from pid_dexpi_aas.dexpi.fields import spec_fields
    dexpi = {}
    for f in spec_fields(type(obj).__name__):
        if f.custom:
            continue
        v = _value(getattr(obj, f.name, None))
        if v not in (None, "", {}):
            dexpi[f.name] = v
    custom = {}
    for a in getattr(obj, "customAttributes", None) or []:
        name, v = getattr(a, "attributeName", None), getattr(a, "value", None)
        if name and v not in (None, "") and name not in ("TagName", "PlantSectionName"):
            custom[name] = v
    out = {}
    if dexpi:
        out["dexpi"] = dexpi
    if custom:
        out["custom"] = custom
    return out


def _tag(obj) -> str:
    cls = type(obj).__name__
    if cls == "ProcessInstrumentationFunction":
        cat = getattr(obj, "processInstrumentationFunctionCategory", None) or ""
        mod = getattr(obj, "processInstrumentationFunctionModifier", None) or ""
        num = getattr(obj, "processInstrumentationFunctionNumber", None) or ""
        return f"{cat}{mod} {num}".strip()
    for attr in ("tagName", "pipingComponentName"):
        v = getattr(obj, attr, None)
        if v:
            return v
    for a in getattr(obj, "customAttributes", None) or []:
        if getattr(a, "attributeName", "") in ("TagName", "ConnectorName"):
            return str(a.value)
    return ""


def _load(xml: bytes):
    from pid_dexpi_aas.dexpi.proteus_patch import PatchedProteusSerializer
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "m.xml").write_bytes(xml)
        return PatchedProteusSerializer().load(Path(tmp), "m")


def read_plant(xml: bytes) -> dict:
    """Items, pipes and instrument lines of a DEXPI file, by object:
    {"items": {id: obj}, "pipes": [(a, b, segment)], "measure": [(inst, at)], "signals": [(a, b)]}."""
    model = _load(xml)
    cm = model.conceptualModel
    # what pyDEXPI's reader drops (an off-page connector's custom attributes and reference):
    # straight from the XML, by element ID (= the object's proteusId)
    import xml.etree.ElementTree as ET
    extra = {}
    for el in ET.fromstring(xml).iter():
        if el.tag != "PipeOffPageConnector" or not el.get("ID"):
            continue
        attrs = {g.get("Name"): g.get("Value") for ga in el.findall("GenericAttributes[@Set='CustomAttributes']")
                 for g in ga.findall("GenericAttribute")}
        ref = el.find("PipeOffPageConnectorReference")
        if ref is not None:
            for g in ref.iter("GenericAttribute"):
                if g.get("Name") == "ReferencedDrawingNumberAssignmentClass":
                    attrs["ConnectedDrawing"] = g.get("Value")
        extra[el.get("ID")] = attrs
    items, owner = {}, {}
    for eq in cm.taggedPlantItems or []:
        items[eq.id] = eq
        for n in getattr(eq, "nozzles", None) or []:
            owner[n.id] = eq.id
    node = lambda o: None if o is None else owner.get(o.id, o.id)
    pipes = []
    for system in cm.pipingNetworkSystems or []:
        for seg in system.segments or []:
            for it in seg.items or []:
                items.setdefault(it.id, it)
            pairs = [(c.sourceItem, c.targetItem) for c in seg.connections or []]
            # pyDEXPI's reader does not always give a segment's links to its two ends back as
            # connections: the chain source -> items -> target says them too
            chain = [o for o in [getattr(seg, "sourceItem", None), *(seg.items or []), getattr(seg, "targetItem", None)]
                     if o is not None]
            pairs += list(zip(chain, chain[1:]))
            seen = set()
            for sa, sb in pairs:
                a, b = node(sa), node(sb)
                for o in (sa, sb):
                    if o is not None and o.id not in owner:
                        items.setdefault(o.id, o)
                if a and b and a != b and (a, b) not in seen and (b, a) not in seen:
                    seen.add((a, b))
                    pipes.append((a, b, seg))
    measure, signals = [], []
    for pif in cm.processInstrumentationFunctions or []:
        items[pif.id] = pif
        for g in pif.processSignalGeneratingFunctions or []:
            loc = getattr(g, "sensingLocation", None)
            if loc is not None:
                if hasattr(loc, "connections"):   # measuring on a pipe segment: onto that pipe
                    measure.append((pif.id, ("segment", id(loc), node((loc.items or [None])[0]))))
                    continue
                at = node(loc)
                if at in items or at in owner.values():
                    measure.append((pif.id, at))
        for f in pif.signalConveyingFunctions or []:
            src, tgt = getattr(f, "source", None), getattr(f, "target", None)
            ids = []
            for o in (src, tgt):
                if o is None:
                    ids.append(None)
                elif type(o).__name__ == "ProcessSignalGeneratingFunction":
                    ids.append(None)   # the measuring line: drawn from what it measures
                elif type(o).__name__ == "ActuatingFunction":   # acts on a valve: the line goes to it
                    loc = getattr(o, "actuatingLocation", None)
                    ids.append(node(loc) if loc is not None else None)
                else:
                    ids.append(node(o))
            if ids[0] and ids[1] and ids[0] != ids[1]:
                signals.append((ids[0], ids[1]))
        for af in pif.actuatingFunctions or []:
            loc = getattr(af, "actuatingLocation", None)
            reached = any(getattr(f, "target", None) is af for f in pif.signalConveyingFunctions or [])
            if loc is not None and not reached:
                signals.append((pif.id, node(loc)))
    return {"items": items, "pipes": pipes, "measure": measure, "signals": signals, "model": model,
            "extra": extra}


# ------------------------------------------------------------------ the layout ---
def layout(plant: dict) -> dict:
    """{item id: (x, y)} - process items left to right by flow, instruments above."""
    import networkx as nx
    items = plant["items"]
    is_inst = lambda i: type(items[i]).__name__ == "ProcessInstrumentationFunction"
    G = nx.DiGraph()
    for i in items:
        if not is_inst(i):
            G.add_node(i)
    for a, b, _ in plant["pipes"]:
        if a in G and b in G:
            G.add_edge(a, b)
    # cycles (recycle loops): the edges that close one go backwards; the rest is a DAG
    D = nx.DiGraph(G)
    while True:
        try:
            cyc = nx.find_cycle(D)
        except nx.NetworkXNoCycle:
            break
        D.remove_edge(*cyc[-1][:2])
    layer = {}
    for v in nx.topological_sort(D):
        layer[v] = max((layer[u] + 1 for u in D.predecessors(v)), default=0)
    last = max(layer.values(), default=0)
    for v in D:
        if type(items[v]).__name__ == "FlowOutPipeOffPageConnector" and not list(D.successors(v)):
            layer[v] = last   # streams leaving the sheet: at its right edge
        if type(items[v]).__name__ == "FlowInPipeOffPageConnector" and not list(D.predecessors(v)):
            layer[v] = 0
    # order within layers: start from a walk along the pipes, then barycentre sweeps
    by_layer = defaultdict(list)
    seen = set()
    for comp in sorted(nx.weakly_connected_components(D), key=len, reverse=True):
        start = sorted(comp, key=lambda v: layer[v])
        for v in nx.dfs_preorder_nodes(D.to_undirected(), start[0]):
            if v not in seen:
                seen.add(v)
                by_layer[layer[v]].append(v)
    pos = {v: k for L in by_layer.values() for k, v in enumerate(L)}
    U = D.to_undirected()
    for sweep in range(8):
        rng = range(1, last + 1) if sweep % 2 == 0 else range(last - 1, -1, -1)
        for L in rng:
            side = (lambda v: [u for u in U[v] if layer[u] < L]) if sweep % 2 == 0 else (lambda v: [u for u in U[v] if layer[u] > L])
            bary = {v: (sum(pos[u] for u in side(v)) / len(side(v))) if side(v) else pos[v] for v in by_layer[L]}
            by_layer[L].sort(key=lambda v: bary[v])
            for k, v in enumerate(by_layer[L]):
                pos[v] = k
    # rows: each item level with what feeds it where it can be, never above its neighbour
    y = {}
    for L in range(last + 1):
        floor = -math.inf
        for v in by_layer[L]:
            feeds = sorted(y[u] for u in U[v] if u in y)
            want = feeds[len(feeds) // 2] if feeds else (floor + DY if floor > -math.inf else 0)
            y[v] = max(want, floor + DY)
            floor = y[v]
    # each layer as wide as what is in it: a column of small valves and fittings is narrow
    width = lambda v: _shape(_class(items[v]))[1]
    xs, x = {}, 0.0
    for L in range(last + 1):
        wL = max((width(v) for v in by_layer[L]), default=20)
        if L:
            x += prev / 2 + wL / 2 + GAP
        xs[L], prev = x, wL
    xy = {v: (xs[layer[v]], y[v]) for v in D}
    # instruments: above what they measure, stacked; one fed by an instrument, above that one
    anchor = {}
    for inst, at in plant["measure"]:
        anchor.setdefault(inst, at[2] if isinstance(at, tuple) else at)
    for a, b in plant["signals"]:
        for i, j in ((b, a), (a, b)):
            if i in items and is_inst(i) and i not in anchor and j in items:
                anchor[i] = j
    stack = defaultdict(int)
    placed = set()

    def put(i, depth=0):
        if i in placed or depth > 6:
            return
        at = anchor.get(i)
        if at is not None and at in items and is_inst(at):
            put(at, depth + 1)
        if at in xy:
            ax, ay = xy[at]
            stack[at] += 1
            xy[i] = (ax + (stack[at] - 1) * 0 if not is_inst(at) else ax + 50 * (stack[at] - 1),
                     ay - INST_DY * (1 if not is_inst(at) else 1) * (stack[at] if not is_inst(at) else 1))
            placed.add(i)
    for i in items:
        if is_inst(i):
            put(i)
    rest = [i for i in items if i not in xy]
    bottom = max((p[1] for p in xy.values()), default=0) + DY
    for k, i in enumerate(rest):   # nothing to place them by: a row underneath
        xy[i] = (k * 60, bottom)
    return xy


# ------------------------------------------------------------------ draw.io output ---
def _seg_dist(p, a, b) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    L = ax * ax + ay * ay
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L))
    return math.dist(p, (a[0] + t * ax, a[1] + t * ay))


def _centre_line(own: dict, seg, a, b, xy) -> list | None:
    """The drawn centre line of `seg` that runs from item a to item b (its ends on their
    boxes, within a few units), oriented a -> b; None if the drawing has none."""
    def box_dist(p, i):
        if i not in xy:
            return 1e9
        cx, cy = xy[i]
        w, h = own["size"].get(i, (6, 6))
        return math.hypot(max(abs(p[0] - cx) - w / 2, 0), max(abs(p[1] - cy) - h / 2, 0))
    def fit(lines, tol):
        best = None
        for pl in lines:
            for line in (pl, pl[::-1]):
                d = box_dist(line[0], a) + box_dist(line[-1], b)
                if best is None or d < best[0]:
                    best = (d, line)
        return best[1] if best and best[0] <= tol else None
    # its own segment's lines first; a network's lines can sit on a neighbouring segment
    # (then only one that ends right on both)
    return fit(own["lines"].get(getattr(seg, "proteusId", None) or "", []), 16) or fit(own["every"], 4)


def _class(obj) -> str:
    cls = type(obj).__name__
    return "(off-page connector)" if cls in ("FlowInPipeOffPageConnector", "FlowOutPipeOffPageConnector") else cls


UNITS = 4.0   # draw.io units per mm (the file's drawing is in mm)


def drawn(xml: bytes, plant: dict) -> dict | None:
    """The file's own drawing, when it has one (Proteus graphics): {"xy": {item: centre},
    "size": {item: (w, h)}, "style": {item: draw.io style with its catalogue symbol},
    "lines": {segment proteusId: [polyline]}} in draw.io units; None without one."""
    import base64
    import xml.etree.ElementTree as ET

    from pid_dexpi_aas.render.dexpi_svg import dexpi_svg
    root = ET.fromstring(xml)
    cat = root.find("ShapeCatalogue")
    catalogue = {e.get("ComponentName") or e.get("ID"): e for e in (cat if cat is not None else [])}
    by_id = {el.get("ID"): el for el in root.iter() if el.get("ID")}
    num = lambda e, k: float(e.get(k, 0) or 0)
    tops = [num(m, "Y") for m in root.iter("Max")]
    if not tops:
        return None
    H = max(tops)
    P = lambda x, y: (x * UNITS, (H - y) * UNITS)
    out = {"xy": {}, "size": {}, "style": {}, "lines": {}}
    symbols = {}
    for i, obj in plant["items"].items():
        el = by_id.get(getattr(obj, "proteusId", None) or "")
        if el is None:
            continue
        loc = el.find("Position/Location")
        ext = el.find("Extent")
        if loc is None:
            continue
        lx, ly = num(loc, "X"), num(loc, "Y")
        if ext is not None and ext.find("Min") is not None:
            mn, mx = ext.find("Min"), ext.find("Max")
            x0, y0, x1, y1 = num(mn, "X"), num(mn, "Y"), num(mx, "X"), num(mx, "Y")
        else:
            x0, y0, x1, y1 = lx - 4, ly - 4, lx + 4, ly + 4
        (cx, cy) = P((x0 + x1) / 2, (y0 + y1) / 2)
        out["xy"][i] = (cx, cy)
        out["size"][i] = (max((x1 - x0) * UNITS, 4), max((y1 - y0) * UNITS, 4))
        sym = catalogue.get(el.get("ComponentName"))
        if sym is None:
            continue
        pos = el.find("Position")
        key = (el.get("ComponentName"), ET.tostring(pos), round(x0 - lx, 2), round(y0 - ly, 2), round(x1 - lx, 2), round(y1 - ly, 2))
        if key not in symbols:   # the symbol on its own, cut out at the item's extent
            pm = ET.Element("PlantModel")
            ET.SubElement(pm, "ShapeCatalogue").append(sym)
            ET.SubElement(pm, "Item", ComponentName=el.get("ComponentName")).append(pos)
            mini = ET.tostring(pm)
            cat.append(sym) if sym not in list(cat) else None   # (append moved it out of the catalogue)
            el.insert(0, pos) if pos not in list(el) else None
            svg = dexpi_svg(mini, view=(x0, y0, x1, y1), background=False)
            symbols[key] = base64.b64encode(svg.encode()).decode() if svg else None
        if symbols[key]:
            out["style"][i] = (f"shape=image;image=data:image/svg+xml,{symbols[key]};imageAspect=0;html=1;fontSize=10;"
                               "verticalLabelPosition=bottom;verticalAlign=top;")
    for el in root.iter("PipingNetworkSegment"):
        lines = [[P(num(c, "X"), num(c, "Y")) for c in cl.findall("Coordinate")] for cl in el.findall("CenterLine")]
        out["lines"][el.get("ID")] = [ln for ln in lines if len(ln) >= 2]
    out["every"] = [ln for ls in out["lines"].values() for ln in ls]
    out["signal_lines"] = []   # (polyline, measuring?) of instrument lines (InformationFlow)
    for el in root.iter("InformationFlow"):
        measuring = el.get("ComponentClass") == "MeasuringLineFunction"
        for cl in el.findall("CenterLine"):
            ln = [P(num(c, "X"), num(c, "Y")) for c in cl.findall("Coordinate")]
            if len(ln) >= 2:
                out["signal_lines"].append((ln, measuring))
    if len(out["xy"]) < 0.5 * max(1, len(plant["items"])):
        return None   # too little of a drawing to go by
    return out


def dexpi_to_drawio(xml: bytes, title: str = "from DEXPI") -> tuple[str, dict]:
    """(draw.io XML, report) for DEXPI file `xml`: drawn as its own drawing has it when it has
    one, else laid out from the plant data."""
    plant = read_plant(xml)
    items = plant["items"]
    xy = layout(plant)
    own = drawn(xml, plant)
    if own:
        # where the drawing puts things; a tee, where its pipes' centre lines meet
        placed = own["xy"]
        for i, obj in items.items():
            if i in placed:
                xy[i] = placed[i]
            elif type(obj).__name__ == "PipeTee" and "position" not in own.get("tees", {}).get(i, {}):
                # where one of its pipes' drawn ends lands on another drawn pipe, away from any
                # item - the junction nearest the items it joins
                legs = [(a, b, seg) for a, b, seg in plant["pipes"] if i in (a, b)]
                every = own["every"]
                near_ids = [b if a == i else a for a, b, _ in legs]
                def touches(pl):
                    return any(math.hypot(max(abs(q[0] - placed[n][0]) - own["size"][n][0] / 2, 0),
                                          max(abs(q[1] - placed[n][1]) - own["size"][n][1] / 2, 0)) <= 6
                               for n in near_ids if n in placed and n in own["size"] for q in (pl[0], pl[-1]))
                lines = [pl for pl in every if touches(pl)]
                near = [b if a == i else a for a, b, _ in legs]
                anchors = [placed[n] for n in near if n in placed]
                def off_items(p):
                    return all(math.hypot(max(abs(p[0] - placed[n][0]) - own["size"][n][0] / 2, 0),
                                          max(abs(p[1] - placed[n][1]) - own["size"][n][1] / 2, 0)) > 6
                               for n in placed if n in own["size"])
                cands = [p for pl in lines for p in (pl[0], pl[-1])
                         if off_items(p) and any(_seg_dist(p, q, r) < 2 for ln in every if p not in (ln[0], ln[-1])
                                                 for q, r in zip(ln, ln[1:]))]
                if cands:
                    cx = sum(p[0] for p in anchors) / len(anchors) if anchors else 0
                    cy = sum(p[1] for p in anchors) / len(anchors) if anchors else 0
                    best = min(cands, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
                    reach = max((math.hypot(p[0] - cx, p[1] - cy) for p in anchors), default=0)
                    # never far from what it joins: else between them
                    xy[i] = best if math.hypot(best[0] - cx, best[1] - cy) <= max(reach, 40) else (cx, cy)
                elif anchors:
                    xy[i] = (sum(p[0] for p in anchors) / len(anchors), sum(p[1] for p in anchors) / len(anchors))
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    ids = {}
    streams = defaultdict(int)
    for k, (i, obj) in enumerate(items.items()):
        if i not in xy or type(obj).__name__ == "PipeTee":   # a tee: where the pipes meet (below)
            continue
        cls = _class(obj)
        style, w, h = _shape(cls)
        more = plant["extra"].get(getattr(obj, "proteusId", None) or "", {})
        tag = _tag(obj) or more.get("ConnectorName", "")
        if not tag and cls == "(off-page connector)":   # a stream needs a name to be one
            way = "Inlet" if type(obj).__name__ == "FlowInPipeOffPageConnector" else "Outlet"
            streams[way] += 1
            tag = f"{way} {streams[way]}"
        if cls == "ProcessInstrumentationFunction":
            letters, _, number = tag.partition(" ")
            label = f"{html.escape(letters)}<br>{html.escape(number)}"
            loc = _value(getattr(obj, "location", None))
            if loc == "central":
                style = style.replace("mounting=field", "mounting=room")
        else:
            label = html.escape(tag)
        if own and i in own["style"]:
            style = own["style"][i] + ("verticalLabelPosition=middle;verticalAlign=middle;"
                                       if cls == "ProcessInstrumentationFunction" else "")
        if own and i in own["size"]:
            w, h = own["size"][i]
        x, y = xy[i]
        cid = f"d{k}"
        ids[i] = cid
        data = {"label": label, "dexpi_class": cls, "dexpi_auto": "0", "notes": "from the DEXPI file"}
        if tag and cls != "PipeReducer":   # a reducer's "80/50" is its size, not a tag
            data.update(tag=tag, tag_auto="0")
        specs = _specs(obj)
        ref = getattr(obj, "connectorReference", None)
        if ref is not None and getattr(ref, "referencedDrawingNumber", None):
            specs.setdefault("custom", {})["ConnectedDrawing"] = ref.referencedDrawingNumber
        for k2, v2 in more.items():
            if k2 != "ConnectorName" and v2:
                specs.setdefault("custom", {}).setdefault(k2, v2)
        if specs:
            data["specs"] = json.dumps(specs, sort_keys=True)
        attrs = " ".join(f"{a}={quoteattr(str(v))}" for a, v in data.items())
        geo = f'<mxGeometry x="{x - w / 2:.1f}" y="{y - h / 2:.1f}" width="{w}" height="{h}" as="geometry"/>'
        cells.append(f'<object id="{cid}" {attrs}><mxCell style={quoteattr(style)} vertex="1" parent="1">{geo}</mxCell></object>')
    n_pipes = n_sig = 0
    from pydexpi.dexpi_classes.pydantic_classes import PipingNetworkSegmentFlowClassification as Flow
    # pipes. A tee is not a shape of its own: as on any draw.io P&ID, one pipe runs through
    # the tee's point (a waypoint) and the branch ends on it - the draw.io route reads that as
    # the tee (a pipe network), so DEXPI gets its PipeTee back
    tees = {i for i, o in items.items() if type(o).__name__ == "PipeTee" and i in xy}
    pieces, seen_pair = [], set()
    for a, b, seg in plant["pipes"]:
        if (a in ids or a in tees) and (b in ids or b in tees) and (a, b) not in seen_pair:
            seen_pair.add((a, b))
            pieces.append([a, b, seg])
    partner: dict = {}   # (piece index, tee) -> the piece it continues into through that tee
    for t in sorted(tees):
        legs = [k for k, (a, b, _) in enumerate(pieces) if t in (a, b)]
        ins = [k for k in legs if pieces[k][1] == t]
        outs = [k for k in legs if pieces[k][0] == t]
        pair = (ins[0], outs[0]) if ins and outs else tuple(legs[:2]) if len(legs) >= 2 else None
        if pair:
            partner[(pair[0], t)], partner[(pair[1], t)] = pair[1], pair[0]
    line_numbers = set()
    tee_pt = lambda t: xy[t]
    # runs: pieces joined through the tees they pass, walked from an end no run continues past
    runs, used = [], set()
    def walk(k, node):
        seq, ks = [node], []
        while k is not None and k not in used:
            used.add(k)
            a, b, _s = pieces[k]
            other = b if node == a else a
            seq.append(other)
            ks.append(k)
            node, k = other, (partner.get((k, other)) if other in tees else None)
        return seq, ks
    starts = [(k, e) for k, (a, b, _s) in enumerate(pieces) for e in (a, b)
              if not (e in tees and (k, e) in partner)]
    for k, e in starts + [(k, pieces[k][0]) for k in range(len(pieces))]:
        if k not in used:
            seq, ks = walk(k, e)
            # the way most of its pieces flow (source -> target)
            forward = sum(1 for i, kk in enumerate(ks) if pieces[kk][0] == seq[i])
            if forward * 2 < len(ks):
                seq, ks = seq[::-1], ks[::-1]
            runs.append((seq, ks))
    through = {t: f"p{k0}" for k0, (seq, _ks) in enumerate(runs) for t in seq[1:-1]}   # tee -> its run
    run_of_seg = {}
    for k0, (seq, ks) in enumerate(runs):
        for kk in ks:
            run_of_seg.setdefault(id(pieces[kk][2]), (f"p{k0}", seq))
    for k0, (seq, ks) in enumerate(runs):
        src, tgt, pts = seq[0], seq[-1], [tee_pt(t) for t in seq[1:-1]]
        segs = [pieces[kk][2] for kk in ks]
        seg = segs[0]
        flow = getattr(seg, "flowDirection", None)
        arrow = ("endArrow=block;endFill=1;endSize=5;startArrow=block;startFill=1;startSize=5;" if flow == Flow.DualFlowPipingNetworkSegment
                 else "endArrow=block;endFill=1;endSize=5;" if flow is not None else "endArrow=none;")
        style = arrow + "html=1;rounded=0;edgeStyle=orthogonalEdgeStyle;"
        specs = _specs(seg)
        attrs = f' specs={quoteattr(json.dumps(specs, sort_keys=True))}' if specs else ""
        label = next((getattr(sg, "segmentNumber", None) for sg in segs if getattr(sg, "segmentNumber", None)), "") or ""
        if label in line_numbers:
            label = ""   # each line number written once, on the first stretch of its pipe
        line_numbers.add(label)
        if label:
            style += "fontSize=9;labelBackgroundColor=#ffffff;"
        # a branch ends on the pipe running through its tee: attached to that line itself
        terms = "".join(f' {side}="{ids[n]}"' if n in ids else f' {side}="{through[n]}"' if n in through else ""
                        for side, n in (("source", src), ("target", tgt)))
        geo = "".join(f'<mxPoint x="{tee_pt(n)[0]:.1f}" y="{tee_pt(n)[1]:.1f}" as="{side}Point"/>'
                      for side, n in (("source", src), ("target", tgt)) if n in tees)
        if own:   # as drawn: each piece along the centre line that runs between its two ends
            drawn_pts, ok = [], True
            for n_i, kk in enumerate(ks):
                a_, b_ = seq[n_i], seq[n_i + 1]
                pl = _centre_line(own, pieces[kk][2], a_, b_, xy)
                if pl is None:
                    ok = False
                    break
                drawn_pts += pl[1:-1] + ([xy[b_]] if n_i < len(ks) - 1 else [])
            if ok:
                pts = drawn_pts
                style = style.replace("edgeStyle=orthogonalEdgeStyle;", "edgeStyle=none;")
        if pts:
            geo += '<Array as="points">' + "".join(f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in pts) + "</Array>"
        cells.append(f'<object id="p{k0}" label={quoteattr(label)}{attrs}><mxCell style="{style}" edge="1" parent="1"{terms}>'
                     f'<mxGeometry relative="1" as="geometry">{geo}</mxGeometry></mxCell></object>')
        n_pipes += 1
    for k, (inst, at) in enumerate(plant["measure"]):
        if isinstance(at, tuple):   # on a pipe segment: the line attached to that pipe
            run = run_of_seg.get(at[1])
            if inst in ids and run:
                pid, seq = run
                pts = [xy[n] for n in seq if n in xy]
                x, y = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2) if len(pts) > 1 else pts[0]
                cells.append(f'<mxCell id="m{k}" style="endArrow=none;html=1;rounded=0;edgeStyle=orthogonalEdgeStyle;" '
                             f'edge="1" parent="1" source="{ids[inst]}" target="{pid}"><mxGeometry relative="1" as="geometry">'
                             f'<mxPoint x="{x:.1f}" y="{y:.1f}" as="targetPoint"/></mxGeometry></mxCell>')
            continue
        if inst in ids and at in ids:
            cells.append(f'<mxCell id="m{k}" style="endArrow=none;html=1;rounded=0;edgeStyle=orthogonalEdgeStyle;" edge="1" '
                         f'parent="1" source="{ids[inst]}" target="{ids[at]}"><mxGeometry relative="1" as="geometry"/></mxCell>')
        elif inst in ids and at in through:   # measuring on the pipe where a tee was: onto that pipe
            x, y = tee_pt(at)
            cells.append(f'<mxCell id="m{k}" style="endArrow=none;html=1;rounded=0;edgeStyle=orthogonalEdgeStyle;" edge="1" '
                         f'parent="1" source="{ids[inst]}" target="{through[at]}"><mxGeometry relative="1" as="geometry">'
                         f'<mxPoint x="{x:.1f}" y="{y:.1f}" as="targetPoint"/></mxGeometry></mxCell>')
    for k, (a, b) in enumerate(plant["signals"]):
        if a in ids and b in ids:
            cells.append(f'<mxCell id="s{k}" style="endArrow=none;dashed=1;html=1;rounded=0;edgeStyle=orthogonalEdgeStyle;" '
                         f'edge="1" parent="1" source="{ids[a]}" target="{ids[b]}"><mxGeometry relative="1" as="geometry"/></mxCell>')
            n_sig += 1
    body = "".join(cells)
    xml_out = (f'<mxfile><diagram name={quoteattr(title)} id="dexpi"><mxGraphModel dx="0" dy="0" grid="1" gridSize="10" '
               f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" math="0" shadow="0">'
               f'<root>{body}</root></mxGraphModel></diagram></mxfile>')
    counts = defaultdict(int)
    for i, obj in items.items():
        counts[_class(obj)] += 1
    return xml_out, {"items": len(ids), "pipes": n_pipes, "signal_lines": n_sig, "classes": dict(counts),
                     "instruments": sum(1 for o in items.values() if type(o).__name__ == "ProcessInstrumentationFunction")}
