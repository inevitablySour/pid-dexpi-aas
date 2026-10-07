"""Draw labelled symbols and connections on top of each drawing, for checking by eye.

For every docs/examples/pid2graph_export/*.graphml: the matching draw.io drawing as a flow map
(pid_dexpi_aas.render.flow_map - pipes along their drawn routes, coloured by how their flow
direction is known, arrowheads downstream), with each exported symbol's box coloured by
its training class and labelled with its tag. Needs the draw.io desktop app.

Usage (main venv):
    .venv/bin/python ml/pid2graph/data/render_labels.py [--out label_check/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

COLORS = {"valve": (0, 128, 128), "instrumentation": (67, 99, 216), "tank": (240, 50, 230),
          "pump": (145, 30, 180), "general": (40, 160, 60), "inlet/outlet": (245, 130, 49), "arrow": (230, 25, 75)}


def render(graphml: Path, drawio: Path, out_png: Path, max_side: int = 3200) -> dict:
    """Class-coloured boxes from the export on the shared flow map (pid_dexpi_aas.render.flow_map)."""
    from pid_dexpi_aas.render.flow_map import flow_map
    from pid_dexpi_aas.drawio.parser import parse_drawio_file
    from pid_dexpi_aas.review import from_drawio

    diagram = parse_drawio_file(drawio)
    g = nx.read_graphml(graphml)
    tags = {s.id: s.tag for s in from_drawio(diagram).symbols}
    boxes = []
    for n, a in g.nodes(data=True):
        sid = n[1:]  # export node ids are "n" + draw.io cell id
        tag = tags.get(sid)
        boxes.append((sid, (float(a["xmin"]), float(a["ymin"]), float(a["xmax"]), float(a["ymax"])),
                      COLORS.get(a["label"], (0, 0, 0)), a["label"] + (f": {tag}" if tag and tag != a["label"] else ""), 4))
    got = flow_map(diagram, drawio, boxes=boxes, max_side=max_side)
    if got is None:
        raise SystemExit("needs the draw.io app (ml/tools/drawio.AppImage) for line routes")
    im, stats = got
    im.save(out_png)
    fs = diagram.flow
    return {"symbols": g.number_of_nodes(), "connections": g.number_of_edges(),
            "pipe legs (arrow/implied/unknown)": (fs.by_arrow, fs.implied, fs.unknown) if fs else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exports", default=str(ROOT / "docs" / "examples" / "pid2graph_export"))
    ap.add_argument("--out", default=str(ROOT / "label_check"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    drawios = {p.stem.strip(): p for p in (ROOT / "subprojects").glob("*/*.drawio")}
    for gm in sorted(Path(args.exports).glob("*.graphml")):
        src = drawios.get(gm.stem.strip())
        if src is None:
            continue
        target = out / f"{gm.stem.strip()}.png"
        print(target.name, render(gm, src, target), flush=True)


if __name__ == "__main__":
    main()
