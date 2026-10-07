"""Turn the team's draw.io P&IDs into PID2Graph-format training/test data (images + labels).

Runs in the MAIN venv (needs the draw.io parser and review layer), plus the draw.io
desktop app for rendering (ml/tools/drawio.AppImage, see ml/README.md). For each file:

  1. Render it to PNG with draw.io itself (scale 1, no border) - the team's real style:
     grey-shaded vessels, filled valves, coloured columns, which the synthetic training
     data never shows.
  2. Labels come from export_pid2graph.labelled_symbols - the same labelling as the
     (regenerated) docs/examples/pid2graph_export files: only real symbols, with their
     PID2Graph class; connections keep their line type (dashed = "non-solid").
  3. draw.io's export origin is NOT simply the min of the parsed geometry (stray points
     stretch it), so the draw.io -> pixel offset is found by sliding the symbol boxes
     over the rendered ink and keeping the best overlap (measured: LightUp's naive origin
     was 56 px off; registered boxes sit on their symbols).
  4. Write the whole sheet (Complete/<subset>/<name>.png + .graphml, for sheet-level
     evaluation) and 1500 px patches at a 500 px stride (Patched/<subset>/<name>/<n>.*,
     PID2Graph's own layout), keeping symbols >= 50% inside a patch (clipped) and
     connections whose two ends are both kept.
  5. Flow arrows (unless --no-arrows): every arrowhead visible in the rendering is labelled
     "arrow" and splits the line it sits on, as in OPEN100 (labelled_symbols(arrows=True)).
     Team / Team v3 were made without them.

Usage:
    .venv/bin/python ml/pid2graph/data/make_team_data.py --out data/PID2Graph \
        --test "P&ID gramflow reactor V3"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pid_dexpi_aas.drawio.decode import decode_drawio  # noqa: E402
from align import register_offset  # noqa: E402
from pid_dexpi_aas.image.export_pid2graph import labelled_symbols  # noqa: E402
from pid_dexpi_aas.drawio.line_routes import DRAWIO_APP  # noqa: E402  (finds the installed app)
from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved  # noqa: E402
from pid_dexpi_aas.image.training_export import dir_writer, write_sheet  # noqa: E402


def render_png(drawio_path: Path, out_png: Path) -> None:
    subprocess.run([str(DRAWIO_APP), "--no-sandbox", "-x", "-f", "png", "-b", "0", "-s", "1",
                    "-o", str(out_png), str(drawio_path)], check=True, capture_output=True)


def process(drawio_path: Path, out_root: Path, subset: str, tile: int, stride: int, arrows: bool = True) -> dict:
    diagram = parse_drawio_xml_resolved(decode_drawio(drawio_path)[0], source_file=drawio_path.name, drawio_path=drawio_path)
    syms, edges = labelled_symbols(diagram)  # the same labelling as docs/examples/pid2graph_export

    name = drawio_path.stem.strip()
    with tempfile.TemporaryDirectory() as tmp:
        png_tmp = Path(tmp) / "sheet.png"
        render_png(drawio_path, png_tmp)
        ox, oy, cov = register_offset([b for _, _, b, _ in syms], png_tmp)
        sheet = Image.open(png_tmp).convert("RGB")
    if arrows:  # the heads are fitted to the rendered ink, so this comes after registration
        dark = np.asarray(sheet.convert("L")) < 128

        def ink(points):
            inside = [dark[int(round(y - oy)), int(round(x - ox))] for x, y in points
                      if 0 <= round(y - oy) < dark.shape[0] and 0 <= round(x - ox) < dark.shape[1]]
            return float(np.mean(inside)) if inside else 0.0
        syms, edges = labelled_symbols(diagram, arrows=True, ink=ink)
    px = [(sid, lab, (b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy)) for sid, lab, b, _ in syms]
    # the same writer as pictures reviewed in the app (pid_dexpi_aas.image.training_export)
    n = write_sheet(dir_writer(out_root), sheet, name, subset, px, edges, tile, stride)
    W, H = sheet.size
    return {"file": drawio_path.name, "subset": subset, "size": (W, H), "symbols": len(px), "edges": len(edges),
            "arrows": sum(lab == "arrow" for _, lab, _ in px),
            "patches": n, "offset": (round(ox), round(oy)), "ink_coverage": round(cov, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "PID2Graph"))
    ap.add_argument("--test", nargs="*", default=["P&ID gramflow reactor V3"],
                    help="file stems held out as the team test set")
    ap.add_argument("--skip", nargs="*", default=["PFD combined set up 301122"],
                    help="file stems left out entirely (the PFD is a simplified flow diagram, not a P&ID)")
    ap.add_argument("--subset", default="Team",
                    help="subset name; the held-out drawing goes to '<subset> test'. Use a new name "
                         "(e.g. 'Team v2') to regenerate without touching data a running training reads")
    ap.add_argument("--no-arrows", action="store_true", help="no flow-arrow labels (as Team / Team v3)")
    ap.add_argument("--tile", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=500)
    args = ap.parse_args()
    for path in sorted((ROOT / "subprojects").glob("*/*.drawio")):
        if path.stem.strip() in [k.strip() for k in args.skip]:
            continue
        subset = f"{args.subset} test" if path.stem.strip() in [t.strip() for t in args.test] else args.subset
        print(process(path, Path(args.out), subset, args.tile, args.stride, not args.no_arrows), flush=True)


if __name__ == "__main__":
    main()
