"""Run the full P&ID -> DEXPI pipeline on one .drawio file.

Usage:
    .venv/bin/python scripts/run_pipeline.py "path/to/your.drawio" output_dir/

Decodes the drawio file (if needed), parses it, builds a DexpiModel, prints the
validate-by-counting report, writes Proteus XML to <output_dir>/<name>.xml, and
re-renders the built model as a P&ID-style SVG at <output_dir>/<name>.svg for visual
verification against the original drawing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pid_dexpi_aas.render.build_diagram import build_diagram
from pid_dexpi_aas.dexpi.build_model import build_model
from pid_dexpi_aas.drawio.decode import decode_drawio
from pid_dexpi_aas.drawio.parser import parse_drawio_xml_resolved
from pid_dexpi_aas.render.render_svg import render_diagram_svg
from pid_dexpi_aas.dexpi.serialize import save_and_verify
from pid_dexpi_aas.dexpi.validate_by_counting import format_report, report_for_diagram


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    drawio_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_variants = decode_drawio(drawio_path)
    xml_text = xml_variants[0]  # first <diagram> in the file

    diagram = parse_drawio_xml_resolved(xml_text, source_file=drawio_path.name)
    result = build_model(diagram)

    report = report_for_diagram(diagram, result)
    print(format_report(report))
    print()

    output_name = drawio_path.stem
    reloaded = save_and_verify(result.model, output_dir, output_name)
    print(f"Wrote {output_dir / (output_name + '.xml')}")
    print(f"Round-trip verified: equipment={len(reloaded.conceptualModel.taggedPlantItems)}, "
          f"piping systems={len(reloaded.conceptualModel.pipingNetworkSystems)}, "
          f"instrumentation={len(reloaded.conceptualModel.processInstrumentationFunctions)}")

    diagram_obj = build_diagram(diagram, result, result.model)
    svg_path = render_diagram_svg(diagram_obj, output_dir / f"{output_name}.svg")
    print(f"Wrote {svg_path} (re-rendered P&ID for visual verification)")


if __name__ == "__main__":
    main()
