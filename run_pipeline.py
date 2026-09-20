"""Run the full P&ID -> DEXPI pipeline on one .drawio file.

Usage:
    .venv/bin/python run_pipeline.py "path/to/your.drawio" output_dir/

Decodes the drawio file (if needed), parses it, builds a DexpiModel, prints the
validate-by-counting report, and writes Proteus XML to <output_dir>/<name>.xml.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pid_dexpi_aas.build_model import build_model
from pid_dexpi_aas.decode_drawio import decode_drawio
from pid_dexpi_aas.parser_drawio import parse_drawio_xml
from pid_dexpi_aas.serialize import save_and_verify
from pid_dexpi_aas.validate_by_counting import format_report, report_for_diagram


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    drawio_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_variants = decode_drawio(drawio_path)
    xml_text = xml_variants[0]  # first <diagram> in the file

    diagram = parse_drawio_xml(xml_text, source_file=drawio_path.name)
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


if __name__ == "__main__":
    main()
