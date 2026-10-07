"""End-to-end smoke test across all 7 team drawio files: parse -> build -> serialize
(with the piping/instrumentation patch) -> reload -> validate-by-counting.
"""
from pathlib import Path

import pytest

from pid_dexpi_aas.dexpi.build_model import build_model
from pid_dexpi_aas.drawio.parser import parse_drawio_file
from pid_dexpi_aas.dexpi.serialize import save_and_verify
from pid_dexpi_aas.dexpi.validate_by_counting import report_for_diagram

DECODED_DIR = Path(__file__).parent / "fixtures" / "decoded_xml"
ALL_FILES = sorted(DECODED_DIR.glob("*.xml"))


@pytest.mark.parametrize("xml_path", ALL_FILES, ids=lambda p: p.name)
def test_full_pipeline_on_each_team_file(xml_path, tmp_path):
    diagram = parse_drawio_file(xml_path)
    result = build_model(diagram)

    report = report_for_diagram(diagram, result)
    assert report.fully_accounted_for

    reloaded = save_and_verify(result.model, tmp_path, xml_path.stem)
    assert len(reloaded.conceptualModel.taggedPlantItems) == len(
        result.model.conceptualModel.taggedPlantItems
    )
