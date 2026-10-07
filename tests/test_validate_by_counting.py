from pathlib import Path

from pid_dexpi_aas.dexpi.validate_by_counting import report_for_directory, report_for_file

DECODED_DIR = Path(__file__).parent / "fixtures" / "decoded_xml"
FIXTURE = DECODED_DIR / "PFD combined set up 301122__diagram0.xml"


def test_single_file_report_is_fully_accounted_for():
    report = report_for_file(FIXTURE)
    assert report.fully_accounted_for
    assert report.built_count + report.skipped_count == report.total_elements


def test_all_seven_team_files_are_fully_accounted_for():
    reports = report_for_directory(DECODED_DIR)
    assert len(reports) == 7
    for report in reports:
        assert report.fully_accounted_for, f"{report.source_file} lost/gained elements"


def test_stream_names_no_longer_pile_onto_a_tank():
    report = report_for_file(FIXTURE)
    # The PFD tank used to collect its tag plus the stream names "Product" and "Gasses"
    # (tag_association.py's documented limitation); stream names a line ends on are now
    # off-page connectors, so no item carries several tags here.
    assert report.multi_tag_shapes == []
