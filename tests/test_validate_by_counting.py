from pathlib import Path

from pid_dexpi_aas.validate_by_counting import report_for_directory, report_for_file

DECODED_DIR = Path(__file__).parent.parent / "handoff" / "decoded_xml"
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


def test_report_flags_multi_tag_shapes_for_manual_review():
    report = report_for_file(FIXTURE)
    # The known tank-with-three-tags case from the PFD smoke fixture (see
    # tag_association.py's documented limitation).
    assert len(report.multi_tag_shapes) >= 1
