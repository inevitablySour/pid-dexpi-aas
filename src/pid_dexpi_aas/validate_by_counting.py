"""'Validate by counting': the week 5-6 methodology deliverable (self-defined, no
published method exists for this - see HANDOFF.md's 'Decided approach').

For every input vertex/edge, report whether it was mapped (and to what DEXPI class) or
skipped (and why). Nothing is silently dropped - build_model.py's BuildRecord already
tracks this per element; this module turns it into a report and asserts full coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pid_dexpi_aas.build_model import BuildResult, build_model
from pid_dexpi_aas.parser_drawio import ParsedDiagram, parse_drawio_file
from pid_dexpi_aas.tag_association import build_tag_lookup


@dataclass
class FileReport:
    source_file: str
    shape_count: int
    edge_count: int
    built_count: int
    skipped_count: int
    multi_tag_shapes: list[str]  # shape ids with >1 associated tag - needs manual review
    skipped_details: list[str]  # "<element_id> (<kind>): <reason>"

    @property
    def total_elements(self) -> int:
        return self.shape_count + self.edge_count

    @property
    def fully_accounted_for(self) -> bool:
        return self.built_count + self.skipped_count == self.total_elements


def report_for_diagram(diagram: ParsedDiagram, result: BuildResult) -> FileReport:
    tag_lookup = build_tag_lookup(diagram)
    multi_tag_shapes = [
        shape_id for shape_id, tags in tag_lookup.items() if len(tags) > 1
    ]

    built = [r for r in result.records if r.status == "built"]
    skipped = [r for r in result.records if r.status == "skipped"]

    return FileReport(
        source_file=diagram.source_file,
        shape_count=len(diagram.shapes),
        edge_count=len(diagram.edges),
        built_count=len(built),
        skipped_count=len(skipped),
        multi_tag_shapes=multi_tag_shapes,
        skipped_details=[f"{r.element_id} ({r.element_kind}): {r.reason}" for r in skipped],
    )


def report_for_file(path: str | Path) -> FileReport:
    diagram = parse_drawio_file(path)
    result = build_model(diagram)
    return report_for_diagram(diagram, result)


def format_report(report: FileReport) -> str:
    lines = [
        f"=== {report.source_file} ===",
        f"shapes: {report.shape_count}, edges: {report.edge_count}, "
        f"total: {report.total_elements}",
        f"built: {report.built_count}, skipped: {report.skipped_count}",
        f"fully accounted for: {report.fully_accounted_for}",
    ]
    if report.multi_tag_shapes:
        lines.append(
            f"shapes with multiple associated tags (needs manual review): "
            f"{len(report.multi_tag_shapes)}"
        )
    if report.skipped_details:
        lines.append("skipped elements:")
        lines.extend(f"  - {detail}" for detail in report.skipped_details)
    return "\n".join(lines)


def report_for_directory(directory: str | Path) -> list[FileReport]:
    directory = Path(directory)
    return [report_for_file(path) for path in sorted(directory.glob("*.xml"))]


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("handoff/decoded_xml")
    reports = report_for_directory(target) if target.is_dir() else [report_for_file(target)]
    for report in reports:
        print(format_report(report))
        print()
