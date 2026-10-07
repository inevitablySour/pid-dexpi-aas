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
    # Built EDGE connections broken down by how trustworthy their resolution is - see
    # build_model.BuildRecord.confidence and parser_drawio.Edge's *_resolution fields.
    # "explicit": the drawio file states the connection directly - ground truth.
    # "exit_fraction": a tight (<=10 unit) geometric match - high confidence but inferred.
    # "nearest_shape": a looser proximity guess - confirmed (by manual cross-checking
    #   against source P&ID images) capable of producing wrong connections; review before
    #   trusting.
    # "mixed": one endpoint explicit, the other inferred.
    connections_by_confidence: dict[str, int]

    @property
    def total_elements(self) -> int:
        return self.shape_count + self.edge_count

    @property
    def fully_accounted_for(self) -> bool:
        return self.built_count + self.skipped_count == self.total_elements

    @property
    def inferred_connection_count(self) -> int:
        """Built connections that are NOT ground truth from the file - these should be
        spot-checked against the original drawing before the DEXPI output is trusted."""
        return sum(
            count
            for confidence, count in self.connections_by_confidence.items()
            if confidence != "explicit"
        )


def report_for_diagram(diagram: ParsedDiagram, result: BuildResult) -> FileReport:
    tag_lookup = build_tag_lookup(diagram)
    multi_tag_shapes = [
        shape_id for shape_id, tags in tag_lookup.items() if len(tags) > 1
    ]

    built = [r for r in result.records if r.status == "built"]
    skipped = [r for r in result.records if r.status == "skipped"]

    connections_by_confidence: dict[str, int] = {}
    for r in built:
        if r.element_kind == "edge":
            connections_by_confidence[r.confidence] = (
                connections_by_confidence.get(r.confidence, 0) + 1
            )

    return FileReport(
        source_file=diagram.source_file,
        shape_count=len(diagram.shapes),
        edge_count=len(diagram.edges),
        built_count=len(built),
        skipped_count=len(skipped),
        multi_tag_shapes=multi_tag_shapes,
        skipped_details=[f"{r.element_id} ({r.element_kind}): {r.reason}" for r in skipped],
        connections_by_confidence=connections_by_confidence,
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
    if report.connections_by_confidence:
        explicit = report.connections_by_confidence.get("explicit", 0)
        lines.append(
            f"connections: {explicit} explicit (from the file) / "
            f"{report.inferred_connection_count} inferred (please spot-check against "
            f"the original drawing) - breakdown: {report.connections_by_confidence}"
        )
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
