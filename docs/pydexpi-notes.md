# pyDEXPI 1.2.0: gaps and bugs, and how this project works around them

Background for `src/pid_dexpi_aas/dexpi/proteus_patch.py` and `src/pid_dexpi_aas/render/` - read before changing the pinned pydexpi version.

Also patched since these notes were first written: the exporter now writes instrument wiring (measuring and signal lines, sensors, actuators), and the reader accepts choice fields written by value (pyDEXPI's writer and reader disagreed, and a choice field made an object lose all its attributes on reload). pyDEXPI's reader still ignores custom attributes; they are written to the file correctly.

## Proteus export is equipment-only: pyDEXPI 1.2.0's Proteus exporter is equipment-only

**Confirmed by direct testing and by reading `process-intelligence-research/pyDEXPI` on
GitHub (`master` branch, same as the installed release - not fixed upstream yet):**
`ProteusExporter`/`PlantModelExporter`
(`pydexpi/loaders/proteus_serializer/proteus_exporter/exporter_modules.py`) only
implements exporter classes for `MetaData`, `Equipment`, and `Nozzle`. There is no
exporter for `PipingNetworkSystem` or `ProcessInstrumentationFunction` - calling
`ProteusSerializer().save()` on a model with piping/instrumentation data silently drops
it; equipment survives, piping and instrumentation do not. The Proteus **parser** (reading
XML in) does support these classes, so this is a one-way gap in the writer only.

The pyDEXPI README frames full Proteus export (including piping) as anticipated future
work, not a documented bug - there's no GitHub issue open for it as of 2026-09-20. `1.2.0`
is the latest PyPI release (`1.1.0`, `1.0.0` are the only older versions); no newer release
fixes this.

**Decision (2026-09-20):** rather than block the pipeline on this, or fall back to
pyDEXPI's `JsonSerializer` (which is NOT a DEXPI-standard format - it's pyDEXPI's own
pydantic-dump format, unreadable by any other DEXPI tool, and therefore not a real
substitute when the actual requirement is a DEXPI model), we are writing a **local patch**:
new exporter classes for `PipingNetworkSystem`/`ProcessInstrumentationFunction`, following
the exact pattern of the existing `EquipmentExporter`/`NozzleExporter`, producing Proteus
XML shaped to round-trip through pyDEXPI's own Proteus **parser** (validated via
build -> export -> reload -> diff, not just "it writes a file").

**pydexpi is pinned to `==1.2.0`** (see `pyproject.toml`) specifically because this patch
reaches into `exporter_modules.py` internals (`ModuleContext`, `register_submodule`,
`ExporterModule.compositional_pass`) that are not a documented public API. If the pyDEXPI
authors implement real piping/instrumentation export themselves (likely, per their own
README), a version bump could silently stop our patch from being called, or break it
loudly. **Do not bump the pydexpi version without re-running the full round-trip test
suite (`tests/test_proteus_patch.py`) and re-checking whether upstream has
shipped real piping/instrumentation export** - if they have, delete our patch and switch
back to the stock `ProteusSerializer`.

Location of the patch: `src/pid_dexpi_aas/dexpi/proteus_patch.py`.

## The re-drawn DEXPI picture (SVG)

`scripts/run_pipeline.py` also writes a `<name>.svg` re-rendering the built model back into a
picture, for visual "does this match my drawing" verification alongside the raw XML.

**pyDEXPI ships no symbol geometry at all** - confirmed by direct inspection:
`svg_loader.py`'s renderer only dispatches on primitive type (polygon/polyline/ellipse/
text/etc), never on DEXPI class name, and no shape-catalogue data ships in the package or
exists in any sample/test data. `symbol_library.py` is a small, independently hand-drawn
ISO-10628/ISA-5.1-style symbol set (bowtie valves, circular pumps/instruments, capsule
vessels, a pentagon off-page connector, etc) covering every DEXPI class `mapping.py` can
produce - not copied from any source, so no licensing concerns.

**Also confirmed and worked around: a pyDEXPI 1.2.0 viewBox bug.**
`DrawDiagram.draw_svg()` resets its bounds tracking to `±inf` on every call and rebuilds
the SVG `viewBox` from each primitive's raw *local* coordinates (our symbol library's
normalized 0-100 box) - never accounting for the `<g transform="translate(...)
scale(...)">` that places each symbol on the diagram. In practice this collapsed the
viewBox to roughly 0-100 regardless of the actual diagram size. `render_svg.py` works
around this by rendering normally via `DrawDiagram`, then overwriting the root SVG's
width/height/viewBox using the correct global bounds `build_diagram.py` already computed
from the original drawio coordinates.

**Also confirmed: `DexpiModel.diagram` is a math-convention (Y increases upward)
coordinate space**, opposite to drawio/mxGraph's screen convention (Y increases
downward) - `build_diagram.py` negates Y throughout; get this wrong and everything
renders with large negative coordinates and an empty-looking canvas.

**Also confirmed and fixed: pyDEXPI 1.2.0's `convert_stroke`/`convert_polygon` never
render dashed lines or filled polygons.** `convert_stroke` checks `stroke.dashStyle.value
== "Dashed"`/`"Dotted"`, but the real `DashStyle` enum members are `Dash`/`Dot` - that
condition can never be true, so every stroke renders solid. `convert_polygon` computes
`fill` via `"none" if ... == "Solid" else "none"` - both branches return `"none"`, so a
filled polygon (needed for arrowhead triangles) always renders unfilled. Subclassing
`DrawDiagram` does NOT fix this: `_create_content_group` hands off to nested
`DrawRepresentationGroup`/`DrawShapeUsage` drawers that each construct their own plain
`SvgRenderer` internally, with no dependency-injection point - `svg_renderer_patch.py`
monkeypatches `SvgRenderer.convert_stroke`/`convert_polygon` directly (at class level,
applied once on import) so every renderer instance gets the fix regardless of which
drawer class constructs it.

**Also confirmed: pyDEXPI 1.2.0's Proteus round-trip is broken for off-page connectors,
in the library itself, not just our exporter.** `PipeOffPageConnector` (the class the
parser reconstructs from a `<PipeOffPageConnector>` XML tag) implements neither
`PipingSourceItem` nor `PipingTargetItem` - only its `FlowIn`/`FlowOutPipeOffPageConnector`
subclasses do, and the parser never produces those. `PipingNetworkSegmentParser.
reference_pass` unconditionally assigns any adjacent segment item (including a plain
`PipeOffPageConnector`) as a `PipingConnection`'s `sourceItem`/`targetItem`, which
pydantic then rejects. This means ANY Proteus file with an off-page connector adjacent to
an internal connection fails to fully reload in this pyDEXPI version - confirmed via
direct testing, independent of our exporter's correctness. Our exporter still emits a
schema-correct `<PipeOffPageConnector>` (used by `build_model.py`'s
`_get_or_build_off_page_connector`, for edges terminating at a text-only destination like
"Fume hood [atm]" or "safe location" - real, distinct connections that would otherwise be
silently dropped since nothing else builds a DEXPI object for a floating text label); the
rest of the model (all equipment, all other segments, instrumentation) survives reload
fully - only that one segment's off-page-connector-side `sourceItem`/`targetItem` comes
back `None` after a save-then-reload round-trip. See `tests/test_proteus_patch.py`'s
`test_known_parser_bugs_produce_only_expected_errors` for the exact confirmed error.

Coverage/known limits: a shape with multiple associated tags (see
`tag_association.py`'s documented limitation) shows all of them comma-joined on one
label, which is a visible cue that the shape needs manual review, not a rendering bug.
Piping connections route as right-angle ("elbow") lines between the two shapes' real,
computed bounding boxes (`build_diagram.py`'s `_route_orthogonal`/
`compute_local_bounding_box`), not the diagram's original routed paths.
