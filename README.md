# P&ID -> DEXPI -> AAS pipeline

INCEPTION/CHILL research minor. Converts the team's plant drawings (draw.io P&ID files,
plus MEM4CHEM's PowerPoint P&ID) into DEXPI models via pyDEXPI, for later packaging into
an AAS Type 1 submodel (IDTA 02012-1-0).

See `handoff/HANDOFF.md` for full project context, decisions, and literature.

## Setting up on another machine

```bash
git clone https://github.com/inevitablySour/pid-dexpi-aas.git && cd pid-dexpi-aas
./launch_app.sh              # creates .venv, installs, opens the app (draw.io mode works now)
./scripts/get_drawio.sh      # Linux: draw.io AppImage for line routes, tees and the flow map
                             # (macOS/Windows: install draw.io normally - found automatically)
```

For image and hybrid (draw.io + image) mode, also:

```bash
./ml/pid2graph/setup.sh      # ML venv; without an NVIDIA GPU it installs a CPU-only build
gh auth login && ./ml/get_model.sh   # trained model (GitHub release asset) -> ml/models/
```

Without draw.io the parser still works, but lines that end on other lines (tees) are not
joined and there is no flow map.

GPU: `setup.sh` picks the build from what it finds. An NVIDIA GPU with its normal driver
(`nvidia-smi` works) gets GPU PyTorch, and the app's "Device" defaults to `cuda` - no
CUDA toolkit needed to run the model. The CUDA toolkit (`nvcc`) is only needed to train.
No NVIDIA GPU (Apple Silicon, AMD, Intel graphics): CPU, a few minutes per sheet.
`setup.sh` is a bash script: on Windows run it in WSL2.

## Review app: draw.io, image, or both -> reviewed DEXPI

```bash
.venv/bin/streamlit run app.py
```

Upload a draw.io file, a P&ID image, or both (sidebar):

| input | where symbols/connections come from |
|---|---|
| draw.io | the file (same DEXPI output as the plain pipeline - tested on all 7 team files) |
| image | trained model (`ml/`) + EasyOCR tags + tag-prefix classifier |
| both | draw.io is the truth; the image checks it (`image check` column) and adds model-only detections as unticked rows |

Every symbol is a row showing its tag, DEXPI class, and **how it was identified** (draw.io
shape, tag prefix, or model class). Edit tag/class/include, add missed symbols (**＋ Add item** on the map, then drag a
box around the symbol; text the OCR read inside it becomes its tag) or connections (drawn
with right angles around the other items, like a draw.io connector), then **Build & validate DEXPI** (export + re-load round trip) and download
the XML. Reviews can be saved as JSON and resumed.

- Image mode needs the ML env (`ml/pid2graph/setup.sh`) and a checkpoint; set
  `PID_MODEL_CHECKPOINT` to override the default path. ~15 s per sheet on GPU.
- Tag prefixes -> classes live in `mapping.py` (`VALVE_TAG_PREFIX_MAP`,
  `EQUIPMENT_TAG_PREFIX_MAP`, ISA letters). Team-specific guesses (GV, XV, X, H, ...) are
  marked "confirm with team".
- Connections from the image model are currently unreliable (edge-label bug in
  training, see `ml/README.md`) - the app warns about this in image-only mode.

Code: `src/pid_dexpi_aas/review.py` (review rows, cross-check, -> DEXPI),
`ocr_tags.py` (OCR tokens -> tags -> symbols), `ml/pid2graph/infer_sheet.py` (model + OCR,
runs in the ML venv).

## Sharing the app from this PC (free)

```bash
./scripts/serve_public.sh            # prints a public https link and a password
APP_PASSWORD='...' ./scripts/serve_public.sh
```

A free Cloudflare quick tunnel (no account; `cloudflared` is downloaded to `tools/` on
first use) gives the app a public link while the script runs; the app itself listens on
127.0.0.1 only. Testers need the link and the password. Model settings are hidden, one
image is processed at a time, and image mode uses the GPU unless another process (a
training run) is computing on it - then the CPU, a few minutes per sheet. The link
changes on every start and stops when the PC sleeps or the script ends.

## DEXPI -> AAS (IDTA 02012-1-0)

The review app's build step also gives an `.aasx`: an Asset Administration Shell (Type 1)
with the IDTA 02012-1-0 Submodel "Information Model for P&I Diagrams based on DEXPI":
`PlantMetadata`, and per drawing a `Model01` holding `ModelMetadata`, the DEXPI XML
(`ModelFile`), the drawing as SVG (`ModelRepresentation`) and a `MappingDirectory` - every
tagged item (equipment, valves, instruments; nozzles as sub tags) with its DEXPI local id
and a relationship to a stable global id `<prefix>:asset:<tag>` for the digital twins.
Every package is checked with the IDTA's aas-test-engines.

Several drawings of one plant in one AAS (Model01, Model02, ...):

```bash
.venv/bin/python -m pid_dexpi_aas.aas_export subprojects/Lumen/*.drawio -o lumen.aasx \
    --process-plant-name "Lumen gramflow setups" --project-name "INCEPTION CHILL"
```

AAS metamodel 3.0 (default) or 3.1 (`--aas-version 3.1`, or the app's AAS details). 3.0 is
what IDTA 02012-1-0 was published for and is checked with the IDTA's aas-test-engines plus
aas-core3.0; 3.1 is the same content in the 3.1 schema, verified with aas-core3.1 (the IDTA
test engine does not read 3.1 yet). Built with basyx-python-sdk 2.0.x (2.1+ writes only 3.1).

## Known limitation: pyDEXPI 1.2.0's Proteus exporter is equipment-only

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
suite (`tests/test_serialize_patch.py` once written) and re-checking whether upstream has
shipped real piping/instrumentation export** - if they have, delete our patch and switch
back to the stock `ProteusSerializer`.

Location of the patch: `src/pid_dexpi_aas/proteus_patch.py`.

## P&ID re-render (SVG)

`run_pipeline.py` also writes a `<name>.svg` re-rendering the built model back into a
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
