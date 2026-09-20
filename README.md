# P&ID -> DEXPI -> AAS pipeline

INCEPTION/CHILL research minor. Converts the team's plant drawings (draw.io P&ID files,
plus MEM4CHEM's PowerPoint P&ID) into DEXPI models via pyDEXPI, for later packaging into
an AAS Type 1 submodel (IDTA 02012-1-0).

See `handoff/HANDOFF.md` for full project context, decisions, and literature.

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

Location of the patch: `src/pid_dexpi_aas/proteus_patch.py` (once written).
