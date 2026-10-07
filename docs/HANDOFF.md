# P&ID to DEXPI to AAS: handoff to local Claude Code CLI

Written 2026-09-20 by a cloud Claude session for Joel Kumi, INCEPTION/CHILL research minor.
Drop this whole folder into a local git repo and open it with Claude Code CLI. Everything
below is what the cloud session had confirmed working before the handoff; nothing here is
speculative except the "Next steps" section.

## Project context

Three-person team, INCEPTION/CHILL research minor: Nora Nagy-Bozor (Distillation), Dennis
Dzoba (Extraction), Joel Kumi (Infrastructure). Goal: convert the team's plant drawings into
DEXPI models, then package into an Asset Administration Shell (AAS) so the extraction and
distillation digital twins can consume the data directly instead of each parsing a P&ID
themselves.

### Decided approach

- Source format is Draw.io / mxGraph XML, not CAD, not scanned images. The shapes and
  connectivity are already structured data, so there is no image-recognition step needed
  for the team's own drawings.
- Pipeline: parse the mxGraph XML directly, map the shape vocabulary onto DEXPI equipment
  classes, build/serialise the model with pyDEXPI (Goldstein et al. 2025) rather than
  reimplementing DEXPI handling from scratch.
- Validation method (self-defined): "validate by counting", every shape mapped, every
  connector accounted for, with a discrepancy list for anything that does not map.
- AAS target: Type 1 (static, standards-compliant container per IDTA 02012-1-0), not a
  live/reactive twin.
- CE620 (liquid-liquid extraction) is conventional equipment and maps cleanly. CE630
  (rotating multi-chamber extractor) is the stretch case: open question is whether DEXPI
  1.4's CustomEquipment / Custom-prefixed fallback classes are detailed enough, or whether
  a new subclass should be proposed.
- A separate, exploratory task (week 6-7) will test symbol/topology image recognition on
  the team's own drawings, to scope what generalising to non-standard drawings would need
  later. This does NOT replace the XML pipeline; it is a feasibility side-quest.

### Timeline

1. Week 4-5 (now): build the pipeline, working P&ID to DEXPI conversion on the team's own
   drawings.
2. Week 5-6: validate by counting.
3. Week 6-7: symbol/topology recognition feasibility test (exploratory, separate track).
4. Week 7-8: package into an AAS Submodel per IDTA 02012-1-0.
5. Week 8-9: write up the Submodel-to-twins integration.

### Key literature

- Fillinger et al. 2017: why a neutral exchange format (DEXPI) exists at all.
- Cameron et al. 2024 (DEXPI Process), Tolksdorf et al. 2025 (DEXPI 2.0): current state of
  the standard, beyond just P&ID.
- Kim et al. 2022: best published image-to-DEXPI pipeline, 96%+ precision, but on ~75k
  annotated symbols of conventional CAD-style drawings, ~2.4 hr/diagram. This is why we are
  not doing image recognition for our own drawings.
- Ma et al. 2026: names heterogeneous drafting styles as a bottleneck to generalisation.
  This is the literature version of the real problem with the LightUp reactor drawing style.
- Zhang et al. 2025: the three AAS types, used to justify targeting Type 1.
- Goldstein et al. 2025: pyDEXPI itself.

Full condensed literature review is in the project doc
`claude/pid-dexpi-aas-literature-condensed.md` (claude.ai Project "CHILL Minor Project") if
you need it locally, ask your local Claude to fetch/recreate it from what's here, since the
CLI cannot open the claude.ai Project directly.

## What is actually done so far (confirmed working)

1. `pip install pydexpi` works cleanly (pulls in pydexpi 1.1.0, plus pyflowsheet, sfiles2,
   plotly, etc as deps). Relevant modules:
   - `pydexpi.dexpi_classes.equipment` — 159 equipment classes (see `equipment_classes.txt`)
   - `pydexpi.dexpi_classes.piping` — piping components including valve classes (see
     `piping_classes.txt`)
   - `pydexpi.dexpi_classes.instrumentation` — instrumentation classes (see
     `instrumentation_classes.txt`)
   - `pydexpi.loaders.proteus_serializer.ProteusSerializer` — can (likely) write real
     DEXPI/Proteus XML output directly. NOT YET INSPECTED IN DETAIL, this is the next thing
     to check (`ProteusSerializer.__init__` signature, its write method).
   - `pydexpi.loaders.json_serializer`, `pydexpi.loaders.graph_loader`,
     `pydexpi.loaders.ml_graph_loader` also exist, not yet inspected.
   - `pydexpi.toolkits.model_toolkit`, `pydexpi.toolkits.piping_toolkit`,
     `pydexpi.toolkits.instrumentation_toolkit`, `pydexpi.toolkits.base_model_utils` exist,
     not yet inspected. These are almost certainly the helper functions for building a
     DexpiModel programmatically (adding equipment, connecting nozzles, etc) rather than
     hand-building pydantic objects. Read these before writing the mapper.
   - Top-level model class: `pydexpi.dexpi_classes.dexpiModel` re-exports `ConceptualModel`
     and `DexpiModel` from `pydantic_classes`.

2. Got real source material from Joel: 8 `.drawio` files across three sub-projects
   (`MEM4CHEM`, `Lumen`, `LightUp`), all in `drawio_originals/`. `LightUp` is the one named
   in the literature doc as "the LightUp reactor P&ID" (HMI screenshot style, bespoke lab
   modules) so it is the good stress-test case, not just the easy case.

3. Decoded all 8 `.drawio` files into plain mxGraph XML (drawio stores the diagram as
   base64(deflate(urlencode(xml))) inside `<diagram>...</diagram>`). Decoder is
   `decode.py` in this folder, already tested working on all 8 files. Usage:
   `python3 decode.py "<file>.drawio" <output_dir>`. Decoded output for all 8 files is
   already in `decoded_xml/` so you do not have to re-run it, though you should re-run it
   if Joel sends newer drawio files.

4. **Key finding**: all 8 files use the *official draw.io P&ID shape library*
   (`mxgraph.pid.*`, `mxgraph.pid2valves.*`, `mxgraph.pid2inst.*`, `mxgraph.pid2misc.*`),
   not freeform/custom shapes. This is a finite, well-known vocabulary, much better starting
   point than expected. Style strings look like:
   `style="shape=mxgraph.pid2valves.valve;..."`

   Observed shape vocabulary and rough frequency (across the 8 files, will vary per file,
   full counts were printed during analysis, not saved to a file, rerun `analyze.py` logic
   below if you want exact per-file counts):
   - `mxgraph.pid2valves.valve` (most common single equipment shape, ~12-33 per file)
   - `mxgraph.pid2inst.discInst` (discrete instrument bubbles, ~15-20 per file, these carry
     tag text like "PC 202 H2", "PI 201 N2" inside an HTML `<table>`/`<font>` blob in the
     `value` attribute — needs HTML stripping to get the instrument tag)
   - `mxgraph.pid.vessels.container,_tank,_cistern`
   - `mxgraph.pid.vessels.gas_bottle`
   - `mxgraph.pid.vessels.pressurized_vessel`
   - `mxgraph.pid.vessels.jacketed_mixing_vessel`
   - `mxgraph.pid.vessels.reactor`
   - `mxgraph.pid.heat_exchangers.shell_and_tube_heat_exchanger_2`
   - `mxgraph.pid.pumps.peristaltic`
   - `mxgraph.pid.pumps_-_iso.pump_(liquid)`
   - `mxgraph.pid2misc.column`
   - `mxgraph.electrical.opto_electronics.lamp` (LED indicator, not process equipment)
   - `offPageConnector`, `trapezoid`, `cylinder3`, `mxgraph.arrows2.arrow` (misc/generic)

   IMPORTANT: roughly half of all vertices in every file have NO `shape=` token at all
   (plain rectangles). These are almost all text labels/tags (e.g. "V101", "T202", "PSV201",
   "150 bar", "safe location", "Fume hood [atm]"), not equipment shapes themselves. They sit
   near, but are NOT the same mxCell as, the actual equipment shape they label. Any mapper
   has to treat these as tag/annotation text and associate them to the nearest real
   equipment shape (by geometry proximity, or by following edges/parent-child mxCell
   relationships), not as equipment in their own right.

5. Did NOT yet write the actual parser/mapper/serializer code. This is the actual next
   task, described below.

## Next steps (not yet started, do this)

1. **Inspect the pyDEXPI API properly** before writing anything:
   ```python
   from pydexpi.loaders.proteus_serializer import ProteusSerializer
   from pydexpi.toolkits import model_toolkit, piping_toolkit, instrumentation_toolkit, base_model_utils
   # inspect signatures, docstrings, example usage. Check for a GitHub repo/README/examples
   # for pyDEXPI (Goldstein et al. 2025) — there is likely an examples folder upstream.
   ```
   Also check pypi/GitHub for pyDEXPI's own example scripts, this saves guessing the model
   construction API (equipment ports/nozzles, piping network segments, how a DexpiModel is
   assembled and validated before serialisation).

2. **Write the mxGraph parser** (`parser.py`): given a decoded XML string, extract every
   `mxCell` with `vertex="1"` (shapes) and `edge="1"` (connectors/piping), capturing: id,
   style string (parse out `shape=` token), value (HTML-stripped text), geometry (x, y,
   width, height), and for edges: source/target cell ids.

3. **Write the shape-to-DEXPI mapping table** (`mapping.py`): a dict from mxgraph shape
   token (e.g. `mxgraph.pid.vessels.reactor`) to a pyDEXPI equipment/piping/instrumentation
   class. Use `equipment_classes.txt`, `piping_classes.txt`, `instrumentation_classes.txt`
   in this folder as the candidate list. Suggested first-pass mapping (verify against
   pyDEXPI class fields before trusting):
   - `mxgraph.pid.vessels.reactor` -> a Vessel/PressureVessel subtype, check if DEXPI has a
     dedicated Reactor class (it does NOT appear in the 159-class dump, likely needs
     `CustomVessel` or plain `Vessel` with a name/tag noting it is a reactor).
   - `mxgraph.pid.vessels.container,_tank,_cistern` -> `Tank`
   - `mxgraph.pid.vessels.pressurized_vessel` -> `PressureVessel`
   - `mxgraph.pid.vessels.gas_bottle` -> likely `PressureVessel` or `CustomVessel`, gas
     bottle has no exact DEXPI equivalent, flag as a discrepancy.
   - `mxgraph.pid.vessels.jacketed_mixing_vessel` -> `Vessel` + `Agitator`/`Mixer`
     composition, check pyDEXPI's actual composition pattern for jacketed/agitated vessels.
   - `mxgraph.pid.heat_exchangers.shell_and_tube_heat_exchanger_2` -> `TubularHeatExchanger`
   - `mxgraph.pid.pumps.peristaltic` -> no exact DEXPI class, likely `CustomPump`
   - `mxgraph.pid.pumps_-_iso.pump_(liquid)` -> generic `Pump`
   - `mxgraph.pid2misc.column` -> `ProcessColumn`
   - `mxgraph.pid2valves.valve` -> generic valve, but the *value*/sub-style may encode the
     specific valve type (gate, ball, check, safety/relief -- PSV-tagged ones in the labels
     are clearly `SafetyValveOrFitting`/`SpringLoadedGlobeSafetyValve`, GV-tagged ones look
     like gate valves, CV-tagged ones look like check valves). Use the adjacent tag text
     (V101, GV201, PSV201, CV201, XV101, FV201) as the actual signal for valve subtype, the
     shape token alone is not enough. Build a tag-prefix convention table:
     - `PSV` -> SafetyValveOrFitting / spring-loaded safety valve
     - `GV` -> GateValve (needs confirming with Joel/team, this is inferred from usage not
       from a legend, check if there is a legend/title block in the drawings)
     - `CV` -> CheckValve
     - `XV` -> likely a shutoff/isolation valve, or an actuated on/off valve, needs
       confirming with team
     - `FV` -> flow control valve
     - `V` alone -> generic manual valve, StraightwayValve or GateValve as default
   - `mxgraph.pid2inst.discInst` -> `ProcessInstrumentationFunction`/discrete instrument,
     tag prefix (PC, PI, TC, TI, FI, etc, standard ISA loop-tag letters: first letter is
     measured variable, second is function e.g. C=control, I=indicate) drives the specific
     instrumentation class. This needs its own small ISA-5.1 tag parser.
   - `mxgraph.electrical.opto_electronics.lamp`, generic arrows, trapezoid, cylinder3,
     offPageConnector -> likely NOT process equipment, decide per-file whether to skip or
     map to a signal/off-page connector class (`SignalOffPageConnector` exists in
     `instrumentation_classes.txt` and is probably the right target for
     `offPageConnector` cells specifically).

4. **Write the tag-association step**: for every no-shape/plain-rect vertex with text, find
   the nearest real equipment/valve/instrument vertex (by bounding-box distance, or by
   checking if an edge already connects them, or by mxCell `parent` grouping if the drawio
   file groups a shape with its label as parent/child) and attach the text as that
   equipment's tag/name. Do this before building the DEXPI model, not after, so tags end up
   on the right DEXPI object's `tagName`/`name` field.

5. **Build the DEXPI model**: use `model_toolkit`/`piping_toolkit` (once inspected per step
   1) to construct a `DexpiModel`, adding one equipment/valve/instrument object per mapped
   vertex, and one piping/signal connection per mapped edge (edge source/target cell ids map
   back to the DEXPI objects created from those vertices).

6. **Serialise with `ProteusSerializer`** to real DEXPI/Proteus XML output. Confirm output
   validates against the DEXPI schema if pyDEXPI exposes a validator, or at minimum confirm
   it opens without error in any DEXPI-aware viewer if the team has access to one.

7. **Validate by counting**: write a report that lists, for every input vertex/edge, whether
   it was mapped, to what DEXPI class, or why it was skipped/unmapped (discrepancy list).
   This is the actual "validation method" deliverable for week 5-6, do not skip it, it is
   explicitly called out as self-defined methodology since no published method exists.

8. Test against ALL 8 files, not just one, since they vary in complexity (smallest is
   `PFD combined set up 301122.drawio`, 16 vertices/13 edges, good smoke test; largest is
   `P&ID Lumen combined set-up (1) (1) (1).drawio`, 147 vertices/127 edges).

## Files in this folder

- `HANDOFF.md` — this file
- `decode.py` — working drawio decoder (base64/deflate/urldecode), tested on all 8 files
- `drawio_originals/` — all 8 original `.drawio` files plus their PDF/PNG exports (exports
  are for reference/visual comparison only, do not parse these, parse the `.drawio`/`.xml`)
- `decoded_xml/` — pre-decoded plain mxGraph XML for all 8 files, ready to parse
- `equipment_classes.txt`, `piping_classes.txt`, `instrumentation_classes.txt` — full class
  lists pulled from the installed pyDEXPI 1.1.0 package, use these as the mapping target
  vocabulary

## Environment note

`pip install pydexpi` (with `--break-system-packages` if pip complains about an externally
managed environment) is all that was needed to get pyDEXPI 1.1.0 working. No other setup
was required.

## Addendum, 2026-09-20 (local build session)

Local build (parser, mapping, tag-association, model construction) completed against
pyDEXPI **1.2.0** (newer than the 1.1.0 above; pinned in `pyproject.toml`). Full API
inspection of `model_toolkit`/`piping_toolkit`/`instrumentation_toolkit`/`ProteusSerializer`
resolved all "NOT YET INSPECTED" items above - see `src/pid_dexpi_aas/build_model.py` for
the confirmed working construction pattern.

**New blocking finding**: pyDEXPI 1.2.0's Proteus *exporter* only supports
MetaData/Equipment/Nozzle - piping and instrumentation are silently dropped on
`ProteusSerializer().save()`. Confirmed against GitHub `master`, not just the PyPI release;
no open issue, framed as future work in their README. Full details and the chosen
workaround (a local exporter patch, see `src/pid_dexpi_aas/proteus_patch.py`) are in the
top-level `README.md`. **Do not treat `pydexpi`'s Proteus export as trustworthy for
piping/instrumentation without checking that file first.**

Also confirmed this session: **MEM4CHEM has no `.drawio` file** - it's a `P&ID.pptx` with
native structured shapes/text/lines but purely image-based equipment symbols (no semantic
naming), so it needs the week 6-7 image-recognition side-quest to get equipment
*classification* at all, not just as a stretch-goal exploration. See
`subprojects/MEM4CHEM/`.
