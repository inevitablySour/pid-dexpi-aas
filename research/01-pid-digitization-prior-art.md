# P&ID digitization: prior art and how others handle ambiguity

Research question this answered: *our pipeline resolves loosely-drawn connections using
proximity heuristics, which produce silent errors. Is that a flaw in our approach, or is
it inherent to the problem?*

**Finding: it is inherent.** Every real system found uses the same class of heuristic,
and none of them trust it without human review.

---

## Kim et al. 2022 — the reference pipeline

**Citation:** Kim, H., Lee, W., Kim, M., Moon, Y., Lee, T., Cho, M., Mun, D.
"End-to-end digitization of image format piping and instrumentation diagrams at an
industrially applicable level." *Journal of Computational Design and Engineering*,
9(4):1298-1326, 2022.
https://academic.oup.com/jcde/article/9/4/1298/6611631

**Status:** UNVERIFIED (abstract + secondary sources read; full text not fetched)

This is the paper cited in our own project literature review as the best published
image-to-DEXPI pipeline. Relevant detail: their connectivity step is a **distance
tolerance proximity heuristic** — the same family as ours. Reported approach: for each
line-symbol pair, an edge is added if the symbol's bounding box and the line's endpoints
are within a distance tolerance; ambiguous text is attached to the nearest symbol or line.

They add substantial **preprocessing** to make that heuristic work: splitting lines at
crossings, extending line endpoints by a small buffer "to provide a margin of error," and
removing spurious penetrated segments.

**Applicability:** high. This validates the general approach but also shows that the
known-good published method has the same failure mode we hit. They do not report a
confidence gate or self-loop guard — the two bugs we found in our own implementation may
well exist latently in theirs.

---

## Microsoft ISE — production engagement writeup

**Citation:** Microsoft ISE Developer Blog, "Engineering Document (P&ID) Digitization."
https://devblogs.microsoft.com/ise/engineering-document-pid-digitization/

**Status:** UNVERIFIED (search snippet + secondary summary)

The most operationally honest source found. A real production engagement using Azure
Document Intelligence and proximity/bounding-box matching with endpoint buffer extension.

Reported outcomes:
- ~80% recall on assets and connections
- **~10% mis-association error rate when symbols are crowded**
- Stated mitigation is *process, not algorithm*: "it is strongly recommended that all
  extracted assets and connections undergo human verification before using the generated
  assets graph."

**Applicability:** very high, and the single most useful data point for setting
expectations. A well-funded commercial pipeline with commercial OCR still lands at ~80%
and mandates human review. This directly supports our decision to surface inferred vs.
explicit connections rather than presenting all extractions with equal confidence.

---

## Schulze Balhorn et al. — rule-based autocorrection on DEXPI graphs

**Citation:** Schulze Balhorn, L., et al. "Rule-Based Autocorrection of Piping and
Instrumentation Diagrams (P&IDs) on Graphs." arXiv:2502.18493, 2025.
https://arxiv.org/abs/2502.18493

**Status:** UNVERIFIED (abstract/secondary; full text not fetched)

Built on pyDEXPI — same ecosystem we target. Rather than proximity guessing, they define
**33 rule graphs**: graph-pattern-matching rules (subgraph isomorphism, citing Cordella
et al. 2004's VF2 algorithm) that detect and correct *specific known structural defects*
in an already-parsed DEXPI graph.

Philosophy differs fundamentally from proximity heuristics: encode known-good/known-bad
topological patterns as domain rules, and only apply a correction when a pattern matches
unambiguously — no distance-based guessing.

**Applicability:** high, and arguably the best model for how to improve our pipeline.
Instead of tuning distance thresholds, encode P&ID grammar (e.g. "a valve has exactly 2
piping connections", "an edge whose resolved source and target are identical is suspect").
Domain grammar is a stronger confidence signal than geometry.

---

## pyDEXPI

**Citation:** Goldstein, D.P., Schulze Balhorn, L., Alimin, A.A., Schweidtmann, A.M.
"pyDEXPI: A Python framework for piping and instrumentation diagrams using the DEXPI
information model." Proceedings of ESCAPE35, 2025.
Repo: https://github.com/process-intelligence-research/pyDEXPI
Paper PDF: https://psecommunity.org/wp-content/plugins/wpor/includes/file/2506/LAPSE-2025.0371-1v1.pdf

**Status:** VERIFIED (source code of installed v1.2.0 read directly and extensively)

The library our pipeline builds on. Important caveats discovered by direct source
inspection, documented fully in the project `README.md`:

- Proteus **exporter** only implements MetaData/Equipment/Nozzle. Piping and
  instrumentation are silently dropped on `save()`. Worked around in `proteus_patch.py`.
- `svg_loader.py` dash-style check compares against `"Dashed"`/`"Dotted"` but the real
  enum values are `Dash`/`Dot` — dashed lines never render. Worked around in
  `svg_renderer_patch.py`.
- `convert_polygon`'s fill logic has both ternary branches returning `"none"` — filled
  polygons never fill.
- `PipeOffPageConnector` implements neither `PipingSourceItem` nor `PipingTargetItem`,
  but the Proteus parser assigns it as a `PipingConnection` endpoint — pydantic rejects
  this, so any Proteus file with an off-page connector adjacent to an internal connection
  fails to fully reload.

**Applicability:** direct dependency. These are real, reproducible defects in v1.2.0,
pinned in `pyproject.toml` for that reason.

---

## Handwritten circuit diagrams — closest analog to informal drawings

**Citation:** Bayer, J., et al. "Modular Graph Extraction for Handwritten Circuit Diagram
Images." arXiv:2402.11093, 2024. https://arxiv.org/html/2402.11093v1
Predecessor: "Instance Segmentation Based Graph Extraction for Handwritten Circuit
Diagrams." arXiv:2301.03155. https://arxiv.org/pdf/2301.03155

**Status:** UNVERIFIED (search snippet only)

Different domain, same underlying difficulty (informally hand-drawn, loosely connected).
Notable design choice: they avoid the floating-endpoint problem by changing *what is
detected* — connector endpoints and junction points are detected as first-class objects,
and edges are formed by segmentation-based wire tracing that only counts a connection
when a traced wire physically touches two detected regions. Ambiguity is resolved in the
*perception* stage where pixel-level touching is verifiable, rather than later from
abstract coordinates.

**Applicability:** the design principle transfers even though the method doesn't (our
input is already-structured XML, not raster): prefer whatever signal is closest to
ground truth over inferred proximity. In our case that means the XML's own
`source=`/`target=` attributes over any geometric inference.

---

## draw.io / mxGraph connection conventions

**Citation:** drawio-mcp XML reference.
https://github.com/jgraph/drawio-mcp/blob/main/shared/xml-reference.md
Related bug report thread: draw.io Google Group, "Lines are not hooked correctly to nodes."
https://groups.google.com/g/drawio/c/1MlY_rqns-8

**Status:** UNVERIFIED (search snippet), but independently CONFIRMED against our own files
by direct XML inspection.

There is **no** draw.io connectivity-repair, lint, or cleanup feature. What does exist is
a documented convention for what a properly-connected edge looks like:

- Properly snapped edge: `mxCell` has `source="<cellId>"` and/or `target="<cellId>"`.
- Unsnapped endpoint: only an `mxPoint as="sourcePoint"`/`as="targetPoint"` with raw x,y
  and no `source`/`target` attribute.
- Proper connections often also carry `exitX/exitY/exitDx/exitDy` and
  `entryX/entryY/entryDx/entryDy` in the style string — normalized (0-1) attachment
  points on the shape's bounding box perimeter.

**Applicability:** this is the single most actionable finding for our project. It means
we can *reliably detect* which edges are risky (no `source`/`target` attribute) even
though we cannot reliably repair them. It also gives a concrete instruction for the
source-file authors: drag each loose connector end until it snaps (green highlight), and
drawio will write proper `source`/`target` attributes.

Confirmed in our files: roughly half of all edges across every team file lack these
attributes.

---

## Summary judgement

No source reviewed claims silent, fully-automatic, trustworthy P&ID connectivity
extraction from imperfect input. The consistent pattern across every real deployment —
academic and commercial — is: **automate the confident majority, explicitly surface
everything else for human review.** Our pipeline's trust-tier reporting (explicit vs.
exit_fraction vs. nearest_shape) follows this pattern.
