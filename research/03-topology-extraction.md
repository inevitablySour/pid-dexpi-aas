# Topology/connectivity extraction: VLM approaches and vector-input prior art

Research question: is there a better approach to connectivity extraction than the
proximity heuristics this pipeline currently uses — specifically, does a "~83% VLM
connectivity accuracy" claim hold up, and does anyone solve this problem from
structured/vector input rather than raster images?

**Headline correction:** the ~83% figure traces to a real preprint, but it was
mis-attributed to the wrong arXiv ID in earlier notes, and **83% is symbol association,
not connectivity** — the actual connectivity number in that source is 82% ("line
association"), on an unspecified real-industrial corpus, unverified beyond the abstract.

**Headline finding, most load-bearing for this project:** in the one paper whose method
was fully read (Gadekar et al., below), replacing adaptive endpoint snapping with a
**fixed 10-pixel tolerance collapsed edge F1 from 0.742 to 0.193** — a ~4x degradation
from exactly the kind of single-distance-threshold heuristic this pipeline has been using
(`_MAX_RESOLUTION_DISTANCE = 40.0` in `parser_drawio.py`). This is the strongest published
evidence that our threshold-tuning approach is fundamentally the wrong lever to pull.

---

## Part 1 — the ~83% claim, corrected

**Citation:** Ali, R.H., Park, C.Y., Aryal, A., Jeong, H.D., Lee, G. "P&ID Information
Extraction through a Vision Language Model for Digital Twins." SSRN preprint, posted
16 January 2026. SSRN abstract ID 6083108. DOI 10.2139/ssrn.6083108.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6083108

**Status:** PARTIALLY VERIFIED — abstract only; SSRN blocks automated full-text fetch
(HTTP 403). **Not peer-reviewed.**

Method (per abstract): training-free framework combining classical CV preprocessing with
few-shot VLM prompting. Three stages: (1) density-based segmentation partitioning the
diagram into bounded-complexity windows, (2) geometric line reconstruction, (3) few-shot
VLM prompting to extract equipment and connectivity.

Reported numbers:
- 83% symbol association accuracy
- **82% line association accuracy** (the connectivity-relevant figure — not 83%)
- 73% for an unnamed "state-of-the-art supervised" comparison baseline
- Validated on "real industrial datasets" (sheet count not disclosed in the abstract)
- Stated failure thresholds: total entropy > 2x10^6, edge complexity > 0.5

**Caveats:** "line association accuracy" is undefined at the level available (unknown
whether per-edge, per-node-pair, or per-line-object; unknown whether precision, recall,
or matched accuracy). Sheet count and provenance of "real industrial" data unknown. The
73% baseline is unnamed and unauditable.

**Note on a separate, unrelated "83%":** the PID2Graph benchmark (Part 6) reports 83.63%
node AP for Relationformer — this is a different number about a different task from a
different paper. Two unrelated 83s exist in this literature; do not conflate them.

**Applicability to this project: LOW-MODERATE.** Stages 1-2 of Ali et al.'s pipeline exist
to recover vector primitives from raster pixels — work our mxGraph XML already gives us.
Only stage 3 (few-shot VLM prompting) would transfer, and its entropy/edge-complexity
failure thresholds suggest windowing a diagram before prompting rather than passing a
whole sheet — a transferable idea even if the exact numbers are not.

---

## Part 2 — the paper actually worth building on

**Citation:** Gadekar, A., Srinivas, ?., Runkana, V. "Grounded and Faithful P&ID
Reasoning: Constraining Vision-Language Models with Recovered Evidence Graphs."
arXiv:2609.05880v1, 5 September 2026. https://arxiv.org/html/2609.05880

**Status:** VERIFIED — full HTML read.

**Core thesis:** VLMs describe P&ID sheets fluently but invent and drop process
connections, which can reverse isolation/reachability decisions. Fix: recover an explicit
graph deterministically first, then constrain the VLM to answer questions only by
querying that graph through read-only operators — never asking it to infer topology
directly.

**Dataset:** Digitize-PID — synthetic, 500 raster sheets, 32 symbol classes, solid and
dashed lines, word boxes; 100-sheet test partition. Also introduces **TopoPID-VQA**, a
3,000-question topology benchmark (easy/medium/hard).

**Key numbers:**
- Exact process-connection edge F1 = **0.742** on the auto-recovered graph
- Connectivity-consistency F1 = 0.958
- Attributed micro-F1 (nodes + edges + tags) = 0.801
- Grounding (querying the recovered graph) lifts VLM QA accuracy by **+33.8 to +37.6
  percentage points** over an ungrounded VLM (36.7-41.3% -> 74.3-76.0%)
- Stock cloud OCR/object-detection APIs (Azure, Textract, GCP) score **near-zero** on
  P&ID symbol detection and line segmentation

**★ The single most actionable finding of this whole research pass:** in their ablation,
a fixed 10-pixel snap tolerance for endpoint matching collapses exact edge F1 from
**0.742 to 0.193** — roughly a 4x degradation, caused purely by using one global distance
threshold instead of an adaptive/context-dependent one.

**Direct implication for this codebase:** `parser_drawio.py`'s `_MAX_RESOLUTION_DISTANCE`
(a single fixed 40-unit cutoff applied uniformly) is the exact anti-pattern this ablation
demonstrates is catastrophic, not just suboptimal. This is published, quantified
justification for replacing it with a multi-signal, adaptive approach (see Part 5a) rather
than continuing to tune the single number.

**Applicability: HIGH conceptually**, despite raster input. The snap-tolerance ablation
and the "recover graph, then constrain the VLM to query it" architecture both transfer
directly to vector/XML input. Their ungrounded-VLM baseline (36.7-41.3% QA accuracy) is
also a strong argument against ever asking an LLM to infer topology directly from a
diagram description.

---

## Part 3 — VLM topology extraction: the sobering general-purpose baseline

**Citation:** Guo, B., Zhao, X., Liu, Y., Cheng, W., Chen, S., Li, D., Morimoto, M.,
Kuroda, T., Metaxas, D., Chen, H. "TopoAgent: A Structure-Aware Perception-to-Reasoning
Framework for Diagram-to-Graph Topology Extraction with Large Vision-Language Models."
arXiv:2608.28701v1. https://arxiv.org/abs/2608.28701

**Status:** VERIFIED — full HTML read.

**Benchmark:** TopoBench-180 — 180 human-verified diagram-graph pairs (54 web-style
directed-arrow diagrams, 126 network-style bidirectional/undirected diagrams). General
structural diagrams, not P&ID.

**Metric (worth adopting):** node-level F1 for entity identification; edge-level
precision/recall/F1 where an edge counts correct **only when the (source, target) tuple
exactly matches ground truth** within a node's relationship group.

**Results (selected):**

| Setting | Method | Node F1 | Edge F1 |
|---|---|---|---|
| Web-style | TopoAgent | 95.6 | 45.8 |
| Web-style | GPT-4o | 93.3 | 32.8 |
| Network-style | TopoAgent | 87.3 | 58.1 |
| Network-style | GPT-4o | 85.2 | 53.5 |

**The structural lesson:** node F1 sits at 85-96 across every method; edge F1 sits at
33-58. Nodes (symbols) are near-solved; edges (connectivity) are the wall. This matches
this project's own experience exactly — symbol identification from the `shape=` token is
essentially exact, while topology resolution is where every real failure has occurred.

Input: raster images (their own SVGs were collected during dataset construction, then
rasterized anyway for the benchmark).

**Applicability: HIGH as an expectation-setting baseline.** If this project ever routed
diagrams through a VLM as rendered images, ~33-58 edge F1 is the realistic prior our
structured XML should be expected to beat.

---

## Part 4 — structured/vector input (the closest methodological match found)

### VectorGraphNET

**Citation:** Carrara, A., Nousias, S., Borrmann, A. "VectorGraphNET: Graph Attention
Networks for Accurate Segmentation of Complex Technical Drawings." arXiv:2410.01336;
*Journal of Computing in Civil Engineering* 39(6), DOI 10.1061/JCCEE5.CPENG-6508.
https://arxiv.org/abs/2410.01336 | https://ascelibrary.org/doi/10.1061/JCCEE5.CPENG-6508

**Status:** PARTIALLY VERIFIED (abstract/metadata; results corroborated across multiple
listings, full paper body not read).

**This is the closest methodological match to this project found in the literature.**
Pipeline: PDF -> SVG -> a feature-rich graph capturing relationships between vector
entities using geometric information -> graph attention transformer with hierarchical
label definitions -> line-level semantic segmentation.

Results: proprietary TUM dataset (76 architectural drawings) — overall accuracy 0.97,
macro-F1 0.82. Public FloorplanCAD benchmark — SOTA weighted F1 89%.

**Applicability: HIGHEST of anything found.** "Each vector entity becomes a graph node,
edges encode geometric relationships, learn over that graph" transfers almost verbatim
from SVG to mxGraph XML. Important scope limit: this is *semantic segmentation*
(labelling each line's class/role), which is closer to solving our decorative-vs-real
linework problem (Part 5c) than to solving connectivity resolution itself. It is also
supervised — needs labelled training data.

### ChatP&ID (structured-input, DEXPI-native)

**Citation:** Alimin, A.A., Schweidtmann, A.M. "GraphRAG for Engineering Diagrams:
ChatP&ID Enables LLM Interaction with P&IDs." arXiv:2603.22528, 23 March 2026.
https://arxiv.org/abs/2603.22528

**Status:** VERIFIED — full HTML read. (Note: this is NOT the source of the 83% claim —
that attribution was corrected in Part 1.)

DEXPI-format P&IDs -> knowledge graph -> graph-based retrieval -> natural-language QA.
GPT-5 with conceptual-graph representation: 0.94 avg accuracy vs. 0.76 reading the raw
flowsheet image directly (+18 percentage points). GPT-5-mini + ContextRAG: 0.91 accuracy
at $0.004/task, 85% token-cost reduction.

**Methodological weaknesses to flag if cited:** evaluated on a **single** P&ID (the
published DEXPI standard reference example DEXPIEX01.xml — synthetic, not industrial) with
**19 QA pairs** across 4 categories, each configuration run twice. The graph-vs-image
comparison is not apples-to-apples (tooling and representation are confounded).
Critically: **connectivity here is read out of already-structured DEXPI attributes
(`send_to`, `control`, `manipulate`), not extracted from geometry.**

**Applicability: HIGH for the direction, not the method.** It is the best available
evidence that structured input beats raster for a given diagram (+18pp on the *same*
underlying diagram). But it demonstrates that QA is easier *given* correct topology — it
does not solve topology extraction, since DEXPI hands over the edges already.

**Where this project actually sits:** between raster (no primitives at all) and DEXPI
(semantic edges already given). We have explicit shape and line coordinates (better than
raster) but no semantic connection references (worse than DEXPI). The task is genuinely
"geometry -> topology," which is a different, harder problem than ChatP&ID's.

---

## Part 5 — the three specific sub-problems

### 5a. Dangling / near-miss / unsnapped endpoints

**Primary evidence:** the Gadekar et al. snap-tolerance ablation (Part 2): fixed 10px
tolerance -> 0.193 edge F1, vs. 0.742 for their adaptive approach. Cite this one for the
"why not just tune the threshold" question.

**Citation (patent, UNVERIFIED — full text not read):** US Patent 8,578,346, "System and
method to validate and repair process flow drawings."
https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8578346

Identifies dangling edges in filtered flow-graphs and computes probable fixes;
connectors whose endpoints lie near nodes or other lines are recursively identified as
part of candidate edges; **confidence is scored from containment, nearness, AND geometric
orientation heuristics together**, not distance alone.

**Applicability: HIGH conceptually.** A multi-signal confidence score (containment +
proximity + orientation) is exactly the adaptive approach the Gadekar ablation vindicates,
and is a concrete, implementable upgrade over this pipeline's single-distance-threshold
`resolve_floating_edge_endpoints`.

**Note:** the researching agent found no academic paper dedicated specifically to
near-miss connection inference in loosely/hand-drawn *structured* diagrams — this
project's exact problem setting appears to be under-served in the literature; the
relevant techniques exist only as components inside larger pipelines, or as patents.

### 5b. Reconstructing one logical (dashed) line drawn as many disconnected segments

**Citation:** Stürmer, J.M., Graumann, M., Koch, T. "From Engineering Diagrams to Graphs:
Digitizing P&IDs with Transformers." arXiv:2411.13929v2; IEEE DSAA 2025.
https://arxiv.org/abs/2411.13929

**Status:** VERIFIED — HTML read.

Their dashed-line method: small line segments are assumed to be dashes of one dashed
line, **clustered with DBSCAN, and merged**. Their modular baseline assigns line
endpoints to the closest symbol or nearest corresponding endpoint of another line; 3+
endpoints meeting creates a "crossing node"; a bend between two linked segments creates
an "ankle node."

**Applicability: HIGH.** DBSCAN over segment midpoints/endpoints is cheap and directly
implementable on our own XML geometry — this is a concrete, buildable improvement over
the ad hoc chain-merging attempted earlier this session (which was abandoned as
unreliable). The crossing/ankle node typology is also a clean data model to adopt.

**Citation (title-only match, UNVERIFIED, high priority to read):** Prusty, S.S.,
Agarwal, A., Iyenger, S. "Advanced Integration of Discrete Line Segments in Digitized
P&ID for Continuous Instrument Connectivity." arXiv:2505.11976, 17 May 2025.
https://arxiv.org/abs/2505.11976

By title, the single most precisely on-point paper for this project's fragmented-segment
problem. Abstract discloses only the goal (merging detected line segments to link
instruments); no method, dataset, or numbers surfaced. **Recommended as the highest-
priority manual follow-up read** — full PDF was not accessible to automated fetch.

**Supporting technique (UNVERIFIED, search snippet):** classical two-stage endpoint
merging — segments are "proximal" if any of the four pairwise endpoint separations falls
below a tolerance that scales with image size; merging then collapses similar segments
using tolerances on **both angle and distance** (not distance alone). Applicability:
HIGH — angle-plus-distance similarity is the right test for merging a dashed run, matching
the earlier finding this session that the TIC 203 dashed loop's fragments were evenly
spaced and collinear (a real angle+distance signature), unlike the PSV decorative
clusters (irregular, criss-crossing, no consistent angle).

### 5c. Decorative symbol linework vs. real connections

Best answer: VectorGraphNET (Part 4) — line-level semantic segmentation over vector
entities is precisely the right formulation (macro-F1 0.82).

**Structural shortcut specific to our draw.io XML** (not from the literature, but implied
by it): decorative strokes are typically part of a shape's own style/geometry, while
connections are separate `mxCell` elements with `edge="1"` and (ideally) source/target
attributes. This project's own finding this session — that decorative valve-symbol
clusters are tight (all endpoints within one shape's bounding box) while real connections
span to a distant, different shape — is consistent with this being substantially an XML/
geometry classification problem rather than one needing ML.

---

## Part 6 — datasets, benchmarks, code (for future reference)

- **PID2Graph** (VERIFIED). First publicly available P&ID dataset with real-world data
  and full graph annotations (symbols + connections). 12 manually annotated real sheets
  (OPEN100 nuclear reactor) + 700+ synthetic. Zenodo: https://zenodo.org/records/14803338
  Benchmark: Relationformer 83.63% node AP / 75.46% edge mAP on real data (this 83.63% is
  unrelated to the Ali et al. 83% in Part 1 — different paper, different metric).
- **Digitize-PID** (VERIFIED via multiple sources). 500 synthetic P&IDs, 32 symbol
  classes. arXiv:2109.03794. Mirror: https://huggingface.co/datasets/hamzas/digitize-pid-yolo
- **TopoPID-VQA** (VERIFIED). 3,000 topology QA pairs (Gadekar et al.). Release status
  unconfirmed.
- **TopoBench-180** (VERIFIED). 180 diagram-graph pairs, general structural diagrams
  (Guo et al.).
- **FloorplanCAD** — public vector-drawing segmentation benchmark (used by
  VectorGraphNET).
- **Azure-Samples/digitization-of-piping-and-instrument-diagrams** (UNVERIFIED, repo not
  read). Microsoft's own open-source reference implementation, includes a graph-
  construction stage. https://github.com/Azure-Samples/digitization-of-piping-and-instrument-diagrams
  Worth reading `docs/architecture.md` directly for its line-to-symbol association logic
  — this is the same team behind the ISE devblog writeup cited in `01-pid-digitization-prior-art.md`.
- **Roboflow "PID Connect"** (UNVERIFIED). https://universe.roboflow.com/pid-connect/p-id-symbols
- Also noted, not detailed: Zhu, B., Duong, S., Vyas, J., Mercangöz, M. "From P&ID
  Drawings to Process Graphs: A Multimodal Language Model Approach." arXiv:2607.19568,
  21 July 2026 — infers connectivity from tag semantics and process-design rules in a
  separate LLM pass, rather than from geometry. Conceptually a complement to (not a
  replacement for) geometric inference. No metrics accessible.
- Survey for orientation (UNVERIFIED, not read): "Artificial intelligence for piping and
  instrumentation diagrams: a review and perspective," *Frontiers of Chemical Science and
  Engineering*, DOI 10.1007/s11705-026-2685-z.

---

## Synthesis: what this means for the pipeline

1. **The original "~83% VLM connectivity" claim should not be repeated as stated.**
   Corrected form: Ali et al. (SSRN preprint, Jan 2026, non-peer-reviewed) report 83%
   symbol association and 82% line association on unspecified real industrial data,
   full text unverified.

2. **Do not expect a VLM to read topology off a rendered image of these diagrams.**
   Corroborated independently by two papers: edge F1 33-58% on general diagrams (Guo et
   al.), ungrounded VLM QA accuracy 36.7-41.3% (Gadekar et al.), near-zero for stock cloud
   vision APIs on P&ID-specific symbol/line detection.

3. **Structured input is a real advantage, quantified, but for a different problem than
   ours.** ChatP&ID's +18pp (graph vs. raster) is the cleanest evidence for the direction,
   but its topology is read from existing DEXPI attributes, not inferred from geometry —
   its result shows correct-topology-in beats raster-in for QA, not that topology
   extraction itself is solved.

4. **The single highest-value, directly actionable finding:** replace the fixed-distance-
   threshold approach (`_MAX_RESOLUTION_DISTANCE`) with a multi-signal confidence score
   (containment + proximity + orientation, per US 8,578,346) and DBSCAN-based clustering
   for fragmented dashed lines using angle-and-distance similarity (per Stürmer et al.),
   rather than continuing to tune one global number. The Gadekar ablation (0.742 -> 0.193
   from exactly this kind of change, in the opposite direction) is direct, quantified,
   published evidence that this class of fix matters far more than threshold-tuning.

5. **Adopt a real metric.** Exact-tuple edge precision/recall/F1 within node relationship
   groups (Guo et al.) and/or Hungarian-matched edge mAP (Stürmer et al./PID2Graph),
   reporting node and edge accuracy *separately* — conflating them is exactly how the 83%
   figure got mistaken for a connectivity number in the first place.

6. **Highest-priority manual follow-up:** arXiv:2505.11976 (Prusty et al.) — by title the
   most precisely on-point paper for the fragmented-dashed-segment problem this session
   hit directly; full PDF not accessible to automated fetch.
