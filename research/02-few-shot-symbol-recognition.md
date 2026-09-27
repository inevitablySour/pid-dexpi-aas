# Few-shot / one-shot symbol recognition for P&IDs

Research question: *is it easier to train a model to recognise symbols and topology than
to hand-write parsing heuristics? Specifically, is ~65-70% accuracy achievable from one
training sample?*

**Short answer: the one-shot figure is real and actually better than 65-70% — but only
for SYMBOL RECOGNITION, which is not this project's bottleneck. There is no comparable
few-shot evidence for topology extraction, which is.**

---

## OSSR-PID — the closest paper to the one-shot claim

**Citation:** Paliwal, S., Sharma, M., et al. "OSSR-PID: One-Shot Symbol Recognition in
P&ID Sheets using Path Sampling and GCN." arXiv:2109.03849, 2021.
https://arxiv.org/abs/2109.03849

**Status:** UNVERIFIED (abstract and secondary sources; full text not fetched)

**What "one-shot" means here:** one prototypical example *image per symbol class*. Not
one whole annotated P&ID, and not one-shot for a single class — every class gets exactly
one training exemplar.

**Method:** samples boundary pixels of a symbol into a graph, embeds with a Dynamic Graph
CNN + ResNet-34, trains with ArcFace loss for embedding separability, classifies via
nearest-prototype matching.

**Dataset:** 100 *synthetic* P&ID diagrams, plus a private 12-sheet real-world set used
only for comparison against prior fully-supervised work.

**Reported accuracy progression (average symbol classification accuracy, one example per
class):**

| Configuration | Accuracy |
|---|---|
| DGCNN + cross-entropy loss | ~77.07% |
| DGCNN + ArcFace loss | ~85.78% |
| DGCNN + ResNet-34 + ArcFace | ~85.98% |

**Critical scope limitation:** symbol **classification/localization only**. The paper does
not attempt line or connectivity extraction — it treats pipelines as fixed context, not
as a graph-construction target.

**Applicability to this project: LOW.** Two reasons. First, the measured number is on
clean synthetic diagrams; real hand-drawn input would be lower. Second and more
importantly, **our pipeline already gets symbol recognition for free and essentially
perfectly** — the `shape=mxgraph.pid.*` style token in the draw.io XML is unambiguous
ground truth. Few-shot symbol learning would solve a problem we do not have.

---

## Few-Shot Symbol Detection in Engineering Drawings

**Citation:** Jamieson, L., Elyan, E., Moreno-García, C.F. "Few-Shot Symbol Detection in
Engineering Drawings." *Applied Artificial Intelligence*, 38(1), 2024.
DOI: 10.1080/08839514.2024.2406712

**Status:** UNVERIFIED — paywalled at Tandfonline, 403 on ResearchGate. The exact numeric
mAP table could NOT be retrieved. If precise numbers are needed for a write-up, obtain
the PDF via library access or the RGU institutional repository.

Standard N-way K-shot object detection framing (fine-tuning based, compared against a
TFA/R-101 baseline), K from 1 to ~9-10 shots per class, on **real industrial engineering
drawings** (general engineering drawings, not necessarily P&ID specifically).

Reports mAP for novel classes (nAP), base classes (bAP), and combined. Search snippets
suggested nAP improvements of 35-50 points over baseline at low shot counts, and some
novel classes reaching recall 1.00 at ~9 shots — **these specific figures are unverified.**

This paper is the most plausible origin of a "65-70%" figure (a low-shot, novel-classes-only
mAP configuration would plausibly land there), but that is inference, not a confirmed
citation.

**Applicability: LOW**, for the same reason as OSSR-PID — it is detection (localization +
classification), not connectivity.

---

## The critical caveat: symbol accuracy does not transfer to topology

This is the most important finding of this research pass.

**Every** few-shot/one-shot P&ID paper found is scoped to symbol classification or
detection. None address topology/connectivity extraction within their few-shot evaluation.

In the broader literature, topology extraction — line tracing, junction resolution,
disambiguating crossing and branching lines, matching line endpoints to symbol ports — is
treated as a **separate, downstream problem**, solved either by hand-written rules or by
fully-supervised GNNs that assume substantial labelled graph-structure training data.
See `03-topology-extraction.md`.

**No published few-shot or one-shot benchmark for P&ID topology extraction was found.**

Implication for any write-up: quoting "~65-70% (or ~86%) with one training sample" as an
expectation for *end-to-end* digitization would be a conflation. It is a symbol-recognition
number. Topology extraction is structurally harder — combinatorial 2D line/junction
reasoning rather than single-object classification — and has no evidenced few-shot
solution.

---

## No standard benchmark exists

There is no widely-adopted few-shot P&ID benchmark — no miniImageNet-equivalent for this
domain. Each paper uses its own private or synthetic dataset (100 synthetic diagrams; 12
real sheets; RGU's industrial drawing set), so cross-paper comparability is weak.

Related: **SynthPID** generates topology-preserving synthetic P&ID training data
specifically because real labelled connectivity data is scarce — implicitly confirming
that connectivity ground truth is the hard, under-resourced part of this field.
arXiv:2604.16513, https://arxiv.org/html/2604.16513v1 — **UNVERIFIED** (search snippet).

---

## Conclusion for this project

Training a symbol recogniser would be solving an already-solved problem (our symbol
identification comes free from structured XML tokens and is effectively 100% accurate).
The bottleneck is topology, and the literature offers no few-shot shortcut for it.

This is a genuinely discouraging finding for "just train a model instead" as a strategy —
it does not target the actual failure mode.
