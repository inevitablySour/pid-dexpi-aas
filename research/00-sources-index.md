# Research sources index

Literature gathered while building the P&ID -> DEXPI pipeline. Organised by the problem
each source speaks to. Every entry records what was actually verified vs. what came from
a search snippet only, because several claims in this space turned out to be weaker than
they first appeared.

Files in this directory:

- `00-sources-index.md` (this file) — master index
- `01-pid-digitization-prior-art.md` — how others digitize P&IDs, and how they handle
  ambiguity (the "is our approach sane?" question)
- `02-few-shot-symbol-recognition.md` — one-shot/few-shot symbol recognition evidence
- `03-topology-extraction.md` — connectivity/topology extraction, incl. VLM approaches
- `04-dexpi-semantics.md` — DEXPI spec questions (self-loops, recycles, nozzles)

## Status legend

- **VERIFIED** — the source was fetched and read directly.
- **UNVERIFIED** — only seen via search-result snippet; treat the specific numbers as
  provisional until the source is read.

## Why this directory exists

Four claims during this project turned out, on checking, to be materially different from
how they were first stated:

1. pyDEXPI's Proteus exporter was assumed to round-trip a full DEXPI model. Direct
   source inspection showed it only exports MetaData/Equipment/Nozzle — piping and
   instrumentation are silently dropped. (See `README.md`.)
2. A "~65-70% accuracy with one training sample" figure for P&ID ML turned out to be in
   the right ballpark of real one-shot *symbol recognition* results (actually ~86% in the
   strongest paper), but there is no comparable few-shot evidence for *topology*
   extraction — the harder problem this project actually has. (See
   `02-few-shot-symbol-recognition.md`.)
3. A "~83% VLM connectivity accuracy, training-free" claim traced to a real (but
   non-peer-reviewed, unverified-in-full) preprint, but 83% turned out to be *symbol*
   association — the actual connectivity number in the same source is 82%. A second,
   unrelated "83%" (node AP, not connectivity, from a different paper entirely) exists in
   this literature too — easy to conflate. (See `03-topology-extraction.md`.)
4. Self-loops in the extracted graph were initially treated as pure extraction noise to
   discard. Research into the DEXPI spec and pyDEXPI's own source showed real
   recirculation/reflux/kickback loops are a legitimate, common P&ID feature — but DEXPI
   cannot represent them as literal equipment-level self-loops at all (Equipment is not a
   valid segment source/target type); they must be two distinct nozzles on the same
   equipment. This pipeline currently builds only one nozzle per equipment, so it cannot
   yet distinguish a real recirculation loop from a resolution artifact. (See
   `04-dexpi-semantics.md`.)

All four cases argue for recording exact measurement/spec conditions alongside every
claim, not just the headline number.
