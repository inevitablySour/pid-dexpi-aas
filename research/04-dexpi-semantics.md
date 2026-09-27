# DEXPI semantics: self-loops, recirculation, and what downstream consumers expect

Research question, raised directly by the user: *self-loops have to be properly
identified as such in DEXPI, because they mean something for the experiment and for the
downstream models that will consume these DEXPI-formatted P&IDs.*

**Bottom line: the user's instinct was right that self-loops matter, and wrong (in the
specific implementation sense) that this pipeline should model them as literal
self-loops. DEXPI's own type system makes an equipment-level self-loop structurally
impossible — recirculation must be modelled as a segment between two DIFFERENT nozzles
of the same equipment. The self-loops rejected as "invalid" earlier this session were
correctly rejected AT THE NOZZLE LEVEL, but the pipeline currently has no equipment-level
abstraction where a legitimate self-loop could even be represented, so this needs new
work, not just a change to what gets rejected.**

---

## Part 1 — real process features that self-connect equipment

**Pump minimum-flow / kickback recycle.** The minimum-flow recirculation line is taken
off downstream of the pump discharge (check valve) and returns to the pump's suction
vessel or suction drum.

Sources (SEMI-VERIFIED — search snippet text, not independently fetched in full):
- EnggCyclopedia, "Typical P&ID arrangement for pumps."
  https://enggcyclopedia.com/2011/03/typical-pid-arrangement-pumps/
- Inst Tools, "P&ID Guidelines for Pumps." https://instrumentationtools.com/pid-guidelines-pumps/
- Eng-Tips forum thread, "Pump Minimum Flow Recycle Return Line" (UNVERIFIED, forum).
  https://www.eng-tips.com/threads/pump-minimum-flow-recycle-return-line.478117/

**Jacketed vessel circulation.** A jacket has "an inlet and an outlet [to] provide for
the circulation of liquid within the jacket" — a same-vessel, same-chamber-type loop.

- Wikipedia, "Jacketed vessel" (SEMI-VERIFIED). https://en.wikipedia.org/wiki/Jacketed_vessel
- US4473171A, "Nozzle construction for jacketed pressure vessels" (SEMI-VERIFIED).
  https://patents.google.com/patent/US4473171A/en

Note: DEXPI models a jacket as a `Chamber` on the equipment — VERIFIED, the
`ChamberDescription` attribute's own example value is literally "jacket chamber."
https://dexpi.org/static/pid_specification_1.4/reference/Equipment/Chamber.html
This gives a finer-grained disambiguation: a jacket loop is same-equipment, different-
chamber.

**Reflux.** Column overhead vapour condenses, collects in a reflux drum, and part returns
to the column — a column-level self-connection routed via condenser + drum + pump.
SEMI-VERIFIED (ScienceDirect topic page, snippet only).
https://www.sciencedirect.com/topics/engineering/reflux-condenser

**Answer to "is this ever drawn as a bare line leaving and re-entering the same symbol,
with no intermediate equipment?"** No evidence of that as normal practice was found.
Real recirculation always routes through at least one in-line element (pump, control
valve, restriction orifice, exchanger). This means:

- At the raw nozzle/component level: **never a literal self-loop** — there is always an
  intervening `PipingComponent` and a distinct destination `Nozzle`.
- At an abstraction level where in-line piping is collapsed into edges and equipment
  becomes a single node: **a literal, legitimate self-loop.**

**This is the crux:** whether a self-loop is an artifact or a real feature is a function
of the abstraction level chosen, not of the physical plant. A pipeline that only ever
operates at the raw-shape/nozzle level (which is what this project currently does) will
never see a legitimate equipment-level self-loop — because that abstraction doesn't exist
in the model yet. The self-loops this session rejected earlier (before the decorative-
cluster fix) were, without exception, checked at the raw resolved-shape level, which is
the correct level to reject them at.

---

## Part 2 — what the DEXPI spec actually says (this is the decisive part)

All entries in this section marked VERIFIED were fetched and read directly from
dexpi.org/dexpi.plants-and-bytes.de.

**`PipingNetworkSegment`** (DEXPI P&ID Specification v1.4). VERIFIED.
https://dexpi.org/static/pid_specification_1.4/reference/Piping/PipingNetworkSegment.html
- `SourceItem`: type `PipingSourceItem`, multiplicity 0..1.
- `TargetItem`: type `PipingTargetItem`, multiplicity 0..1.
- No constraint stated prohibiting identical source and target. No mention of recycle,
  loop, or self-reference anywhere on the page. This is spec **silence**, not permission
  — important to state precisely, not oversold.

**`PipingSourceItem`** (v1.3). VERIFIED.
https://dexpi.plants-and-bytes.de/reference/Piping/PipingSourceItem.html
- Abstract class. **Concrete subtypes: `FlowInPipeOffPageConnector`, `Nozzle`,
  `PipingComponent`, `PropertyBreak`.**
- **`Equipment` is NOT a subtype.** This is the decisive structural fact: a
  `PipingNetworkSegment` can never name a vessel (or any equipment) directly as its
  source or target — only a nozzle, an in-line piping component, a property break, or an
  off-page connector.

**`Nozzle`** (v1.4). VERIFIED.
https://dexpi.org/static/pid_specification_1.4/reference/Equipment/Nozzle.html
- `Nozzle` inherits from **both** `PipingSourceItem` and `PipingTargetItem` — the nozzle,
  not the equipment body, is the actual piping attachment point in the model.

**`Equipment` / `NozzleOwner`** (v1.4). VERIFIED.
https://dexpi.org/static/pid_specification_1.4/reference/Equipment/Equipment.html
https://dexpi.org/static/pid_specification_1.4/reference/Equipment/NozzleOwner.html
- Equipment's supertypes are `ChamberOwner`, `NozzleOwner`, `TaggedPlantItem`,
  `TransmissionDriver` — no piping-item supertype.
- `NozzleOwner.Nozzles`: multiplicity `*` (many nozzles per equipment), opposite
  multiplicity 0..1 (each nozzle belongs to at most one owner).

**`PipingNetworkSystem`** (v1.3). VERIFIED.
https://dexpi.plants-and-bytes.de/reference/Piping/PipingNetworkSystem.html
- No topology constraints stated anywhere (no rule on loops, cycles, branching, or
  connectivity validity). The DEXPI spec pages are attribute/composition-oriented; they
  do not validate topology at all.

### Conclusion for the modelling question

- **Can a segment have the identical object as both source and target?** The spec does
  not forbid it in prose, but it is **unreachable for equipment** specifically, since
  equipment is never a valid source/target type in the first place. Naming the *same
  nozzle* as both source and target would be a degenerate, zero-length segment with no
  engineering meaning — and this is independently confirmed by pyDEXPI's own toolkit code
  (Part 3): `connect_piping_network_segment` explicitly raises `ValueError` when a
  segment references "an internal object as source/target," calling it corrupting.
- **How must a recirculation loop be modelled?** As one or more segments running from
  Nozzle A of equipment X, through intervening piping components, to Nozzle B of the
  *same* equipment X. Not a self-loop at the nozzle level. A self-loop only appears if
  and when nozzles are later collapsed/projected onto their parent equipment.

**Direct implication for this pipeline's implementation:** the correct fix is not "allow
self-loops through" at the current build stage — build_model.py currently works at
essentially the nozzle/shape level with no equipment-collapsing abstraction layer at all.
A genuine same-equipment recirculation in the source drawing should currently manifest as
**two distinct edges to two distinct nozzles of the same vessel shape**, not one edge
whose resolved source and target are the identical shape ID. If parser_drawio.py's
resolution genuinely produces same-shape-same-nozzle results, per this research that is
correctly invalid, not a suppressed real feature — *unless* the pipeline is extended to
build multiple nozzles per equipment (it currently builds exactly one nozzle per
equipment, per `build_model.py`'s `_build_equipment`) and resolve edges to a *specific*
nozzle rather than just the parent shape. That is the real, correctly-scoped next step,
covered under "implications for this codebase" below.

---

## Part 3 — pyDEXPI's own behaviour confirms this reading (verified from source, not docs)

**Citation:** pyDEXPI, https://github.com/process-intelligence-research/pyDEXPI
(`pydexpi/loaders/graph_loader.py`, `pydexpi/toolkits/piping_toolkit.py` — VERIFIED, read
directly from the `master` branch source, not from the paper, which returned HTTP 429).

`GraphLoader` builds a `networkx.MultiDiGraph` (VERIFIED) — a container that permits both
parallel edges and self-loops natively; no restriction is imposed at this level.

`GraphAbstractor.build_process_graph` retains nozzles explicitly (source comment: `# No
Equipment Collapsing`) — so no equipment-level self-loop can arise at this abstraction
tier; recirculation still shows up as a path through distinct nozzles.

`GraphAbstractor.build_conceptual_graph` is where equipment-level projection happens:

```python
conceptual_loader.remove_nodes_by_label(
    label=[class_type.__name__ for class_type in et.get_equipment_internal_classes()])
conceptual_loader.remove_nodes_by_label("Nozzle", stitch=True)
```

`remove_nodes_by_label(..., stitch=True)`: nodes with exactly 2 connected edges get
stitched into a direct edge between their neighbours before removal. Removing all
`Nozzle` nodes with stitching is exactly the "project nozzle-level topology onto parent
equipment" operation that produces equipment-level self-loops for a recirculation line —
**and this stitching has no self-loop guard** (confirmed: unconditionally calls
`self.plant_graph.add_edge(neighbor1, neighbor2, ...)`).

By contrast, the *general-purpose* node-collapsing routines (`collapse_node_to_node`,
`collapse_node_to_node_by_label`) **do** guard against manufacturing new self-loops, with
this comment repeated four times in the source:

```python
if pred_node != to_node_id:  # Avoid self-loops unless they existed
```

**This is the single best piece of evidence available:** the library's own authors
distinguish between *manufacturing a new, spurious* self-loop while collapsing
(guarded against) and *preserving a pre-existing* self-loop (deliberately allowed to
survive). That is a direct statement, in code, that self-loops are legitimate data in a
P&ID graph once you're at the equipment-abstraction level — not an error condition to be
designed away.

`piping_toolkit.py`'s `connect_piping_network_segment` raises `ValueError` specifically
when a segment references an "internal object as source/target," calling it corrupting.
No cycle-detection or "self_loop"/"recycle" logic exists anywhere in pyDEXPI's toolkits
(confirmed by grep, case-insensitive, across `piping_toolkit.py` and
`pydexpi/syndata/graph_pattern.py`) — consistent with cycles/self-loops being legal and
simply unvalidated, not specially handled either way.

---

## Part 4 — downstream consumers preserve cycles

**SFILES 2.0** — Vogel, G., Hirtreiter, E., Schulze Balhorn, L., Schweidtmann, A.M.
"SFILES 2.0: An extended text-based flowsheet representation." *Optimization and
Engineering* 24, 2911-2933 (2023); arXiv:2208.00778. VERIFIED — HTML fetched and read.
https://arxiv.org/abs/2208.00778 | https://arxiv.org/html/2208.00778v2

Recycle notation is first-class, not an afterthought: `#` marks the start of a recycle,
`<#` marks its directed end (example: `(u)(v)<1(w)(x)1` denotes a recycle flowing from
node `(x)` back to `(v)`). For **fully cyclic processes** (e.g. refrigeration cycles,
where every node has in-degree > 0), SFILES 2.0 uses a virtual-node trick: the
lowest-ranked non-outlet node is connected to a virtual node so a single traversal from
that virtual node still works.

**Important scope limitation, stated precisely:** SFILES 2.0's own text does not mention
*self-loops* (an edge from a unit to itself) anywhere — because SFILES operates on
unit-operation graphs where intervening units (pump, heat exchanger) are themselves
nodes, so a recirculation is a *cycle*, not a self-loop, at that representation's chosen
granularity. This is evidence that **cycles must survive conversion**, not direct evidence
about literal self-loops specifically. Don't over-cite it for the self-loop question.

**Rule-based autocorrection** — Schulze Balhorn, L., Seijsener, N., Dao, K., Kim, M.,
Goldstein, D.P., Driessen, G.H.M., Schweidtmann, A.M. "Rule-based autocorrection of
Piping and Instrumentation Diagrams (P&IDs) on graphs." arXiv:2502.18493. VERIFIED — HTML
fetched and read. https://arxiv.org/abs/2502.18493

Their graph nodes are tagged plant items, piping components (e.g. valves), actuators, and
process instrumentation — **nozzles are not nodes** at all in this representation. Only 5
of their stated 33 rules are actually enumerated in the paper (globe-valve sizing, level
instrument on vessel, strainer on pump suction, check valve on pump discharge, block
valves/drain on pump suction/discharge) — **none of the 33 rules disclosed concerns
self-loops, cycles, or node-degree/connection-count constraints**, and the remaining 28
are not published, so this can't be stated with full certainty for the whole rule set.

**The strongest single citation for this project's write-up:** their own illustrative
case study contains a real recirculation-to-vessel: "fluid is discharged by another
pump and is partly recycled to the vessel... additional thermal conditioning for
temperature control of the vessel" — and this is not flagged, corrected, or discussed as
anomalous by any of their rules. The leading graph-based P&ID *validation* paper treats
recirculation-to-the-same-vessel as unremarkable, normal P&ID content.

---

## Explicit unresolved questions (do not overclaim past these)

1. **DEXPI is silent, not permissive, on segment self-identity.** No spec page states
   whether `sourceItem == targetItem` is legal or illegal. State only: (a) equipment is
   structurally excluded as a source/target type, so equipment-level self-reference is
   impossible at the segment-object level; (b) nozzle-level identity is rejected in
   practice by pyDEXPI's own segment-integrity checks, as corrupting.
2. **No DEXPI reference example of a recycle line was found.** The spec has no worked
   recycle example, so there's no "blessed" canonical pattern to cite — the "Nozzle A to
   Nozzle B of the same equipment" model is a sound inference from the type system, not
   a directly-cited normative example.
3. **DEXPI defines no topology validity rules at all** (no acyclicity requirement, no
   "valve must have exactly 2 connections"). Such rules, if wanted, would need to come
   from downstream tooling (pyDEXPI toolkits, or a custom rule set like the 33-rule
   paper's), not from the standard itself.
4. **The 33 autocorrection rules are not published in full** — only 5 of 33 are
   enumerated. Cannot rule out that one of the other 28 concerns loops/degree.
5. The pyDEXPI ESCAPE-35 paper PDF could not be fetched (HTTP 429); all pyDEXPI
   behavioural claims above come from directly reading the source code on GitHub, not
   from the paper.
6. Several engineering-practice sources for kickback/jacket/reflux are trade references
   and forums (EnggCyclopedia, Inst Tools, Eng-Tips, Wikipedia) — solid for orientation,
   but a formal write-up should prefer a textbook citation (e.g. Turton et al.,
   *Analysis, Synthesis and Design of Chemical Processes*) for the underlying process
   engineering claim.

---

## Implications for this codebase

This is a genuinely different, larger piece of work than anything patched so far this
session, and it should be scoped as its own task rather than folded into "fix the
self-loop rejection":

1. **`build_model.py` currently gives every piece of equipment exactly one `Nozzle`**
   (see `_build_equipment`). This makes it structurally impossible to represent a real
   recirculation loop correctly even if the source drawing has one — there is only one
   attachment point per vessel to connect *anything* to, so two distinct connections to
   the same vessel already collide on the same nozzle today, regardless of the self-loop
   question. Modelling recirculation correctly requires building multiple nozzles per
   equipment shape (one per distinct edge connection point on that shape), matching how
   many actual pipe connections the drawio shape has.
2. **A same-shape-same-*resolved-nozzle* edge should stay rejected** — that is the
   correctly-identified degenerate case per pyDEXPI's own validity check.
3. **A same-shape-*different*-nozzle edge should be accepted and built as a real
   segment**, not discarded — this is the legitimate recirculation case, and doing so
   requires the multi-nozzle change in (1) before it's even representable.
4. Given today's pipeline resolves edges to a *shape*, not a *specific nozzle on that
   shape*, none of the self-loops found and rejected earlier this session can currently
   be distinguished into "genuine recirculation" vs. "resolution artifact" — that
   distinction requires the nozzle-level granularity this pipeline doesn't have yet. This
   is worth being explicit about in the discrepancy report: today's "self-loop, invalid"
   category may be silently swallowing real recirculation loops from the source
   drawings, and that should be flagged as a known gap, not quietly left as-is.
