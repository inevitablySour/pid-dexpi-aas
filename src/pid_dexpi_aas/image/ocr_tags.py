"""Turn raw OCR tokens into plant tags, and attach tags to detected symbols.

Input tokens come from ml/pid2graph/pipeline/infer_sheet.py (EasyOCR): {"text", "conf", "box"}
with box = [x1, y1, x2, y2] in image pixels. Two P&ID conventions shape this module:

  - Instrument bubbles stack the ISA letters above the loop number ("PI" over "102"),
    and OCR returns those as two separate tokens - they must be joined into "PI 102".
  - A tag sits either inside its symbol (instrument bubbles) or next to it (valves,
    equipment), so attachment prefers "inside" and otherwise takes the nearest tag
    within a distance proportional to the symbol's size - never force-matching a far
    tag, the same rule tag_association.py applies to draw.io labels.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# ISA letters / equipment prefix, e.g. "PI", "TIC", "PSV", "V", "LED".
_LETTERS = re.compile(r"^[A-Z]{1,4}$")
# Loop / item number, e.g. "102", "201/2", "N2" (a medium written under the letters).
_NUMBER = re.compile(r"^(\d{1,4}[A-Z]?(/\d+)?|[A-Z]\d{1,2})$")
# A complete tag on one line, e.g. "GV204", "PSV-101", "LED 201", "TI 201/2". At least two
# digits, so a medium like "N2"/"H2" (written under an instrument's letters) is left free
# to join its bubble instead of standing alone as a fake tag.
_FULL_TAG = re.compile(r"^([A-Z]{1,4})[- ]?(\d{2,4}[A-Z]?(/\d+)?)$")


@dataclass
class OcrTag:
    text: str  # normalized, e.g. "PI 102", "GV204"
    box: tuple[float, float, float, float]
    conf: float
    parts: list[str] = field(default_factory=list)  # raw token texts it was built from

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


# OCR's look-alike characters, by which side of a tag they are on: the letters (ISA letters,
# equipment prefix) are never digits, the number never letters (bar one final letter, 201A).
_AS_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A", "|": "I"}
_AS_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "|": "1", "S": "5", "B": "8", "Z": "2",
             "G": "6", "T": "7"}
_TAG_SHAPE = re.compile(r"^([A-Z]{1,4})([- ]?)(\d{1,4}[A-Z]?(/\d+)?)$")


def repair_tag(text: str) -> str:
    """A token read as a tag with look-alike characters on the wrong side - "P5V201" ->
    "PSV201", "TlC-3O1" -> "TIC-301", "8PR 401" -> "BPR 401": each split into 1-4 letters
    and a number is tried, the look-alikes swapped to their side, and the split needing
    the fewest swaps wins (a known ISA / equipment prefix breaks ties). Unchanged when it
    already is a tag or cannot be one."""
    from pid_dexpi_aas.dexpi.mapping import classify_tag

    t = text.strip().upper().replace("_", "-")
    # a number alone (the "102" under a bubble's "PI") is half a tag, never to be made letters
    if _TAG_SHAPE.match(t) or not any(c.isdigit() for c in t) or len(t) > 12 or not re.search(r"[A-Z]", t):
        return t
    best = None
    for k in range(1, 5):
        head, tail = t[:k], t[k:]
        sep = tail[:1] if tail[:1] in "- " else ""
        tail = tail[len(sep):]
        letters = "".join(_AS_LETTER.get(c, c) for c in head)
        # the number: look-alikes to digits, but a last letter may stay (201A)
        body, last = (tail[:-1], tail[-1]) if len(tail) > 1 and tail[-1].isalpha() and tail[-1] not in _AS_DIGIT else (tail, "")
        number = "".join(_AS_DIGIT.get(c, c) for c in body) + last
        cand = f"{letters}{sep}{number}"
        if not _TAG_SHAPE.match(cand):
            continue
        swaps = sum(a != b for a, b in zip(head + body, letters + number[:len(body)]))
        known = classify_tag(f"{letters} {number}")[0] is not None
        key = (swaps, not known, k)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1] if best else t


def snap_to_known(text: str, known: list[str]) -> str:
    """A read only look-alike characters off a tag known to be on this drawing (the spec
    sheet's, the draw.io file's) is that tag - "PSV2O1" -> "PSV201", "P5V201" -> "PSV201" -
    when exactly one known tag is that close. Nothing else: tags one letter apart are often
    both real (V202 / XV202 / GV202 - measured: a one-character-off rule turned right reads
    wrong). Compared by the tag part ("PC 201 N2" -> PC201), without spaces and hyphens."""
    flat = lambda x: re.sub(r"[\s\-_]", "", x.upper())

    def key(x: str) -> str:
        m = re.search(r"[A-Z]{1,4}[\s-]?\d{1,4}[A-Z]?(?:/\d+)?", x.upper())
        return flat(m.group(0)) if m else flat(x)
    t = flat(text)
    by_key: dict[str, str] = {}  # its tag part as written ("TIC 301")
    for k in known:
        if k:
            m = re.search(r"[A-Z]{1,4}[\s-]?\d{1,4}[A-Z]?(?:/\d+)?", k.upper())
            by_key.setdefault(key(k), m.group(0) if m else k.upper())
    if not t or t in by_key or len(t) < 3:
        return text
    alike = lambda a, b: a == b or _AS_DIGIT.get(a) == b or _AS_DIGIT.get(b) == a or _AS_LETTER.get(a) == b or _AS_LETTER.get(b) == a
    hits = [k for f, k in by_key.items() if len(f) == len(t) and all(alike(x, y) for x, y in zip(t, f))]
    return hits[0] if len(hits) == 1 else text


def normalize_token(text: str) -> str:
    """Uppercase, trim, and fix the classic letter/digit OCR confusions in the number
    part only (O->0, I/l->1), never in the letter prefix where 'O'/'I' are legitimate."""
    # Lowercase "l" in a letter prefix is almost always a misread "I" ("Fl" -> "FI").
    t = re.sub(r"^([A-Za-z]{1,4})", lambda m: m.group(1).replace("l", "I"), text.strip())
    t = t.upper().replace("_", "-")
    m = re.match(r"^([A-Z]{1,4})([- ]?)(.*)$", t)
    if m and m.group(3) and any(c.isdigit() for c in m.group(3)):
        number = m.group(3).replace("O", "0").replace("I", "1").replace("L", "1")
        return f"{m.group(1)}{m.group(2)}{number}"
    if re.fullmatch(r"[\dOIl/]+", text.strip()) and any(c.isdigit() for c in text):
        return text.strip().replace("O", "0").replace("I", "1").replace("l", "1")
    return t


def _box(token: dict) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in token["box"])  # type: ignore[return-value]


def assemble_tags(tokens: list[dict], min_conf: float = 0.3, known: list[str] | None = None) -> list[OcrTag]:
    """Build tags from tokens: one-line tags as-is, stacked letters+number pairs joined.
    Look-alike characters are put right (repair_tag); with known (the drawing's tags, from
    a spec sheet or the draw.io file) a near-miss read becomes that tag (snap_to_known)."""
    kept = [t for t in tokens if float(t["conf"]) >= min_conf]
    items = [(repair_tag(normalize_token(t["text"])), _box(t), float(t["conf"])) for t in kept]
    # tag letters are capitals; a word written in lower case ("Bursting disc") is a note -
    # bar a misread "l" for "I" ("Fl")
    worded = {i for i, t in enumerate(kept) if any(c.islower() for c in t["text"].replace("l", "I"))}
    used: set[int] = set()
    tags: list[OcrTag] = []

    for i, (text, box, conf) in enumerate(items):
        if _FULL_TAG.match(text):
            tags.append(OcrTag(text, box, conf, [text]))
            used.add(i)

    # Stacked pairs: a letters-only token directly above a number token, horizontally
    # overlapping, with a vertical gap under one line height.
    letters = [i for i, (t, _, _) in enumerate(items) if i not in used and i not in worded and _LETTERS.match(t)]
    numbers = [i for i, (t, _, _) in enumerate(items) if i not in used and _NUMBER.match(t)]
    pairs = []
    for li in letters:
        lt, lb, lc = items[li]
        line_h = lb[3] - lb[1]
        for ni in numbers:
            nt, nb, nc = items[ni]
            gap = nb[1] - lb[3]
            overlap = min(lb[2], nb[2]) - max(lb[0], nb[0])
            if -0.5 * line_h <= gap <= 1.0 * line_h and overlap > 0:
                pairs.append((gap, li, ni))
    for _, li, ni in sorted(pairs):
        if li in used or ni in used:
            continue
        lt, lb, lc = items[li]
        nt, nb, nc = items[ni]
        box = (min(lb[0], nb[0]), min(lb[1], nb[1]), max(lb[2], nb[2]), max(lb[3], nb[3]))
        tags.append(OcrTag(f"{lt} {nt}", box, min(lc, nc), [lt, nt]))
        used.update((li, ni))

    # Leftover bare ISA letters (number unreadable) are still useful for classification.
    for i in letters:
        if i not in used and len(items[i][0]) >= 2:
            tags.append(OcrTag(items[i][0], items[i][1], items[i][2], [items[i][0]]))
            used.add(i)
    if known:
        for tag in tags:
            tag.text = snap_to_known(tag.text, known)
    return tags


def _distance_to_box(point: tuple[float, float], box) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


# what kind of tag a detected class takes (model classes -> classify_tag kinds)
_CLASS_KIND = {"tank": "equipment", "pump": "equipment", "general": "equipment", "valve": "valve",
               "instrumentation": "instrument"}


def attach_tags(symbol_boxes: dict[str, tuple[float, float, float, float]], tags: list[OcrTag],
                max_gap_factor: float = 1.0, classes: dict[str, str] | None = None) -> dict[str, tuple[OcrTag, str]]:
    """Assign each symbol at most one tag, each tag to at most one symbol.

    Returns {symbol_id: (tag, how)} with how = "inside" | "nearby". Candidates are
    scored by distance from tag center to symbol box (0 when inside); a "nearby" tag
    must be within max_gap_factor x the symbol's larger side. Greedy, best first.

    classes ({symbol id: detected class}): a tag whose kind does not fit the symbol (an
    instrument tag "FI 102" on a tank, "V101" on a valve) counts as half the symbol's size
    further away - before this, "FI 102" next to V101 could take the tank box and leave
    "V101" on nothing.
    """
    from pid_dexpi_aas.dexpi.mapping import classify_tag

    kinds = [classify_tag(t.text)[0] for t in tags]
    candidates = []
    for sid, box in symbol_boxes.items():
        size = max(box[2] - box[0], box[3] - box[1])
        want = _CLASS_KIND.get((classes or {}).get(sid))
        for ti, tag in enumerate(tags):
            d = _distance_to_box(tag.center, box)
            if d <= max_gap_factor * size:
                misfit = want and kinds[ti] and kinds[ti] != want
                candidates.append((d + (0.5 * size if misfit else 0.0), sid, ti))
    result: dict[str, tuple[OcrTag, str]] = {}
    taken: set[int] = set()
    for d, sid, ti in sorted(candidates):
        if sid in result or ti in taken:
            continue
        result[sid] = (tags[ti], "inside" if _distance_to_box(tags[ti].center, symbol_boxes[sid]) == 0 else "nearby")
        taken.add(ti)
    return result
