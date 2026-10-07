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


def assemble_tags(tokens: list[dict], min_conf: float = 0.3) -> list[OcrTag]:
    """Build tags from tokens: one-line tags as-is, stacked letters+number pairs joined."""
    items = [(normalize_token(t["text"]), _box(t), float(t["conf"])) for t in tokens if float(t["conf"]) >= min_conf]
    used: set[int] = set()
    tags: list[OcrTag] = []

    for i, (text, box, conf) in enumerate(items):
        if _FULL_TAG.match(text):
            tags.append(OcrTag(text, box, conf, [text]))
            used.add(i)

    # Stacked pairs: a letters-only token directly above a number token, horizontally
    # overlapping, with a vertical gap under one line height.
    letters = [i for i, (t, _, _) in enumerate(items) if i not in used and _LETTERS.match(t)]
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
