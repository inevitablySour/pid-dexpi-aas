"""draw.io coordinates -> rendered-image pixels, found by registration (numpy + PIL only,
so both the main venv and the ML venv can import it)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def register_offset(boxes: list[tuple[float, float, float, float]], png: Path, search: int = 40, f: int = 4):
    """draw.io origin (ox, oy) such that pixel = drawio - origin, by best ink coverage."""
    im = Image.open(png).convert("L")
    W, H = im.size
    ink = (np.asarray(im.resize((W // f, H // f))) < 200).astype(np.float32)
    h, w = ink.shape
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)

    def coverage(ox, oy):
        hit = tot = 0.0
        for bx in boxes:
            a, b = max(int((bx[1] - oy) / f), 0), min(int((bx[3] - oy) / f) + 1, h)
            c, e = max(int((bx[0] - ox) / f), 0), min(int((bx[2] - ox) / f) + 1, w)
            if a < b and c < e:
                hit += ink[a:b, c:e].sum()
                tot += (b - a) * (e - c)
            else:
                tot += 1
        return hit / max(tot, 1.0)

    score, dx, dy = max((coverage(x0 + dx * f, y0 + dy * f), dx, dy)
                        for dy in range(-search, search + 1) for dx in range(-search, search + 1))
    return x0 + dx * f, y0 + dy * f, score
