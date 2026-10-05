"""Turn any P&ID picture into dark ink on white - what the model was trained on.

draw.io exports made in dark theme have WHITE lines on a transparent background. Laid on
white (the usual way to drop transparency) the lines vanish and only the filled shapes
remain, so the model drew large, wrong boxes around them. Screenshots of a dark-themed
page have the same problem without the transparency. Both are detected from the ink's
brightness and inverted.

numpy + Pillow only, so ml/pid2graph/infer_sheet.py (ML venv) and the app share it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def _edges(rgb: Image.Image) -> int:
    """How much linework is visible: strong brightness steps between neighbouring pixels."""
    g = np.asarray(rgb.convert("L").reduce(2), dtype=np.int16)
    return int((np.abs(np.diff(g, axis=0)) > 60).sum() + (np.abs(np.diff(g, axis=1)) > 60).sum())


def to_dark_on_white(im: Image.Image) -> Image.Image:
    """RGB image with dark linework on a white background.

    Transparent: laid on white AND on black - the lines are visible on one of the two
    (dark lines on white, light lines on black), the one with more edges; a light-on-black
    result is inverted. (Not judged from the ink's brightness: normal exports carry white
    fills - instrument bubbles, vessels - which made that misfire.) Opaque: a mostly dark
    picture (a dark-page screenshot) is inverted."""
    if im.mode in ("RGBA", "LA", "P"):
        rgba = im.convert("RGBA")
        on = {}
        for colour in ("white", "black"):
            bg = Image.new("RGBA", rgba.size, colour)
            bg.alpha_composite(rgba)
            on[colour] = bg.convert("RGB")
        if _edges(on["black"]) > 1.3 * _edges(on["white"]):
            return ImageOps.invert(on["black"])
        return on["white"]
    rgb = im.convert("RGB")
    small = np.asarray(rgb.resize((max(1, rgb.width // 8), max(1, rgb.height // 8))))
    return ImageOps.invert(rgb) if np.median(small.mean(axis=2)) < 90 else rgb
